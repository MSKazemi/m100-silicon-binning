"""Slow path: raw per-core temperature series -> committed per-socket thermal summaries.

The census of thermal_all_nodes.py pooled every pair correlation into one 24x24 matrix and then
discarded the sockets. That made two things impossible: uncertainty at the level at which the
data are dependent (socket, node, rack), and any check of composition, because a pooled cell is
averaged over whichever sockets happen to have both of its positions active. This script keeps
the sockets. It applies exactly the paper's selection and estimator (both sockets of the node
fully configured in the month, 16 cores each with > 20,000 samples, every fifth 20 s sample,
common mode removed, double-centred) and writes one residual correlation matrix per socket,
plus the same at native 20 s cadence and lagged cross-correlations with the lag measured in
real 20 s steps.

Input : THERM_DIR/year_month=<YM>/plugin=ipmi_pub/metric=p{0,1}_core{0..23}_temp/a_0.parquet
        (default THERM_DIR = <repo>/ipmi, gitignored). Extract with
        tar -xf 22-08.tar --wildcards 'year_month=22-08/plugin=ipmi_pub/metric=p?_core*_temp'
Stage 1 converts each partition once into a dense uint8 grid on disk (THERM_DIR/grid_<YM>/,
        nodes x 20 s slots, 0 = missing; 6.3 GB for a 31-day month), resumable per partition.
        Memory-mapped, so the page cache rather than the process holds the data.
Stage 2 computes every socket from the grids.
Output: analysis/thermal_socket_meta_<YM>.csv   one row per socket
        analysis/thermal_socket_c5_<YM>.npy     (nsock, 120) float32, every 5th sample (census)
        analysis/thermal_socket_c1_<YM>.npy     (nsock, 120) float32, every sample
        analysis/thermal_socket_lag_<YM>.npy    (nsock, 2, 9, 4) float32 lagged means
            axis 1: 0 = native cadence, lag in 20 s steps; 1 = decimated rows, lag in rows
                    (the latter is what lag_mechanism.py computed)
            axis 2: lags LAGS;  axis 3: within, samequad, crossquad, autocorrelation
The 120 values are the upper triangle (np.triu_indices(16, 1)) over the socket's sorted
active cores, listed in the meta file.
"""
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('THERM_MONTH', '22-08')
TDIR = Path(os.environ.get('THERM_DIR', root / 'ipmi'))
base = TDIR / f'year_month={YM}' / 'plugin=ipmi_pub'
GRID = TDIR / f'grid_{YM}'
OUT = root / 'analysis'
NC, MINSAMP, DECIM, STEP_MS = 24, 20000, 5, 20_000
LAGS = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
KINDS = ['within', 'samequad', 'crossquad']
NPROC = int(os.environ.get('THERM_PROCS', '6'))

# ------------------------------------------------------------------ selection (as the paper)
dd = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
ok = dd.groupby(['node', 'socket'])['core'].nunique()
full = ok[ok == 16].reset_index().groupby('node').size()
nodes = np.array(sorted(full[full == 2].index), dtype=int)
t0 = pd.Timestamp(dd['day'].min()).tz_convert('UTC').normalize()
ndays = pd.Timestamp(dd['day'].max()).tz_convert('UTC').normalize().day - t0.day + 1
del dd, ok, full
T = ndays * 86400 * 1000 // STEP_MS
T0 = int(t0.value // 1_000_000)                      # ms since epoch
assert (T0 // STEP_MS) % DECIM == 0, 'decimation phase would differ from the paper'
LUT = np.full(1000, -1, dtype=np.int64)
LUT[nodes] = np.arange(len(nodes))
D5 = np.arange(T) % DECIM == 0
PAIRS16 = list(zip(*tl.IU16))


def grid_path(sk, c):
    return GRID / f'p{sk}_core{c}.u8'


def build_grid(sk, c):
    """Partition -> (nodes, T) uint8 grid on disk. Returns (samples off the grid, duplicates)."""
    f = base / f'metric=p{sk}_core{c}_temp' / 'a_0.parquet'
    if not f.exists():
        raise SystemExit(f'missing partition {f} -- refusing to run on a partial extraction')
    g = np.zeros((len(nodes), T), dtype=np.uint8)
    mis = n_in = 0
    for bt in pq.ParquetFile(f).iter_batches(batch_size=4_000_000,
                                             columns=['timestamp', 'value', 'node']):
        nd = pc.cast(bt.column('node'), pa.int32()).to_numpy()
        ts = bt.column('timestamp').cast(pa.int64()).to_numpy()
        val = bt.column('value').to_numpy()
        keep = (nd >= 0) & (nd < 1000)
        row = LUT[np.where(keep, nd, 0)]
        keep &= row >= 0
        off = ts[keep] - T0
        mis += int((off % STEP_MS != 0).sum())
        slot = off // STEP_MS
        inr = (slot >= 0) & (slot < T)
        g[row[keep][inr], slot[inr]] = np.clip(val[keep][inr], 1, 255).astype(np.uint8)
        n_in += int(inr.sum())
    dup = n_in - int(np.count_nonzero(g))
    tmp = grid_path(sk, c).with_suffix('.tmp')
    g.tofile(tmp)
    tmp.rename(grid_path(sk, c))                     # atomic: a killed run leaves no half grid
    return mis, dup


_MM = {}


def grids():
    if not _MM:
        for sk in (0, 1):
            for c in range(NC):
                _MM[(sk, c)] = np.memmap(grid_path(sk, c), dtype=np.uint8, mode='r',
                                         shape=(len(nodes), T))
    return _MM


def lagmat(Z0, v, L):
    """16x16 matrix of mean z_a(t) z_b(t+L) over the t for which rows t and t+L are both valid.
    Z0 holds standardised residuals with zeros on invalid rows, so invalid products vanish and
    only the count needs the mask. Element [a, b] is lag_mechanism.py's lagcorr(z_a, z_b, L)."""
    if L == 0:
        A_, B_, m = Z0, Z0, v
    elif L > 0:
        A_, B_, m = Z0[:-L], Z0[L:], v[:-L] & v[L:]
    else:
        A_, B_, m = Z0[-L:], Z0[:L], v[-L:] & v[:L]
    return (A_.T @ B_) / m.sum()


def standardised(R):
    R = R - R.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
    return R / R.std(0)


def one_node(i):
    G = grids()
    node = int(nodes[i])
    out = []
    for sk in (0, 1):
        X = np.stack([np.asarray(G[(sk, c)][i]) for c in range(NC)], axis=1)   # (T, 24), 0 = missing
        cnt5 = np.count_nonzero(X[D5], axis=0)
        cols = np.flatnonzero(cnt5 > MINSAMP)
        if len(cols) != 16:
            continue
        Xa = X[:, cols]
        valid = (Xa > 0).all(1)
        v5 = valid & D5
        if v5.sum() < MINSAMP:
            continue
        C5, sd5 = tl.residual_corr(Xa[v5])
        if (sd5 == 0).any():
            continue
        C1, _ = tl.residual_corr(Xa[valid])
        kinds = {k: [(a, b) for a, b in PAIRS16 if tl.pair_kind(cols[a], cols[b]) == k]
                 for k in KINDS}
        lag = np.full((2, len(LAGS), 4), np.nan, dtype='float64')
        # native: residual on valid rows, put back on the time grid, lag in real 20 s steps
        Z = np.zeros((len(Xa), 16))
        Z[valid] = standardised(Xa[valid].astype('float64'))
        # decimated: rows of the joined decimated table, lag in rows (lag_mechanism.py)
        Zr = standardised(Xa[v5].astype('float64'))
        vr = np.ones(len(Zr), dtype=bool)
        for li, L in enumerate(LAGS):
            for mode, (Zm, vm) in enumerate(((Z, valid), (Zr, vr))):
                Lm = lagmat(Zm, vm, L)
                for ki, k in enumerate(KINDS):
                    if kinds[k]:
                        lag[mode, li, ki] = np.mean([Lm[a, b] for a, b in kinds[k]])
                lag[mode, li, 3] = np.mean(np.diag(Lm))
        out.append(dict(node=node, socket=sk, active='-'.join(map(str, cols)),
                        rows5=int(v5.sum()), rows1=int(valid.sum()),
                        mean_temp=float(Xa[valid].mean()),
                        n_within=len(kinds['within']), n_samequad=len(kinds['samequad']),
                        n_crossquad=len(kinds['crossquad']),
                        c5=C5[tl.IU16].astype('float32'), c1=C1[tl.IU16].astype('float32'),
                        lag=lag.astype('float32')))
    return out


if __name__ == '__main__':
    print(f'[{YM}] nodes with both sockets fully configured: {len(nodes)}; '
          f'grid {T} x 20 s from {t0.date()}', flush=True)
    GRID.mkdir(parents=True, exist_ok=True)
    tc = time.time(); mis = dup = 0
    for sk in (0, 1):
        for c in range(NC):
            p = grid_path(sk, c)
            if p.exists() and p.stat().st_size == len(nodes) * T:
                continue
            m, d = build_grid(sk, c); mis += m; dup += d
        print(f'  grids for p{sk} ready ({time.time() - tc:.0f}s)', flush=True)
    print(f'  samples off the 20 s grid: {mis};  duplicate slots collapsed: {dup} '
          f'(partitions converted in this run only)', flush=True)
    tc = time.time()
    with get_context('fork').Pool(NPROC) as pool:
        res = [r for chunk in pool.imap(one_node, range(len(nodes)), chunksize=8) for r in chunk]
    res.sort(key=lambda r: (r['node'], r['socket']))           # the paper's order: node, then p0, p1
    print(f'  sockets processed: {len(res)} ({time.time() - tc:.0f}s)', flush=True)
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ('c5', 'c1', 'lag')}
                         for r in res])
    meta['rack'] = meta.node // 20
    meta.to_csv(OUT / f'thermal_socket_meta_{YM}.csv', index=False)
    np.save(OUT / f'thermal_socket_c5_{YM}.npy', np.stack([r['c5'] for r in res]))
    np.save(OUT / f'thermal_socket_c1_{YM}.npy', np.stack([r['c1'] for r in res]))
    np.save(OUT / f'thermal_socket_lag_{YM}.npy', np.stack([r['lag'] for r in res]))
    print(f'  wrote thermal_socket_{{meta,c5,c1,lag}}_{YM}', flush=True)
