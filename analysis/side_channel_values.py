"""Does compact renumbering really close the channel? A test of the temperature VALUES.

side_channel.py shows that the presence channel -- which sensors exist -- is closed only by
removing the per-core position: socket aggregates, or renumbering the surviving cores
compactly to 0..15. Compact renumbering removes the presence channel by construction, but the
paper's own thermal probe shows that per-core temperatures carry physical adjacency: after
removing the socket's common mode, neighbouring cores co-vary more than distant ones. If a
harvested slice sits between two consecutive surviving slices, those two slices are physically
farther apart than their compact indices suggest, and their residual coupling should be weaker.
This script measures how much of the map that leaks when only compact indices are published.

For each socket with a 16-core map stable over month YM we build the native 20 s series of its
16 active cores, compute the residual (double-centred) correlation matrix exactly as the
thermal probe does, and look only at what a compact-renumbering publisher would still release:
the 8 surviving slices in order, with the harvested ones removed. For each of the 7 boundaries
between consecutive surviving slices, the feature is the mean residual correlation of the 2x2
cross-slice core pairs, and the label is whether >= 1 harvested slice lies in between.

  1. AUC of that feature for "harvested slice in between", node-clustered bootstrap CI.
  2. Two decoders over all C(12,4) = 495 candidate maps, fitted on half of the nodes and
     scored on the other half (2-fold by node): (a) a Gaussian likelihood of the 7 boundary
     features given each boundary's gap; (b) template matching of the socket's whole 16x16
     residual matrix against the fleet's 24x24 physical-core template restricted to each
     candidate's cores (this uses 2-D floorplan effects too). Each with a uniform or a
     per-slice-marginal prior; reported against the prior alone.

Input: extracted core partitions (argv[1]) as for side_channel_raw.py, or the monthly tar in
M100_RAW. Output: analysis/side_channel_values.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('SC_MONTH', '22-08')
RAW = Path(os.environ.get('M100_RAW', '/home/mohsen/exadata/raw'))
NSEL = int(os.environ.get('SC_NODES', '300'))
MINROWS = 100000                      # >= ~23 days of complete 20 s rows
rng = np.random.default_rng(41)
t_start = time.time()

tmp = None
if len(sys.argv) > 1:
    base = Path(sys.argv[1])
else:
    tmp = Path(tempfile.mkdtemp(prefix='sc_val_', dir=os.environ.get('SC_TMP')))
    subprocess.run(['tar', '-xf', str(RAW / f'{YM}.tar'), '-C', str(tmp), '--wildcards',
                    '--touch', '--warning=no-timestamp',
                    f'year_month={YM}/plugin=ipmi_pub/metric=p0_core*_temp',
                    f'year_month={YM}/plugin=ipmi_pub/metric=p1_core*_temp'],
                   capture_output=True, text=True)
    base = tmp
part = base / f'year_month={YM}' / 'plugin=ipmi_pub'
if sum((part / f'metric=p{s}_core{c}_temp' / 'a_0.parquet').exists()
       for s in (0, 1) for c in range(24)) != 48:
    sys.exit('not all 48 core partitions present; refusing to run')

# ------------------------------------------------------------------ sockets and maps
d = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
d['day'] = pd.to_datetime(d['day'], utc=True)
days = sorted(d.day.unique())
t0 = int(pd.Timestamp(days[0]).value // 10**6)
MS = len(days) * 4320
sets = d.groupby(['node', 'socket', 'day'])['core'].apply(frozenset)
stable = sets.groupby(['node', 'socket']).agg(lambda s: s.nunique() == 1 and len(s.iloc[0]) == 16)
maps = sets.groupby(['node', 'socket']).first()[stable]
nodes_ok = sorted({n for n, _ in maps.index})
sel = sorted(rng.choice(nodes_ok, min(NSEL, len(nodes_ok)), replace=False).tolist())
socks = [(n, s) for (n, s) in maps.index if n in set(sel)]
print(f'month {YM}: {len(maps):,} sockets with a stable 16-core map; sampled {len(sel)} nodes '
      f'-> {len(socks)} sockets')

# ------------------------------------------------------------------ native 20 s series
NN = int(d.node.max()) + 1
X = np.zeros((len(socks), MS, 16), dtype=np.uint8)                   # 0 = missing
row_of = np.full((NN, 2), -1, dtype=np.int64)
col_of = np.full((NN, 2, 24), -1, dtype=np.int64)
for i, (n, s) in enumerate(socks):
    row_of[n, s] = i
    for j, c in enumerate(sorted(maps[(n, s)])):
        col_of[n, s, c] = j
for s in (0, 1):
    for c in range(24):
        nl = [n for (n, ss) in socks if ss == s and col_of[n, s, c] >= 0]
        if not nl:
            continue
        pf = pq.ParquetFile(part / f'metric=p{s}_core{c}_temp' / 'a_0.parquet')
        for bt in pf.iter_batches(batch_size=2_000_000, columns=['timestamp', 'value', 'node']):
            nd = pc.cast(bt.column('node'), 'int32').to_numpy()
            keep = (nd >= 0) & (nd < NN)
            keep[keep] = col_of[nd[keep], s, c] >= 0
            if not keep.any():
                continue
            ts = (bt.column('timestamp').cast('int64').to_numpy()[keep] - t0) // 20000
            val = bt.column('value').to_numpy()[keep]
            nd = nd[keep]
            ok = (ts >= 0) & (ts < MS)
            X[row_of[nd[ok], s], ts[ok], col_of[nd[ok], s, c]] = np.clip(val[ok], 1, 255)
print(f'series loaded ({time.time() - t_start:.0f} s)')
if tmp is not None:
    shutil.rmtree(tmp, ignore_errors=True)


def resid_corr(A):
    """Residual correlation: remove the per-timestamp socket mean, then the per-core mean."""
    R = A.astype(np.float64)
    R -= R.mean(1, keepdims=True)
    R -= R.mean(0, keepdims=True)
    sd = R.std(0)
    if (sd == 0).any():
        return None
    return (R.T @ R) / len(R) / np.outer(sd, sd)


rows = []                                                  # one row per socket
for i, k in enumerate(socks):
    A = X[i]
    A = A[(A > 0).all(1)]
    if len(A) < MINROWS:
        continue
    C = resid_corr(A)
    if C is None:
        continue
    act = sorted({c // 2 for c in maps[k]})                # 8 surviving slices, in order
    feats, gaps = [], []
    for j in range(7):
        a = [2 * j, 2 * j + 1]; b = [2 * j + 2, 2 * j + 3]   # compact core indices
        feats.append(C[np.ix_(a, b)].mean())
        gaps.append(act[j + 1] - act[j] - 1)
    rows.append({'node': k[0], 'socket': k[1], 'feat': feats, 'gap': gaps, 'C': C,
                 'cores': sorted(maps[k]),
                 'lead': act[0], 'trail': 11 - act[-1], 'harv': sorted(set(range(12)) - set(act)),
                 'n_rows': int(len(A))})
del X
print(f'sockets analysed: {len(rows)} (>= {MINROWS:,} complete rows each)')

F = np.array([r['feat'] for r in rows]); G = np.array([r['gap'] for r in rows])
node_of = np.array([r['node'] for r in rows])
y = (G > 0).ravel(); x = F.ravel(); nrep = np.repeat(node_of, 7)


def auc(score, label):
    """P(score of a negative > score of a positive): low coupling flags a harvested gap."""
    from scipy.stats import rankdata
    r = rankdata(-score)
    npos, nneg = label.sum(), (~label).sum()
    return (r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)


A0 = auc(x, y)
un = np.unique(node_of)
boot = []
for _ in range(2000):
    pick = rng.choice(un, len(un), replace=True)
    idx = np.concatenate([np.flatnonzero(nrep == n) for n in pick])
    boot.append(auc(x[idx], y[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print('\n=== 1. does residual coupling reveal a harvested slice between surviving slices? ===')
for g in range(5):
    v = x[G.ravel() == g]
    if len(v):
        print(f'  gap {g} harvested slice(s): mean coupling {v.mean():+.4f}  (n={len(v):,})')
print(f'  AUC (gap >= 1 vs gap 0) = {A0:.3f}, node-clustered 95% bootstrap CI [{lo:.3f}, {hi:.3f}]')

# ------------------------------------------------------------------ 2. decoder
CAND = [c for c in combinations(range(12), 4)]
cand_gaps, cand_harv = [], []
for h in CAND:
    act = [s for s in range(12) if s not in h]
    cand_gaps.append([act[j + 1] - act[j] - 1 for j in range(7)])
    cand_harv.append(h)
cand_gaps = np.array(cand_gaps)                             # (495, 7)
truth_idx = np.array([CAND.index(tuple(r['harv'])) for r in rows])

fold = np.isin(node_of, un[rng.permutation(len(un))[: len(un) // 2]]).astype(int)
hits = {'uniform': 0, 'marginal': 0, 'prior_only': 0, 'template_uniform': 0,
        'template_marginal': 0}
overlap = {k: 0 for k in hits}
rank_true = []
cand_cores = [np.array([c for c in range(24) if c // 2 not in h]) for h in CAND]
iu = np.triu_indices(16, 1)
cand_set = [set(h) for h in CAND]
for k in (0, 1):
    tr, te = fold != k, fold == k
    mu = np.zeros(5); sd = np.zeros(5)
    for g in range(5):
        v = F[tr][G[tr] == g]
        v = v if len(v) > 5 else F[tr][G[tr] >= 1]
        mu[g], sd[g] = v.mean(), v.std() + 1e-6
    pr = np.full(12, 1e-3)
    for r, t in zip(rows, tr):
        if t:
            pr[r['harv']] += 1
    pr /= tr.sum()
    logprior = np.array([np.log(pr[list(h)]).sum() + np.log(1 - pr[[s for s in range(12) if s not in h]]).sum()
                         for h in CAND])
    Tsum = np.zeros((24, 24)); Tn = np.zeros((24, 24)); Tss = np.zeros((24, 24))
    for r, t in zip(rows, tr):
        if t:
            c = np.array(r['cores'])
            Tsum[np.ix_(c, c)] += r['C']; Tss[np.ix_(c, c)] += r['C'] ** 2; Tn[np.ix_(c, c)] += 1
    Tm = Tsum / np.maximum(Tn, 1)
    Tv = np.maximum(Tss / np.maximum(Tn, 1) - Tm ** 2, 1e-4)
    tmpl = np.stack([Tm[np.ix_(cc, cc)][iu] for cc in cand_cores])       # (495, 120)
    tvar = np.stack([Tv[np.ix_(cc, cc)][iu] for cc in cand_cores])
    for i in np.flatnonzero(te):
        ll = (-0.5 * ((F[i][None, :] - mu[cand_gaps]) / sd[cand_gaps]) ** 2
              - np.log(sd[cand_gaps])).sum(1)
        lt = (-0.5 * (rows[i]['C'][iu][None, :] - tmpl) ** 2 / tvar - 0.5 * np.log(tvar)).sum(1)
        pick = {'uniform': np.argmax(ll), 'marginal': np.argmax(ll + logprior),
                'prior_only': np.argmax(logprior), 'template_uniform': np.argmax(lt),
                'template_marginal': np.argmax(lt + logprior)}
        for key, j in pick.items():
            hits[key] += int(j == truth_idx[i])
            overlap[key] += len(cand_set[j] & cand_set[truth_idx[i]])
        rank_true.append(int((lt + logprior > (lt + logprior)[truth_idx[i]]).sum()) + 1)
N = len(rows)
print('\n=== 2. decode the whole map from the values alone (2-fold by node) ===')
print('   decoder                              exact map    harvested slices right (of 4)')
for key, lab in [('prior_only', 'per-slice prior alone (no data)'),
                 ('uniform', 'boundary coupling, uniform prior'),
                 ('marginal', 'boundary coupling, per-slice prior'),
                 ('template_uniform', 'full-matrix template, uniform prior'),
                 ('template_marginal', 'full-matrix template, per-slice prior')]:
    print(f'   {lab:36s} {hits[key] / N:6.3f} ({hits[key]:3d}/{N})   {overlap[key] / N:5.2f}')
print(f'  median rank of the true map (template + prior) among 495: {np.median(rank_true):.0f}')
print('  (presence, as published: exact for every socket.)')

res = {'month': YM, 'nodes_sampled': len(sel), 'sockets': N,
       'auc_gap_vs_adjacent': A0, 'auc_ci95_node_bootstrap': [lo, hi],
       'mean_coupling_by_gap': {int(g): float(x[G.ravel() == g].mean())
                                for g in range(5) if (G.ravel() == g).any()},
       'n_by_gap': {int(g): int((G.ravel() == g).sum()) for g in range(5)},
       'decoder_exact': {k: v / N for k, v in hits.items()},
       'decoder_mean_harvested_slices_right_of_4': {k: v / N for k, v in overlap.items()},
       'decoder_median_rank_of_truth': float(np.median(rank_true))}
# The wall-clock runtime is printed, never written into the JSON: this file is pinned in
# DATA.sha256, and a timing field would make the checksum change on every re-run even when every
# number is bit-identical.
(root / 'analysis' / 'side_channel_values.json').write_text(json.dumps(res, indent=1))
print(f'\nwrote analysis/side_channel_values.json ({round(time.time() - t_start, 1)} s)')
