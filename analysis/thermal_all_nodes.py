"""Thermal analysis over EVERY node, not a sample.

thermal_v2.py used 120 nodes and the lag/grid studies 60-90, bounded by RAM: holding 48
full-month core series for ~980 nodes at once does not fit. That is an implementation limit, not
a scientific one, and a referee is entitled to ask whether the sample drove the result.

This removes the limit by processing nodes in chunks: for each chunk we read the 48 metrics
restricted to those nodes, reduce each socket to its residual correlation matrix, accumulate the
pooled sums, and discard. Memory is bounded by the chunk, not the fleet. Everything is computed
at native 20 s cadence, with the common mode removed per socket, as in the paper.
"""
import os, gc, pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('THERM_MONTH', '22-08')
base = Path(os.environ.get('THERM_DIR', root / '.therm')) / f'year_month={YM}' / 'plugin=ipmi_pub'
NC, NP, CHUNK, MINSAMP = 24, 12, 120, 100000

dd = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
ok = dd.groupby(['node', 'socket'])['core'].nunique()
full = ok[ok == 16].reset_index().groupby('node').size()
allnodes = sorted(full[full == 2].index)
del dd, ok, full
print(f'nodes with both sockets fully configured in {YM}: {len(allnodes)}  (processing ALL)')

Csum = np.zeros((NC, NC)); Cn = np.zeros((NC, NC))
WITHIN, CROSS1 = [], []
per_socket = []
nsock = 0
for ci in range(0, len(allnodes), CHUNK):
    chunk = set(str(x) for x in allnodes[ci:ci + CHUNK])
    series = {}
    for s in (0, 1):
        for c in range(NC):
            f = base / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
            if not f.exists(): continue
            d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
            d = d[d.node.isin(chunk)]
            series[(s, c)] = {n: v.set_index('timestamp')['value'].astype('float32')
                              for n, v in d.groupby('node', observed=True)}
            del d
    for node in sorted(chunk, key=int):
        for s in (0, 1):
            cols = {c: series.get((s, c), {}).get(node) for c in range(NC)}
            cols = {c: v for c, v in cols.items() if v is not None and len(v) > MINSAMP}
            if len(cols) != 16: continue
            X = pd.DataFrame(cols).dropna()
            if len(X) < MINSAMP: continue
            idx = np.array(sorted(cols)); R = X.values.astype('float64')
            R = R - R.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
            sd = R.std(0)
            if (sd == 0).any(): continue
            C = (R.T @ R) / len(R) / np.outer(sd, sd)
            nsock += 1
            w = [C[a, b] for a in range(16) for b in range(a+1, 16)
                 if abs(idx[a]-idx[b]) == 1 and idx[a]//2 == idx[b]//2]
            x = [C[a, b] for a in range(16) for b in range(a+1, 16)
                 if abs(idx[a]-idx[b]) == 1 and idx[a]//2 != idx[b]//2]
            WITHIN += w; CROSS1 += x
            if w and x: per_socket.append(np.mean(w) - np.mean(x))
            for a in range(16):
                for b in range(16):
                    Csum[idx[a], idx[b]] += C[a, b]; Cn[idx[a], idx[b]] += 1
            del X, R, C
    del series; gc.collect()
    print(f'  ...{min(ci+CHUNK, len(allnodes))}/{len(allnodes)} nodes, {nsock} sockets', flush=True)

CORR = Csum / np.maximum(Cn, 1); np.fill_diagonal(CORR, 1.0)
np.save(root / 'analysis' / 'thermal_corr_allnodes.npy', CORR)
rng = np.random.default_rng(5)
def ci(v, n=4000):
    v = np.asarray(v); m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print(f'\n=== ALL-NODE thermal result ({nsock} sockets) ===')
for lab, v in [('within-slice sibling', WITHIN), ('cross-slice neighbour', CROSS1)]:
    m, lo, hi = ci(v)
    print(f'  {lab:<24} r = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n = {len(v):,}')
d = np.mean(WITHIN) - np.mean(CROSS1)
comb = np.array(WITHIN + CROSS1); nw = len(WITHIN)
null = np.array([(lambda q: q[:nw].mean() - q[nw:].mean())(rng.permutation(comb))
                 for _ in range(4000)])
ps = np.array(per_socket)
print(f'  contrast {d:+.4f}, permutation p = {(np.abs(null) >= abs(d)).mean():.5f}')
print(f'  sockets where within > cross: {(ps>0).sum()}/{len(ps)} ({100*(ps>0).mean():.1f}%)')

S = np.zeros((NP, NP))
for a in range(NP):
    for b in range(NP):
        if a != b: S[a, b] = np.mean([CORR[2*a+i, 2*b+j] for i in (0, 1) for j in (0, 1)])
obs = np.array([S[a, b] for a, b in combinations(range(NP), 2)])
dist = np.array([abs(a-b) for a, b in combinations(range(NP), 2)])
print(f'  1x12 floorplan fit (cross-slice pairs only): {np.corrcoef(dist, obs)[0,1]:+.3f}')
