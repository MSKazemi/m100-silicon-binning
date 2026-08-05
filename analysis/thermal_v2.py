"""Thermal topology probe, v2 -- addresses reviewer items Mo2, Mo3, Mo4.

v1 used 40 nodes, socket p0 only, in 2020-04 -- a month with just three reporting days.
v2 uses a full 31-day month, BOTH sockets, many more nodes, and reports bootstrap CIs.
It also tests explicitly whether the secondary bump near gap 8 in the pooled curve is real
or an artefact of pooling.
"""
import os, sys, pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent
YM = os.environ.get('THERM_MONTH', '22-08')
base = Path(os.environ.get('THERM_DIR', root.parent / '.therm')) / f'year_month={YM}' / 'plugin=ipmi_pub'
rng = np.random.default_rng(31)
NC, NP = 24, 12
MINSAMP = 8000           # >= ~6 days on the 1-minute aligned grid
NSEL = 120               # nodes sampled; bounded by RAM, stated rather than silent

# Memory-lean ingest: a full month of 48 core metrics does not fit in RAM at once, so read
# each metric ONCE, immediately restrict to the sampled nodes, downsample 3x (correlation is
# unaffected at this cadence), and keep only float32 values keyed by node.
# Select nodes from the daily table, not from core 0: core 0 exists on only ~39% of sockets
# (slice 0 is the most-harvested), so sampling from it would bias the population.
dd = pd.read_parquet(root.parent / 'daily' / f'daily_{YM}.parquet')
ok = dd.groupby(['node', 'socket'])['core'].nunique()
full = ok[ok == 16].reset_index().groupby('node').size()
cand = sorted(full[full == 2].index)           # both sockets fully configured
sel = set(str(cand[i]) for i in rng.choice(len(cand), min(NSEL, len(cand)), replace=False))
print(f'nodes with both sockets fully configured: {len(cand)}, sampled {len(sel)}')
del dd, ok, full

series = {}                       # (socket, core) -> {node: Series}
for s in (0, 1):
    for c in range(NC):
        f = base / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        d = d[d.node.isin(sel)]
        # Align by flooring to a 1-minute grid and averaging. A row-stride downsample would
        # pick different timestamps for different cores and destroy the cross-core join.
        d['t'] = d['timestamp'].dt.floor('1min')
        g = {n: v.groupby('t')['value'].mean().astype('float32')
             for n, v in d.groupby('node', observed=True)}
        series[(s, c)] = g
        del d
print(f'metrics loaded: {len(series)}')

per_node = []            # (node, socket, corr matrix restricted to its 16 cores, idx)
Csum = np.zeros((NC, NC)); Cn = np.zeros((NC, NC))
WITHIN, CROSS1, NEAR, FAR = [], [], [], []
for node in sorted(sel, key=int):
    for s in (0, 1):
        cols = {}
        for c in range(NC):
            g = series.get((s, c))
            if g is None: continue
            v = g.get(node)
            if v is not None and len(v) > MINSAMP:
                cols[c] = v
        if len(cols) != 16: continue
        X = pd.DataFrame(cols).dropna()
        if len(X) < MINSAMP: continue
        idx = np.array(sorted(cols)); R = X.values
        R = R - R.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
        sd = R.std(0)
        if (sd == 0).any(): continue
        C = (R.T @ R) / len(R) / np.outer(sd, sd)
        per_node.append((node, s, C, idx))
        for a in range(16):
            for b in range(16):
                Csum[idx[a], idx[b]] += C[a, b]; Cn[idx[a], idx[b]] += 1
        for a in range(16):
            for b in range(a + 1, 16):
                g = abs(idx[a] - idx[b]); r = C[a, b]
                if g == 1:
                    (WITHIN if idx[a] // 2 == idx[b] // 2 else CROSS1).append(r)
                if g <= 2: NEAR.append((node, s, r))
                if g >= 12: FAR.append((node, s, r))
print(f'sockets analysed: {len(per_node)} (both sockets, {YM}, >= {MINSAMP} samples each)')
CORR = Csum / np.maximum(Cn, 1); np.fill_diagonal(CORR, 1.0)

def boot_ci(v, n=4000):
    v = np.asarray(v); idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print('\n--- the same-distance control (the load-bearing result) ---')
for name, v in [('within-slice sibling (gap 1)', WITHIN), ('cross-slice neighbour (gap 1)', CROSS1)]:
    m, lo, hi = boot_ci(v)
    print(f'  {name:<32} r = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n = {len(v)}')
d = np.array(WITHIN).mean() - np.array(CROSS1).mean()
comb = np.array(WITHIN + CROSS1); nw = len(WITHIN)
null = np.array([(lambda p: p[:nw].mean() - p[nw:].mean())(rng.permutation(comb)) for _ in range(4000)])
print(f'  difference {d:+.4f}, permutation p = {(np.abs(null) >= abs(d)).mean():.5f}')

print('\n--- pooled correlation vs core index distance, with bootstrap CI ---')
gap = {}
for a, b in combinations(range(NC), 2): gap.setdefault(abs(a - b), []).append(CORR[a, b])
rows = []
for g in sorted(gap):
    m, lo, hi = boot_ci(gap[g])
    rows.append((g, m, lo, hi, len(gap[g])))
    print(f'  gap {g:2d}: r = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  (n={len(gap[g])})')
np.save(root / 'thermal_corr_v2.npy', CORR)
pd.DataFrame(rows, columns=['gap', 'r', 'lo', 'hi', 'n']).to_csv(root / 'thermal_gap_v2.csv', index=False)

print('\n--- Mo3: is the secondary bump near gap 8 real? ---')
# per-socket test: does each socket individually show corr(gap 7..9) > corr(gap 4..6)?
cnt = 0; tot = 0; diffs = []
for node, s, C, idx in per_node:
    mid = [C[a, b] for a in range(16) for b in range(a+1, 16) if 4 <= abs(idx[a]-idx[b]) <= 6]
    bump = [C[a, b] for a in range(16) for b in range(a+1, 16) if 7 <= abs(idx[a]-idx[b]) <= 9]
    if mid and bump:
        tot += 1; dd = np.mean(bump) - np.mean(mid); diffs.append(dd)
        if dd > 0: cnt += 1
diffs = np.array(diffs)
m, lo, hi = boot_ci(diffs)
print(f'  sockets where mean r(gap 7-9) > r(gap 4-6): {cnt}/{tot} ({100*cnt/tot:.1f}%)')
print(f'  mean difference {m:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]')
print('  -> a consistent per-socket bump is NOT explainable by pooling across'
      ' heterogeneous active sets;\n     it is what a 2-D floorplan would produce and a 1-D one would not.'
      if lo > 0 else '  -> not resolvable; consistent with pooling noise.')

print('\n--- socket p0 vs p1: does the topology replicate? ---')
for s in (0, 1):
    w = [r for (n, ss, C, idx) in per_node if ss == s
         for a in range(16) for b in range(a+1, 16)
         if abs(idx[a]-idx[b]) == 1 and idx[a]//2 == idx[b]//2 for r in [C[a, b]]]
    x = [r for (n, ss, C, idx) in per_node if ss == s
         for a in range(16) for b in range(a+1, 16)
         if abs(idx[a]-idx[b]) == 1 and idx[a]//2 != idx[b]//2 for r in [C[a, b]]]
    print(f'  p{s}: within-slice {np.mean(w):+.4f} (n={len(w)}) vs '
          f'cross-slice {np.mean(x):+.4f} (n={len(x)})')
