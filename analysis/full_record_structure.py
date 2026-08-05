"""Re-run every STRUCTURAL result over the full 31-month record (reviewer finding M3).

Previously §IV-B..IV-G were estimated from a single month (2020-04). Since 89.5% of sockets
never change configuration, each socket is summarised by the MODAL harvest map over all its
clean socket-days; sockets that do change contribute their dominant configuration. Also emits
per-month marginals so staticity can be shown rather than asserted.
"""
import pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(2024)
NP, EXPECT = 12, 4320

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
f = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, f], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
dfc = df.merge(clean, on=['node', 'socket', 'day'])
print(f'clean socket-days: {len(clean):,}   sockets: {clean.groupby(["node","socket"]).ngroups}')

sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()
sets['pair'] = sets['c'].apply(lambda s: frozenset(x // 2 for x in s))

# ---------- modal harvest map per socket, over the whole record ----------
rows, keys, ndays = [], [], []
for (n, s), g in sets.groupby(['node', 'socket']):
    vc = g['pair'].value_counts()
    act = vc.index[0]
    v = np.ones(NP, dtype=np.int8); v[sorted(act)] = 0
    if v.sum() != 4: continue
    rows.append(v); keys.append((int(n), s)); ndays.append(vc.iloc[0])
M = np.array(rows); nid = np.array([k[0] for k in keys]); rack = nid // 20
print(f'sockets with a well-defined modal map: {len(M)}  '
      f'(median {int(np.median(ndays))} days supporting the mode)')

# ---------- pattern diversity + marginals ----------
pats = pd.Series([tuple(np.flatnonzero(r)) for r in M])
print(f'\ndistinct patterns: {pats.nunique()} of {len(list(combinations(range(NP),4)))}')
print(f'most common: {pats.value_counts().index[0]} at {100*pats.value_counts().iloc[0]/len(M):.1f}%')
marg = M.mean(0)
print('\nmarginals %: ' + ' '.join(f'{100*x:.1f}' for x in marg))
chi2 = (((M.sum(0) - len(M)*4/NP) ** 2) / (len(M)*4/NP)).sum()
print(f'chi2 uniformity = {chi2:.1f} (df=11)')

# ---------- curveball clustering ----------
def curveball(mat, n):
    rs = [set(np.flatnonzero(r)) for r in mat]; R = len(rs)
    for _ in range(n):
        i, j = rng.integers(0, R, 2)
        if i == j: continue
        A, B = rs[i], rs[j]
        if not (A - B) or not (B - A): continue
        pool = list((A - B) | (B - A)); rng.shuffle(pool)
        na = (A & B) | set(pool[:len(A) - len(A & B)]); nb = (A | B) - na
        if len(na) == len(A) and len(nb) == len(B): rs[i], rs[j] = na, nb
    out = np.zeros_like(mat)
    for r, s in enumerate(rs): out[r, list(s)] = 1
    return out
def cooc(m): mm = m.astype(np.int32); return mm.T @ mm
def gap(m):
    tot = k = 0
    for r in m:
        idx = np.flatnonzero(r)
        for x, y in combinations(idx, 2): tot += abs(x - y); k += 1
    return tot / k

obs_c, obs_g = cooc(M), gap(M)
NS = 200; nc = np.zeros((NS, NP, NP)); ng = np.zeros(NS); cur = M.copy()
for i in range(NS):
    cur = curveball(cur, 20000)
    assert (cur.sum(0) == M.sum(0)).all() and (cur.sum(1) == 4).all()
    nc[i] = cooc(cur); ng[i] = gap(cur)
Z = (obs_c - nc.mean(0)) / (nc.std(0) + 1e-9)
print(f'\nmean index gap: obs {obs_g:.3f} vs null {ng.mean():.3f}+/-{ng.std():.3f} '
      f'-> z = {(obs_g-ng.mean())/ng.std():+.2f}')
# Bonferroni over the C(12,2)=66 pair tests (reviewer finding Mo1)
crit = 3.83  # two-sided alpha=0.05/66 -> |z|>3.83
sig = [(i, j, Z[i, j]) for i, j in combinations(range(NP), 2) if abs(Z[i, j]) > crit]
print(f'co-harvest pairs significant at Bonferroni |z|>{crit} ({len(sig)} of 66):')
for i, j, z in sorted(sig, key=lambda x: -abs(x[2]))[:10]:
    print(f'   {i:2d}&{j:2d}  z={z:+6.1f}   |dk|={abs(i-j)}')
adj = [abs(i-j) for i, j, z in sig if z > 0]
print(f'   positive-excess pairs with |dk|<=2: {sum(1 for d in adj if d<=2)}/{len(adj)}')

# ---------- lot changepoint ----------
best = max(((cp, abs((M[rack < cp, 0].mean() - M[rack >= cp, 0].mean()) /
                     np.sqrt(M[rack < cp, 0].var()/max((rack < cp).sum(),1) +
                             M[rack >= cp, 0].var()/max((rack >= cp).sum(),1))))
            for cp in range(3, 46)), key=lambda x: x[1])
cp, tstat = best
A, B = M[rack < cp], M[rack >= cp]
print(f'\nlot changepoint at rack {cp} (node {cp*20}), Welch t = {tstat:.1f}')
print(f'  Lot A n={len(A)}  P(slice0)={100*A[:,0].mean():.1f}%  patterns={len(set(map(tuple,map(np.flatnonzero,A))))}')
print(f'  Lot B n={len(B)}  P(slice0)={100*B[:,0].mean():.1f}%  patterns={len(set(map(tuple,map(np.flatnonzero,B))))}')

# ---------- per-month marginals (staticity evidence + figure data) ----------
sets['month'] = sets['day'].dt.strftime('%y-%m')
recs = []
for m, g in sets.groupby('month'):
    mm = np.ones((len(g), NP), dtype=np.int8)
    for i, p in enumerate(g['pair'].values): mm[i, sorted(p)] = 0
    recs.append([m, len(g)] + list(mm.mean(0)))
PM = pd.DataFrame(recs, columns=['month', 'socketdays'] + [f's{k}' for k in range(NP)])
PM.to_csv(root / 'analysis' / 'per_month_marginals.csv', index=False)
sd = PM[[f's{k}' for k in range(NP)]].std(0)
print(f'\nper-month marginal stability: max across-month SD over the 12 slices = '
      f'{100*sd.max():.2f} pp (slice {int(sd.idxmax()[1:])})')
print(f'   month-to-month range for slice 0: '
      f'{100*PM.s0.min():.1f}% .. {100*PM.s0.max():.1f}%')
np.save(root / 'analysis' / 'harvest_maps_full.npy', M)
np.save(root / 'analysis' / 'harvest_keys_full.npy', np.array(keys))
print('\nsaved harvest_maps_full.npy, harvest_keys_full.npy, per_month_marginals.csv')
