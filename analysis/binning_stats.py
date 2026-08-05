"""Statistical characterisation of disabled core-pair patterns across the M100 fleet."""
import pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent
rng = np.random.default_rng(12345)
NP = 12  # core pairs per socket

def pair_matrix(ym):
    """binary matrix M[socket, pair] = 1 if that core-pair is DISABLED"""
    df = pd.read_parquet(root / f'counts_{ym}.parquet')
    df['pair'] = df['core'] // 2
    rows, keys = [], []
    for (node, sock), g in df.groupby(['node', 'socket']):
        present = set(g['pair'])
        v = np.ones(NP, dtype=np.int8)
        v[sorted(present)] = 0
        if v.sum() == 4:                      # exactly 4 disabled pairs -> clean socket
            rows.append(v); keys.append((node, sock))
    return np.array(rows), keys

M, keys = pair_matrix('20-04')
S = len(M)
print(f'sockets with exactly 4 disabled pairs: {S}')
print(f'possible patterns C(12,4) = {len(list(combinations(range(NP),4)))}')

# ---------- 1. pattern diversity ----------
pats = [tuple(np.flatnonzero(r)) for r in M]
vc = pd.Series(pats).value_counts()
print(f'observed distinct patterns: {vc.nunique() if False else len(vc)}')
print(f'most common pattern {vc.index[0]} seen {vc.iloc[0]}x ({100*vc.iloc[0]/S:.1f}%)')
# expected distinct under uniform sampling of 495 cells with S draws
exp_distinct = 495 * (1 - (1 - 1/495)**S)
print(f'expected distinct if uniform over 495: {exp_distinct:.0f}')

# ---------- 2. marginal disable rate per pair ----------
marg = M.mean(axis=0)
print('\n--- P(pair disabled) by pair index (cores 2k,2k+1) ---')
for k in range(NP):
    print(f'  pair {k:2d} (cores {2*k:2d},{2*k+1:2d}): {100*marg[k]:5.1f}%   {"#"*int(100*marg[k]/2)}')
print(f'  uniform expectation = 4/12 = 33.3%')
obs = M.sum(axis=0); exp = S * 4 / NP
chi2 = ((obs - exp) ** 2 / exp).sum()
print(f'  chi2 uniformity = {chi2:.1f} (df=11, crit_0.001=31.3) -> {"NON-uniform" if chi2>31.3 else "uniform"}')

# ---------- 3. curveball null preserving row AND column sums ----------
def curveball(mat, n_swaps):
    m = mat.copy()
    rows = [set(np.flatnonzero(r)) for r in m]
    R = len(rows)
    for _ in range(n_swaps):
        i, j = rng.integers(0, R, 2)
        if i == j: continue
        a, b = rows[i], rows[j]
        ab, ba = a - b, b - a
        if not ab or not ba: continue
        swap = min(len(ab), len(ba))
        pool = list(ab | ba)
        rng.shuffle(pool)
        na = (a & b) | set(pool[:len(a) - len(a & b)])
        nb = (a | b) - na
        if len(na) == len(a) and len(nb) == len(b):
            rows[i], rows[j] = na, nb
    out = np.zeros_like(mat)
    for r, s in enumerate(rows): out[r, list(s)] = 1
    return out

def cooc(mat):
    m = mat.astype(np.int32)          # int8 matmul overflows at 127 -> must widen
    return m.T @ m

def mean_gap(mat):
    """mean |index distance| between disabled pairs within a socket"""
    tot, n = 0.0, 0
    for r in mat:
        idx = np.flatnonzero(r)
        for x, y in combinations(idx, 2):
            tot += abs(x - y); n += 1
    return tot / n

print('\n--- null model: curveball (preserves per-socket count=4 AND per-pair marginals) ---')
NSIM = 200
obs_c = cooc(M)
obs_gap = mean_gap(M)
null_c = np.zeros((NSIM, NP, NP)); null_gap = np.zeros(NSIM)
cur = M.copy()
for s in range(NSIM):
    cur = curveball(cur, 20000)
    assert (cur.sum(axis=1) == 4).all(), 'curveball broke row sums'
    assert (cur.sum(axis=0) == M.sum(axis=0)).all(), 'curveball broke column sums'
    null_c[s] = cooc(cur); null_gap[s] = mean_gap(cur)

print(f'mean index-gap between disabled pairs: observed {obs_gap:.3f}, '
      f'null {null_gap.mean():.3f} +/- {null_gap.std():.3f}  '
      f'-> z = {(obs_gap-null_gap.mean())/null_gap.std():+.2f}')

z = (obs_c - null_c.mean(axis=0)) / (null_c.std(axis=0) + 1e-9)
print('\nco-disable z-scores (observed vs null), |z|>3 flagged:')
flag = [(i, j, obs_c[i, j], null_c.mean(axis=0)[i, j], z[i, j])
        for i, j in combinations(range(NP), 2) if abs(z[i, j]) > 3]
if not flag:
    print('  none — disabled pairs are mutually independent given the marginals')
for i, j, o, e, zz in sorted(flag, key=lambda t: -abs(t[4]))[:15]:
    print(f'  pair {i:2d} & {j:2d}: obs {o:4.0f} vs null {e:6.1f}  z={zz:+.1f}')

# ---------- 4. are the two sockets of a node independent? ----------
d = {}
for (node, sock), r in zip(keys, M): d[(node, sock)] = r
both = [n for (n, s) in d if s == 0 and (n, 1) in d]
same = sum(1 for n in both if tuple(np.flatnonzero(d[(n,0)])) == tuple(np.flatnonzero(d[(n,1)])))
ov = np.array([len(set(np.flatnonzero(d[(n,0)])) & set(np.flatnonzero(d[(n,1)]))) for n in both])
print(f'\n--- socket p0 vs p1 within the same node (n={len(both)}) ---')
print(f'identical disabled set: {same} ({100*same/len(both):.1f}%)')
print(f'mean overlap: {ov.mean():.3f} pairs')
# null: independent draws from observed pattern distribution
sim = []
arr0 = [set(np.flatnonzero(d[(n,0)])) for n in both]
arr1 = [set(np.flatnonzero(d[(n,1)])) for n in both]
for _ in range(200):
    perm = rng.permutation(len(both))
    sim.append(np.mean([len(arr0[i] & arr1[p]) for i, p in enumerate(perm)]))
sim = np.array(sim)
print(f'null (shuffled pairing): {sim.mean():.3f} +/- {sim.std():.3f} -> z = {(ov.mean()-sim.mean())/sim.std():+.2f}')

# ---------- 5. does the pattern depend on node id (rack position)? ----------
nid = np.array([int(n) for (n, s) in keys])

def mwu_z(x, y):
    """Mann-Whitney U, normal approximation with tie correction."""
    allv = np.concatenate([x, y])
    r = pd.Series(allv).rank().values
    n1, n2 = len(x), len(y)
    U = r[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, cnt = np.unique(allv, return_counts=True)
    tie = (cnt ** 3 - cnt).sum()
    n = n1 + n2
    sd = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    return (U - mu) / sd

print('\n--- pattern vs node id (proxy for rack/chassis position) ---')
zs = []
for k in range(NP):
    dis = nid[M[:, k] == 1]; ok = nid[M[:, k] == 0]
    z = mwu_z(dis, ok); zs.append(z)
    print(f'  pair {k:2d}: median nodeid disabled {np.median(dis):6.1f} vs enabled {np.median(ok):6.1f}  z={z:+.2f}')
print(f'  max |z| = {max(abs(np.array(zs))):.2f} (Bonferroni crit for 12 tests @0.05 = 2.87)')
