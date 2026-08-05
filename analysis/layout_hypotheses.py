"""Which physical arrangement of the 12 slices best explains the thermal coupling?

Free MDS on the pooled matrix is noisy, so instead we score explicit candidate floorplans:
predicted physical distance vs observed thermal dissimilarity, using only the pairs that
are least affected by common-mode-removal bias (both cores of DIFFERENT slices).
"""
import numpy as np
from pathlib import Path
from itertools import combinations, permutations

root = Path(__file__).resolve().parent
import os
CORR = np.load(root / os.environ.get('THERM_MATRIX', 'thermal_corr_24x24.npy'))
NC, NP = 24, 12
rng = np.random.default_rng(5)

# slice-level thermal similarity: mean corr over the 4 cross-slice core pairs
S = np.zeros((NP, NP))
for a in range(NP):
    for b in range(NP):
        if a == b: continue
        S[a, b] = np.mean([CORR[2*a+i, 2*b+j] for i in (0, 1) for j in (0, 1)])
obs = np.array([S[a, b] for a, b in combinations(range(NP), 2)])

def score(coords, label):
    d = np.array([np.linalg.norm(np.subtract(coords[a], coords[b]))
                  for a, b in combinations(range(NP), 2)])
    r = np.corrcoef(d, obs)[0, 1]
    print(f'  {label:<34} corr(distance, thermal r) = {r:+.3f}')
    return r

print('Candidate floorplans (more negative = better: closer slices are hotter together)')
cands = {
    '1 x 12 linear':        [(k, 0) for k in range(NP)],
    '2 x 6 row-major':      [(k % 6, k // 6) for k in range(NP)],
    '2 x 6 boustrophedon':  [(k % 6 if k < 6 else 5 - (k - 6), k // 6) for k in range(NP)],
    '3 x 4 row-major':      [(k % 4, k // 4) for k in range(NP)],
    '4 x 3 row-major':      [(k % 3, k // 3) for k in range(NP)],
    '6 x 2 row-major':      [(k % 2, k // 2) for k in range(NP)],
    '2 x 6 mirrored halves':[(k, 0) if k < 6 else (11 - k, 1) for k in range(NP)],
}
res = {lab: score(c, lab) for lab, c in cands.items()}
best = min(res, key=res.get)
print(f'\nbest candidate: {best} ({res[best]:+.3f})')

# permutation baseline: how good is a RANDOM assignment of slices to 1x12 positions?
lin = [(k, 0) for k in range(NP)]
null = []
for _ in range(4000):
    p = rng.permutation(NP)
    coords = [lin[p[k]] for k in range(NP)]
    d = np.array([np.linalg.norm(np.subtract(coords[a], coords[b]))
                  for a, b in combinations(range(NP), 2)])
    null.append(np.corrcoef(d, obs)[0, 1])
null = np.array(null)
r_lin = res['1 x 12 linear']
print(f'\nrandom-permutation null for 1x12: mean {null.mean():+.3f}, sd {null.std():.3f}')
print(f'  identity ordering r = {r_lin:+.3f} -> z = {(r_lin-null.mean())/null.std():+.2f}, '
      f'empirical p = {(null <= r_lin).mean():.4f}')

# exhaustive search over orderings is 12!/2 -- too many; do a greedy + restarts hill-climb
def order_score(p):
    coords = [(int(np.where(p == k)[0][0]), 0) for k in range(NP)]
    d = np.array([abs(coords[a][0] - coords[b][0]) for a, b in combinations(range(NP), 2)])
    return np.corrcoef(d, obs)[0, 1]
bestp, bestv = None, 1
for _ in range(200):
    p = rng.permutation(NP); v = order_score(p)
    improved = True
    while improved:
        improved = False
        for i, j in combinations(range(NP), 2):
            q = p.copy(); q[i], q[j] = q[j], q[i]
            w = order_score(q)
            if w < v: p, v, improved = q, w, True
    if v < bestv: bestp, bestv = p.copy(), v
print(f'\nbest 1-D ordering found by hill-climbing: r = {bestv:+.3f}')
print(f'  slice order = {list(bestp)}')
print(f'  identity    = {list(range(NP))}')
print(f'  Spearman(best order, identity) = '
      f'{np.corrcoef(np.argsort(bestp), np.arange(NP))[0,1]:+.3f}')

# ------------------------------------------------------------------ figure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
FIG = root.parent / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5.4, 2.6))
labs = list(res) + ['best ordering (hill-climb)']
vals = [res[l] for l in res] + [bestv]
cols = ['#2b6cb0'] * len(res) + ['#718096']
cols[list(res).index('1 x 12 linear')] = '#c53030'
y = np.arange(len(labs))
ax.barh(y, vals, color=cols)
ax.axvspan(null.mean() - 2*null.std(), null.mean() + 2*null.std(), color='k', alpha=.12,
           label='random-ordering null ($\\pm2\\sigma$)')
ax.axvline(null.mean(), color='k', lw=.8)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
ax.set_xlabel('corr(physical distance, thermal coupling)   — more negative = better fit')
ax.legend(fontsize=6.5, frameon=False, loc='lower left')
ax.invert_yaxis()
fig.savefig(FIG / 'f5_layout_hypotheses.pdf'); plt.close(fig)
old = FIG / 'f5_mds_layout.pdf'
if old.exists(): old.unlink()
print('\nwrote', FIG / 'f5_layout_hypotheses.pdf')
