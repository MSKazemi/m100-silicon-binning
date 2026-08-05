"""Rack-clustered uncertainty for the clustering parameter gamma (review finding R2-M2).

The fit in analysis/yield_model.py treats sockets as i.i.d. draws. They are not: the paper itself
reports that same-rack sockets share more harvested slices than chance (z = +13.9), and each node
contributes two sockets. The i.i.d. standard errors are therefore optimistic.

This resamples RACKS with replacement (the natural cluster: hardware arrives and is installed by
rack), refits gamma on each resample, and reports the clustered SE alongside the i.i.d. one.
"""
import numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
nid = K[:, 0]; rack = nid // 20
NP, B = 12, 400
rng = np.random.default_rng(2026)

PATS = list(combinations(range(NP), 4))
A = np.zeros((len(PATS), NP)); ADJ = np.zeros(len(PATS))
for i, t in enumerate(PATS):
    A[i, list(t)] = 1
    ADJ[i] = sum(1 for a, b in combinations(t, 2) if abs(a - b) == 1)
IDX = {tuple(t): i for i, t in enumerate(PATS)}

def fit_gamma(Msub, iters=2500, lr=0.06):
    idx = [IDX[tuple(np.flatnonzero(r))] for r in Msub]
    n_k = A[idx].sum(0); s_adj = ADJ[idx].sum(); N = len(idx)
    beta = np.zeros(NP); gamma = 0.0
    for _ in range(iters):
        lin = A @ beta + gamma * ADJ; lin -= lin.max()
        w = np.exp(lin); w /= w.sum()
        beta += lr * (n_k / N - A.T @ w)
        gamma += lr * (s_adj / N - ADJ @ w)
        beta -= beta.mean()
    return gamma

def boot(Msub, racksub, label, iid_se):
    g_hat = fit_gamma(Msub)
    racks = np.unique(racksub)
    vals = []
    for _ in range(B):
        pick = rng.choice(racks, len(racks), replace=True)
        rows = np.concatenate([np.flatnonzero(racksub == r) for r in pick])
        vals.append(fit_gamma(Msub[rows]))
    v = np.array(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f'{label:<22} gamma = {g_hat:+.4f}   i.i.d. SE {iid_se:.4f}   '
          f'clustered SE {v.std():.4f}   inflation x{v.std()/iid_se:.2f}')
    print(f'{"":<22} 95% clustered CI [{lo:+.4f}, {hi:+.4f}]')
    return g_hat, v

print(f'rack-clustered bootstrap, B = {B} resamples of {len(np.unique(rack))} racks\n')
gA, vA = boot(M[rack < 22], rack[rack < 22], 'Lot A (racks 0-21)', 0.0437)
gB, vB = boot(M[rack >= 22], rack[rack >= 22], 'Lot B (racks 22-48)', 0.0400)
gAll, vAll = boot(M, rack, 'Whole fleet', 0.0290)

d = gA - gB
# bootstrap the DIFFERENCE by resampling racks within each lot independently
dv = vA[:min(len(vA), len(vB))] - vB[:min(len(vA), len(vB))]
z_iid = d / np.sqrt(0.0437**2 + 0.0400**2)
z_cl = d / dv.std()
print(f'\nlot difference gamma_A - gamma_B = {d:+.4f}')
print(f'  i.i.d.      z = {z_iid:+.2f}')
print(f'  clustered   z = {z_cl:+.2f}   (bootstrap SE {dv.std():.4f})')
print(f'  95% CI on the difference [{np.percentile(dv, 2.5):+.4f}, {np.percentile(dv, 97.5):+.4f}]')
print(f'\n-> the lots {"DO" if abs(z_cl) > 2.58 else "do NOT"} differ in clustering strength '
      f'once socket dependence is accounted for.')
