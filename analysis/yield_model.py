"""Fit a clustering parameter to the observed harvest maps, per procurement lot.

The yield literature models defect COUNTS per die with a negative binomial whose parameter
alpha controls clustering (alpha -> infinity recovers Poisson, i.e. no clustering)
[Koren-Koren-Stapper 1993]. We cannot fit that directly: the SKU fixes the count at exactly
four harvested slices, so the count distribution is not observable. What IS observable is the
conditional SPATIAL arrangement given four.

We therefore fit the natural conditional analogue -- a pairwise-interaction (Strauss-type)
model on the 12 slice positions, conditioned on exactly four being harvested:

    P(S) proportional to  exp( sum_{k in S} beta_k  +  gamma * adj(S) )

with adj(S) = number of index-adjacent pairs in S. beta_k absorbs the per-slice propensity
(the slice-0 asymmetry); gamma is the clustering parameter of interest. gamma > 0 means
harvested slices attract, i.e. defects cluster; gamma = 0 is conditional independence.

The normalising constant is an exact sum over all C(12,4) = 495 patterns, so the likelihood
and its gradient are exact -- no MCMC, no approximation.
"""
import numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')   # 1 = harvested
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
NP = 12
PATS = list(combinations(range(NP), 4))
A = np.zeros((len(PATS), NP))                              # design: slice indicators
ADJ = np.zeros(len(PATS))                                  # adjacency statistic
for i, t in enumerate(PATS):
    A[i, list(t)] = 1
    ADJ[i] = sum(1 for a, b in combinations(t, 2) if abs(a - b) == 1)

def stats(Msub):
    idx = [PATS.index(tuple(np.flatnonzero(r))) for r in Msub]
    return A[idx].sum(0), ADJ[idx].sum(), len(idx)

def fit(Msub, gamma_free=True, iters=4000, lr=0.05):
    n_k, s_adj, N = stats(Msub)
    beta = np.zeros(NP); gamma = 0.0
    for _ in range(iters):
        lin = A @ beta + gamma * ADJ
        lin -= lin.max()
        w = np.exp(lin); w /= w.sum()
        g_beta = n_k / N - A.T @ w
        g_gamma = s_adj / N - ADJ @ w
        beta += lr * g_beta
        if gamma_free: gamma += lr * g_gamma
        beta -= beta.mean()                                # identifiability
    lin = A @ beta + gamma * ADJ; lin -= lin.max()
    w = np.exp(lin); logZ = np.log(w.sum()) + (A @ beta + gamma * ADJ).max()
    ll = (n_k @ beta + gamma * s_adj) - N * logZ
    return beta, gamma, ll, N

def report(name, Msub):
    b1, g1, ll1, N = fit(Msub, True)
    b0, g0, ll0, _ = fit(Msub, False)
    lr_stat = 2 * (ll1 - ll0)                              # 1 df
    # observed vs model-expected adjacency
    lin = A @ b1 + g1 * ADJ; lin -= lin.max(); w = np.exp(lin); w /= w.sum()
    obs_adj = stats(Msub)[1] / N
    lin0 = A @ b0; lin0 -= lin0.max(); w0 = np.exp(lin0); w0 /= w0.sum()
    print(f'\n=== {name}  (n = {N}) ===')
    print(f'  gamma (clustering)      = {g1:+.4f}')
    print(f'  LR test gamma = 0       : chi2(1) = {lr_stat:.1f}  '
          f'-> {"clustering CONFIRMED" if lr_stat > 10.83 else "not significant"} (p<0.001 crit 10.83)')
    print(f'  mean adjacent pairs     : observed {obs_adj:.4f}, '
          f'model {ADJ @ w:.4f}, no-clustering model {ADJ @ w0:.4f}')
    print(f'  per-slice beta (higher = more often harvested):')
    print('    ' + ' '.join(f'{x:+.2f}' for x in b1))
    return g1, lr_stat, N

nid = K[:, 0]; rack = nid // 20
gA, lrA, nA = report('Lot A  (racks 0-21)', M[rack < 22])
gB, lrB, nB = report('Lot B  (racks 22-48)', M[rack >= 22])
gAll, lrAll, nAll = report('Whole fleet', M)

# --- is gamma different between lots? ---
# approximate SE of gamma from the observed-information of the 1-parameter profile
def se_gamma(Msub, beta, gamma):
    lin = A @ beta + gamma * ADJ; lin -= lin.max(); w = np.exp(lin); w /= w.sum()
    var = (ADJ ** 2) @ w - (ADJ @ w) ** 2
    return 1.0 / np.sqrt(len(Msub) * var)
bA, _, _, _ = fit(M[rack < 22]); bB, _, _, _ = fit(M[rack >= 22])
seA, seB = se_gamma(M[rack < 22], bA, gA), se_gamma(M[rack >= 22], bB, gB)
z = (gA - gB) / np.sqrt(seA ** 2 + seB ** 2)
print(f'\n=== lot comparison ===')
print(f'  gamma_A = {gA:+.4f} +/- {seA:.4f}')
print(f'  gamma_B = {gB:+.4f} +/- {seB:.4f}')
print(f'  difference z = {z:+.2f}  -> '
      f'{"lots DIFFER in clustering" if abs(z) > 2.58 else "no significant difference in clustering"}')
print(f'\ninterpretation: gamma > 0 means harvested slices attract each other beyond what the')
print(f'per-slice propensities explain -- the conditional signature of spatially correlated')
print(f'defects. beta captures the slice-0 asymmetry separately, so the two effects do not mix.')
