"""Posterior-predictive checks for the pairwise-interaction model.

A likelihood-ratio test says gamma is not zero. It does not say the model is adequate, and a
model that fits the adjacency statistic it was fitted on while failing everywhere else would
still pass that test. So we simulate from the fitted model and compare it to the data on
quantities it was NOT fitted to:

  1. the full pattern-frequency distribution (495 cells),
  2. the pairwise co-harvest matrix, as a residual heatmap,
  3. the distribution of the maximum gap between harvested slices,
  4. the number of distinct patterns realised.

The model is fitted on the per-slice counts and the adjacency total, so (1)-(4) are genuine
out-of-statistic checks. Where the model misses, the miss is the interesting part: it bounds how
much of the structure "propensity plus adjacency" actually explains, which is exactly the
question behind the paper's claim that the slice-0 asymmetry and the spatial clustering are
distinct phenomena.

This also speaks to a caveat the paper must keep: a positive gamma is a statement about residual
spatial association, NOT evidence that a defect process rather than a binning policy produced it.
Both generate clustering; the fit cannot separate them, and the paper should not claim it does.
"""
from itertools import combinations
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
NP, NDRAW = 12, 500
rng = np.random.default_rng(29)

PATS = list(combinations(range(NP), 4))
A = np.zeros((len(PATS), NP))
ADJ = np.zeros(len(PATS))
for i, t in enumerate(PATS):
    A[i, list(t)] = 1
    ADJ[i] = sum(1 for a, b in combinations(t, 2) if abs(a - b) == 1)
PIDX = {t: i for i, t in enumerate(PATS)}


def fit(Msub, gamma_free=True, iters=4000, lr=0.05):
    idx = [PIDX[tuple(np.flatnonzero(r))] for r in Msub]
    n_k, s_adj, N = A[idx].sum(0), ADJ[idx].sum(), len(idx)
    beta, gamma = np.zeros(NP), 0.0
    for _ in range(iters):
        lin = A @ beta + gamma * ADJ
        lin -= lin.max()
        w = np.exp(lin)
        w /= w.sum()
        beta += lr * (n_k / N - A.T @ w)
        if gamma_free:
            gamma += lr * (s_adj / N - ADJ @ w)
        beta -= beta.mean()
    return beta, gamma


def probs(beta, gamma):
    lin = A @ beta + gamma * ADJ
    lin -= lin.max()
    w = np.exp(lin)
    return w / w.sum()


def maxgap(t):
    return max(t[i + 1] - t[i] for i in range(len(t) - 1))


MAXGAP = np.array([maxgap(t) for t in PATS])


def check(name, Msub):
    idx = np.array([PIDX[tuple(np.flatnonzero(r))] for r in Msub])
    N = len(idx)
    beta, gamma = fit(Msub)
    p = probs(beta, gamma)
    beta0, _ = fit(Msub, gamma_free=False)
    p0 = probs(beta0, 0.0)

    print(f'\n=== {name}  (n = {N}, gamma = {gamma:+.3f}) ===')

    # --- 1. pattern frequencies: chi-square style discrepancy, calibrated by simulation
    obs_cnt = np.bincount(idx, minlength=len(PATS))

    def disc(cnt, pp):
        e = N * pp
        return float(((cnt - e) ** 2 / np.maximum(e, 1e-9)).sum())

    d_obs, d_obs0 = disc(obs_cnt, p), disc(obs_cnt, p0)
    sim = np.array([disc(np.bincount(rng.choice(len(PATS), N, p=p), minlength=len(PATS)), p)
                    for _ in range(NDRAW)])
    sim0 = np.array([disc(np.bincount(rng.choice(len(PATS), N, p=p0), minlength=len(PATS)), p0)
                     for _ in range(NDRAW)])
    print(f'  pattern-frequency discrepancy   observed {d_obs:8.1f}   '
          f'replicated {sim.mean():7.1f} +/- {sim.std():5.1f}   ppp = {(sim >= d_obs).mean():.3f}')
    print(f'    same, no-clustering model     observed {d_obs0:8.1f}   '
          f'replicated {sim0.mean():7.1f} +/- {sim0.std():5.1f}   ppp = {(sim0 >= d_obs0).mean():.3f}')

    # --- 2. distinct patterns realised
    rep_u = np.array([len(np.unique(rng.choice(len(PATS), N, p=p))) for _ in range(NDRAW)])
    print(f'  distinct patterns               observed {len(np.unique(idx)):8d}   '
          f'replicated {rep_u.mean():7.1f} +/- {rep_u.std():5.1f}   '
          f'ppp = {(rep_u >= len(np.unique(idx))).mean():.3f}')

    # --- 3. max gap between harvested slices
    obs_mg = MAXGAP[idx].mean()
    rep_mg = np.array([MAXGAP[rng.choice(len(PATS), N, p=p)].mean() for _ in range(NDRAW)])
    print(f'  mean max-gap                    observed {obs_mg:8.3f}   '
          f'replicated {rep_mg.mean():7.3f} +/- {rep_mg.std():5.3f}   '
          f'ppp = {(rep_mg >= obs_mg).mean():.3f}')

    # --- 4. residual pairwise co-harvest matrix
    obs_pair = np.zeros((NP, NP))
    for i in idx:
        t = PATS[i]
        for a, b in combinations(t, 2):
            obs_pair[a, b] += 1
            obs_pair[b, a] += 1
    exp_pair = np.zeros((NP, NP))
    for i, pp in enumerate(p):
        t = PATS[i]
        for a, b in combinations(t, 2):
            exp_pair[a, b] += N * pp
            exp_pair[b, a] += N * pp
    with np.errstate(divide='ignore', invalid='ignore'):
        resid = (obs_pair - exp_pair) / np.sqrt(np.maximum(exp_pair, 1e-9))
    np.fill_diagonal(resid, 0.0)
    np.save(root / 'analysis' / f'ppc_resid_{name.split()[0].lower()}.npy', resid)
    iu = np.triu_indices(NP, 1)
    worst = np.argsort(-np.abs(resid[iu]))[:4]
    print(f'  pairwise residual (Pearson)     max |r| = {np.abs(resid[iu]).max():.2f}, '
          f'rms = {np.sqrt((resid[iu] ** 2).mean()):.2f}')
    print('    largest: ' + ', '.join(
        f'({iu[0][w]},{iu[1][w]}) {resid[iu][w]:+.1f}' for w in worst))
    return resid


rack = K[:, 0] // 20
check('LotA (racks 0-21)', M[rack < 22])
check('LotB (racks 22-48)', M[rack >= 22])
check('Fleet (all sockets)', M)

print('\n=== reading ===')
print('  The adjacency model reproduces the coarse structure it was fitted to. Where the')
print('  replicated discrepancy does not cover the observed one, the model is incomplete --')
print('  a single adjacency term does not capture the whole spatial pattern.')
print('  Note what none of this settles: clustering is equally consistent with a correlated')
print('  DEFECT process and with a binning POLICY that fuses in contiguous blocks. gamma')
print('  measures residual spatial association; it does not identify its cause.')
