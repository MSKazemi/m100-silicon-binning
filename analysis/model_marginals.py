"""Uniformity of the per-slice harvest rates, tested with the four-per-socket constraint built in.

Sec. IV-C tests the twelve per-slice counts n_k against the uniform expectation N/3 with a Pearson
chi-square on 11 df. That reference distribution is for multinomial counts. Here every socket
contributes exactly four DISTINCT slices, i.e. a draw without replacement, so under H0 (each map an
independent uniform 4-subset of 12)

    Cov(n) = N (8/33) (I - J/12),

a finite-population factor (12-4)/(12-1) = 8/11 below the multinomial covariance. The Pearson
statistic X2 is then distributed as (8/11) chi2_11, not chi2_11: the published test was
CONSERVATIVE, and the corrected statistic is X2 * 11/8 ~ chi2_11. This script reports:

  1. the published X2 and the corrected (11/8) X2 with their chi2_11 p-values;
  2. an exact Monte-Carlo test that draws every socket's map as a uniform 4-subset
     (B draws, p = (b+1)/(B+1)), which also verifies the 8/11 factor empirically;
  3. a rack-cluster-robust Wald test of uniformity, so the conclusion does not rest on sockets
     being independent (Sec. IV-G shows same-rack sockets are not).
Writes analysis/model_marginals.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, f as fdist

root = Path(__file__).resolve().parent.parent
NP, KSEL, B = 12, 4, 20_000


def pearson(n, N):
    e = N * KSEL / NP
    return float(((n - e) ** 2 / e).sum())


def mc_null(N, B, rng, chunk=500):
    """Pearson X2 for B data sets of N independent uniform 4-subsets of 12."""
    out = np.empty(B)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        keys = rng.random((b, N, NP))
        pick = np.argpartition(keys, KSEL, axis=2)[:, :, :KSEL]
        cnt = np.zeros((b, NP))
        for k in range(KSEL):
            cnt += np.eye(NP)[pick[:, :, k]].sum(1)
        e = N * KSEL / NP
        out[s:s + b] = ((cnt - e) ** 2 / e).sum(1)
    return out


def cluster_wald(M, clusters):
    """Wald test of E[M_k] = 1/3 for all k with a CR1 cluster-robust covariance, on the
    11-dimensional subspace orthogonal to the constant (rows always sum to 4)."""
    N = len(M)
    xbar = M.mean(0)
    d = xbar - KSEL / NP
    labs = np.unique(clusters)
    G = len(labs)
    S = np.array([(M[clusters == g] - xbar).sum(0) for g in labs])
    V = (S.T @ S) / N ** 2 * G / (G - 1)
    P = np.eye(NP) - 1.0 / NP                                    # projector, rank 11
    W = float(d @ np.linalg.pinv(P @ V @ P, rcond=1e-10) @ d)
    q = NP - 1
    Fst = W * (G - q) / (q * (G - 1))
    se0 = float(np.sqrt(V[0, 0]))
    return {'W': W, 'df': q, 'p_chi2': float(chi2.sf(W, q)), 'clusters': int(G),
            'F': float(Fst), 'F_df': [q, int(G - q)], 'p_F': float(fdist.sf(Fst, q, G - q)),
            'slice0_rate': float(xbar[0]), 'slice0_se_cluster': se0,
            'slice0_z_vs_third': float((xbar[0] - KSEL / NP) / se0)}


if __name__ == '__main__':
    M = np.load(root / 'analysis' / 'harvest_maps_full.npy').astype(float)
    K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
    rack = K[:, 0].astype(int) // 20
    N = len(M)
    assert (M.sum(1) == KSEL).all()
    n = M.sum(0)
    x2 = pearson(n, N)
    x2c = x2 * (NP - 1) / (NP - KSEL)
    print(f'sockets {N}; slice-0 rate {100 * n[0] / N:.1f}% against uniform {100 * KSEL / NP:.1f}%')
    print(f'Pearson X2 = {x2:.1f} (df = 11, multinomial reference, as published)')
    print(f'fixed-cardinality correction: Var of each count is 8/11 of the multinomial value, so '
          f'X2 ~ (8/11) chi2_11 under H0;')
    print(f'  corrected statistic (11/8) X2 = {x2c:.1f} on 11 df, p = {chi2.sf(x2c, 11):.1e}')

    rng = np.random.default_rng(4)
    null = mc_null(N, B, rng)
    b = int((null >= x2).sum())
    p_mc = (b + 1) / (B + 1)
    print(f'exact Monte-Carlo null ({B:,} data sets of {N} uniform 4-subsets):')
    print(f'  null X2 mean {null.mean():.3f} (theory 8/11 x 11 = 8.000; multinomial would be 11), '
          f'var {null.var():.2f} (theory 2 x (8/11)^2 x 11 = {2 * (8 / 11) ** 2 * 11:.2f}); '
          f'max {null.max():.1f}')
    print(f'  0.999 quantile {np.quantile(null, .999):.1f} vs (8/11) chi2_11 '
          f'{8 / 11 * chi2.isf(.001, 11):.1f} vs chi2_11 {chi2.isf(.001, 11):.1f}')
    print(f'  observed {x2:.1f} exceeded by {b} of {B:,} -> p = (b+1)/(B+1) = {p_mc:.1e}')

    cw = cluster_wald(M, rack)
    print(f'rack-cluster-robust Wald test ({cw["clusters"]} racks): W = {cw["W"]:.1f} on 11 df '
          f'(p = {cw["p_chi2"]:.1e}); small-sample F = {cw["F"]:.1f} on (11, {cw["F_df"][1]}) df, '
          f'p = {cw["p_F"]:.1e}')
    print(f'  slice 0: {100 * cw["slice0_rate"]:.1f}% +/- {100 * cw["slice0_se_cluster"]:.1f} pp '
          f'(rack-clustered SE), z = {cw["slice0_z_vs_third"]:+.1f} against 33.3%')
    print('-> NON-uniform under every reference distribution')
    json.dump({'N': N, 'counts': n.tolist(), 'pearson_x2': x2, 'pearson_df': 11,
               'fpc_factor': (NP - KSEL) / (NP - 1), 'x2_corrected': x2c,
               'p_corrected_chi2': float(chi2.sf(x2c, 11)),
               'mc_B': B, 'mc_exceed': b, 'mc_p': p_mc, 'mc_null_mean': float(null.mean()),
               'mc_null_var': float(null.var()), 'mc_null_max': float(null.max()),
               'mc_null_q999': float(np.quantile(null, .999)), 'cluster_wald_rack': cw},
              open(root / 'analysis' / 'model_marginals.json', 'w'), indent=1)
    print('wrote analysis/model_marginals.json')
