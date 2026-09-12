"""Fit a clustering parameter to the observed harvest maps, per lot (Table VIII).

The yield literature models defect COUNTS per die with a negative binomial whose parameter
alpha controls clustering (alpha -> infinity recovers Poisson, i.e. no clustering)
[Koren-Koren-Stapper 1993]. We cannot fit that directly: the SKU fixes the count at exactly
four harvested slices, so the count distribution is not observable. What IS observable is the
conditional SPATIAL arrangement given four.

We therefore fit a conditional pairwise-interaction (Strauss-type) model on the 12 slice
positions. For a harvested set S, a subset of {0,...,11} with |S| = 4,

    P(S) = exp( sum_{k in S} beta_k  +  gamma * adj(S) ) / Z,
    adj(S) = number of unordered pairs {k, k+1} with both k and k+1 in S.

Because |S| is fixed, beta is identified only up to an additive constant; we impose
sum_k beta_k = 0. beta_k summarises per-slice propensity (the slice-0 asymmetry); gamma > 0 means
harvested slices are index-adjacent more often than those propensities alone predict. That is a
statement of residual spatial ASSOCIATION: a correlated defect process and a binning policy that
fuses contiguous blocks would both produce it, and this fit cannot tell them apart.

Z is an exact sum over all C(12,4) = 495 patterns, so the log-likelihood, gradient and Hessian are
exact. The MLE is computed by Newton's method, cross-checked against scipy's L-BFGS-B and against
the fixed-step loop the published numbers originally came from, and every fit asserts a gradient
norm below 1e-8. Writes analysis/model_fit.json.
"""
import json
from pathlib import Path

import numpy as np

import model_core as mc

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')   # 1 = harvested
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
GTOL = 1e-8


def report(name, Msub):
    counts = mc.pattern_counts(Msub)
    N = int(counts.sum())
    f1, f0 = mc.fit_newton(counts, True), mc.fit_newton(counts, False)
    lb1, lb0 = mc.fit_lbfgs(counts, True), mc.fit_lbfgs(counts, False)
    fx1, fx0 = mc.fit_fixed_step(counts, True), mc.fit_fixed_step(counts, False)
    for f in (f1, f0):
        assert f['grad_norm'] < GTOL, f'{name}: Newton did not converge'
    for f in (lb1, lb0):
        assert f['grad_norm'] < 1e-6, f'{name}: L-BFGS did not converge'
    assert abs(f1['beta'].sum()) < 1e-12
    d_lbfgs = max(np.abs(f1['beta'] - lb1['beta']).max(), abs(f1['gamma'] - lb1['gamma']))
    d_fixed = max(np.abs(f1['beta'] - fx1['beta']).max(), abs(f1['gamma'] - fx1['gamma']))
    assert d_lbfgs < 1e-6, f'{name}: Newton and L-BFGS disagree by {d_lbfgs:.2e}'
    lr_stat = 2 * (f1['ll'] - f0['ll'])                     # 1 df
    se_model = float(np.sqrt(f1['inv_info'][-1, -1]))
    se_cond = float(mc.se_gamma_conditional(counts, f1['beta'], f1['gamma']))
    obs_adj = mc.ADJ @ counts / N
    eig = f1['hess_eig']
    print(f'\n=== {name}  (n = {N}) ===')
    print(f'  gamma (clustering)      = {f1["gamma"]:+.4f}')
    print(f'  LR test gamma = 0       : chi2(1) = {lr_stat:.1f}  '
          f'-> {"residual adjacency SIGNIFICANT" if lr_stat > 10.83 else "not significant"} '
          f'(p<0.001 crit 10.83)')
    print(f'  mean adjacent pairs     : observed {obs_adj:.4f}, '
          f'model {mc.ADJ @ f1["p"]:.4f}, no-clustering model {mc.ADJ @ f0["p"]:.4f}')
    print(f'  per-slice beta (sum-to-zero constraint; higher = more often harvested):')
    print('    ' + ' '.join(f'{x:+.2f}' for x in f1['beta']))
    print(f'  optimiser: Newton {f1["iterations"]} it, |grad|/n = {f1["grad_norm"]:.1e}; '
          f'L-BFGS max|diff| {d_lbfgs:.1e}; legacy fixed-step max|diff| {d_fixed:.1e} '
          f'(its |grad|/n = {fx1["grad_norm"]:.1e})')
    print(f'  Hessian (per socket, identified 12-dim parameterisation): eigenvalues '
          f'{eig.min():.3f} .. {eig.max():.3f}, condition number {eig.max() / eig.min():.1f}')
    print(f'  i.i.d. SE(gamma): model-based {se_model:.4f} '
          f'(beta held fixed, as previously printed: {se_cond:.4f})')
    return {'n': N, 'gamma': f1['gamma'], 'beta': f1['beta'].tolist(),
            'beta_sum': float(f1['beta'].sum()), 'loglik': f1['ll'], 'loglik_gamma0': f0['ll'],
            'lr_chi2': lr_stat, 'obs_mean_adj': obs_adj, 'model_mean_adj': float(mc.ADJ @ f1['p']),
            'nocluster_mean_adj': float(mc.ADJ @ f0['p']),
            'newton_iterations': f1['iterations'], 'grad_norm': f1['grad_norm'],
            'lbfgs_max_abs_diff': d_lbfgs, 'lbfgs_grad_norm': lb1['grad_norm'],
            'fixed_step_max_abs_diff': d_fixed, 'fixed_step_grad_norm': fx1['grad_norm'],
            'fixed_step_lr_chi2': 2 * (fx1['ll'] - fx0['ll']),
            'hessian_eig_min': float(eig.min()), 'hessian_eig_max': float(eig.max()),
            'hessian_condition': float(eig.max() / eig.min()),
            'se_gamma_model_based': se_model, 'se_gamma_beta_fixed': se_cond}


if __name__ == '__main__':
    nid = K[:, 0]
    rack = nid // 20
    out = {'lotA': report('Lot A  (racks 0-21)', M[rack < 22]),
           'lotB': report('Lot B  (racks 22-48)', M[rack >= 22]),
           'fleet': report('Whole fleet', M)}
    gA, gB = out['lotA']['gamma'], out['lotB']['gamma']
    seA, seB = out['lotA']['se_gamma_model_based'], out['lotB']['se_gamma_model_based']
    z = (gA - gB) / np.hypot(seA, seB)
    z_old = (gA - gB) / np.hypot(out['lotA']['se_gamma_beta_fixed'],
                                 out['lotB']['se_gamma_beta_fixed'])
    out['lot_difference_iid'] = {'difference': gA - gB, 'z_model_based': z,
                                 'z_beta_fixed_superseded': z_old}
    print(f'\n=== lot comparison (i.i.d. sockets; see gamma_clustered_se.py for rack clustering) ===')
    print(f'  gamma_A = {gA:+.4f} +/- {seA:.4f}')
    print(f'  gamma_B = {gB:+.4f} +/- {seB:.4f}')
    print(f'  difference z = {z:+.2f}  (the beta-fixed SE previously used gave {z_old:+.2f}) -> '
          f'{"lots DIFFER in clustering" if abs(z) > 2.58 else "no significant difference"}')
    json.dump(out, open(root / 'analysis' / 'model_fit.json', 'w'), indent=1, default=float)
    print('\ninterpretation: gamma > 0 means harvested slices are index-adjacent more often than the')
    print('per-slice propensities predict -- residual spatial association, which a correlated defect')
    print('process and a block-fusing binning policy would both produce. beta and gamma are separable')
    print('summaries of the maps, not an identified mechanism. wrote analysis/model_fit.json')
