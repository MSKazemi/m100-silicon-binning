"""Where the residual adjacency lives: within POWER9 quads or across them.

The one-gamma pairwise model (yield_model.py) spreads a single adjacency term over all eleven
index-adjacent slice pairs, and its model-replicate check (ppc.py) shows where that fails: it
under-predicts co-harvest at pairs (0,1), (6,7), ... and over-predicts it at (3,4), (5,6), (7,8),
(9,10). POWER9 groups its 24 cores into six four-core quads (EQ chiplets), cores 4q..4q+3, so the
slices 2q and 2q+1 share a quad and the slices 2q+1 and 2q+2 do not. This script splits the
adjacency statistic accordingly,

    P(S) ~ exp( sum_{k in S} beta_k + g_in * adj_in(S) + g_x * adj_x(S) ),
    adj_in(S) = #{q : 2q, 2q+1 in S},   adj_x(S) = #{q : 2q+1, 2q+2 in S},

(sum beta = 0), and reports, per lot and for the fleet: both gammas with model-based and
rack-clustered bootstrap SEs, the likelihood-ratio test of g_in = g_x (i.e. the one-gamma model),
and the pattern-frequency discrepancy of both models against refitted parametric replicates, so it
is visible how much of the misfit the split removes and how much remains.
Descriptive only: which process fuses within quads is not identified here. Writes
analysis/model_quad.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

import model_core as mc

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
rack = K[:, 0].astype(int) // 20
XB = mc.A @ mc.L
X1 = np.column_stack([XB, mc.ADJ])
X2 = np.column_stack([XB, mc.ADJ_IN, mc.ADJ_X])
B_BOOT, B_REP = 1000, 2000
rng = np.random.default_rng(2027)


def analyse(sel, label):
    idx = mc.pattern_index(M[sel])
    rk = rack[sel]
    counts = np.bincount(idx, minlength=mc.NPAT).astype(float)
    N = int(counts.sum())
    f1, f2 = mc.fit_design(X1, counts), mc.fit_design(X2, counts)
    assert f1['grad_norm'] < 1e-8 and f2['grad_norm'] < 1e-8
    lr = 2 * (f2['ll'] - f1['ll'])
    se_model = np.sqrt(np.diag(f2['inv_info']))[-2:]
    labs = np.unique(rk)
    RC = np.array([np.bincount(idx[rk == g], minlength=mc.NPAT) for g in labs], dtype=float)
    boot = np.empty((B_BOOT, 2))
    for b in range(B_BOOT):
        fb = mc.fit_design(X2, RC[rng.integers(0, len(labs), len(labs))].sum(0), f2['theta'])
        assert fb['grad_norm'] < 1e-8
        boot[b] = fb['theta'][-2:]
    se_rack = boot.std(0, ddof=1)
    diff_boot = boot[:, 0] - boot[:, 1]
    d1, d2 = mc.pearson_discrepancy(counts, f1['p']), mc.pearson_discrepancy(counts, f2['p'])
    rep1, rep2 = np.empty(B_REP), np.empty(B_REP)
    for b in range(B_REP):
        c1 = rng.multinomial(N, f1['p']).astype(float)
        c2 = rng.multinomial(N, f2['p']).astype(float)
        rep1[b] = mc.pearson_discrepancy(c1, mc.fit_design(X1, c1, f1['theta'])['p'])
        rep2[b] = mc.pearson_discrepancy(c2, mc.fit_design(X2, c2, f2['theta'])['p'])
    O = mc.PAIRMAT.T @ counts
    ratio = {}
    for name, kind in (('within_quad', 0), ('cross_quad', 1)):
        ratio[name] = {f'{a},{b}': float(O[j]) for j, (a, b) in enumerate(mc.PAIRS)
                       if b - a == 1 and a % 2 == kind}
    res = {'label': label, 'n': N, 'gamma_one': float(f1['theta'][-1]),
           'gamma_in': float(f2['theta'][-2]), 'gamma_x': float(f2['theta'][-1]),
           'se_model': se_model.tolist(), 'se_rack_boot': se_rack.tolist(),
           'diff': float(f2['theta'][-2] - f2['theta'][-1]),
           'diff_se_rack_boot': float(diff_boot.std(ddof=1)),
           'diff_z_rack': float((f2['theta'][-2] - f2['theta'][-1]) / diff_boot.std(ddof=1)),
           'lr_in_eq_x': float(lr), 'lr_p': float(chi2.sf(lr, 1)),
           'discrepancy_one_gamma': d1, 'rep_one_gamma': [float(rep1.mean()), float(rep1.std(ddof=1))],
           'p_one_gamma': mc.mc_pvalue(rep1, d1),
           'discrepancy_two_gamma': d2, 'rep_two_gamma': [float(rep2.mean()), float(rep2.std(ddof=1))],
           'p_two_gamma': mc.mc_pvalue(rep2, d2), 'adjacent_pair_counts': ratio}
    print(f'\n=== {label}  (n = {N}) ===')
    print(f'  one gamma            {res["gamma_one"]:+.3f}')
    print(f'  within-quad  g_in    {res["gamma_in"]:+.3f}  (model SE {se_model[0]:.3f}, rack bootstrap '
          f'{se_rack[0]:.3f})')
    print(f'  cross-quad   g_x     {res["gamma_x"]:+.3f}  (model SE {se_model[1]:.3f}, rack bootstrap '
          f'{se_rack[1]:.3f})')
    print(f'  g_in - g_x = {res["diff"]:+.3f}, rack-bootstrap z = {res["diff_z_rack"]:+.1f};  '
          f'LR test of one gamma: chi2(1) = {lr:.1f}, p = {res["lr_p"]:.1e}')
    print(f'  pattern-frequency discrepancy: one gamma {d1:.1f} (replicates {rep1.mean():.1f} +/- '
          f'{rep1.std(ddof=1):.1f}, p = {res["p_one_gamma"]:.1e}); two gammas {d2:.1f} (replicates '
          f'{rep2.mean():.1f} +/- {rep2.std(ddof=1):.1f}, p = {res["p_two_gamma"]:.1e})')
    return res


if __name__ == '__main__':
    out = {'lotA': analyse(rack < 22, 'Lot A (racks 0-21)'),
           'lotB': analyse(rack >= 22, 'Lot B (racks 22-48)'),
           'fleet': analyse(np.ones(len(M), bool), 'Whole fleet'),
           'B_boot': B_BOOT, 'B_rep': B_REP}
    json.dump(out, open(root / 'analysis' / 'model_quad.json', 'w'), indent=1)
    print('\nreading: the residual adjacency is concentrated on slice pairs that share a POWER9 quad;')
    print('splitting gamma removes part of the misfit but not all of it (see the two-gamma p).')
    print('wrote analysis/model_quad.json')
