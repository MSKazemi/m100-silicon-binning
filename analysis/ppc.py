"""Parametric model-replicate checks of the pairwise-interaction model ("ppc" = parametric
predictive check; not posterior-predictive -- no parameter posterior is sampled).

A likelihood-ratio test says gamma is not zero. It does not say the model is adequate, and a
model that fits the adjacency statistic it was fitted on while failing everywhere else would
still pass that test. So we simulate replicate data sets from the fitted model and compare them
to the data on quantities it was NOT fitted to:

  1. the full pattern-frequency distribution (495 cells, Pearson discrepancy),
  2. the number of distinct patterns realised,
  3. the mean of the largest gap between harvested slices,
  4. the 66-cell pairwise co-harvest matrix, as residuals.

The model is fitted on the per-slice counts and the adjacency total, so (1)-(4) are genuine
out-of-statistic checks. Two replicate schemes are run, B each:
  plug-in    replicates drawn from the fitted model and compared with that same fit (as before);
  refitted   parametric bootstrap: each replicate is drawn from the fit AND refitted, and its
             statistic is computed against its own refit -- exactly what was done to the data,
             so parameter uncertainty and the fitted constraints are carried into the reference.
Monte-Carlo p-values are (b+1)/(B+1).

Residuals. (observed - expected)/sqrt(expected) is not a standard-normal deviate here: cells are
correlated, a co-harvest count is binomial-like rather than Poisson, and the fit forces every
row of the residual matrix and the adjacent-pair total to sum to zero. Each cell is therefore
standardised by the refitted replicate distribution of its own residual,
    z_cal = (r_obs - mean(r*)) / sd(r*),   r = O - E(theta_hat),
with a two-sided tail p-value per cell and a max-|z| family-wise bound over the 66 cells. The
Pearson matrices are still written (ppc_resid_<pop>.npy) for the current Fig. 10; the calibrated
ones are ppc_resid_<pop>_simcal.npy. Results: analysis/model_ppc.json.

What none of this settles: clustering is equally consistent with a correlated DEFECT process and
with a binning POLICY that fuses in contiguous blocks. gamma measures residual spatial association;
it does not identify its cause.
"""
import json
from pathlib import Path

import numpy as np

import model_core as mc

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
NDRAW = 10_000
rng = np.random.default_rng(29)
iu = np.triu_indices(mc.NP, 1)


def pv_upper(null, obs):
    return mc.mc_pvalue(null, obs, 'upper')


def pv_two(null, obs):
    return min(1.0, 2 * min(mc.mc_pvalue(null, obs, 'upper'), mc.mc_pvalue(null, obs, 'lower')))


def check(name, Msub):
    counts = mc.pattern_counts(Msub)
    rc = mc.replicate_checks(counts, NDRAW, rng)
    N, f1, O, E = rc['N'], rc['fit'], rc['O'], rc['E']
    rep, z_cal, z_rep, sd_r = rc['rep'], rc['z_cal'], rc['z_rep'], rc['sd_r']
    d_obs, d_obs0, u_obs, mg_obs = rc['d_obs'], rc['d_obs0'], rc['u_obs'], rc['mg_obs']
    d_plug, d_ref, d_plug0, d_ref0 = rep['d_plug'], rep['d_ref'], rep['d_plug0'], rep['d_ref0']
    u_rep, mg_rep = rep['u'], rep['mg']
    p_cell = (1 + (np.abs(z_rep) >= np.abs(z_cal)).sum(0)) / (NDRAW + 1)
    maxz = np.abs(z_rep).max(1)
    p_fwer = (1 + (maxz[:, None] >= np.abs(z_cal)[None]).sum(0)) / (NDRAW + 1)
    crit = float(np.quantile(maxz, 0.95))
    z_plug = rc['r_obs'] / rc['r_plug'].std(0, ddof=1)
    pearson = rc['pearson']

    tag = name.split()[0].lower()
    np.save(root / 'analysis' / f'ppc_resid_{tag}.npy', mc.to_matrix(pearson))
    np.save(root / 'analysis' / f'ppc_resid_{tag}_simcal.npy', mc.to_matrix(z_cal))

    res = {'n': N, 'gamma': f1['gamma'], 'B': NDRAW,
           'discrepancy': {'observed': d_obs, 'refit_mean': float(d_ref.mean()),
                           'refit_sd': float(d_ref.std(ddof=1)), 'p_refit': pv_upper(d_ref, d_obs),
                           'plugin_mean': float(d_plug.mean()), 'plugin_sd': float(d_plug.std(ddof=1)),
                           'p_plugin': pv_upper(d_plug, d_obs)},
           'discrepancy_no_clustering': {'observed': d_obs0, 'refit_mean': float(d_ref0.mean()),
                                         'refit_sd': float(d_ref0.std(ddof=1)),
                                         'p_refit': pv_upper(d_ref0, d_obs0),
                                         'plugin_mean': float(d_plug0.mean()),
                                         'plugin_sd': float(d_plug0.std(ddof=1))},
           'distinct_patterns': {'observed': u_obs, 'rep_mean': float(u_rep.mean()),
                                 'rep_sd': float(u_rep.std(ddof=1)),
                                 'p_upper': pv_upper(u_rep, u_obs), 'p_two_sided': pv_two(u_rep, u_obs)},
           'mean_max_gap': {'observed': mg_obs, 'rep_mean': float(mg_rep.mean()),
                            'rep_sd': float(mg_rep.std(ddof=1)),
                            'p_lower': mc.mc_pvalue(mg_rep, mg_obs, 'lower'),
                            'p_two_sided': pv_two(mg_rep, mg_obs)},
           'residual_fwer_crit_abs_z': crit,
           'residual_cells_fwer_sig': int((p_fwer <= 0.05).sum()),
           'residual_rms_z_cal': float(np.sqrt((z_cal ** 2).mean())),
           'residual_rms_pearson': float(np.sqrt((pearson ** 2).mean())),
           'cells': [{'pair': list(mc.PAIRS[j]), 'observed': float(O[j]), 'expected': float(E[j]),
                      'pearson': float(pearson[j]), 'z_cal': float(z_cal[j]),
                      'z_plugin': float(z_plug[j]), 'sd_ratio_vs_sqrt_expected':
                      float(sd_r[j] / np.sqrt(E[j])), 'p_cell': float(p_cell[j]),
                      'p_fwer': float(p_fwer[j])} for j in range(mc.NPAIR)]}
    D, U, G = res['discrepancy'], res['distinct_patterns'], res['mean_max_gap']
    print(f'\n=== {name}  (n = {N}, gamma = {f1["gamma"]:+.3f}, B = {NDRAW:,} replicates) ===')
    print(f'  pattern-frequency discrepancy   observed {d_obs:8.1f}   refitted replicates '
          f'{D["refit_mean"]:7.1f} +/- {D["refit_sd"]:5.1f}  p = {D["p_refit"]:.1e}   '
          f'(plug-in {D["plugin_mean"]:.1f} +/- {D["plugin_sd"]:.1f}, p = {D["p_plugin"]:.1e})')
    print(f'    same, no-clustering model     observed {d_obs0:8.1f}   refitted replicates '
          f'{d_ref0.mean():7.1f} +/- {d_ref0.std(ddof=1):5.1f}')
    print(f'  distinct patterns               observed {u_obs:8d}   replicated '
          f'{U["rep_mean"]:7.1f} +/- {U["rep_sd"]:5.1f}   p(>=) = {U["p_upper"]:.3f}, '
          f'two-sided {U["p_two_sided"]:.3f}')
    print(f'  mean max-gap                    observed {mg_obs:8.3f}   replicated '
          f'{G["rep_mean"]:7.3f} +/- {G["rep_sd"]:5.3f}   p(<=) = {G["p_lower"]:.1e}')
    order = np.argsort(-np.abs(z_cal))[:6]
    print(f'  co-harvest residuals: simulation-calibrated z (refitted replicates), Pearson in '
          f'brackets; replicate sd / sqrt(expected) = '
          f'{(sd_r / np.sqrt(E)).min():.2f}..{(sd_r / np.sqrt(E)).max():.2f}')
    print('    ' + ', '.join(f'({mc.PAIRS[j][0]},{mc.PAIRS[j][1]}) {z_cal[j]:+.1f} '
                             f'[{pearson[j]:+.1f}] p={p_cell[j]:.1e}' for j in order))
    print(f'    cells beyond the max-|z| family-wise bound |z| > {crit:.2f}: '
          f'{res["residual_cells_fwer_sig"]} of 66;  rms z_cal {res["residual_rms_z_cal"]:.2f} '
          f'vs rms Pearson {res["residual_rms_pearson"]:.2f}')
    return res


if __name__ == '__main__':
    rack = K[:, 0] // 20
    out = {'LotA': check('LotA (racks 0-21)', M[rack < 22]),
           'LotB': check('LotB (racks 22-48)', M[rack >= 22]),
           'Fleet': check('Fleet (all sockets)', M)}
    json.dump(out, open(root / 'analysis' / 'model_ppc.json', 'w'), indent=1)
    print('\n=== reading ===')
    print('  The adjacency model reproduces the coarse structure it was fitted to. Where the')
    print('  replicated discrepancy does not cover the observed one, the model is incomplete --')
    print('  a single adjacency term does not capture the whole spatial pattern.')
    print('  Note what none of this settles: clustering is equally consistent with a correlated')
    print('  DEFECT process and with a binning POLICY that fuses in contiguous blocks. gamma')
    print('  measures residual spatial association; it does not identify its cause.')
    print('  wrote ppc_resid_*.npy (Pearson), ppc_resid_*_simcal.npy (calibrated), model_ppc.json')
