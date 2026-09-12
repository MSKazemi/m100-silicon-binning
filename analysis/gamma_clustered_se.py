"""Cluster-aware uncertainty for the clustering parameter gamma, with stability diagnostics.

The fit in analysis/yield_model.py treats sockets as i.i.d. draws. They are not: same-rack sockets
share more harvested slices than chance (z = +13.9) and each node contributes two sockets. This
script reports gamma's standard error under every reasonable dependence assumption, so that the
lot comparison can be read against all of them rather than against one convenient choice:

  model-based   inverse Fisher information of the full (beta, gamma) model -- i.i.d. sockets
                (the earlier script printed 1/sqrt(N Var adj), which holds beta fixed and
                understates the SE; that value is shown as `conditional` for the record)
  HC0           socket-level sandwich (robust to misspecification, still i.i.d.)
  CR1 node      sandwich clustered on node (the two sockets of a node)
  CR1 rack      sandwich clustered on rack, small-sample factor G/(G-1)
  jackknife     delete-one-rack jackknife, every fit redone
  bootstrap     rack pairs bootstrap, B resamples of the racks within each lot, every fit redone;
                SE with its Monte-Carlo error, percentile, BCa and normal intervals
  wild score    rack wild score bootstrap (Webb six-point weights, Kline & Santos 2012)

Racks are node_id // 20. The four nodes 980-983 have no documented cabinet; their two mapped
sockets fall in the last group (index 49) and are kept as one extra cluster of Lot B, so Lot A
has 22 clusters, Lot B 28 and the fleet 50. A sensitivity run without them is also reported.
Writes analysis/model_gamma_se.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

import model_core as mc

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
nid = K[:, 0].astype(int)
rack = nid // 20
IDX = mc.pattern_index(M)
B_BOOT, B_WILD = 2000, 9999
rng = np.random.default_rng(2026)


def group_counts(idx, labels):
    labs = np.unique(labels)
    C = np.zeros((len(labs), mc.NPAT))
    for j, g in enumerate(labs):
        C[j] = np.bincount(idx[labels == g], minlength=mc.NPAT)
    return labs, C


def sandwich(idx, labels, fit, small_sample=True):
    labs, S = mc.cluster_scores(idx, labels, fit)
    G = len(labs)
    Hinv = fit['inv_info']
    V = Hinv @ (S.T @ S) @ Hinv
    if small_sample:
        V *= G / (G - 1)
    return float(np.sqrt(V[-1, -1])), G


def bca(boot, est, jack):
    z0 = norm.ppf(np.clip((boot < est).mean(), 1e-6, 1 - 1e-6))
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)
    out = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        out.append(float(np.percentile(boot, 100 * norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))))
    return out, float(z0), float(a)


def analyse(sel, label):
    idx, rk, nd = IDX[sel], rack[sel], nid[sel]
    counts = np.bincount(idx, minlength=mc.NPAT).astype(float)
    fit = mc.fit_newton(counts)
    assert fit['grad_norm'] < 1e-8, 'MLE did not converge'
    g_hat = fit['gamma']
    se_model = float(np.sqrt(fit['inv_info'][-1, -1]))
    se_cond = float(mc.se_gamma_conditional(counts, fit['beta'], g_hat))
    se_hc0, _ = sandwich(idx, np.arange(len(idx)), fit, small_sample=False)
    se_node, G_node = sandwich(idx, nd, fit)
    se_rack, G = sandwich(idx, rk, fit)

    labs, RC = group_counts(idx, rk)
    jack = np.array([mc.fit_newton(counts - RC[j], theta0=fit['theta'])['gamma']
                     for j in range(G)])
    se_jack = float(np.sqrt((G - 1) / G * ((jack - jack.mean()) ** 2).sum()))

    boot = np.empty(B_BOOT)
    for b in range(B_BOOT):
        pick = rng.integers(0, G, G)
        f = mc.fit_newton(RC[pick].sum(0), theta0=fit['theta'])
        assert f['grad_norm'] < 1e-8
        boot[b] = f['gamma']
    se_boot = float(boot.std(ddof=1))
    batches = boot.reshape(20, -1).std(1, ddof=1)
    se_boot_mcse = float(batches.std(ddof=1) / np.sqrt(len(batches)))
    pct = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    bca_ci, z0, acc = bca(boot, g_hat, jack)

    _, S = mc.cluster_scores(idx, rk, fit)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    W = webb[rng.integers(0, 6, (B_WILD, G))]
    wild = g_hat + (W @ S @ fit['inv_info'].T)[:, -1]
    se_wild = float(wild.std(ddof=1))

    res = {'label': label, 'n': int(counts.sum()), 'clusters_rack': int(G),
           'clusters_node': int(G_node), 'gamma': float(g_hat),
           'se': {'model_based': se_model, 'conditional_beta_fixed': se_cond, 'hc0': se_hc0,
                  'cr1_node': se_node, 'cr1_rack': se_rack, 'jackknife_rack': se_jack,
                  'bootstrap_rack': se_boot, 'bootstrap_rack_mcse': se_boot_mcse,
                  'wild_score_rack': se_wild},
           'ci95': {'percentile': pct, 'bca': bca_ci,
                    'normal_boot': [g_hat - 1.96 * se_boot, g_hat + 1.96 * se_boot],
                    't_jackknife': [g_hat - tdist.ppf(.975, G - 1) * se_jack,
                                    g_hat + tdist.ppf(.975, G - 1) * se_jack]},
           'bca_z0': z0, 'bca_accel': acc,
           'boot_skew': float(((boot - boot.mean()) ** 3).mean() / boot.std() ** 3),
           'wald_z_gamma0_rack': float(g_hat / se_boot),
           'jackknife_range': [float(jack.min()), float(jack.max())]}
    s = res['se']
    print(f'\n{label:<22} n = {res["n"]:5d}   racks {G:2d}   gamma = {g_hat:+.4f}')
    print(f'  SE  model-based {s["model_based"]:.4f}   (conditional, beta fixed: '
          f'{s["conditional_beta_fixed"]:.4f})   HC0 {s["hc0"]:.4f}   CR1-node {s["cr1_node"]:.4f}')
    print(f'      CR1-rack {s["cr1_rack"]:.4f}   jackknife-rack {s["jackknife_rack"]:.4f}   '
          f'bootstrap-rack {s["bootstrap_rack"]:.4f} (MC error {s["bootstrap_rack_mcse"]:.4f})   '
          f'wild-score {s["wild_score_rack"]:.4f}')
    print(f'  95% CI  percentile [{pct[0]:+.4f}, {pct[1]:+.4f}]   BCa [{bca_ci[0]:+.4f}, '
          f'{bca_ci[1]:+.4f}]   (z0 {z0:+.3f}, a {acc:+.4f}, bootstrap skew {res["boot_skew"]:+.2f})')
    print(f'  rack-clustered Wald z for gamma = 0: {res["wald_z_gamma0_rack"]:+.1f}')
    return res, boot, jack


def compare(rA, rB, bootA, bootB, jackA, jackB):
    d = rA['gamma'] - rB['gamma']
    out = {'difference': d, 'z': {}, 'ci95': {}}
    for key in ('model_based', 'hc0', 'cr1_node', 'cr1_rack', 'jackknife_rack', 'wild_score_rack',
                'conditional_beta_fixed'):
        out['z'][key] = d / np.hypot(rA['se'][key], rB['se'][key])
    dv = bootA - bootB                                   # independent resamples of disjoint racks
    out['z']['bootstrap_rack'] = d / dv.std(ddof=1)
    out['se_bootstrap'] = float(dv.std(ddof=1))
    out['ci95']['percentile'] = [float(x) for x in np.percentile(dv, [2.5, 97.5])]
    df = min(rA['clusters_rack'], rB['clusters_rack']) - 1
    sej = np.hypot(rA['se']['jackknife_rack'], rB['se']['jackknife_rack'])
    out['ci95']['t_jackknife_df%d' % df] = [d - tdist.ppf(.975, df) * sej,
                                             d + tdist.ppf(.975, df) * sej]
    out['p_two_sided_t_jackknife'] = float(2 * tdist.sf(abs(d / sej), df))
    out['boot_fraction_leq0'] = float((dv <= 0).mean())
    print(f'\nlot difference gamma_A - gamma_B = {d:+.4f}')
    for k, z in out['z'].items():
        print(f'  z ({k:<22}) = {z:+.2f}')
    print(f'  95% CI  bootstrap percentile [{out["ci95"]["percentile"][0]:+.4f}, '
          f'{out["ci95"]["percentile"][1]:+.4f}]   jackknife-t(df={df}) '
          f'[{out["ci95"]["t_jackknife_df%d" % df][0]:+.4f}, '
          f'{out["ci95"]["t_jackknife_df%d" % df][1]:+.4f}]')
    print(f'  bootstrap resamples with gamma_A <= gamma_B: {int((dv <= 0).sum())} of {len(dv)}')
    return out


if __name__ == '__main__':
    print(f'rack-clustered uncertainty for gamma: B = {B_BOOT} rack-bootstrap refits, '
          f'B = {B_WILD} wild-score draws, delete-one-rack jackknife')
    rA, bA, jA = analyse(rack < 22, 'Lot A (racks 0-21)')
    rB, bB, jB = analyse(rack >= 22, 'Lot B (racks 22-48)')
    rF, bF, jF = analyse(np.ones(len(M), bool), 'Whole fleet')
    cmp_ = compare(rA, rB, bA, bB, jA, jB)

    documented = nid < 980
    print('\n--- sensitivity: the two sockets of the undocumented nodes 980-983 removed ---')
    rB2, bB2, jB2 = analyse((rack >= 22) & documented, 'Lot B, nodes < 980')
    cmp2 = compare(rA, rB2, bA, bB2, jA, jB2)

    json.dump({'B_boot': B_BOOT, 'B_wild': B_WILD, 'lotA': rA, 'lotB': rB, 'fleet': rF,
               'lot_difference': cmp_,
               'sensitivity_without_nodes_980_983': {'lotB': rB2, 'lot_difference': cmp2}},
              open(root / 'analysis' / 'model_gamma_se.json', 'w'), indent=1, default=float)
    zc = cmp_['z']['bootstrap_rack']
    print(f'\n-> the lots {"DO" if min(abs(z) for z in cmp_["z"].values()) > 2.58 else "may not"} '
          f'differ in clustering strength under every dependence model (smallest |z| = '
          f'{min(abs(z) for z in cmp_["z"].values()):.2f}; rack bootstrap z = {zc:+.2f}).')
    print('wrote analysis/model_gamma_se.json')
