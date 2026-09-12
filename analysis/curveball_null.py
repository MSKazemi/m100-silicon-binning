"""Pair-level co-harvest inference against an exchangeable curveball null, with mixing diagnostics.

The paper's pair tests (Sec. IV-D) standardised each of the 66 co-harvest counts by the mean and SD
of 200 chained curveball draws and applied a normal-theory cut |z| > 3.83. Three problems:
  * 200 draws estimate a mean and an SD, not a tail near 0.05/66, so the normal approximation is
    assumed rather than checked, and pair counts moved with the seed (16-18 pairs);
  * the two-sided Bonferroni critical value for 66 tests at alpha = 0.05 is 3.37, not 3.83
    (3.83 corresponds to a family-wise alpha of about 0.0085);
  * the curveball trade as implemented dropped the shared columns from the second row and then
    rejected the trade, so only trades between rows with no harvested slice in common ever ran
    (about 10% of attempts); see model_core.curveball_seq_legacy.
This script replaces that with a test whose error rate depends on none of them:

1. LEGACY  -- reproduces the published run exactly (seed 2024, 200 draws x 20,000 attempted
   trades of the legacy sampler, chained from the observed matrix) plus four more seeds.
2. MIXING  -- burn-in traces from the observed matrix and from a maximally structured start with
   the same margins, lagged autocorrelation of all 66 cells and the mean gap, and Gelman-Rubin
   R-hat across the two starts, for the legacy, corrected sequential and global samplers.
3. NULL    -- B exchangeable null matrices by the Besag-Clifford (1989) parallel construction: run
   the (reversible) chain k steps from the observed matrix to a hidden state x*, then B independent
   k-step chains from x*. Under H0 the observed matrix and the B draws are exchangeable, so every
   rank-based p-value (b+1)/(B+1) is exact whatever the mixing time. Main run: B = 100,000 with
   vectorised global curveball trades; confirmation: B = 10,000 with the corrected sequential
   sampler at 20,000 trades.
4. TESTS   -- per-pair two-sided Monte-Carlo p with Bonferroni over 66 pairs, and single-step
   max-|z| (Westfall-Young) family-wise adjusted p; effect sizes log2(observed/expected) with the
   pointwise and simultaneous null spread; a check of the normal-tail approximation.
5. STRATA  -- sensitivity: the same test with trades restricted to within each lot, so each lot keeps
   its own per-slice marginals and a two-population mixture cannot create pair association.

Writes analysis/model_curveball.json. MODEL_WORKERS sets the process count (default 12); the null
draws are split into MODEL_CHUNKS (default 12) independently seeded chunks whatever the process
count, so the output does not depend on the number of workers.
"""
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm, kurtosis, skew

import model_core as mc

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy').astype(bool)
KEYS = np.load(root / 'analysis' / 'harvest_keys_full.npy')
RACK = KEYS[:, 0].astype(int) // 20
LOT = (RACK >= 22).astype(int)
OBS = mc.cooc_upper(M)
WORKERS = int(os.environ.get('MODEL_WORKERS', 12))
CHUNKS = int(os.environ.get('MODEL_CHUNKS', 12))            # fixes the random streams
B_MAIN, K_GLOBAL = int(os.environ.get('MODEL_B', 100_000)), 40
B_SEQ, K_SEQ = int(os.environ.get('MODEL_B_SEQ', 10_000)), 20_000
ALPHA = 0.05
NTEST = mc.NPAIR
Z_BONF = float(norm.isf(ALPHA / NTEST / 2))                 # 3.368
Z_PAPER = 3.83


def pool():
    # 'spawn', not 'fork': forking a process that already runs threads (BLAS, pytest, a notebook)
    # can deadlock the child, and Python 3.12 warns about it. Workers receive their inputs (the
    # hidden matrix, seeds, strata) as arguments and re-read only the committed .npy maps at import,
    # so the draws are identical under either start method.
    return ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('spawn'))


def split(n, parts):
    q, r = divmod(n, parts)
    return [q + (i < r) for i in range(parts) if q + (i < r) > 0]


# ------------------------------------------------------------------ workers
def _stepper(kind, rng):
    if kind == 'legacy':
        return lambda x, n: mc.curveball_seq_legacy(x, n, rng)
    if kind == 'seq':
        return lambda x, n: mc.curveball_seq(x, n, rng)
    return lambda x, n: mc.curveball_global(x, n, rng)


def _w_global(args):
    xstar, n, k, seed, strata = args
    rng = np.random.default_rng(seed)
    out = np.empty((n, NTEST), dtype=np.int32)
    for i in range(n):
        out[i] = mc.cooc_upper(mc.curveball_global(xstar, k, rng, strata))
    return out


def _w_seq(args):
    xstar, n, k, seed = args
    rng = np.random.default_rng(seed)
    x8 = xstar.astype(np.int8)
    out = np.empty((n, NTEST), dtype=np.int32)
    for i in range(n):
        out[i] = mc.cooc_upper(mc.curveball_seq(x8, k, rng))
    return out


def _w_legacy(seed):
    """The published procedure: 200 draws chained from the observed matrix, 20,000 trades each."""
    rng = np.random.default_rng(seed)
    cur = M.astype(np.int8)
    null = np.empty((200, NTEST))
    st = {}
    for s in range(200):
        cur = mc.curveball_seq_legacy(cur, 20000, rng, st)
        assert (cur.sum(1) == 4).all() and (cur.sum(0) == M.sum(0)).all()
        null[s] = mc.cooc_upper(cur)
    return null, st


def _w_trace(args):
    """Burn-in trace of the mean gap (every `every` trades) from a given start, then `ndraw`
    draws spaced `space` trades -- for R-hat. kind = 'seq' (trades) or 'glob' (global trades)."""
    start, kind, n_trace, every, ndraw, space, seed = args
    rng = np.random.default_rng(seed)
    step = _stepper(kind, rng)
    x = start.copy() if kind == 'glob' else start.astype(np.int8)
    trace = [mc.mean_gap(mc.cooc_upper(x))]
    for _ in range(n_trace):
        x = step(x, every)
        trace.append(mc.mean_gap(mc.cooc_upper(x)))
    draws = []
    for _ in range(ndraw):
        x = step(x, space)
        draws.append(mc.mean_gap(mc.cooc_upper(x)))
    return np.array(trace), np.array(draws)


def _w_acf(args):
    """A long stationary run recording all 66 cells every `every` steps, for autocorrelation."""
    kind, burn, nrec, every, seed = args
    rng = np.random.default_rng(seed)
    step = _stepper(kind, rng)
    x = M.copy() if kind == 'glob' else M.astype(np.int8)
    x = step(x, burn)
    rec = np.empty((nrec, NTEST), dtype=np.int32)
    for i in range(nrec):
        x = step(x, every)
        rec[i] = mc.cooc_upper(x)
    return rec


# ------------------------------------------------------------------ analysis pieces
def legacy():
    seeds = [2024, 1, 2, 3, 4]
    with pool() as ex:
        runs = list(ex.map(_w_legacy, seeds))
    out = []
    for seed, (null, st) in zip(seeds, runs):
        mu, sd = null.mean(0), null.std(0)
        z = (OBS - mu) / (sd + 1e-9)
        g = mc.mean_gap(null)
        zg = (mc.mean_gap(OBS) - g.mean()) / g.std()
        sig = np.abs(z) > Z_PAPER
        near_pos = [mc.PAIRS[j] for j in np.flatnonzero(sig & (z > 0) & (mc.PAIR_DIST <= 2))]
        out.append({'seed': seed, 'gap_null_mean': float(g.mean()), 'gap_null_sd': float(g.std()),
                    'gap_z': float(zg), 'n_sig_383': int(sig.sum()),
                    'n_pos_383': int((sig & (z > 0)).sum()),
                    'n_neg_383': int((sig & (z < 0)).sum()),
                    'n_near_pos_383': len(near_pos), 'near_pos_383': near_pos,
                    'n_sig_337': int((np.abs(z) > Z_BONF).sum()),
                    'z01': float(z[0]), 'null_mean': mu.tolist(), 'null_sd': sd.tolist(),
                    'trades_executed_fraction': st['executed'] / (200 * 20000)})
        print(f'  seed {seed:5d}: gap null {g.mean():.3f} +/- {g.std():.3f}  z = {zg:+.2f}   '
              f'|z|>3.83: {sig.sum():2d} ({out[-1]["n_pos_383"]}+/{out[-1]["n_neg_383"]}-), '
              f'positive at |dk|<=2: {len(near_pos)}   |z|>3.37: {out[-1]["n_sig_337"]}   '
              f'trades executed {100 * out[-1]["trades_executed_fraction"]:.1f}%')
    return out


def acf(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    v = (x * x).mean()
    return [float((x[:-L] * x[L:]).mean() / v) if L < len(x) else float('nan') for L in lags]


def mixing(ref_mean, ref_sd):
    res = {}
    gr = mc.gale_ryser_start(M.sum(1), M.sum(0))
    assert (gr.sum(0) == M.sum(0)).all() and (gr.sum(1) == 4).all()
    print(f'  alternative start (greedy Gale-Ryser, same margins): mean gap '
          f'{mc.mean_gap(mc.cooc_upper(gr)):.3f} vs observed {mc.mean_gap(OBS):.3f}; '
          f'stationary {ref_mean:.3f} +/- {ref_sd:.4f}')
    # --- sequential sampler (the paper's): trace every 1,000 trades for 60,000, then 40 draws
    #     spaced 20,000 trades, 8 chains from each start
    specs = {'legacy': dict(n_trace=160, every=1000, ndraw=40, space=20000),
             'seq': dict(n_trace=60, every=1000, ndraw=40, space=20000),
             'glob': dict(n_trace=60, every=1, ndraw=200, space=K_GLOBAL)}
    for kind, sp in specs.items():
        jobs = [(st, kind, sp['n_trace'], sp['every'], sp['ndraw'], sp['space'], 1000 + 17 * i)
                for i, st in enumerate([M] * 8 + [gr] * 8)]
        with pool() as ex:
            out = list(ex.map(_w_trace, jobs))
        tr = np.array([o[0] for o in out])                    # (16, n_trace+1)
        dr = np.array([o[1] for o in out])                    # (16, ndraw)
        unit = 'global trades' if kind == 'glob' else 'trades'
        steps = np.arange(sp['n_trace'] + 1) * sp['every']
        dev = np.abs(tr - ref_mean) / ref_sd                  # in stationary SDs
        within = np.all(dev < 3, axis=0)
        ok = [bool(within[i:].all()) for i in range(len(within))]
        first = int(steps[ok.index(True)]) if any(ok) else -1
        # the strict criterion above is set by single-chain excursions; this one asks when the mean
        # of each start group (8 chains) is within 1 stationary SD and stays there
        gdev = np.abs(np.vstack([tr[:8].mean(0), tr[8:].mean(0)]) - ref_mean) / ref_sd
        gwithin = np.all(gdev < 1, axis=0)
        gok = [bool(gwithin[i:].all()) for i in range(len(gwithin))]
        first_group = int(steps[gok.index(True)]) if any(gok) else -1
        # Gelman-Rubin on the post-burn-in draws, chains = 16 (8 per start)
        m, n = dr.shape
        W = dr.var(1, ddof=1).mean()
        Bv = n * dr.mean(1).var(ddof=1)
        rhat = float(np.sqrt(((n - 1) / n * W + Bv / n) / W))
        diff = float(dr[:8].mean() - dr[8:].mean())
        se_diff = float(np.sqrt(dr[:8].var(ddof=1) / dr[:8].size + dr[8:].var(ddof=1) / dr[8:].size))
        chk = {int(s): [float(tr[:8, j].mean()), float(tr[8:, j].mean())]
               for j, s in enumerate(steps) if s in (0, sp['every'] * 2, sp['every'] * 5,
                                                     sp['every'] * 10, sp['every'] * 20,
                                                     sp['every'] * 40, sp['every'] * 60,
                                                     sp['every'] * 100, sp['every'] * 160)}
        res[kind] = {'unit': unit, 'trace_checkpoints_mean_gap_[observed_start,gr_start]': chk,
                     'all_16_chains_within_3sd_from': first,
                     'both_start_group_means_within_1sd_from': first_group, 'rhat_mean_gap': rhat,
                     'start_difference_after_burnin': diff, 'start_difference_se': se_diff}
        print(f'  [{kind}] all 16 chains within 3 stationary SD of the null mean from {first:,} '
              f'{unit} on (both start-group means within 1 SD from {first_group:,}); '
              f'R-hat (observed vs structured starts) = {rhat:.4f}; '
              f'mean-gap difference between starts {diff:+.4f} +/- {se_diff:.4f}')
    # --- autocorrelation along long stationary runs
    with pool() as ex:
        fl = ex.submit(_w_acf, ('legacy', 200000, 2000, 1000, 76))
        fs = ex.submit(_w_acf, ('seq', 60000, 2000, 1000, 77))
        fg = ex.submit(_w_acf, ('glob', 100, 20000, 1, 78))
        rl, rs, rg = fl.result(), fs.result(), fg.result()
    for kind, rec, lags, unit in (('legacy', rl, [1, 2, 5, 10, 20, 40], 1000),
                                  ('seq', rs, [1, 2, 5, 10, 20, 40], 1000),
                                  ('glob', rg, [1, 2, 5, 10, 20, 40], 1)):
        stats = np.column_stack([rec, mc.mean_gap(rec)])
        A_ = np.array([acf(stats[:, j], lags) for j in range(stats.shape[1])])
        res[kind]['acf_lags_steps'] = [L * unit for L in lags]
        res[kind]['acf_mean_gap'] = A_[-1].tolist()
        res[kind]['acf_max_abs_over_67_stats'] = np.abs(A_).max(0).tolist()
        lagtxt = ', '.join(f'{L * unit:,}: {g:+.3f}/{mx:.3f}' for L, g, mx in
                           zip(lags, A_[-1], np.abs(A_).max(0)))
        print(f'  [{kind}] lag ({"global trades" if kind == "glob" else "trades"}): '
              f'ACF of mean gap / max |ACF| over 66 cells + gap -> {lagtxt}')
    return res


def exchangeable(kind, B, k, seed, strata=None):
    ss = np.random.SeedSequence(seed)
    kids = ss.spawn(CHUNKS + 1)
    rng0 = np.random.default_rng(kids[0])
    if kind == 'glob':
        xstar = mc.curveball_global(M, k, rng0, strata)
        jobs = [(xstar, n, k, kids[i + 1], strata) for i, n in enumerate(split(B, CHUNKS))]
        fn = _w_global
    else:
        xstar = mc.curveball_seq(M.astype(np.int8), k, rng0).astype(bool)
        jobs = [(xstar, n, k, kids[i + 1]) for i, n in enumerate(split(B, CHUNKS))]
        fn = _w_seq
    t = time.time()
    with pool() as ex:
        null = np.vstack(list(ex.map(fn, jobs)))
    return null, time.time() - t


def infer(null, label):
    allm = np.vstack([OBS[None].astype(float), null.astype(float)])
    mu, sd = allm.mean(0), allm.std(0, ddof=1)
    Z = (allm - mu) / sd
    z0, Zn = Z[0], Z[1:]
    Bn = len(Zn)
    p_pair = (1 + (np.abs(Zn) >= np.abs(z0) - 1e-12).sum(0)) / (Bn + 1)
    T = np.abs(Zn).max(1)
    p_maxT = (1 + (T[:, None] >= np.abs(z0)[None] - 1e-12).sum(0)) / (Bn + 1)
    crit_maxT = float(np.quantile(T, 1 - ALPHA))
    bonf = p_pair <= ALPHA / NTEST
    fwer = p_maxT <= ALPHA
    g_obs, g_null = mc.mean_gap(OBS), mc.mean_gap(null)
    zg = (g_obs - g_null.mean()) / g_null.std(ddof=1)
    p_gap = mc.mc_pvalue(g_null, g_obs, 'lower')
    lr = np.log2(OBS / mu)
    q = np.quantile(np.log2(np.maximum(null, .5) / mu), [0.025, 0.975], axis=0)
    band = [np.log2(np.maximum(mu - crit_maxT * sd, .5) / mu), np.log2((mu + crit_maxT * sd) / mu)]
    tail = {'frac_null_abs_z_gt_3.37': float((np.abs(Zn) > Z_BONF).mean()),
            'nominal_3.37': ALPHA / NTEST,
            'frac_null_abs_z_gt_3.83': float((np.abs(Zn) > Z_PAPER).mean()),
            'nominal_3.83': float(2 * norm.sf(Z_PAPER)),
            'skew_range': [float(x) for x in (skew(Zn, 0).min(), skew(Zn, 0).max())],
            'excess_kurtosis_range': [float(x) for x in (kurtosis(Zn, 0).min(),
                                                          kurtosis(Zn, 0).max())],
            'maxT_95_quantile': crit_maxT,
            'normal_independent_maxT_95': float(norm.isf((1 - (1 - ALPHA) ** (1 / NTEST)) / 2))}
    normal = np.abs(z0) > Z_BONF
    near_pos_fwer = [mc.PAIRS[j] for j in np.flatnonzero(fwer & (z0 > 0) & (mc.PAIR_DIST <= 2))]
    near_pos_bonf = [mc.PAIRS[j] for j in np.flatnonzero(bonf & (z0 > 0) & (mc.PAIR_DIST <= 2))]
    far_pos_fwer = [mc.PAIRS[j] for j in np.flatnonzero(fwer & (z0 > 0) & (mc.PAIR_DIST > 2))]
    near_neg_fwer = [mc.PAIRS[j] for j in np.flatnonzero(fwer & (z0 < 0) & (mc.PAIR_DIST <= 3))]
    near_all = [mc.PAIRS[j] for j in np.flatnonzero(mc.PAIR_DIST <= 2)]
    pairs = []
    for j, (a, b) in enumerate(mc.PAIRS):
        pairs.append({'pair': [a, b], 'dk': int(b - a), 'observed': int(OBS[j]),
                      'null_mean': float(mu[j]), 'null_sd': float(sd[j]), 'z': float(z0[j]),
                      'log2_ratio': float(lr[j]), 'ratio': float(2 ** lr[j]),
                      'null_log2_q025': float(q[0, j]), 'null_log2_q975': float(q[1, j]),
                      'simult_log2_lo': float(band[0][j]), 'simult_log2_hi': float(band[1][j]),
                      'p_mc': float(p_pair[j]), 'p_maxT': float(p_maxT[j]),
                      'sig_bonferroni_mc': bool(bonf[j]), 'sig_maxT': bool(fwer[j]),
                      'sig_normal_337': bool(normal[j])})
    s = {'label': label, 'B': int(Bn), 'gap_observed': float(g_obs),
         'gap_null_mean': float(g_null.mean()), 'gap_null_sd': float(g_null.std(ddof=1)),
         'gap_z': float(zg), 'gap_p_mc': float(p_gap), 'bonferroni_alpha': ALPHA / NTEST,
         'n_sig_bonferroni_mc': int(bonf.sum()),
         'n_sig_bonferroni_mc_pos': int((bonf & (z0 > 0)).sum()),
         'n_sig_maxT': int(fwer.sum()), 'n_sig_maxT_pos': int((fwer & (z0 > 0)).sum()),
         'n_sig_maxT_neg': int((fwer & (z0 < 0)).sum()),
         'n_sig_normal_337': int(normal.sum()), 'n_sig_normal_383': int((np.abs(z0) > Z_PAPER).sum()),
         'near_pos_maxT': near_pos_fwer, 'near_pos_bonferroni_mc': near_pos_bonf,
         'far_pos_maxT': far_pos_fwer, 'near_neg_maxT_dk_le3': near_neg_fwer,
         'all_near_pairs_dk_le2': near_all,
         'mean_log2_ratio_dk_le2': float(lr[mc.PAIR_DIST <= 2].mean()),
         'mean_log2_ratio_dk_gt2': float(lr[mc.PAIR_DIST > 2].mean()),
         'max_ratio': float(2 ** lr.max()), 'tail_check': tail, 'pairs': pairs}
    print(f'\n  [{label}] B = {Bn:,}: mean gap {g_obs:.3f} vs null {g_null.mean():.3f} +/- '
          f'{g_null.std(ddof=1):.4f}  -> z = {zg:+.2f}, Monte-Carlo p = {p_gap:.1e} (= 1/(B+1))')
    print(f'    pairs significant: Monte-Carlo Bonferroni {bonf.sum()} '
          f'({(bonf & (z0 > 0)).sum()}+), max-|z| FWER {fwer.sum()} '
          f'({(fwer & (z0 > 0)).sum()}+/{(fwer & (z0 < 0)).sum()}-, crit |z| > {crit_maxT:.3f}), '
          f'normal |z|>3.37: {normal.sum()}, normal |z|>3.83: {(np.abs(z0) > Z_PAPER).sum()}')
    print(f'    positive-excess pairs at |dk|<=2 (FWER): {len(near_pos_fwer)} -> {near_pos_fwer}')
    print(f'    positive-excess pairs at |dk|>2  (FWER): {far_pos_fwer}')
    print(f'    deficits at |dk|<=3 (FWER): {near_neg_fwer}')
    print(f'    normal tail check: null P(|z|>3.37) = {tail["frac_null_abs_z_gt_3.37"]:.2e} '
          f'(nominal {ALPHA / NTEST:.2e}); P(|z|>3.83) = {tail["frac_null_abs_z_gt_3.83"]:.2e} '
          f'(nominal {tail["nominal_3.83"]:.2e}); skew {tail["skew_range"][0]:+.3f}..'
          f'{tail["skew_range"][1]:+.3f}, excess kurtosis {tail["excess_kurtosis_range"][0]:+.3f}..'
          f'{tail["excess_kurtosis_range"][1]:+.3f}')
    nb = {'within_quad': {}, 'cross_quad': {}}             # index neighbours (k, k+1)
    for j, (a, b) in enumerate(mc.PAIRS):
        if b - a == 1:
            nb['within_quad' if a % 2 == 0 else 'cross_quad'][f'{a},{b}'] = {
                'ratio': float(2 ** lr[j]), 'z': float(z0[j]), 'sig_maxT': bool(fwer[j])}
    s['neighbour_pairs'] = nb
    for key, d in nb.items():
        print(f'    {key:<11} neighbours O/E: ' + ', '.join(
            f'{k} {v["ratio"]:.2f}{"*" if v["sig_maxT"] else ""}' for k, v in d.items())
            + '   (* family-wise significant)')
    print(f'    effect: mean log2(O/E) {s["mean_log2_ratio_dk_le2"]:+.3f} at |dk|<=2 '
          f'({2 ** s["mean_log2_ratio_dk_le2"]:.2f}x), {s["mean_log2_ratio_dk_gt2"]:+.3f} beyond '
          f'({2 ** s["mean_log2_ratio_dk_gt2"]:.2f}x); strongest pair {s["max_ratio"]:.2f}x')
    return s, mu, sd


if __name__ == '__main__':
    t0 = time.time()
    print(f'observed: {len(M)} sockets, row sums {sorted(set(M.sum(1)))}, mean gap '
          f'{mc.mean_gap(OBS):.3f}; two-sided Bonferroni critical |z| for 66 tests = {Z_BONF:.3f} '
          f'(the |z| > {Z_PAPER} cut used previously is a family-wise alpha of '
          f'{NTEST * 2 * norm.sf(Z_PAPER):.4f})')

    print('\n--- 1. published procedure, reproduced (200 chained draws x 20,000 trades) ---')
    leg = legacy()

    print(f'\n--- 3. exchangeable null (Besag-Clifford parallel), global trades, k = {K_GLOBAL} ---')
    null_g, dt = exchangeable('glob', B_MAIN, K_GLOBAL, 20260911)
    assert null_g.shape == (B_MAIN, NTEST)
    main, mu, sd = infer(null_g, 'global curveball')
    print(f'    ({dt:.0f} s on {WORKERS} workers)')

    print(f'\n--- 2. mixing diagnostics ---')
    mix = mixing(main['gap_null_mean'], main['gap_null_sd'])

    print(f'\n--- 3b. confirmation with the paper\'s sequential sampler, k = {K_SEQ:,} trades ---')
    null_s, dt = exchangeable('seq', B_SEQ, K_SEQ, 20260912)
    conf, mu_s, sd_s = infer(null_s, 'sequential curveball')
    dm = (mu_s - mu) / (sd / np.sqrt(B_SEQ))
    rs = sd_s / sd
    agree = int(sum(a['sig_maxT'] == b['sig_maxT'] for a, b in zip(main['pairs'], conf['pairs'])))
    print(f'    ({dt:.0f} s) null means agree: max |difference| = {np.abs(dm).max():.2f} MC standard '
          f'errors; SD ratio sequential/global {rs.min():.3f}..{rs.max():.3f}; '
          f'FWER decisions agree on {agree}/66 pairs')
    conf['vs_global'] = {'max_abs_mean_diff_in_mcse': float(np.abs(dm).max()),
                         'sd_ratio_range': [float(rs.min()), float(rs.max())],
                         'fwer_decisions_agree': agree}

    print('\n--- 5. sensitivity: trades within lot (each lot keeps its own marginals) ---')
    null_l, dt = exchangeable('glob', B_MAIN, K_GLOBAL, 20260913, strata=LOT)
    strat, _, _ = infer(null_l, 'lot-stratified curveball')
    print(f'    ({dt:.0f} s)')

    out = {'z_bonferroni_66_two_sided': Z_BONF, 'z_cut_used_previously': Z_PAPER,
           'familywise_alpha_of_previous_cut': float(NTEST * 2 * norm.sf(Z_PAPER)),
           'legacy_runs': leg, 'mixing': mix, 'k_global': K_GLOBAL, 'k_seq': K_SEQ,
           'main': main, 'confirmation_sequential': conf, 'lot_stratified': strat}
    # The wall-clock runtime is printed below, never written into the JSON: this file is pinned in
    # DATA.sha256, and a timing field would make the checksum change on every re-run even when
    # every number is bit-identical.

    json.dump(out, open(root / 'analysis' / 'model_curveball.json', 'w'), indent=1)
    print(f'\nwrote analysis/model_curveball.json ({time.time() - t0:.0f} s total)')
