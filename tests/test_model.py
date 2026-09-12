"""Tests for the pattern statistics and the pairwise-interaction model (MODEL cluster).

Run:  .venv/bin/python -m pytest -q tests/test_model.py
"""
import sys
from itertools import combinations, product
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))

import model_core as mc  # noqa: E402

M = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy')
K = np.load(ROOT / 'analysis' / 'harvest_keys_full.npy')


# ------------------------------------------------------------------ pattern space
def test_pattern_space_arithmetic():
    assert mc.NPAT == comb(12, 4) == 495
    assert mc.NPAIR == comb(12, 2) == 66
    assert (mc.A.sum(1) == 4).all()
    assert (mc.PAIRMAT.sum(1) == 6).all()
    assert len({tuple(np.flatnonzero(r)) for r in M}) == 443
    assert (M.sum(1) == 4).all() and len(M) == 1962


def test_adj_is_number_of_unordered_k_kplus1_pairs():
    for S, a in zip(mc.PATS, mc.ADJ):
        brute = sum(1 for x, y in combinations(S, 2) if abs(x - y) == 1)
        assert a == brute == sum(1 for k in range(11) if k in S and k + 1 in S)
    assert mc.ADJ.min() == 0 and mc.ADJ.max() == 3


def test_pattern_index_roundtrip():
    idx = mc.pattern_index(M)
    back = mc.A[idx]
    assert (back == M).all()
    with pytest.raises(ValueError):
        mc.pattern_index(np.ones((1, 12)))


def test_mean_gap_from_cooc_matches_direct():
    rows = M[:300]
    direct = np.mean([abs(x - y) for r in rows for x, y in combinations(np.flatnonzero(r), 2)])
    assert mc.mean_gap(mc.cooc_upper(rows)) == pytest.approx(direct, abs=1e-12)


# ------------------------------------------------------------------ pairwise model
def test_beta_is_identified_only_up_to_a_constant():
    rng = np.random.default_rng(0)
    beta, gamma = rng.normal(size=12), 0.4
    p1 = mc.probs(beta, gamma)
    p2 = mc.probs(beta + 3.7, gamma)
    assert np.allclose(p1, p2, atol=1e-15)


def test_gradient_matches_finite_differences():
    counts = mc.pattern_counts(M)
    X = mc.design(True)
    theta = np.random.default_rng(1).normal(scale=0.3, size=X.shape[1])
    N = counts.sum()

    def ll(th):
        _, logZ, _, _ = mc._moments(X, th)
        return th @ (X.T @ counts) - N * logZ

    _, _, mu, _ = mc._moments(X, theta)
    g = X.T @ counts - N * mu
    h = 1e-6
    num = np.array([(ll(theta + h * e) - ll(theta - h * e)) / (2 * h) for e in np.eye(len(theta))])
    assert np.allclose(g, num, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('sel', ['A', 'B', 'fleet'])
def test_newton_converges_and_matches_lbfgs_and_legacy(sel):
    rack = K[:, 0] // 20
    mask = {'A': rack < 22, 'B': rack >= 22, 'fleet': np.ones(len(M), bool)}[sel]
    c = mc.pattern_counts(M[mask])
    f = mc.fit_newton(c)
    lb = mc.fit_lbfgs(c)
    fx = mc.fit_fixed_step(c)
    assert f['grad_norm'] < 1e-8
    assert abs(f['beta'].sum()) < 1e-12
    assert np.abs(f['beta'] - lb['beta']).max() < 1e-6 and abs(f['gamma'] - lb['gamma']) < 1e-6
    assert np.abs(f['beta'] - fx['beta']).max() < 1e-6 and abs(f['gamma'] - fx['gamma']) < 1e-6
    assert f['hess_eig'].min() > 0                      # strictly concave in the identified space


def test_published_table_viii_values():
    rack = K[:, 0] // 20
    gA = mc.fit_newton(mc.pattern_counts(M[rack < 22]))
    gB = mc.fit_newton(mc.pattern_counts(M[rack >= 22]))
    assert round(gA['gamma'], 3) == 0.743 and round(gB['gamma'], 3) == 0.434
    assert round(gA['beta'][0], 2) == 2.51
    lr = 2 * (gA['ll'] - mc.fit_newton(mc.pattern_counts(M[rack < 22]), gamma_free=False)['ll'])
    assert round(lr, 1) == 174.2


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(2)
    beta = rng.normal(scale=0.5, size=12)
    beta -= beta.mean()
    p = mc.probs(beta, 0.6)
    c = rng.multinomial(200_000, p).astype(float)
    f = mc.fit_newton(c)
    se = np.sqrt(np.diag(f['inv_info']))
    assert abs(f['gamma'] - 0.6) < 4 * se[-1]
    assert np.abs(f['beta'] - beta).max() < 0.05


def test_model_based_se_is_calibrated_and_conditional_se_is_too_small():
    """The full-information SE matches the sampling SD of gamma-hat; 1/sqrt(N Var adj), which
    holds beta fixed, is smaller than that SD."""
    rack = K[:, 0] // 20
    f = mc.fit_newton(mc.pattern_counts(M[rack < 22]))
    N = 880
    rng = np.random.default_rng(3)
    g = np.array([mc.fit_newton(rng.multinomial(N, f['p']).astype(float),
                                theta0=f['theta'])['gamma'] for _ in range(600)])
    se_full = np.sqrt(f['inv_info'][-1, -1])
    se_cond = mc.se_gamma_conditional(np.full(1, N), f['beta'], f['gamma'])
    assert abs(g.std(ddof=1) / se_full - 1) < 0.12
    assert se_cond < 0.85 * g.std(ddof=1)


# ------------------------------------------------------------------ marginal test
def test_fixed_cardinality_chisquare_is_8_over_11_chi2_11():
    sys.path.insert(0, str(ROOT / 'analysis'))
    import model_marginals as mm
    null = mm.mc_null(400, 3000, np.random.default_rng(4), chunk=300)
    assert abs(null.mean() - 8.0) < 0.25                 # multinomial theory would give 11
    assert abs(null.var() - 2 * (8 / 11) ** 2 * 11) < 1.8


def test_cluster_wald_is_calm_on_uniform_data():
    import model_marginals as mm
    rng = np.random.default_rng(5)
    X = np.zeros((2000, 12))
    X[np.arange(2000)[:, None], np.argsort(rng.random((2000, 12)), 1)[:, :4]] = 1
    res = mm.cluster_wald(X, np.repeat(np.arange(50), 40))
    assert res['p_chi2'] > 1e-3


# ------------------------------------------------------------------ curveball
def _enumerate(row_sums, col_sums):
    C = len(col_sums)
    opts = [[r for r in product((0, 1), repeat=C) if sum(r) == s] for s in row_sums]
    out = []
    for rows in product(*opts):
        if tuple(np.array(rows).sum(0)) == tuple(col_sums):
            out.append(np.array(rows, dtype=bool).tobytes())
    return out


def _uniformity_pvalue(step, start, states, n_samples, thin, rng):
    index = {s: i for i, s in enumerate(states)}
    x = start.copy()
    for _ in range(50):
        x = step(x, rng)
    hits = np.zeros(len(states))
    for _ in range(n_samples):
        for _ in range(thin):
            x = step(x, rng)
        hits[index[np.asarray(x, dtype=bool).tobytes()]] += 1
    e = n_samples / len(states)
    return chi2.sf(((hits - e) ** 2 / e).sum(), len(states) - 1), (hits > 0).mean()


ROWS, COLS = (3, 2, 2, 3, 2), (3, 3, 2, 2, 2)


@pytest.fixture(scope='module')
def small_space():
    states = _enumerate(ROWS, COLS)
    start = np.frombuffer(states[0], dtype=bool).reshape(len(ROWS), len(COLS)).copy()
    return states, start


def test_global_curveball_is_uniform_on_small_instance(small_space):
    states, start = small_space
    assert len(states) > 50
    p, cover = _uniformity_pvalue(lambda x, r: mc.curveball_global(x, 1, r), start, states,
                                  60 * len(states), 3, np.random.default_rng(6))
    assert cover == 1.0 and p > 1e-3


def test_sequential_curveball_is_uniform_on_small_instance(small_space):
    states, start = small_space
    p, cover = _uniformity_pvalue(lambda x, r: mc.curveball_seq(x, 4, r).astype(bool), start,
                                  states, 60 * len(states), 2, np.random.default_rng(7))
    assert cover == 1.0 and p > 1e-3


def test_legacy_curveball_cannot_trade_overlapping_rows():
    """Rows {0,1}, {0,2}, {1,2} pairwise share a column: the published trade never executes, the
    corrected one reaches all six matrices with these margins."""
    x = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
    rng = np.random.default_rng(8)
    st = {}
    y = mc.curveball_seq_legacy(x, 2000, rng, st)
    assert (y == x).all() and st['executed'] == 0 and st['attempted'] > 0
    seen = {mc.curveball_seq(x, n, np.random.default_rng(n)).tobytes() for n in range(1, 200)}
    assert len(seen) == 6 == len(_enumerate((2, 2, 2), (2, 2, 2)))


@pytest.mark.parametrize('sampler', ['global', 'seq', 'legacy', 'strata'])
def test_curveball_preserves_margins(sampler):
    rng = np.random.default_rng(9)
    lot = (K[:, 0] // 20 >= 22).astype(int)
    if sampler == 'global':
        y = mc.curveball_global(M, 30, rng)
    elif sampler == 'strata':
        y = mc.curveball_global(M, 30, rng, strata=lot)
        for s in (0, 1):
            assert (y[lot == s].sum(0) == M[lot == s].sum(0)).all()
    elif sampler == 'seq':
        y = mc.curveball_seq(M, 20000, rng)
    else:
        y = mc.curveball_seq_legacy(M, 20000, rng)
    assert (y.sum(1) == 4).all() and (y.sum(0) == M.sum(0)).all()
    assert (y != M).any()


def test_mc_pvalue_convention():
    assert mc.mc_pvalue(np.zeros(99), 1.0) == pytest.approx(1 / 100)
    assert mc.mc_pvalue(np.arange(10.0), 4.5) == pytest.approx(6 / 11)
    assert mc.mc_pvalue(np.arange(10.0), 4.5, 'lower') == pytest.approx(6 / 11)


def test_besag_clifford_pvalue_is_valid_under_the_null(small_space):
    """If the observed matrix is itself a stationary draw, the parallel-method p-value satisfies
    P(p <= alpha) <= alpha (here alpha = 1/20, B = 19)."""
    states, start = small_space
    rng = np.random.default_rng(10)
    stat = lambda x: int((x[:, 0] & x[:, 1]).sum())          # co-occurrence of columns 0 and 1
    hits, reps = 0, 600
    for _ in range(reps):
        x0 = mc.curveball_global(start, 25, rng)            # a stationary "observation"
        xs = mc.curveball_global(x0, 5, rng)                # backward run to the hidden state
        null = [stat(mc.curveball_global(xs, 5, rng)) for _ in range(19)]
        hits += mc.mc_pvalue(null, stat(x0)) <= 1 / 20
    assert hits / reps < 1 / 20 + 3 * np.sqrt(0.05 * 0.95 / reps)


def test_exchangeable_null_does_not_depend_on_worker_count(monkeypatch):
    """The null draws are split into a fixed number of seeded chunks, so the process count only
    changes the speed, never the numbers (curveball_null.exchangeable)."""
    import curveball_null as cn
    monkeypatch.setattr(cn, 'CHUNKS', 4)
    runs = []
    for w in (1, 3):
        monkeypatch.setattr(cn, 'WORKERS', w)
        null, _ = cn.exchangeable('glob', 10, 2, 123)
        runs.append(null)
    assert runs[0].shape == (10, mc.NPAIR) and np.array_equal(runs[0], runs[1])
    lot = (K[:, 0] // 20 >= 22).astype(int)
    null, _ = cn.exchangeable('glob', 6, 2, 124, strata=lot)
    assert (null.sum(1) == mc.cooc_upper(M).sum()).all()   # 6 pairs per socket, always


def test_gale_ryser_start_has_the_margins():
    g = mc.gale_ryser_start(M.sum(1), M.sum(0))
    assert (g.sum(1) == 4).all() and (g.sum(0) == M.sum(0)).all()


# ------------------------------------------------------------------ replicate checks
def test_calibrated_residuals_are_standard_under_the_model():
    """Data drawn from the fitted model must give calibrated residual z of unit scale, while the
    Pearson residual is too small for adjacent pairs (its variance is not the mean)."""
    rack = K[:, 0] // 20
    f = mc.fit_newton(mc.pattern_counts(M[rack >= 22]))
    rng = np.random.default_rng(11)
    zs = []
    for _ in range(8):
        c = rng.multinomial(1082, f['p']).astype(float)
        rc = mc.replicate_checks(c, 300, rng)
        zs.append(rc['z_cal'])
        ratio = rc['sd_r'] / np.sqrt(rc['E'])
        adjacent = mc.PAIR_DIST == 1
        assert ratio.max() < 0.95 and ratio[adjacent].mean() < ratio[~adjacent].mean()
    z = np.concatenate(zs)
    assert 0.8 < np.sqrt((z ** 2).mean()) < 1.2
    assert np.abs(z).max() < 4.5


def test_balanced_panel_slice_decoding():
    import balanced_panel as bp
    active_cores = [c for k in range(12) if k not in (0, 3, 4, 11) for c in (2 * k, 2 * k + 1)]
    mask = sum(1 << c for c in active_cores)
    harv = bp.bits([sum(1 << k for k in (0, 3, 4, 11))])[0]
    decoded = np.array([((mask >> (2 * k)) & 3) == 0 for k in range(12)])
    assert (decoded == harv.astype(bool)).all()


# ------------------------------------------------------------------ quad-aware adjacency
def test_quad_adjacency_split_sums_to_adj():
    assert np.array_equal(mc.ADJ_IN + mc.ADJ_X, mc.ADJ)
    S = (0, 1, 2, 3)                                        # two within-quad, one cross-quad pair
    i = mc.PATS.index(S)
    assert mc.ADJ_IN[i] == 2 and mc.ADJ_X[i] == 1


def test_fit_design_reproduces_fit_newton():
    c = mc.pattern_counts(M)
    X1 = np.column_stack([mc.A @ mc.L, mc.ADJ])
    f = mc.fit_design(X1, c)
    g = mc.fit_newton(c)
    assert f['grad_norm'] < 1e-8 and abs(f['theta'][-1] - g['gamma']) < 1e-9
