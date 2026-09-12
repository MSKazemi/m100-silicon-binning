"""Unit tests for analysis/thermal_lib.py (THERMAL cluster, review response 2026-09).

The load-bearing one is the exact ordering search: the paper's "best 1-D ordering" came from a
200-restart hill climb; it is now an exact dynamic programme over prefix sets, and these tests
check it against brute force on every ordering of smaller problems.
"""
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import thermal_lib as tl  # noqa: E402


def _brute(W):
    n = len(W)
    return sorted((tl.arrangement_cost(W, p), p) for p in permutations(range(n)))


def _sym(rng, n, neg=True):
    A = rng.normal(size=(n, n)) if neg else rng.random((n, n))
    W = (A + A.T) / 2
    np.fill_diagonal(W, 0.0)
    return W


@pytest.mark.parametrize('seed', range(6))
def test_exact_arrangement_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    W = _sym(rng, 7)
    cost, order = tl.exact_linear_arrangement(W)
    bf = _brute(W)
    assert cost == pytest.approx(bf[0][0], abs=1e-9)
    assert tl.arrangement_cost(W, order) == pytest.approx(cost, abs=1e-9)


@pytest.mark.parametrize('seed', range(4))
def test_kbest_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    W = _sym(rng, 6)
    kb = tl.kbest_linear_arrangements(W, 60)
    bf = _brute(W)
    np.testing.assert_allclose([c for c, _ in kb], [c for c, _ in bf[:60]], atol=1e-9)
    for c, o in kb:
        assert tl.arrangement_cost(W, o) == pytest.approx(c, abs=1e-9)
        assert sorted(o) == list(range(6))


@pytest.mark.parametrize('seed', range(4))
def test_best_order_is_the_most_negative_pearson_score(seed):
    """Pearson corr(distance, similarity) over all orderings is minimised by the DP optimum."""
    rng = np.random.default_rng(200 + seed)
    n = 7
    S = _sym(rng, n)
    pairs = list(combinations(range(n), 2))
    cv = np.array([S[a, b] for a, b in pairs])
    score, order = tl.best_order(S)
    allscores = [tl.order_score(cv, np.array(p), pairs) for p in permutations(range(n))]
    assert score == pytest.approx(min(allscores), abs=1e-12)


def test_distance_multiset_is_permutation_invariant():
    """Why the reduction is exact: every 1-D ordering has the same multiset of distances."""
    rng = np.random.default_rng(3)
    n = 12
    ref = sorted(abs(a - b) for a, b in combinations(range(n), 2))
    for _ in range(20):
        pos = tl.order_positions(rng.permutation(n))
        assert sorted(abs(pos[a] - pos[b]) for a, b in combinations(range(n), 2)) == ref


def test_exact_12_slice_beats_or_matches_every_random_and_local_ordering():
    rng = np.random.default_rng(9)
    S = _sym(rng, 12)
    cv = tl.cells(np.where(np.eye(12, dtype=bool), np.nan, S))
    score, order = tl.best_order(S)
    for _ in range(3000):
        assert tl.order_score(cv, rng.permutation(12)) >= score - 1e-12
    for i, j in combinations(range(12), 2):              # no single swap improves it
        q = order.copy(); q[i], q[j] = q[j], q[i]
        assert tl.order_score(cv, q) >= score - 1e-12


def test_residual_corr_matches_paper_formula_and_rows_sum_to_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 16)) + rng.normal(size=(500, 1)) * 3
    C, sd = tl.residual_corr(X)
    R = X - X.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
    ref = (R.T @ R) / len(R) / np.outer(R.std(0), R.std(0))
    np.testing.assert_allclose(C, ref)
    cov = C * np.outer(sd, sd)
    np.testing.assert_allclose(cov.sum(1), 0, atol=1e-10)   # the common-mode constraint


def test_pair_kinds():
    assert tl.pair_kind(4, 5) == 'within'
    assert tl.pair_kind(5, 6) == 'samequad'
    assert tl.pair_kind(7, 8) == 'crossquad'
    assert tl.pair_kind(3, 5) is None
    kinds = [tl.pair_kind(i, i + 1) for i in range(23)]
    assert kinds.count('within') == 12 and kinds.count('samequad') == 6
    assert kinds.count('crossquad') == 5


def test_unpack_roundtrip_and_slice_similarity():
    rng = np.random.default_rng(2)
    idx = np.sort(rng.choice(24, 16, replace=False))
    A = rng.normal(size=(16, 16)); C = (A + A.T) / 2; np.fill_diagonal(C, 1)
    M = tl.unpack(C[tl.IU16].astype('float32'), idx)
    np.testing.assert_allclose(M[np.ix_(idx, idx)], C, atol=1e-6)
    S = tl.slice_similarity(M)
    for a, b in combinations(range(12), 2):
        v = [M[2 * a + i, 2 * b + j] for i in (0, 1) for j in (0, 1)]
        if np.isnan(v).any():
            assert np.isnan(S[a, b])
        else:
            assert S[a, b] == pytest.approx(np.mean(v))


def test_pool_support_and_fisher():
    a = np.array([[np.nan, 0.2], [0.4, np.nan]])
    b = np.array([[0.1, 0.2], [np.nan, np.nan]])
    m, s = tl.pool([a, b])
    assert s.tolist() == [[1, 2], [1, 0]]
    assert m[0, 1] == pytest.approx(0.2) and m[0, 0] == pytest.approx(0.1)
    mz, _ = tl.pool([a, b], fisher=True)
    assert mz[0, 1] == pytest.approx(0.2)
    assert np.isnan(m[1, 1])


def test_twoway_recovers_cell_effects_under_socket_offsets():
    rng = np.random.default_rng(4)
    ns, nc = 400, 10
    theta = rng.normal(size=nc); u = rng.normal(size=ns) * 2
    Y = u[:, None] + theta[None, :]
    mask = rng.random((ns, nc)) < 0.4
    # make composition depend on the offset: high-offset sockets observe the first cells
    mask |= (u[:, None] > 1) & (np.arange(nc)[None, :] < 3)
    Y = np.where(mask, Y, np.nan)
    th = tl.twoway_cells(Y)
    np.testing.assert_allclose(th - th.mean(), theta - theta.mean(), atol=1e-6)


def test_poststratified_pool_reweights_composition():
    # stratum 0 (80% of units) sees only cell 0; stratum 1 sees both; the raw pool of cell 1
    # would be all stratum 1, the post-stratified one must still be stratum 1's mean there
    mats = np.array([[[0.0, np.nan]]] * 8 + [[[1.0, 2.0]]] * 2)
    strata = np.array([0] * 8 + [1] * 2)
    out = tl.poststratified_pool(mats, strata)
    assert out[0, 0] == pytest.approx(0.8 * 0.0 + 0.2 * 1.0)
    assert out[0, 1] == pytest.approx(2.0)


def test_mc_pvalue_is_never_zero():
    assert tl.mc_pvalue(np.zeros(999), -1.0) == pytest.approx(1 / 1000)
    assert tl.mc_pvalue(np.array([-2.0, 0.0, 1.0]), -1.0) == pytest.approx(2 / 4)


def test_sign_flip_test():
    rng = np.random.default_rng(0)
    p, m = tl.sign_flip_test(np.ones(20), rng, B=9999)
    assert m == 1.0 and p == pytest.approx(1 / 10000, abs=2e-4)
    p0, _ = tl.sign_flip_test(np.r_[np.ones(10), -np.ones(10)], rng, B=9999)
    assert p0 > 0.3


def test_cluster_resample_keeps_whole_clusters():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
    for _ in range(50):
        idx = tl.cluster_resample(cl, rng)
        drawn = cl[idx]
        for k in np.unique(drawn):
            assert (drawn == k).sum() % (cl == k).sum() == 0


def test_common_mode_removal_cannot_create_an_index_gradient():
    """Exchangeable covariance (no geometry) + real-looking active sets: after the analysis'
    common-mode step every cross-slice residual correlation is identical, so no ordering can
    score better than another -- the projection rescales geometry, it does not invent it."""
    rng = np.random.default_rng(5)
    rho_x, rho_w = 0.3, 0.6
    Sig = np.full((24, 24), rho_x)
    for k in range(12):
        Sig[2 * k, 2 * k + 1] = Sig[2 * k + 1, 2 * k] = rho_w
    np.fill_diagonal(Sig, 1.0)
    for _ in range(20):
        sl = np.sort(rng.choice(12, 8, replace=False))
        act = np.sort(np.r_[2 * sl, 2 * sl + 1])
        Cp = tl.projected_corr(Sig, act)
        M = tl.embed24(Cp, act)
        S = tl.slice_similarity(M)
        v = tl.cells(S)
        v = v[~np.isnan(v)]
        assert np.ptp(v) < 1e-12


def test_projection_commutes_with_relabelling():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(24, 24)); Sig = A @ A.T
    act = np.sort(rng.choice(24, 16, replace=False))
    perm = rng.permutation(16)
    C1 = tl.projected_corr(Sig, act)[np.ix_(perm, perm)]
    C2 = tl.projected_corr(Sig, act[perm])
    np.testing.assert_allclose(C1, C2, atol=1e-10)


def test_canonical_and_kendall():
    o = np.array([3, 2, 1, 0])
    assert tl.canonical(o).tolist() == [0, 1, 2, 3]
    assert tl.kendall_distance_reflect(o) == 0
    assert tl.kendall_distance_reflect(np.array([1, 0, 2, 3])) == 1
    assert tl.spearman_abs(np.array([3, 2, 1, 0])) == pytest.approx(1.0)


def _sockets(rng, n=400):
    SL = np.zeros((n, 12), dtype=bool)
    for i in range(n):
        SL[i, np.sort(rng.choice(12, 8, replace=False))] = True
    return SL


def _projected_S(SL, positions):
    out = []
    for i in range(len(SL)):
        act = np.sort(np.r_[2 * np.flatnonzero(SL[i]), 2 * np.flatnonzero(SL[i]) + 1])
        pos = positions(act)
        K = np.exp(-np.abs(pos[:, None] - pos[None, :]) / 3.0) + np.eye(16)
        P = np.eye(16) - 1.0 / 16
        R = P @ K @ P; sd = np.sqrt(np.diag(R))
        out.append(tl.slice_similarity(tl.embed24(R / np.outer(sd, sd), act)))
    return np.stack(out)


def test_natural_experiment_larger_for_rank_kernel_than_stationary_index_kernel():
    """A stationary kernel in index distance and one in the rank among surviving cores, both
    pushed through the analysis' common-mode step: the statistic is much larger for the latter.
    (Neither is a physical model -- see the dark-source test below.)"""
    rng = np.random.default_rng(12)
    SL = _sockets(rng)
    idx = tl.natural_experiment(_projected_S(SL, lambda a: a.astype(float)), SL)
    logi = tl.natural_experiment(_projected_S(SL, lambda a: np.arange(16.0)), SL)
    assert logi > idx + 0.05
    assert idx < 0.1


def _model_S(SL, corr_of_active):
    out = []
    for i in range(len(SL)):
        act = np.sort(np.r_[2 * np.flatnonzero(SL[i]), 2 * np.flatnonzero(SL[i]) + 1])
        out.append(tl.slice_similarity(tl.embed24(corr_of_active(act), act)))
    return np.stack(out)


def test_dark_harvested_slices_alone_produce_a_natural_experiment_effect():
    """Why the harvested-between contrast is not a physical-versus-logical test: in a purely
    physical source model on IBM's layout, with no logical ordering anywhere, a harvested (dark)
    slice next to one core removes an independent heat source and raises that core's residual
    correlation with distant cores -- clearly more than a stationary field at the same sensors."""
    rng = np.random.default_rng(12)
    SL = _sockets(rng)
    xy = tl.ibm_core_xy()

    def field(a):
        D = np.linalg.norm(xy[a][:, None] - xy[a][None], axis=-1)
        return tl.projected_corr(np.exp(-D / 0.3) + 0.3 * np.eye(16), np.arange(16))

    dark = tl.natural_experiment(_model_S(SL, lambda a: tl.heat_source_corr(a, xy, 0.3, 0.3)), SL)
    stat = tl.natural_experiment(_model_S(SL, field), SL)
    assert dark > 0.07
    assert dark > 2 * stat


def test_heat_source_with_all_sources_equals_projected_source_covariance():
    rng = np.random.default_rng(21)
    xy = tl.ibm_core_xy()
    act = np.sort(rng.choice(24, 16, replace=False))
    D = np.linalg.norm(xy[act][:, None] - xy[act][None], axis=-1)
    G = np.exp(-D / 0.25)
    ref = tl.projected_corr(G @ G.T + 0.5 * np.eye(16), np.arange(16))
    np.testing.assert_allclose(tl.heat_source_corr(act, xy, 0.25, 0.5), ref, atol=1e-12)
    np.testing.assert_allclose(np.diag(ref), 1.0, atol=1e-12)


def test_ibm_layout_topology():
    """IBM Registers Spec v1.2 Fig. 2: quads EQ0/2/4 in the top band, EQ1/3/5 in the bottom band,
    columns left/middle/right; four cores of a quad stacked, EX0 at the outer die edge."""
    topo = tl.ibm_topology()
    assert topo[:, 0].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert topo[:, 1].tolist() == [0, 0, 1, 1] * 3
    xy = tl.ibm_core_xy()
    for q in range(6):
        c = xy[4 * q:4 * q + 4]
        assert np.ptp(c[:, 0]) == 0                                  # one column
        steps = np.diff(c[:, 1])
        assert np.allclose(np.abs(steps), steps[0] if q % 2 == 0 else -steps[0])
        assert (steps > 0).all() if q % 2 == 0 else (steps < 0).all()  # EX0 at the outer edge
    top = xy[[c for c in range(24) if (c // 4) % 2 == 0], 1].max()
    bot = xy[[c for c in range(24) if (c // 4) % 2 == 1], 1].min()
    assert bot > top                                                 # the nest band between
    s = tl.ibm_slice_xy()
    D = np.linalg.norm(s[:, None] - s[None], axis=-1); np.fill_diagonal(D, np.inf)
    for q in range(6):                   # the two slices of a quad are each other's nearest
        assert D[2 * q].argmin() == 2 * q + 1 and D[2 * q + 1].argmin() == 2 * q
    for q in range(5):                   # index neighbours across a quad boundary are not
        a, b = 2 * q + 1, 2 * q + 2
        assert D[a, b] > D[a].min() and D[a, b] > D[b].min()


def test_ibm_scaling_changes_proportions_only():
    base = tl.ibm_core_xy()
    for kw in (dict(col_scale=2.0), dict(gap_scale=0.0), dict(pitch_scale=0.5)):
        xy = tl.ibm_core_xy(**kw)
        assert (np.argsort(xy[:, 0], kind='stable') == np.argsort(base[:, 0], kind='stable')).all()
        for q in range(6):
            assert np.ptp(xy[4 * q:4 * q + 4, 0]) == 0


def test_within_ols_recovers_slopes_under_two_way_fixed_effects():
    rng = np.random.default_rng(22)
    n_s, n_c = 300, 40
    s = rng.integers(0, n_s, 6000); c = rng.integers(0, n_c, 6000)
    u = rng.normal(size=n_s) * 3; v = rng.normal(size=n_c) * 3
    x1 = rng.normal(size=6000) + u[s] * 0.5 + v[c] * 0.5         # regressor correlated with FE
    x2 = rng.normal(size=6000)
    y = u[s] + v[c] + 0.5 * x1 - 0.2 * x2 + rng.normal(size=6000) * 0.01
    b = tl.within_ols(y, np.column_stack([x1, x2]), s, fe=(c,))
    np.testing.assert_allclose(b, [0.5, -0.2], atol=2e-3)
    b1 = tl.within_ols(y - v[c], np.column_stack([x1, x2]), s)
    np.testing.assert_allclose(b1, [0.5, -0.2], atol=2e-3)


def test_natural_experiment_is_zero_without_a_gap_effect():
    rng = np.random.default_rng(13)
    SL = _sockets(rng)
    S = np.where(SL[:, :, None] & SL[:, None, :], 0.2, np.nan)
    assert tl.natural_experiment(S, SL) == pytest.approx(0.0)


def test_column_contrast_separates_ibm_geometry_from_index_order():
    """Inside a die column IBM's diagram puts slice 4c+3 nearer to 4c+1 than 4c+2 is; the index
    (and the rank among surviving cores) puts 4c+2 nearer. A physical field and a dark-source model
    on IBM's layout give a positive contrast, kernels in index or rank order a negative one."""
    rng = np.random.default_rng(31)
    SL = _sockets(rng)
    xy = tl.ibm_core_xy()

    def field(a):
        D = np.linalg.norm(xy[a][:, None] - xy[a][None], axis=-1)
        return tl.projected_corr(np.exp(-D / 0.3) + 0.3 * np.eye(16), np.arange(16))

    for S in (_model_S(SL, field), _model_S(SL, lambda a: tl.heat_source_corr(a, xy, 0.3, 0.3))):
        d, per = tl.column_contrast(S, SL)
        assert np.nanmean(d) > 0.02 and (np.nanmean(per, 0) > 0).all()
    for pos in (lambda a: a.astype(float), lambda a: np.arange(16.0)):
        d, _ = tl.column_contrast(_projected_S(SL, pos), SL)
        assert np.nanmean(d) < -0.02


def test_column_contrast_requires_all_three_slices():
    SL = np.ones((3, 12), dtype=bool)
    SL[1, 2] = False                       # column 0 unusable on socket 1
    SL[2, [2, 6, 10]] = False              # no column usable on socket 2
    S = np.zeros((3, 12, 12))
    for c in range(3):
        S[:, 4 * c + 1, 4 * c + 3] = 0.3; S[:, 4 * c + 1, 4 * c + 2] = 0.1 * (c + 1)
    d, per = tl.column_contrast(S, SL)
    assert d[0] == pytest.approx(np.mean([0.2, 0.1, 0.0]))
    assert d[1] == pytest.approx(np.mean([0.1, 0.0])) and np.isnan(per[1, 0])
    assert np.isnan(d[2])
