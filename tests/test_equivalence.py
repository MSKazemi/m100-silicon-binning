"""Tests for the power-equivalence estimators (analysis/equivalence_core.py) and for the numbers
analysis/equivalence.py writes to analysis/equivalence_fit.json.

The first group checks each estimator on synthetic inputs with a known answer. The last group
guards the corrections made after the external review (2026-09): power rows must carry the map
valid on their own date, the legacy joint construction must still reproduce the number the
paper used to print (+7.86 W), and the core-frequency covariate must not be counted as an
adjustment, because it is a constant in this dataset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))

import equivalence_core as ec  # noqa: E402


def code_of(slices):
    """24-bit active-core code whose active slices are `slices` (both cores of each)."""
    return sum((1 << (2 * s)) | (1 << (2 * s + 1)) for s in slices)


# ------------------------------------------------------------------ map summaries
def test_slice_features_known_maps():
    assert ec.slice_features(code_of(range(8))) == (3.5, 7, 6)
    # active [0,2,4,6,8,9,10,11]: adjacent pairs (8,9),(9,10),(10,11); low half 0,2,4
    assert ec.slice_features(code_of([0, 2, 4, 6, 8, 9, 10, 11])) == (6.25, 3, 3)
    assert ec.active_slices(0b11) == [0]
    with pytest.raises(ValueError):
        ec.slice_features(0)


# ------------------------------------------------------------------ contemporaneous map
def test_epoch_lookup_runs_gaps_and_edges():
    A, B, C = 11, 22, 33
    ck = [1, 1, 1, 1, 1, 1, 2]
    cd = [0, 1, 2, 5, 8, 9, 3]
    cc = [A, A, A, A, B, B, C]
    qk = [1] * 12 + [2, 2, 0]
    qd = list(range(12)) + [3, 4, 3]
    code, how = ec.epoch_lookup(ck, cd, cc, qk, qd)
    exp_code = [A, A, A, A, A, A, -1, -1, B, B, -1, -1, C, -1, -1]
    exp_how = [1, 1, 1, 2, 2, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    assert list(code) == exp_code
    assert list(how) == exp_how


def test_epoch_lookup_order_invariant_and_rejects_duplicates():
    rng = np.random.default_rng(0)
    ck = np.repeat([3, 4], 6)
    cd = np.tile([0, 2, 4, 6, 8, 10], 2)
    cc = np.array([5, 5, 5, 7, 7, 7, 9, 9, 9, 9, 9, 9])
    qk = np.repeat([3, 4], 11)
    qd = np.tile(np.arange(11), 2)
    c1, h1 = ec.epoch_lookup(ck, cd, cc, qk, qd)
    p = rng.permutation(len(ck))
    c2, h2 = ec.epoch_lookup(ck[p], cd[p], cc[p], qk, qd)
    assert (c1 == c2).all() and (h1 == h2).all()
    # socket 3: map 5 on days 0-4, transition gap day 5, map 7 on days 6-10
    assert list(c1[:11]) == [5] * 5 + [-1] + [7] * 5
    with pytest.raises(ValueError):
        ec.epoch_lookup([1, 1], [2, 2], [5, 5], [1], [2])


# ------------------------------------------------------------------ regression + uncertainty
def _clustered_data(G=30, m=40, beta=0.7, seed=1):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), m)
    x = rng.normal(size=G)[g] + 0.5 * rng.normal(size=G * m)      # regressor shared in cluster
    e = 2.0 * rng.normal(size=G)[g] + rng.normal(size=G * m)       # error shared in cluster
    X = np.column_stack([np.ones(G * m), x])
    return X, 1.0 + beta * x + e, g


def test_ols_cluster_boot_point_estimate_and_replicate_identity():
    X, y, g = _clustered_data()
    b, bs, nsing = ec.ols_cluster_boot(X, y, g, B=50, seed=11)
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert nsing == 0
    # replicate 0 must equal OLS on the explicitly resampled rows
    G = len(np.unique(g))
    w = np.random.default_rng(11).multinomial(G, np.full(G, 1.0 / G), size=50)[0]
    rows = np.concatenate([np.tile(np.where(g == c)[0], w[c]) for c in range(G)])
    assert np.allclose(bs[0], np.linalg.lstsq(X[rows], y[rows], rcond=None)[0])


def test_cluster_bootstrap_agrees_with_cr1_and_exceeds_iid():
    X, y, g = _clustered_data(G=40, m=50, seed=3)
    b, bs, _ = ec.ols_cluster_boot(X, y, g, B=3000, seed=5)
    _, se = ec.cluster_robust_se(X, y, g)
    sd_boot = bs[:, 1].std()
    assert abs(sd_boot / se[1] - 1) < 0.25
    e = y - X @ b
    se_iid = np.sqrt(e.var() * np.linalg.inv(X.T @ X)[1, 1])
    assert se[1] > 2 * se_iid            # clustering matters here, as it does across racks


def test_cr1_reduces_to_hc1_with_singleton_clusters():
    rng = np.random.default_rng(7)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = X @ [1, 2, -1] + rng.normal(size=n) * (1 + np.abs(X[:, 1]))
    b, se = ec.cluster_robust_se(X, y, np.arange(n))
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    hc1 = n / (n - 3) * bread @ (X.T * e ** 2) @ X @ bread
    assert np.allclose(se, np.sqrt(np.diag(hc1)))


def test_bootstrap_recovers_true_slope():
    X, y, g = _clustered_data(G=60, m=30, beta=0.7, seed=9)
    b, bs, _ = ec.ols_cluster_boot(X, y, g, B=2000, seed=2)
    lo, hi = ec.percentile_ci(bs[:, 1], 0.95)
    assert lo < 0.7 < hi


# ------------------------------------------------------------------ scales and TOST
def test_scales():
    s = ec.scales([0, 1, 2, 3, 4])
    assert s['range'] == 4 and np.isclose(s['sd'], np.sqrt(2)) and s['iqr'] == 2


def test_tost_and_equivalence_bound_agree():
    assert ec.tost(-1.0, 4.0, 5.0) and not ec.tost(-1.0, 5.2, 5.0) and not ec.tost(-5.1, 0, 5)
    assert ec.eq_bound(-1.0, 5.2) == 5.2 and ec.eq_bound(-6.0, 1.0) == 6.0
    rng = np.random.default_rng(4)
    for _ in range(500):
        lo = rng.normal(0, 4)
        hi = lo + abs(rng.normal(0, 4))
        m = abs(rng.normal(0, 6))
        assert ec.tost(lo, hi, m) == (ec.eq_bound(lo, hi) < m)


def test_wquantile_matches_expanded_sample():
    rng = np.random.default_rng(6)
    vals = rng.normal(size=40)
    w = rng.integers(1, 9, size=40)
    expanded = np.repeat(vals, w)
    for q in (0.05, 0.25, 0.5, 0.95):
        assert ec.wquantile(vals, w, q)[0] == np.quantile(expanded, q, method='inverted_cdf')


def test_vif():
    rng = np.random.default_rng(8)
    n, rho = 200000, -0.9
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    v = ec.vif(np.column_stack([a, b, rng.normal(size=n)]))
    assert abs(v[0] - 1 / (1 - rho ** 2)) < 0.1 and abs(v[2] - 1) < 0.02


def test_partial_slope_times_marginal_range_overstates_a_collinear_feature():
    """The construction the paper used to print: with two collinear summaries of one exposure,
    the joint model's partial slope times the MARGINAL range is not the effect of any
    realisable contrast, and exceeds the single-feature (total) effect."""
    rng = np.random.default_rng(12)
    n = 50000
    f1 = rng.normal(size=n)
    f2 = -0.9 * f1 + np.sqrt(1 - 0.81) * rng.normal(size=n)
    y = 1.0 * f1 + 0.8 * f2 + rng.normal(size=n)          # total effect of f1: 1 - 0.72 = 0.28
    Xj = np.column_stack([np.ones(n), f1, f2])
    bj = np.linalg.lstsq(Xj, y, rcond=None)[0]
    bm = np.linalg.lstsq(Xj[:, :2], y, rcond=None)[0]
    assert abs(bj[1] - 1.0) < 0.05 and abs(bm[1] - 0.28) < 0.03
    assert bj[1] > 3 * bm[1]


# ------------------------------------------------------------------ SLURM placement extraction
def test_parse_layout_socket_split():
    import equivalence_alloc as ea
    assert ea.parse_layout('{668: [20], 669: [0, 1, 16]}') == [(668, 0, 1, False),
                                                                (669, 2, 1, False)]
    assert ea.parse_layout('{292: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, '
                           '18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}') == \
        [(292, 16, 16, False)]
    assert ea.parse_layout('{668: []}') == [(668, 0, 0, True)]
    assert ea.parse_layout('{None: []}') == [] and ea.parse_layout(float('nan')) == []


def test_split_days_is_exact():
    import equivalence_alloc as ea
    d0 = 18500 * 86400
    days, secs = ea.split_days(d0 + 23 * 3600, d0 + 26 * 3600)
    assert days == [d0, d0 + 86400] and secs == [3600, 7200]
    days, secs = ea.split_days(d0, d0 + 86400)                  # exactly one whole day
    assert days == [d0] and secs == [86400]
    rng = np.random.default_rng(3)
    for _ in range(200):
        s = int(rng.integers(0, 10 ** 9))
        e = s + int(rng.integers(1, 3 * 86400))
        days, secs = ea.split_days(s, e)
        assert sum(secs) == e - s and all(0 < x <= 86400 for x in secs)
        assert all(b - a == 86400 for a, b in zip(days, days[1:]))


# ------------------------------------------------------------------ the committed results
RES = ROOT / 'analysis' / 'equivalence_fit.json'


@pytest.fixture(scope='module')
def res():
    if not RES.exists():
        pytest.skip('run analysis/equivalence.py first')
    return json.loads(RES.read_text())


def test_results_schema_and_headline(res):
    assert res['margin'] == 5.0 and res['margins'] == [2.0, 5.0, 10.0]
    h = [r for r in res['rows'] if r['design'] == res['headline']['design']
         and r['model'] == res['headline']['model']]
    assert {r['feature'] for r in h} == set(ec.FEATURES)
    for r in h:
        lo, hi = r['range']['lo'], r['range']['hi']
        assert r['eq_bound'] == pytest.approx(ec.eq_bound(lo, hi))
        for m in res['margins']:
            assert r['tost'][f'{m:g}'] == ec.tost(lo, hi, m)
        # the three scales are the same slope: effect = slope x spread, CI likewise
        for s in ('range', 'sd', 'iqr'):
            assert r[s]['eff'] == pytest.approx(r['slope'] * r['spread'][s])


def test_headline_rows_use_contemporaneous_maps(res):
    """Acceptance test of the review: every power row carries the map valid on its own date."""
    s = res['sample']
    assert res['headline']['design'] == 'epoch'
    assert s['epoch']['node_days'] > 0.9 * s['modal']['node_days']
    assert s['epoch']['own_date_mismatch'] == 0       # independent merge_asof cross-check
    # the legacy join really did pair relabelled days with the mirror-image maps
    assert s['epoch']['mirrored_vs_modal'] > 0 and s['epoch']['nonmodal_node_days'] > 0


def test_claims_the_paper_makes(res):
    h = {r['feature']: r for r in res['rows']
         if r['design'] == 'epoch' and r['model'] == 'single-adjusted'}
    # two of three summaries equivalent at +/-5 W, all three at +/-10 W
    assert sum(r['tost']['5'] for r in h.values()) == 2 and all(r['tost']['10'] for r in h.values())
    assert all(abs(r['sd']['eff']) < 0.5 for r in h.values())
    # every design, every summary inside +/-10 W
    assert res['robustness_max_bound'] < 10
    # the p0 - p1 asymmetry is placement, not slot: slot effect at equal placement spans zero
    lo, hi = res['workload_placement']['slot_ci']
    assert lo < 0 < hi and res['workload_placement']['corr_dP_dA'] > 0.5
    # placement is (nearly) uncorrelated with the map difference
    assert all(abs(v['corr_dA']) < 0.05 for v in res['workload_placement']['features'].values())


def test_legacy_joint_construction_reproduces_published_number(res):
    leg = {r['feature']: r for r in res['rows']
           if r['design'] == 'modal' and r['model'] == 'joint-adjusted'}
    assert leg['meanidx']['range']['eff'] == pytest.approx(7.862, abs=0.01)
    assert leg['meanidx']['slope'] == pytest.approx(1.1648, abs=1e-3)


def test_constant_covariate_is_not_an_adjustment(res):
    assert 'cpu_speed_mean' in res['dropped_constant_covariates']
    assert 'cpu_speed_mean' not in res['covariates']


def test_placement_adjustment_moves_estimates_away_from_zero(res):
    """Sec. IV-L says: where placement is recorded, adjusting for it moves each estimate about
    1 W away from zero (so omitting it biases toward zero there), and no interval passes 5.0 W."""
    for f, v in res['workload_placement']['features'].items():
        shift = abs(v['with']['eff']) - abs(v['without']['eff'])
        assert 0.8 < shift < 1.4, (f, shift)
        assert max(abs(v['with']['lo']), abs(v['with']['hi'])) < 5.05


def test_robustness_statements(res):
    """The sensitivity paragraph: every interval within +/-7.3 W, but not all within +/-5 W, and
    among covariate-adjusted single-summary designs only idle-day adjacency excludes zero."""
    rob = res['robustness']
    assert res['robustness_max_bound'] < 7.35
    assert rob['stable']['meanidx']['eq_bound'] > 6.0            # never-changing nodes: 6.5 W
    excl = sorted((d, f) for d, v in rob.items() if d != 'idle_nodeweighted'
                  for f, r in v.items() if r['lo'] > 0 or r['hi'] < 0)
    assert excl == [('idle', 'nadj'), ('idle_1pct', 'nadj')]
    nw = rob['idle_nodeweighted']['nadj']
    assert nw['lo'] < 0 < nw['hi']                               # one row per node-epoch


def test_joint_fit_over_realised_map_pairs_agrees(res):
    """The joint fit, evaluated over map pairs that occur, spans about 2.2 W from the 5th to the
    95th percentile pair: the +7.9 W came from the contrast, not from the joint model itself."""
    j = res['joint']['epoch']
    assert 2.0 < j['map_p5_p95'] < 2.4 and j['map_p5_p95_hi'] < 4.5
    assert j['corr']['meanidx~lowhalf'] < -0.85


def test_calibration_statements(res):
    cal, wl = res['calibration'], res['workload_placement']
    assert abs(cal['mean_dP_W'] - 6.75) < 0.05 and abs(cal['sd_dP_W'] - 19.9) < 0.1
    assert abs(cal['idle_mean_dP_W']) < 0.5                      # no slot effect when idle
    assert abs(wl['slot_W_at_equal_placement']) < 0.5
    assert 9.0 < cal['idle_sd_node_mean_dP_W'] < 9.4
