"""Unit tests for the longitudinal estimators in analysis/lifecycle_common.py.

Every estimator used by reboot_evidence.py, change_taxonomy.py, lifecycle_os_crosscheck.py and
lifecycle_taxonomy_errors.py is checked here on synthetic inputs with a known answer: brute-force
enumeration for the exact conditional test, a planted permutation for the remap test, a planted
collector outage for epoch-dated reboots, and closed forms elsewhere.
"""
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))
import lifecycle_common as L  # noqa: E402

D0 = pd.Timestamp('2021-01-01', tz='UTC')


def day(k):
    return D0 + pd.Timedelta(days=k)


# ------------------------------------------------------------------ masks, clean days, intervals
def test_mask_roundtrip():
    s = frozenset({0, 1, 4, 5, 22, 23})
    assert L.mask_to_set(L.set_to_mask(s)) == s
    assert L.set_to_mask({0}) == 1 and L.set_to_mask({23}) == 1 << 23


def _daily(rows):
    return pd.DataFrame(rows, columns=['node', 'day', 'n', 'socket', 'core'])


FULL16 = [c for c in range(24) if c // 2 not in (0, 3, 5, 9)]          # 16 cores, 4 slices off


def test_socket_days_and_clean_sets():
    rows = []
    for k in range(3):                                  # three clean days
        rows += [(1, day(k), 4320, 0, c) for c in FULL16]
    rows += [(1, day(3), 4320, 0, c) for c in FULL16[:-1]] + [(1, day(3), 100, 0, FULL16[-1])]
    rows += [(1, day(4), 4320, 0, c) for c in FULL16[:-1]]                  # 15 cores
    t = L.socket_days(_daily(rows))
    assert list(t.active) == [16, 16, 16, 16, 15]
    assert list(t.nfull) == [16, 16, 16, 15, 15]
    sets = L.clean_sets(_daily(rows), t)
    assert len(sets) == 3
    assert L.mask_to_set(sets['mask'].iloc[0]) == frozenset(FULL16)


def test_intervals_gap_and_change():
    m1, m2 = L.set_to_mask(FULL16), L.set_to_mask([c for c in range(24) if c // 2 not in (1, 3, 5, 9)])
    sets = pd.DataFrame({'node': [1, 1, 1, 2], 'socket': [0, 0, 0, 0],
                         'day': [day(0), day(1), day(4), day(0)], 'mask': [m1, m1, m2, m1]})
    iv = L.intervals(sets)
    assert list(iv.gap) == [1, 3]
    assert list(iv.changed) == [False, True]


def test_gap_bin():
    assert list(L.gap_bin([1, 2, 3, 4, 5, 8, 9, 100])) == [0, 1, 2, 2, 3, 3, 4, 4]


# ------------------------------------------------------------------ taxonomy
def _m(harvested):
    return L.set_to_mask([c for c in range(24) if c // 2 not in harvested])


A, B, C2, D = _m({0, 1, 2, 3}), _m({4, 5, 6, 7}), _m({8, 9, 10, 11}), _m({0, 2, 4, 6})


def _sets(spec):
    rows = []
    for node, sock, seq in spec:
        for k, m in seq:
            rows.append((node, sock, day(k), m))
    return pd.DataFrame(rows, columns=['node', 'socket', 'day', 'mask']).sort_values(
        ['node', 'socket', 'day'], ignore_index=True)


def test_classify_events_all_four_kinds():
    sets = _sets([
        (1, 0, [(0, A), (2, B)]), (1, 1, [(0, B), (2, A)]),            # mirror -> RELABEL
        (2, 0, [(0, A), (2, C2)]), (2, 1, [(0, B), (1, B), (2, B)]),   # partner seen, same
        (3, 0, [(0, A), (5, C2)]), (3, 1, [(0, B)]),                   # partner unobserved
        (4, 0, [(0, A), (2, C2)]), (4, 1, [(0, B), (2, D)]),           # both, no mirror
    ])
    E = L.classify_events(sets).set_index('node')
    assert E.loc[1, 'kind'] == 'RELABEL' and E.loc[1, 'dies'] == 0
    assert E.loc[2, 'kind'] == 'CPU-SWAP' and E.loc[2, 'dies'] == 1
    assert E.loc[3, 'kind'] == 'CPU-SWAP?'
    assert E.loc[4, 'kind'] == 'NODE-SWAP' and E.loc[4, 'dies'] == 2


def test_union_relation():
    a, b = {0, 1, 2, 3}, {2, 3, 4, 5}
    assert L.union_relation({0, 1, 2, 3, 4, 5}, a, b) == 'exact'
    assert L.union_relation({0, 1, 4, 5}, a, b) == 'within'
    assert L.union_relation({0, 1, 2, 3, 8}, a, b) == 'outside'
    assert L.union_relation({0, 1}, None, b) is None


def test_half_mirror():
    sa, sb, sc = (L.mask_to_set(x) for x in (A, B, C2))
    assert L.half_mirror({0: (sa, sb, None), 1: (sb, sc, None)})         # p0 takes p1's old
    assert not L.half_mirror({0: (sa, sb, None), 1: (sb, sa, None)})     # full mirror
    assert not L.half_mirror({0: (sa, sc, None), 1: (sb, L.mask_to_set(D), None)})
    assert not L.half_mirror({0: (sa, sb, None)})


def test_collision_probability_closed_form():
    maps = [{0, 1, 2, 3}] * 3 + [{4, 5, 6, 7}, {0, 2, 4, 6}]
    assert L.collision_probability(maps) == pytest.approx(3 * 2 / (5 * 4))
    assert L.cross_collision_probability(maps, [{0, 1, 2, 3}]) == pytest.approx(3 / 5)
    # uniform over all 495 patterns, one each: no two distinct sockets match
    allp = [set(c) for c in combinations(range(12), 4)]
    assert L.collision_probability(allp) == 0.0


def test_common_slice_permutation_planted_and_refuted():
    rng = np.random.default_rng(5)
    sigma = rng.permutation(12)
    pairs = []
    for _ in range(12):
        s = set(rng.choice(12, 4, replace=False).tolist())
        pairs.append((s, {int(sigma[i]) for i in s}))
    found = L.common_slice_permutation(pairs)
    assert found is not None
    assert all({found[i] for i in o} == n for o, n in pairs)
    # overlaps that no bijection can preserve
    assert L.common_slice_permutation([({0, 1, 2, 3}, {4, 5, 6, 7}),
                                       ({0, 1, 2, 4}, {8, 9, 10, 11})]) is None
    # an exchange of two maps is itself a permutation
    assert L.common_slice_permutation([({0, 1, 2, 3}, {4, 5, 6, 7}),
                                       ({4, 5, 6, 7}, {0, 1, 2, 3})]) is not None


def test_common_slice_permutation_exhaustive_small():
    """Against brute force on a 5-slice space with 2-sets."""
    rng = np.random.default_rng(9)
    for _ in range(40):
        pairs = [(set(rng.choice(5, 2, replace=False).tolist()),
                  set(rng.choice(5, 2, replace=False).tolist())) for _ in range(3)]
        brute = any(all({p[i] for i in o} == n for o, n in pairs) for p in permutations(range(5)))
        assert (L.common_slice_permutation(pairs, nslices=5) is not None) == brute


# ------------------------------------------------------------------ core loss (OS cross-check)
def test_core_loss_rules():
    base = frozenset(FULL16)
    full = {c: 4320 for c in base}
    assert L.core_loss(full, base) == set()
    zero = {c: v for c, v in full.items() if c != FULL16[3]}
    assert L.core_loss(zero, base) == {FULL16[3]}
    onset = dict(full); onset[FULL16[5]] = 900
    assert L.core_loss(onset, base) == set()
    assert L.core_loss(onset, base, onset=True) == {FULL16[5]}
    union = dict(full); union[0] = 50                                   # a core outside base
    assert L.core_loss(union, base) == set()
    reboot_day = {c: 2000 for c in base}                                # every core partial
    assert L.core_loss(reboot_day, base, onset=True) == set()


def test_runs():
    r = L.runs([day(0), day(1), day(3), day(4), day(9)])
    assert [len(x) for x in r] == [2, 2, 1]


# ------------------------------------------------------------------ reboot dating
def _boot(rows):
    b = pd.DataFrame(rows, columns=['node', 'day', 'boot_min', 'boot_max', 'boot_ndistinct', 'boot_n'])
    b = b.sort_values(['node', 'day'], ignore_index=True)
    b['prev'] = b.groupby('node')['boot_max'].shift()
    b['rebooted'] = (b.boot_ndistinct > 1) | (b.prev.notna() & (b.boot_max != b.prev))
    return b


def _ep(k, hours=0.0):
    return (day(k) + pd.Timedelta(hours=hours)).timestamp()


def test_epoch_dating_resolves_a_reboot_inside_a_collector_outage():
    """The six 2021-04 cases in miniature: the collector stops part-way through day 7, the node
    reboots later that day, collection resumes on day 20. The day-flag rule dates the reboot to
    day 20; the epoch dates it to day 7."""
    e0, e1 = _ep(-10), _ep(7, 15)
    rows = [(1, day(k), e0, e0, 1, 1440) for k in range(0, 7)]
    rows += [(1, day(7), e0, e0, 1, 400)]                              # collector stops 06:40
    rows += [(1, day(k), e1, e1, 1, 1440) for k in range(20, 25)]      # resumes, new epoch
    boot = _boot(rows)
    iv = pd.DataFrame({'node': [1, 1, 1, 1], 'd0': [day(6), day(1), day(8), day(21)],
                       'd1': [day(8), day(2), day(9), day(22)]})
    dflag = L.dayflag_reboot(iv, boot)
    ep = L.epoch_reboot(iv, boot)
    assert dflag[0] == 0.0 and ep[0] == 1.0          # transition interval: resolved by the epoch
    assert dflag[1] == 0.0 and ep[1] == 0.0          # ordinary quiet interval
    assert np.isnan(dflag[2]) and ep[2] == 0.0       # inside the outage, after the boot: no boot
    assert ep[3] == 0.0


def test_epoch_dating_unresolved_when_a_later_boot_supersedes():
    e0, e2 = _ep(-10), _ep(12, 3)
    rows = [(1, day(k), e0, e0, 1, 1440) for k in range(0, 8)]
    rows += [(1, day(k), e2, e2, 1, 1440) for k in range(20, 22)]
    boot = _boot(rows)
    iv = pd.DataFrame({'node': [1], 'd0': [day(8)], 'd1': [day(10)]})
    assert np.isnan(L.epoch_reboot(iv, boot)[0])     # a boot inside could have been superseded


def test_dayflag_matches_definition_with_same_day_boot():
    e0, e1 = _ep(-3), _ep(2, 10)
    rows = [(1, day(0), e0, e0, 1, 1440), (1, day(1), e0, e0, 1, 1440),
            (1, day(2), e0, e1, 2, 1440), (1, day(3), e1, e1, 1, 1440)]
    boot = _boot(rows)
    iv = pd.DataFrame({'node': [1, 1], 'd0': [day(1), day(2)], 'd1': [day(2), day(3)]})
    assert list(L.dayflag_reboot(iv, boot)) == [1.0, 0.0]
    assert list(L.epoch_reboot(iv, boot)) == [1.0, 0.0]


# ------------------------------------------------------------------ inference
def _enumerate_tail(stratum, is_t, y):
    """Brute force: permute transition labels within strata over all arrangements."""
    stratum, is_t, y = map(np.asarray, (stratum, is_t, y))
    obs = int((y * is_t).sum())
    dists = []
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        t = int(is_t[idx].sum())
        vals = [int(y[list(c)].sum()) for c in combinations(idx, t)]
        dists.append(np.bincount(vals) / len(vals))
    tot = np.array([1.0])
    for d in dists:
        tot = np.convolve(tot, d)
    mean = float((np.arange(len(tot)) * tot).sum())
    var = float(((np.arange(len(tot)) - mean) ** 2 * tot).sum())
    return float(tot[obs:].sum()), mean, var, obs


def test_stratified_hypergeom_exact_equals_enumeration():
    stratum = [0] * 7 + [1] * 6 + [2] * 5 + [3] * 3
    is_t = [1, 1, 0, 0, 0, 0, 0] + [1, 0, 0, 0, 0, 0] + [1, 1, 0, 0, 0] + [1, 0, 0]
    y = [1, 1, 1, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0] + [1, 0, 1, 0, 0] + [0, 0, 0]
    r = L.stratified_hypergeom_test(stratum, is_t, y, B=50_000, rng=np.random.default_rng(1))
    p, mean, var, obs = _enumerate_tail(stratum, is_t, y)
    assert r['observed'] == obs
    assert r['p_exact'] == pytest.approx(p, rel=1e-9)
    assert r['null_mean'] == pytest.approx(mean, rel=1e-9)
    assert r['null_sd'] == pytest.approx(np.sqrt(var), rel=1e-9)
    assert r['p_mc'] == pytest.approx((r['b'] + 1) / (r['B'] + 1))
    assert abs(r['p_mc'] - p) < 0.01
    # stratum 3 has no reboots: uninformative, contributes nothing variable
    assert r['informative_strata'] == 3


def test_stratified_hypergeom_detects_within_stratum_association_only():
    """Transitions sit in a stratum with a high base rate; within it they are ordinary."""
    rng = np.random.default_rng(3)
    stratum = np.r_[np.zeros(1000, int), np.ones(1000, int)]
    y = np.r_[rng.random(1000) < .05, rng.random(1000) < .6].astype(int)
    is_t = np.zeros(2000, bool)
    is_t[1000:1050] = True                            # all transitions in the high-rate stratum
    pooled = L.stratified_hypergeom_test(np.zeros(2000, int), is_t, y, B=5000, rng=rng)
    strat = L.stratified_hypergeom_test(stratum, is_t, y, B=5000, rng=rng)
    assert pooled['p_exact'] < 1e-3                   # confounded comparison looks significant
    assert strat['p_exact'] > 0.01                    # conditioning removes it


def test_standardized_excess_and_cluster_bootstrap():
    rng = np.random.default_rng(7)
    n = 4000
    cluster = rng.integers(0, 200, n)
    gbin = rng.integers(0, 3, n)
    is_t = rng.random(n) < 0.05
    base = np.array([0.1, 0.3, 0.5])[gbin]
    y = (rng.random(n) < np.where(is_t, np.minimum(base + 0.4, 1), base)).astype(int)
    ex = L.standardized_excess(gbin, is_t, y, nbins=3)
    assert 0.3 < ex < 0.5
    bs = L.cluster_bootstrap_excess(cluster, gbin, is_t, y, B=2000, rng=rng, nbins=3)
    assert bs['point'] == pytest.approx(ex)
    assert bs['lo'] < ex < bs['hi']
    assert bs['clusters'] == 200


def test_cluster_bootstrap_rate():
    rng = np.random.default_rng(11)
    ev = rng.poisson(2.0, 50)
    ex = np.full(50, 100.0)
    r = L.cluster_bootstrap_rate(ev, ex, B=5000, rng=rng)
    assert r['point'] == pytest.approx(ev.sum() / ex.sum())
    assert r['lo'] < r['point'] < r['hi']
    # overdispersed clusters widen the interval relative to evenly spread events
    lumpy = np.zeros(50); lumpy[:5] = ev.sum() / 5
    r2 = L.cluster_bootstrap_rate(lumpy, ex, B=5000, rng=rng)
    assert (r2['hi'] - r2['lo']) > (r['hi'] - r['lo'])


def test_poisson_and_cp_intervals_match_paper():
    lo, hi = L.poisson_ci(96, 4561)
    assert (round(100 * lo, 2), round(100 * hi, 2)) == (1.70, 2.57)
    lo, hi = L.poisson_ci(2, 3991)
    assert (round(1e4 * lo, 2), round(1e4 * hi, 1)) == (0.61, 18.1)
    lo, hi = L.cp_ci(20, 105)
    assert (round(100 * lo), round(100 * hi)) == (12, 28)
