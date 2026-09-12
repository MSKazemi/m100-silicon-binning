"""Tests for the side-channel deletion models (analysis/side_channel_model.py) and for the
numbers analysis/side_channel.py writes to analysis/side_channel_results.json.

The first group validates each closed form against a brute-force simulation of the process it
claims to describe, on small synthetic inputs where the probabilities are neither 0 nor 1.
The last group is a regression guard for the bug found in external review (2026-09): the old
code reported exact recovery 1.000 after deleting 99.9 % of samples; the correct iid value for
one day of healthy telemetry is ~0.81.
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))

from side_channel_model import (SLOTS, decimation_all_phases, deletion_for_target,  # noqa: E402
                                expected_unique_share,
                                mc_iid_thinning, p_exact, p_silent_bursty, p_silent_decimation,
                                p_silent_iid, p_silent_poisson, poisson_curve, run_lengths,
                                simulate_gilbert, slice_counts, slices_revealed_iid,
                                timestamp_deletion_bounds)


def within(est, truth, se, k=4.0):
    return abs(est - truth) <= k * se + 1e-12


# ------------------------------------------------------------------ the reviewer's numbers
def test_reviewer_numbers_all_cores_4320():
    n = np.full(16, 4320)
    assert p_exact(p_silent_iid(n, 0.999)) == pytest.approx(0.8075420, rel=1e-6)
    assert p_exact(p_silent_iid(n, 0.9999)) == pytest.approx(5.26072e-8, rel=1e-5)
    assert p_exact(p_silent_iid(n, 0.99)) == pytest.approx(1.0, abs=1e-15)


def test_poisson_is_the_large_n_limit_of_iid_thinning():
    n = 4320
    for f in (0.99, 0.999, 0.9999):
        lam = n * (1 - f)
        assert p_exact(p_silent_iid(np.full(16, n), f)) == pytest.approx(poisson_curve(lam),
                                                                          rel=2e-3, abs=1e-12)
    assert p_silent_poisson(0.0) == 1.0
    assert poisson_curve(7) == pytest.approx(0.985509, abs=1e-6)


# ------------------------------------------------------------------ iid thinning
def test_iid_analytic_matches_binomial_thinning_mc():
    rng = np.random.default_rng(1)
    counts = rng.integers(10, 80, size=(3000, 16))
    for f in (0.8, 0.9, 0.95):
        an = p_exact(p_silent_iid(counts, f)).mean()
        mc, se = mc_iid_thinning(counts, f, 30, rng)
        assert 0.02 < an < 0.98
        assert within(mc, an, se), (f, an, mc, se)


def test_iid_thinning_of_explicit_timestamps_equals_count_thinning():
    """Deleting individual timestamps and reconstructing the map from what is left gives the
    same recovery rate as the product formula on the counts."""
    rng = np.random.default_rng(2)
    S, slots, f = 4000, 60, 0.9
    present = rng.random((S, 16, slots)) < rng.uniform(0.3, 0.9, size=(S, 16, 1))
    present[:, :, 0] = True                       # every core active: >= 1 sample
    kept = present & (rng.random(present.shape) >= f)
    mc = kept.any(2).all(1).mean()
    an = p_exact(p_silent_iid(present.sum(2), f)).mean()
    se = np.sqrt(an * (1 - an) / S)
    assert 0.05 < an < 0.95
    assert within(mc, an, se)


def test_iid_edge_cases():
    assert np.all(p_silent_iid(np.array([0, 1, 5]), 0.0) == [1.0, 0.0, 0.0])
    assert np.all(p_silent_iid(np.array([0, 1, 5]), 1.0) == 1.0)


# ------------------------------------------------------------------ decimation
def test_decimation_matches_hypergeometric_and_mc():
    slots, K = 60, 4
    for n in (0, 1, 5, 30, 56, 57, 60):
        exact = comb(slots - K, n) / comb(slots, n) if n <= slots - K else 0.0
        assert p_silent_decimation(np.array([n]), K, slots)[0] == pytest.approx(exact, abs=1e-12)
    rng = np.random.default_rng(3)
    n, reps = 12, 200000
    occ = np.argsort(rng.random((reps, slots)), axis=1)[:, :n]     # n distinct random slots
    silent = ~(occ < K).any(1)                                     # kept slots are 0..K-1
    se = np.sqrt(silent.mean() * (1 - silent.mean()) / reps)
    assert within(silent.mean(), p_silent_decimation(np.array([n]), K, slots)[0], se)


def test_decimation_single_slot_and_full_day():
    n = np.array([4320, 4000, 1])
    assert np.allclose(p_silent_decimation(n, 1), 1 - n / SLOTS)
    assert np.all(p_silent_decimation(n, SLOTS) == 0.0)


def test_decimation_rejects_fractional_counts():
    """The original bug fed n * (1 - f) into this formula; that must now fail loudly."""
    with pytest.raises(ValueError):
        p_silent_decimation(np.array([4.32]), SLOTS - 1)


# ------------------------------------------------------------------ bursty deletion
@pytest.mark.parametrize('n,f,L', [(40, 0.9, 3.0), (25, 0.8, 1 / 0.8), (30, 0.95, 10.0),
                                   (15, 0.6, 1.0)])
def test_bursty_closed_form_matches_markov_simulation(n, f, L):
    rng = np.random.default_rng(4)
    reps = 120000
    kept = simulate_gilbert(n, f, L, reps, rng)
    assert kept.mean() == pytest.approx(1 - f, abs=0.01)           # stationary deletion rate
    silent = ~kept.any(1)
    an = p_silent_bursty(np.array([n]), f, L)[0]
    se = np.sqrt(an * (1 - an) / reps)
    assert within(silent.mean(), an, se), (silent.mean(), an, se)


@pytest.mark.parametrize('f,L', [(0.9, 3.0), (0.95, 10.0), (0.99, 15.0)])
def test_gilbert_chain_has_the_stated_rate_and_run_length(f, L):
    """Long chains, so that runs cut by the window edge do not bias the mean run length."""
    rng = np.random.default_rng(5)
    kept = simulate_gilbert(20000, f, L, 300, rng)
    assert kept.mean() == pytest.approx(1 - f, rel=0.05)
    starts = (np.diff(kept.astype(np.int8), axis=1) == 1).sum() + kept[:, 0].sum()
    assert kept.sum() / starts == pytest.approx(L, rel=0.05)


def test_bursty_with_L_equal_1_over_f_is_iid():
    n = np.array([1, 7, 100, 4320])
    for f in (0.5, 0.9, 0.999):
        assert np.allclose(p_silent_bursty(n, f, 1 / f), p_silent_iid(n, f), rtol=1e-9)


def test_bursty_is_more_protective_than_iid_at_equal_deletion():
    n = np.full((10, 16), 4320)
    f = 0.99
    assert p_exact(p_silent_bursty(n, f, 15)).mean() < p_exact(p_silent_iid(n, f)).mean()


def test_bursty_infeasible_parameters_raise():
    with pytest.raises(ValueError):
        p_silent_bursty(np.array([10]), 0.9, 0.5)


# ------------------------------------------------------------------ solver
def test_deletion_for_target_hits_target():
    C = np.full((50, 16), 4320)
    f = deletion_for_target(C, 0.5)
    assert p_exact(p_silent_iid(C, f)).mean() == pytest.approx(0.5, abs=1e-6)
    lam = -np.log(1 - 0.5 ** (1 / 16))
    assert (1 - f) * 4320 == pytest.approx(lam, rel=2e-3)


# ------------------------------------------------------------------ slice-aware attacker
def test_slice_counts_pairs_adjacent_columns():
    c = np.arange(16).reshape(1, 16)
    assert slice_counts(c).tolist() == [[1, 5, 9, 13, 17, 21, 25, 29]]


def test_slice_aware_attacker_needs_one_sample_per_slice():
    """Knowing the slice rule, one surviving sample per active slice gives the exact map, so the
    slice-aware recovery is (1 - f^(2n))^8 and is never below the per-core (1 - f^n)^16."""
    n, f = 4320, 0.999
    core = p_exact(p_silent_iid(np.full(16, n), f))
    sl = p_exact(p_silent_iid(slice_counts(np.full(16, n)), f))
    assert sl == pytest.approx((1 - f ** (2 * n)) ** 8, rel=1e-12)
    assert sl > core


def test_slices_revealed_matches_simulation():
    rng = np.random.default_rng(6)
    S, f = 20000, 0.95
    counts = rng.integers(5, 40, size=(S, 16))
    k_an, cand_an = slices_revealed_iid(counts, f)
    kept = rng.binomial(counts, 1 - f) > 0                        # (S, 16) core seen
    k = kept.reshape(S, 8, 2).any(2).sum(1)                         # revealed active slices
    cand = np.array([comb(12 - kk, 8 - kk) for kk in k], dtype=float)
    assert k.mean() == pytest.approx(k_an.mean(), abs=4 * k.std() / np.sqrt(S))
    assert cand.mean() == pytest.approx(cand_an.mean(), abs=4 * cand.std() / np.sqrt(S))
    # nothing survives -> the prior 495; everything survives -> exactly one map
    k0, c0 = slices_revealed_iid(np.zeros((1, 16)), 0.5)
    k1, c1 = slices_revealed_iid(np.full((1, 16), 10 ** 6), 0.5)
    assert (k0[0], c0[0], k1[0], c1[0]) == (0.0, 495.0, 8.0, 1.0)


# ------------------------------------------------------------------ raw-timestamp helpers
def test_decimation_all_phases_matches_loop_and_single_slot():
    rng = np.random.default_rng(7)
    A = rng.random((50, 16, 120)) < 0.97
    A[:, :, ::7] |= rng.random((50, 1, 18)) < 0.5                  # some structure
    for k in (1, 3, 8, 40, 120):
        loop = np.mean([A[:, :, ph::k].any(2).all(1) for ph in range(k)], axis=0)
        assert np.allclose(decimation_all_phases(A, k), loop)
    # one kept slot per "day": the phase average IS the single-interval recovery
    assert np.allclose(decimation_all_phases(A, 120), A.all(1).mean(1))
    with pytest.raises(ValueError):
        decimation_all_phases(A, 7)


def test_timestamp_deletion_bounds_bracket_simulation():
    rng = np.random.default_rng(8)
    S, T, f = 3000, 200, 0.99
    sock_up = rng.random((S, 1, T)) < 0.9                          # whole-socket gaps
    A = sock_up & (rng.random((S, 16, T)) < 0.995)                  # rare single-core gaps
    lo, hi = timestamp_deletion_bounds(A, f)
    assert np.all(lo <= hi + 1e-15)
    kept = rng.random((S, T)) >= f
    mc = (A & kept[:, None, :]).any(2).all(1).mean()
    se = np.sqrt(mc * (1 - mc) / S)
    assert lo.mean() - 4 * se <= mc <= hi.mean() + 4 * se


def test_expected_unique_share_of_fixed_pseudonyms():
    """A random permutation of the 24 ids sends a fixed 16-set to a uniform 16-of-24 set; the
    expected unique share matches simulation (small universe so collisions are common)."""
    rng = np.random.default_rng(9)
    n, m, k, reps = 100, 10, 3, 400                      # C(10, 3) = 120 possible sets
    base = np.arange(k)
    shares = []
    for _ in range(reps):
        sets = [tuple(sorted(rng.permutation(m)[base])) for _ in range(n)]
        _, c = np.unique(np.array(sets), axis=0, return_counts=True)
        shares.append((c == 1).sum() / n)
    se = np.std(shares) / np.sqrt(reps)
    assert within(np.mean(shares), expected_unique_share(n, comb(m, k)), se)
    assert expected_unique_share(1960, comb(24, 16)) == pytest.approx(0.99734, abs=1e-5)


def test_run_lengths():
    m = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    assert run_lengths(m).tolist() == [2, 1, 3, 9]
    assert run_lengths(m).sum() == m.sum()


# ------------------------------------------------------------------ outputs (regression)
RES = ROOT / 'analysis' / 'side_channel_results.json'


@pytest.mark.skipif(not RES.exists(), reason='run analysis/side_channel.py first')
def test_results_json_has_corrected_dropout_numbers():
    r = json.loads(RES.read_text())
    F = r['iid']['f']
    one_day = dict(zip(F, r['iid']['windows']['1']['p_exact']))
    assert 0.78 < one_day[0.999] < 0.84          # was 1.000 before the fix
    assert one_day[0.9999] < 1e-6                # was 0.61 before the fix
    assert one_day[0.99] > 0.999999
    month = dict(zip(F, r['iid']['windows']['31']['p_exact']))
    assert month[0.9999] > 0.999                 # pooling days defeats heavy deletion
    for m in r['iid']['monte_carlo']:            # analytic and simulated agree
        if 0 < m['mc'] < 1:
            assert within(m['mc'], m['analytic'], m['se'], k=4.0), m
    dec = {x['every_k_slots']: x['model'] for x in r['decimation']}
    assert dec[180] > 0.9999 and 0.99 < dec[4320] < 0.999
    # the slice-aware attacker is never worse off than the per-core one
    for w in r['iid']['windows'].values():
        assert all(s >= c - 1e-12 for s, c in zip(w['p_exact_slice_aware'], w['p_exact']))
    assert dict(zip(F, r['iid']['windows']['1']['p_exact_slice_aware']))[0.999] > 0.99
    part = {(p['days'], p['f']): p for p in r['iid']['partial']}
    assert 4.0 < part[(1, 0.9999)]['mean_slices_revealed'] < 5.2   # exact ~5e-8, yet ~4.6 of 8
    if 'raw' in r:                                  # one kept slot per day == one interval
        raw = r['raw']
        d24 = next(x for x in raw['decimation'] if x['every_k_slots'] == 4320)
        assert d24['measured'] == pytest.approx(raw['window'][0]['p_exact'], abs=1e-9)
        for t in raw['timestamp_deletion']:
            se = np.sqrt(max(t['measured'] * (1 - t['measured']), 1e-12) / t['draws'])
            assert t['bound_lo'] - 4 * se <= t['measured'] <= t['bound_hi'] + 4 * se
