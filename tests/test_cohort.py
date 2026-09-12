"""Unit tests for the cohort estimators in analysis/cohort_models.py."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))
import cohort_models as cm  # noqa: E402


# ------------------------------------------------------------------ map space
def test_pattern_space():
    assert cm.PATTERNS.shape == (495, 4)
    assert (cm.A_SLICE.sum(1) == 4).all()
    # adjacency: {0,1,2,3} has 3 adjacent pairs, {0,2,4,6} has none, {0,1,5,6} has 2
    ids = cm.pattern_ids(np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                   [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]))
    assert list(cm.ADJ[ids]) == [3, 0, 2]
    with pytest.raises(ValueError):
        cm.pattern_ids(np.ones((1, 12), dtype=int))


def _esp4(w):
    """Elementary symmetric polynomial of degree 4, by the standard recursion."""
    e = np.zeros(5)
    e[0] = 1.0
    for x in w:
        e[1:] = e[1:] + x * e[:-1]
    return e[4]


def test_conditional_bernoulli_normaliser():
    rng = np.random.default_rng(0)
    beta = rng.normal(size=12)
    lp = cm.logprob_expfam(cm.features('cb'), beta)
    assert np.isclose(np.exp(lp).sum(), 1.0)
    # P(S) = prod_{k in S} w_k / e_4(w)
    w = np.exp(beta)
    brute = np.array([np.prod(w[list(p)]) for p in cm.PATTERNS]) / _esp4(w)
    assert np.allclose(np.exp(lp), brute)


def test_expfam_recovers_parameters():
    rng = np.random.default_rng(1)
    F = cm.features('pairwise')
    theta = np.concatenate([rng.normal(scale=.6, size=12), [0.7]])
    theta[:12] -= theta[:12].mean()
    p = np.exp(cm.logprob_expfam(F, theta))
    N = 200_000
    draws = rng.choice(cm.NPAT, size=N, p=p)
    T = F[draws].sum(0)
    th, ll = cm.fit_expfam(F, T, N, ridge=1e-6)
    # beta identified only up to a common shift under |S| = 4: compare centred values
    assert np.allclose(th[:12] - th[:12].mean(), theta[:12], atol=0.03)
    assert abs(th[12] - theta[12]) < 0.03
    # moment matching at the MLE
    ph = np.exp(cm.logprob_expfam(F, th))
    assert np.allclose(N * (ph @ F), T, rtol=1e-4, atol=1.0)
    # the MLE's likelihood is at least the truth's
    ll_true = T @ theta - N * np.log(np.exp(F @ theta).sum())
    assert ll >= ll_true - 1e-6


def test_pairwise_matches_yield_model_gamma():
    """Same model as Sec. IV-J: gamma for Lot A should reproduce the paper's +0.743."""
    M = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy')
    K = np.load(ROOT / 'analysis' / 'harvest_keys_full.npy')
    Ma = M[K[:, 0] // 20 < 22]
    F = cm.features('pairwise')
    T = F[cm.pattern_ids(Ma)].sum(0)
    th, _ = cm.fit_expfam(F, T, len(Ma), ridge=1e-6)
    assert abs(th[12] - 0.743) < 0.005


def test_bernoulli_logprob():
    X = np.array([[1, 0], [0, 0]])
    p = np.array([.25, .5])
    assert np.allclose(cm.bernoulli_logprob(X, p),
                       [np.log(.25) + np.log(.5), np.log(.75) + np.log(.5)])


def test_dirichlet_predictive():
    q = np.full(cm.NPAT, 1 / cm.NPAT)
    c = np.zeros(cm.NPAT)
    c[3] = 10
    pr = cm.dirichlet_predictive(c, q, alpha=495)
    assert np.isclose(pr.sum(), 1.0)
    assert np.isclose(pr[3], (10 + 1) / (10 + 495))


# ------------------------------------------------------------------ segmentation
def _bern_cost(blocks):
    R = len(blocks)
    cost = np.full((R + 1, R + 1), -np.inf)
    for i in range(R):
        for j in range(i + 1, R + 1):
            x = np.vstack(blocks[i:j])
            cost[i, j] = cm.bernoulli_logprob(x, x.mean(0)).sum()
    return cost


def test_segment_dp_matches_brute_force_and_recovers_plant():
    rng = np.random.default_rng(2)
    pa, pb = np.full(4, .8), np.full(4, .2)
    blocks = [(rng.random((20, 4)) < (pa if r < 6 else pb)).astype(int) for r in range(14)]
    sizes = np.array([len(b) for b in blocks])
    cost = _bern_cost(blocks)
    res = cm.segment_dp(cost, sizes, kmax=3, min_size=40)
    assert res[2][1] == [6]
    assert np.isclose(res[1][0], cost[0, 14])
    # brute force for k = 2, 3
    best2 = max((cost[0, a] + cost[a, 14], [a]) for a in range(2, 13))
    best3 = max((cost[0, a] + cost[a, b] + cost[b, 14], [a, b])
                for a, b in combinations(range(2, 13), 2) if b - a >= 2)
    assert np.isclose(res[2][0], best2[0]) and res[2][1] == best2[1]
    assert np.isclose(res[3][0], best3[0]) and res[3][1] == best3[1]


def test_reduced_to_original():
    # hold out rack 21; the boundary in the gap (before reduced index 21 = original rack 22)
    assert cm.reduced_to_original(21, 21) == 22
    assert cm.reduced_to_original(21, 5) == 22      # rack 5 removed: reduced 21 = original 22
    assert cm.reduced_to_original(22, 30) == 22     # removal above the cut changes nothing


# ------------------------------------------------------------------ beta-binomial
def test_betabinom_loglik_matches_scipy():
    y, n = np.array([3, 10, 0, 7]), np.array([10, 12, 5, 7])
    mu, rho = .4, .15
    s = 1 / rho - 1
    ref = betabinom.logpmf(y, n, mu * s, (1 - mu) * s).sum()
    assert np.isclose(cm.betabinom_loglik(y, n, mu, rho), ref)


def test_betabinom_two_segment_finds_plant():
    rng = np.random.default_rng(3)
    n = np.full(30, 40)
    mu = np.where(np.arange(30) < 12, .85, .4)
    s = 1 / .05 - 1
    y = betabinom.rvs(n, mu * s, (1 - mu) * s, random_state=rng)
    ll = {c: cm.betabinom_two_segment(y, n, c)[0] for c in range(3, 28)}
    assert max(ll, key=ll.get) == 12
    _, muA, muB, rho = cm.betabinom_two_segment(y, n, 12)
    assert abs(muA - .85) < .08 and abs(muB - .4) < .08 and 0 < rho < .2


# ------------------------------------------------------------------ diversity
def test_entropy_estimators_basic():
    c = np.full(8, 1000)
    assert np.isclose(cm.entropy_plugin(c), 3.0)
    assert np.isclose(cm.entropy_miller_madow(c), 3.0 + 7 / (2 * 8000) / np.log(2))
    assert abs(cm.entropy_chao_shen(c) - 3.0) < 1e-6
    assert np.isclose(cm.entropy_plugin(c, base=np.e), np.log(8))


def test_entropy_bias_correction_helps():
    """On a skewed 495-category law sampled at the fleet's size, the corrected estimators are
    on average closer to the truth than the (downward-biased) plug-in."""
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.full(495, 0.6))
    H = -(p * np.log2(p)).sum()
    pl, mm, cs = [], [], []
    for _ in range(200):
        c = rng.multinomial(1962, p)
        pl.append(cm.entropy_plugin(c))
        mm.append(cm.entropy_miller_madow(c))
        cs.append(cm.entropy_chao_shen(c))
    assert np.mean(pl) < H
    assert abs(np.mean(mm) - H) < abs(np.mean(pl) - H)
    assert abs(np.mean(cs) - H) < abs(np.mean(pl) - H)


def test_hill_and_richness():
    c = np.array([5, 5, 5, 5])
    q0, q1, q2 = cm.hill_numbers(c)
    assert np.allclose([q0, q1, q2], [4, 4, 4])
    assert np.isclose(cm.inverse_simpson_unbiased(c), 20 * 19 / (4 * 5 * 4))
    assert np.isclose(cm.rarefied_richness(c, 20), 4)
    assert np.isclose(cm.rarefied_richness(c, 1), 1)
    rng = np.random.default_rng(5)
    c2 = np.array([30, 10, 5, 2, 1, 1, 1])
    pool = np.repeat(np.arange(len(c2)), c2)
    mc = np.mean([len(set(rng.choice(pool, 12, replace=False))) for _ in range(20000)])
    assert abs(cm.rarefied_richness(c2, 12) - mc) < 0.03
    # Chao1 (bias-corrected): S_obs + (N-1)/N f1(f1-1)/(2(f2+1))
    N = c2.sum()
    assert np.isclose(cm.chao1(c2), 7 + (N - 1) / N * 3 * 2 / (2 * (1 + 1)))


def test_unique_fraction():
    assert cm.unique_fraction(['a', 'b', 'b', 'c']) == 0.5


def test_jackknife_se_of_mean_is_classical_se():
    rng = np.random.default_rng(6)
    x = rng.normal(size=37)
    reps = [np.delete(x, i).mean() for i in range(len(x))]
    assert np.isclose(cm.jack_se(reps), x.std(ddof=1) / np.sqrt(len(x)))


def test_count_based_welch_scan_matches_socket_level():
    """cohort_boundary's rack-count Welch profile equals the socket-level Welch t of
    changepoint.py, and reproduces the paper's argmax (rack 22, |t| = 26.6)."""
    import cohort_boundary as cb
    s0, rack = cb.s0, cb.rack
    prof = cb.welch_profile_counts(cb.y_r, cb.sizes, cb.CANDS)
    for c, t in zip(cb.CANDS, prof):
        a, b = s0[rack < c], s0[rack >= c]
        # changepoint.py's welch_t, including its 1e-9 regulariser under the square root
        ref = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-9)
        assert np.isclose(t, ref, rtol=1e-9)
    k = int(np.argmax(np.abs(prof)))
    assert cb.CANDS[k] == 22 and round(abs(prof[k]), 1) == 26.6


def test_leave_one_rack_out_partition_agreement():
    """The paper's corrected count: with each rack held out, the scan on the other 48 splits
    them exactly as the full-data boundary does in all 49 folds. (The earlier "48 of 49"
    compared boundary NAMES; a boundary in the gap left by a held-out neighbour has two.)"""
    import cohort_boundary as cb
    agree = 0
    for r in range(cb.NRACK):
        rl = np.array([x for x in range(cb.NRACK) if x != r])
        c_red = cb.welch_cut_reduced(rl)
        left = set(rl[:c_red].tolist())
        cut = cm.reduced_to_original(c_red, r)
        # the named boundary and the partition must agree: left side = racks below `cut`
        assert left == {x for x in rl.tolist() if x < cut}
        agree += left == {x for x in rl.tolist() if x < 22}
    assert agree == cb.NRACK
