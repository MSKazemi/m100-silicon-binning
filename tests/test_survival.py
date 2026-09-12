"""Unit tests for the replacement-survival estimators (analysis/survival_core.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import survival_core as sc  # noqa: E402


# ---------------------------------------------------------------- synthetic survival data
def simulate(n=3000, T=12, beta=(0.5, -0.3), link='cloglog', seed=0, late=0.1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    alpha = -4.0 + 0.8 * np.sin(np.arange(T))
    a = np.where(rng.random(n) < late, rng.integers(0, T // 2, n), 0)
    b = np.full(n, T - 1)
    d = np.zeros(n, int)
    for i in range(n):
        for t in range(a[i], T):
            h = sc.inv_link(alpha[t] + X[i] @ np.array(beta), link)
            if rng.random() < h:
                b[i], d[i] = t, 1
                break
    return X, a, b, d, alpha


def rowwise_negll(theta, X, a, b, d, T, link):
    alpha, beta = theta[:T], theta[T:]
    ll = 0.0
    for i in range(len(a)):
        for t in range(a[i], b[i] + 1):
            h = sc.inv_link(alpha[t] + X[i] @ beta, link)
            y = 1 if (d[i] and t == b[i]) else 0
            ll += y * np.log(h) + (1 - y) * np.log1p(-h)
    return -ll


@pytest.mark.parametrize('link', ['cloglog', 'logit'])
def test_fit_matches_bruteforce_likelihood(link):
    X, a, b, d, _ = simulate(n=250, T=6, beta=(0.8, -0.5), link=link, seed=1)
    X = X[:, :2]
    d = d.copy()
    T = 6
    fit = sc.fit_dth(X, a, b, d, T, link=link)
    assert fit['converged']
    ref = minimize(rowwise_negll, np.concatenate([np.full(T, -2.0), np.zeros(2)]),
                   args=(X, a, b, d, T, link), method='BFGS', options={'gtol': 1e-7})
    assert fit['loglik'] == pytest.approx(-ref.fun, abs=1e-5)
    np.testing.assert_allclose(fit['beta'], ref.x[T:], atol=2e-3)
    # the per-unit log-likelihood helper agrees with the fitter
    h = sc.inv_link(fit['alpha'][None, :] + (X @ fit['beta'])[:, None], link)
    assert sc.loglik_rows(h, a, b, d).sum() == pytest.approx(fit['loglik'], abs=1e-8)


def test_cloglog_recovers_hazard_ratios():
    X, a, b, d, _ = simulate(n=6000, T=12, beta=(0.5, -0.3), seed=2)
    fit = sc.fit_dth(X, a, b, d, 12, link='cloglog')
    se = np.sqrt(np.diag(fit['cov']))[12:]
    assert abs(fit['beta'][0] - 0.5) < 3 * se[0]
    assert abs(fit['beta'][1] + 0.3) < 3 * se[1]


def test_penalised_baseline_is_finite_in_eventless_month():
    X, a, b, d, _ = simulate(n=400, T=8, seed=3)
    ev_months = set(b[d == 1])
    T = 10                                              # months 8, 9: nobody at risk
    fit = sc.fit_dth(X, a, b, d, T, pen_base=sc.fe_penalty(T, 1.0))
    assert np.all(np.isfinite(fit['alpha']))
    assert fit['converged']
    assert len(ev_months) >= 1


def test_standardisation_does_not_change_unpenalised_predictions():
    """Affine rescaling of covariates is absorbed by an unpenalised MLE (the v1 'leak' was inert)."""
    X, a, b, d, _ = simulate(n=1500, T=8, seed=4)
    f1 = sc.fit_dth(X, a, b, d, 8)
    Xs = (X - X.mean(0)) / X.std(0)
    f2 = sc.fit_dth(Xs, a, b, d, 8)
    p1 = f1['alpha'][None, :] + (X @ f1['beta'])[:, None]
    p2 = f2['alpha'][None, :] + (Xs @ f2['beta'])[:, None]
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_rcs_basis_shape_and_linear_tail():
    B = sc.rcs_basis(28, [1, 7, 13.5, 20, 26])
    assert B.shape == (28, 5)
    # restricted: linear beyond the last knot -> second differences vanish there
    assert np.allclose(np.diff(B[26:, 2:], 2, axis=0), 0, atol=1e-12)


# ---------------------------------------------------------------- metrics against brute force
def brute_concordance(score, a, b, d, group):
    conc = pairs = 0.0
    for i in range(len(a)):
        if not d[i]:
            continue
        for j in range(len(a)):
            if j == i or group[j] != group[i]:
                continue
            at_risk = a[j] <= b[i] and (b[j] > b[i] or (b[j] == b[i] and d[j] == 0))
            if at_risk:
                pairs += 1
                conc += 1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0
    return conc / pairs


def test_risk_set_concordance_bruteforce():
    rng = np.random.default_rng(5)
    n = 120
    a = rng.integers(0, 3, n)
    b = a + rng.integers(0, 6, n)
    d = rng.integers(0, 2, n)
    score = rng.integers(0, 4, n).astype(float)          # many ties on purpose
    group = rng.integers(0, 3, n)
    assert sc.risk_set_concordance(score, a, b, d, group) == pytest.approx(
        brute_concordance(score, a, b, d, group))


def test_concordance_perfect_and_reversed():
    a = np.zeros(6, int)
    b = np.array([0, 1, 2, 5, 5, 5])
    d = np.array([1, 1, 1, 0, 0, 0])
    risk = -b.astype(float)                              # earlier event = higher risk
    assert sc.risk_set_concordance(risk, a, b, d) == 1.0
    assert sc.risk_set_concordance(-risk, a, b, d) == 0.0


def test_cd_auc_bruteforce():
    rng = np.random.default_rng(6)
    n = 200
    a = np.zeros(n, int)
    b = rng.integers(0, 10, n)
    d = rng.integers(0, 2, n)
    b[d == 0] = 9
    s = rng.normal(size=n)
    tau = 4
    cases = np.where((d == 1) & (b <= tau))[0]
    ctrls = np.where(b > tau)[0]
    ref = np.mean([(s[i] > s[j]) + 0.5 * (s[i] == s[j]) for i in cases for j in ctrls])
    assert sc.cd_auc(s, a, b, d, tau) == pytest.approx(ref)


@pytest.mark.parametrize('tau', [None, 3])
def test_cluster_bootstrap_parts_match_duplicated_data(tau):
    """Cluster multiplicities w in pair_parts == the metric on data with clusters duplicated."""
    rng = np.random.default_rng(10)
    n = 150
    cl = rng.integers(0, 10, n)
    grp = cl % 3                                         # folds are unions of clusters
    a = rng.integers(0, 2, n)
    b = a + rng.integers(0, 6, n)
    d = rng.integers(0, 2, n)
    s = rng.integers(0, 5, n).astype(float)
    clusters = np.arange(10)
    parts = sc.pair_parts(s, a, b, d, cl, clusters, group=grp, tau=tau)
    metric = (lambda *x: sc.risk_set_concordance(*x)) if tau is None else \
        (lambda s_, a_, b_, d_, g_: sc.cd_auc(s_, a_, b_, d_, tau, g_))
    assert sc.weighted_pairs(parts, np.ones(10)) == pytest.approx(metric(s, a, b, d, grp))
    w = np.array([2, 0, 1, 3, 1, 0, 1, 2, 1, 1], float)
    idx = np.concatenate([np.where(cl == c)[0] for c in clusters for _ in range(int(w[c]))])
    assert sc.weighted_pairs(parts, w) == pytest.approx(
        metric(s[idx], a[idx], b[idx], d[idx], grp[idx]))


def test_brier_curve_known_values():
    h = np.full((2, 3), 0.5)
    a = np.array([0, 0])
    b = np.array([1, 2])
    d = np.array([1, 0])
    err, valid = sc.brier_curve(h, a, b, d)
    S = np.array([0.5, 0.25, 0.125])
    np.testing.assert_allclose(err[0], [(1 - S[0]) ** 2, S[1] ** 2, S[2] ** 2])
    np.testing.assert_allclose(err[1], (1 - S) ** 2)
    assert valid.all()


# ---------------------------------------------------------------- no train/test leakage
def test_rarity_uses_training_keys_only():
    train = [(0, 1, 2, 3)] * 5 + [(1, 2, 3, 4)] * 3
    test_a = [(0, 1, 2, 3), (5, 6, 7, 8)]
    test_b = [(0, 1, 2, 3)] * 50 + [(5, 6, 7, 8)] * 50      # a very different held-out fold
    ra = sc.rarity(train, test_a)
    rb = sc.rarity(train, test_b)
    assert ra[0] == rb[0] and ra[1] == rb[-1]              # held-out keys never enter
    # an unseen pattern is finite and rarer than any seen one
    assert np.isfinite(ra[1]) and ra[1] > ra[0]
    n = len(train)
    assert ra[0] == pytest.approx(-np.log((5 + 0.5) / (n + 0.5 * 495)))


def test_design_learns_scaling_on_train_only():
    rng = np.random.default_rng(7)
    keys = [tuple(sorted(rng.choice(12, 4, replace=False))) for _ in range(300)]
    lot = rng.integers(0, 2, 300)
    tr, te = keys[:200], keys[200:]
    Xt, Xe, mu, sd = sc.design(tr, te, lot[:200], lot[200:])
    Xt2, Xe2, mu2, sd2 = sc.design(tr, te[::-1], lot[:200], lot[200:][::-1])
    np.testing.assert_allclose(Xt, Xt2)                      # training design blind to test fold
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(Xe[::-1], Xe2)
    np.testing.assert_allclose(Xt[:, [1, 2, 3]].mean(0), 0, atol=1e-12)
    assert set(np.unique(Xt[:, 0])) <= {0.0, 1.0} and set(np.unique(Xt[:, 4])) <= {0.0, 1.0}


def test_design_ids_matches_design():
    rng = np.random.default_rng(8)
    keys = [tuple(sorted(rng.choice(12, 4, replace=False))) for _ in range(400)]
    lot = rng.integers(0, 2, 400)
    uniq = sorted(set(keys))
    ids = np.array([uniq.index(k) for k in keys])
    F = sc.map_features(uniq)
    A = sc.design(keys[:300], keys[300:], lot[:300], lot[300:])
    B = sc.design_ids(ids[:300], ids[300:], lot[:300], lot[300:], F)
    for x, y in zip(A, B):
        np.testing.assert_allclose(x, y)


def test_fit_without_covariates():
    X, a, b, d, _ = simulate(n=500, T=6, seed=9)
    f0 = sc.fit_dth(np.zeros((500, 0)), a, b, d, 6, pen_base=sc.fe_penalty(6, 1.0))
    assert f0['converged'] and f0['beta'].shape == (0,)


# ---------------------------------------------------------------- die tracking
D = pd.Timestamp('2021-01-10', tz='UTC')


def tr_rows(rows):
    return pd.DataFrame([(n, s, D + pd.Timedelta(days=o), D + pd.Timedelta(days=o + 2), old, new)
                         for (n, s, o, old, new) in rows],
                        columns=['node', 'sock', 'd0', 'd1', 'old', 'new'])


def init(maps):
    return {lab: (m, D) for lab, m in maps.items()}


def test_relabel_moves_no_die():
    tr = tr_rows([(1, 0, 0, 'A', 'B'), (1, 1, 0, 'B', 'A')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert [x['kind'] for x in ev[(1, 0)]] == ['relabel']
    assert [x['kind'] for x in ev[(1, 1)]] == ['relabel']
    assert log['retired'] == 0 and log['spares_in'] == 0


def test_cpu_swap_retires_one_die():
    tr = tr_rows([(1, 0, 0, 'A', 'C')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert ev[(1, 0)][0]['kind'] == 'retired' and ev[(1, 0)][0]['n_out'] == 1
    assert ev[(1, 1)] == []


def test_shift_is_one_departure_not_two():
    # node 113 pattern: p1's die reappears on p0, p1 gets a new die; p0's die is gone
    tr = tr_rows([(1, 0, 0, 'A', 'B'), (1, 1, 1, 'B', 'C')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert ev[(1, 0)][0]['kind'] == 'retired' and ev[(1, 0)][0]['n_out'] == 1
    assert ev[(1, 1)][0]['kind'] == 'relabel'
    assert log['retired'] == 1 and log['spares_in'] == 1


def test_node_swap_retires_both():
    tr = tr_rows([(1, 0, 0, 'A', 'C'), (1, 1, 0, 'B', 'E')])
    ev, _ = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert all(ev[k][0]['kind'] == 'retired' and ev[k][0]['n_out'] == 2 for k in ev)


def test_reciprocal_exchange_between_nodes_is_a_move():
    tr = tr_rows([(1, 0, 0, 'A', 'X'), (2, 1, 0, 'X', 'A')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B', (2, 0): 'Y', (2, 1): 'X'}))
    assert ev[(1, 0)][0]['kind'] == 'moved' and ev[(2, 1)][0]['kind'] == 'moved'
    assert log['retired'] == 0 and log['cross_node_moves'] == 2
    assert log['reciprocal_cross_node'] == 2
    pulled, _ = sc.endpoint_table(ev, pd.DataFrame({'node': [1, 2], 'socket': [0, 1]}), 'pulled')
    retired, _ = sc.endpoint_table(ev, pd.DataFrame({'node': [1, 2], 'socket': [0, 1]}), 'retired')
    assert pulled.notna().all() and retired.isna().all()


def test_swap_after_relabel_is_charged_to_the_die_that_moved_there():
    # RELABEL on day 0, then label p0 changes on day 20: the die leaving is the one first seen on p1
    tr = tr_rows([(1, 0, 0, 'A', 'B'), (1, 1, 0, 'B', 'A'), (1, 0, 20, 'B', 'C')])
    ev, _ = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert [x['kind'] for x in ev[(1, 1)]] == ['relabel', 'retired']
    assert [x['kind'] for x in ev[(1, 0)]] == ['relabel']
    units = pd.DataFrame({'node': [1, 1], 'socket': [0, 1]})
    e, _ = sc.endpoint_table(ev, units, 'pulled')
    assert pd.isna(e.iloc[0]) and e.iloc[1] == D + pd.Timedelta(days=22)


def test_relabel_then_swap_inside_one_episode():
    # node 316 pattern: RELABEL (window 0..2) and a change on p1 (window 2..7) share a day, so they
    # form one episode; the die leaving p1 on day 7 is A, which came from p0 via the relabel
    tr = pd.DataFrame([(1, 0, D, D + pd.Timedelta(days=2), 'A', 'B'),
                       (1, 1, D, D + pd.Timedelta(days=2), 'B', 'A'),
                       (1, 1, D + pd.Timedelta(days=2), D + pd.Timedelta(days=7), 'A', 'C')],
                      columns=['node', 'sock', 'd0', 'd1', 'old', 'new'])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B'}))
    assert log['holder_mismatch'] == 0 and log['episodes'] == 1
    assert [x['kind'] for x in ev[(1, 0)]] == ['relabel', 'retired']
    assert ev[(1, 0)][1]['day'] == D + pd.Timedelta(days=7) and ev[(1, 0)][1]['n_out'] == 1
    assert [x['kind'] for x in ev[(1, 1)]] == ['relabel']


def test_node_pair_exchange_retires_nothing():
    # nodes 142/151 pattern: both dies of each node reappear on the other node the same day
    tr = tr_rows([(1, 0, 0, 'A', 'C'), (1, 1, 0, 'B', 'D'), (2, 0, 0, 'C', 'A'), (2, 1, 0, 'D', 'B')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B', (2, 0): 'C', (2, 1): 'D'}))
    assert log['retired'] == 0 and log['cross_node_moves'] == 4 and log['spares_in'] == 0
    assert log['reciprocal_cross_node'] == 4
    assert all(ev[k][0]['kind'] == 'moved' and ev[k][0]['n_out'] == 2 for k in ev)


def test_one_way_move_between_nodes_is_not_reciprocal():
    # node 1's die A goes to node 2, whose die Y leaves the fleet; node 1 receives a spare S
    tr = tr_rows([(1, 0, 0, 'A', 'S'), (2, 0, 0, 'Y', 'A')])
    ev, log = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B', (2, 0): 'Y', (2, 1): 'Z'}))
    assert ev[(1, 0)][0]['kind'] == 'moved' and ev[(2, 0)][0]['kind'] == 'retired'
    assert log['cross_node_moves'] == 1 and log['reciprocal_cross_node'] == 0
    assert log['retired'] == 1 and log['spares_in'] == 1


def test_endpoint_single_double_censoring():
    tr = tr_rows([(1, 0, 0, 'A', 'C'), (1, 1, 0, 'B', 'E'), (2, 0, 5, 'P', 'Q')])
    ev, _ = sc.track_dies(tr, init({(1, 0): 'A', (1, 1): 'B', (2, 0): 'P', (2, 1): 'R'}))
    units = pd.DataFrame({'node': [1, 1, 2, 2], 'socket': [0, 1, 0, 1]})
    e1, c1 = sc.endpoint_table(ev, units, 'single')
    assert e1.isna().tolist() == [True, True, False, True]
    assert c1.notna().tolist() == [True, True, False, False]
    e2, c2 = sc.endpoint_table(ev, units, 'double')
    assert e2.notna().tolist() == [True, True, False, False]
    assert c2.notna().tolist() == [False, False, True, False]


def test_month_index():
    assert sc.month_index(pd.Timestamp('2020-06-30', tz='UTC')) == 0
    assert sc.month_index(pd.Timestamp('2022-09-01', tz='UTC')) == 27
