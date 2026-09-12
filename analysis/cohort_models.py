"""Estimators behind the cohort (rack-22 boundary) analyses. Pure functions, no I/O.

Kept separate from the scripts that use them so that each estimator is unit-tested
(tests/test_cohort.py) rather than trusted.

Map encoding throughout: X[s, k] = 1 if slice k of socket s is HARVESTED (disabled); every
row has exactly four ones, so a map is one of the C(12,4) = 495 four-subsets of {0..11}.

Contents
  * fixed-cardinality map models over the 495 valid maps (exponential family, exact
    normaliser): the conditional-Bernoulli model P(S) ~ exp(sum_{k in S} beta_k) and the
    paper's pairwise model P(S) ~ exp(sum beta_k + gamma adj(S)), with the independent
    12-Bernoulli model kept only as the reference the paper used;
  * a Dirichlet-smoothed multinomial over the 495 maps;
  * dynamic-programming segmentation of a rack-ordered sequence into k contiguous segments;
  * a rack-level beta-binomial profile likelihood for a single changepoint;
  * entropy estimators (plug-in, Miller-Madow, Chao-Shen) and Hill numbers.
"""
from itertools import combinations

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

NSLICE = 12
PATTERNS = np.array(list(combinations(range(NSLICE), 4)))          # (495, 4), sorted
NPAT = len(PATTERNS)
A_SLICE = np.zeros((NPAT, NSLICE))
A_SLICE[np.arange(NPAT)[:, None], PATTERNS] = 1.0
# adj(S): number of unordered pairs {k, k+1} both in S (no wrap-around), as in yield_model.py
ADJ = np.array([sum(1 for a, b in combinations(p, 2) if abs(a - b) == 1) for p in PATTERNS],
               dtype=float)
_CODE = {tuple(p): i for i, p in enumerate(PATTERNS)}
EPS = 1e-9


# ------------------------------------------------------------------------------ map indexing
def pattern_ids(X):
    """Row-wise index (0..494) of each four-harvested map in PATTERNS."""
    X = np.asarray(X)
    if not (X.sum(1) == 4).all():
        raise ValueError('every map must have exactly four harvested slices')
    return np.array([_CODE[tuple(np.flatnonzero(r))] for r in X], dtype=int)


def features(model):
    """Sufficient-statistic design over the 495 maps for a fixed-cardinality model."""
    if model == 'cb':
        return A_SLICE
    if model == 'pairwise':
        return np.column_stack([A_SLICE, ADJ])
    raise ValueError(model)


# ------------------------------------------------------------------------------ model fitting
def fit_expfam(F, T, N, ridge=1e-2, theta0=None, tol=1e-10, max_iter=100):
    """MAP fit of P(S) ~ exp(theta . F[S]) over the rows of F, from sufficient stats.

    T = sum over sockets of F[S_i]; N = number of sockets. A weak ridge keeps estimates finite
    when a slice is never harvested in a small segment (and fixes the common-shift
    non-identifiability of beta under |S| = 4). Damped Newton; the objective is concave.
    Returns (theta, log-likelihood at theta without the penalty).
    """
    d = F.shape[1]
    th = np.zeros(d) if theta0 is None else np.array(theta0, float)

    def obj(t):
        eta = F @ t
        lz = logsumexp(eta)
        return T @ t - N * lz - 0.5 * ridge * t @ t, eta, lz

    val, eta, lz = obj(th)
    for _ in range(max_iter):
        p = np.exp(eta - lz)
        m = p @ F
        g = T - N * m - ridge * th
        H = N * ((F * p[:, None]).T @ F - np.outer(m, m)) + ridge * np.eye(d)
        step = np.linalg.solve(H, g)
        t_ = 1.0
        while True:
            cand = th + t_ * step
            v2, e2, l2 = obj(cand)
            if v2 >= val - 1e-12 or t_ < 1e-8:
                break
            t_ *= 0.5
        th, val, eta, lz = cand, v2, e2, l2
        if abs(g @ step) < tol:
            break
    ll = T @ th - N * logsumexp(F @ th)
    return th, float(ll)


def logprob_expfam(F, theta):
    """log P(S) for all 495 maps under theta."""
    eta = F @ theta
    return eta - logsumexp(eta)


def bernoulli_logprob(X, p):
    """Independent 12-Bernoulli log-probability of each row (the paper's original score)."""
    p = np.clip(p, EPS, 1 - EPS)
    return (X * np.log(p) + (1 - X) * np.log(1 - p)).sum(1)


def dirichlet_predictive(counts, q, alpha):
    """Posterior-mean map distribution (n_S + alpha q_S) / (N + alpha) over the 495 maps."""
    counts = np.asarray(counts, float)
    return (counts + alpha * q) / (counts.sum() + alpha)


# ------------------------------------------------------------------------------ segmentation
def segment_dp(cost, sizes, kmax, min_size):
    """Best partition of an ordered sequence of R blocks into k = 1..kmax contiguous segments.

    cost[i, j] = in-sample log-likelihood of blocks i..j-1 as one segment (-inf if unusable);
    sizes[r]   = number of sockets in block r; every segment must hold >= min_size sockets.
    Returns {k: (loglik, [cut positions])}, cuts being block indices where a segment starts.
    """
    R = len(sizes)
    csum = np.concatenate([[0], np.cumsum(sizes)])
    dp = np.full((kmax + 1, R + 1), -np.inf)
    bk = np.zeros((kmax + 1, R + 1), dtype=int)
    dp[0, 0] = 0.0
    for s in range(1, kmax + 1):
        for j in range(1, R + 1):
            for i in range(s - 1, j):
                if csum[j] - csum[i] < min_size or not np.isfinite(dp[s - 1, i]):
                    continue
                v = dp[s - 1, i] + cost[i, j]
                if v > dp[s, j]:
                    dp[s, j], bk[s, j] = v, i
    out = {}
    for k in range(1, kmax + 1):
        if not np.isfinite(dp[k, R]):
            continue
        cuts, j = [], R
        for s in range(k, 0, -1):
            i = bk[s, j]
            cuts.append(i)
            j = i
        out[k] = (float(dp[k, R]), sorted(c for c in cuts if c > 0))
    return out


def reduced_to_original(cut, held_out):
    """Map a boundary index found with block `held_out` removed back to original numbering.

    In the reduced sequence, block c is original block c if c < held_out, else c + 1. A
    boundary 'before reduced block c' is therefore 'before original block c + 1' whenever
    c >= held_out. A boundary in the gap left by the held-out block (c == held_out) could as
    well be named held_out; either name gives the same partition of the remaining blocks, so
    fold agreement must be judged on partitions, not names (an earlier changepoint.py count
    compared names and reported "48 of 49" where the partitions agree in all 49 folds).
    """
    return cut + 1 if cut >= held_out else cut


# ------------------------------------------------------------------------------ beta-binomial
def betabinom_loglik(y, n, mu, rho):
    """Sum of beta-binomial log-pmfs, mean mu and intra-cluster correlation rho in (0, 1)."""
    y, n = np.asarray(y, float), np.asarray(n, float)
    s = 1.0 / rho - 1.0
    a, b = mu * s, (1 - mu) * s
    return float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
                        + betaln(y + a, n - y + b) - betaln(a, b)))


def betabinom_two_segment(y, n, cut, grid=None):
    """Profile log-likelihood of a single changepoint at block `cut`, rack-level beta-binomial.

    Segment means mu_A, mu_B and a shared rho are maximised numerically. Returns (ll, muA, muB,
    rho).
    """
    from scipy.optimize import minimize
    y, n = np.asarray(y, float), np.asarray(n, float)
    left = np.arange(len(y)) < cut

    def nll(z):
        muA, muB, rho = 1 / (1 + np.exp(-z))
        return -(betabinom_loglik(y[left], n[left], muA, rho)
                 + betabinom_loglik(y[~left], n[~left], muB, rho))

    pA = np.clip(y[left].sum() / n[left].sum(), .02, .98)
    pB = np.clip(y[~left].sum() / n[~left].sum(), .02, .98)
    best = None
    for r0 in (0.01, 0.05, 0.2):
        z0 = np.log([pA / (1 - pA), pB / (1 - pB), r0 / (1 - r0)])
        res = minimize(nll, z0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 4000})
        if best is None or res.fun < best.fun:
            best = res
    muA, muB, rho = 1 / (1 + np.exp(-best.x))
    return -float(best.fun), float(muA), float(muB), float(rho)


# ------------------------------------------------------------------------------ diversity
def entropy_plugin(counts, base=2.0):
    """Maximum-likelihood (plug-in) Shannon entropy of a count vector."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(base))


def entropy_miller_madow(counts, base=2.0):
    """Plug-in plus the Miller-Madow first-order bias correction (K_obs - 1) / (2N), in nats."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    return entropy_plugin(c, base) + (len(c) - 1) / (2 * c.sum()) / np.log(base)


def entropy_chao_shen(counts, base=2.0):
    """Chao & Shen (2003) coverage-adjusted Horvitz-Thompson entropy estimator."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    N = c.sum()
    f1 = (c == 1).sum()
    if f1 == N:                       # all singletons: coverage adjustment undefined
        f1 = N - 1
    C = 1 - f1 / N
    pa = C * c / N
    la = 1 - (1 - pa) ** N
    return float(-(pa * np.log(pa) / la).sum() / np.log(base))


def hill_numbers(counts):
    """Plug-in Hill numbers of order 0, 1, 2 (richness, exp Shannon, inverse Simpson)."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    p = c / c.sum()
    return (float(len(c)), float(np.exp(-(p * np.log(p)).sum())), float(1 / (p ** 2).sum()))


def inverse_simpson_unbiased(counts):
    """Hill-2 from the unbiased estimator of sum p^2, N(N-1) / sum n(n-1)."""
    c = np.asarray(counts, float)
    N = c.sum()
    return float(N * (N - 1) / (c * (c - 1)).sum())


def chao1(counts):
    """Bias-corrected Chao1 lower-bound estimate of richness."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    N = c.sum()
    f1, f2 = (c == 1).sum(), (c == 2).sum()
    return float(len(c) + (N - 1) / N * f1 * (f1 - 1) / (2 * (f2 + 1)))


def rarefied_richness(counts, m):
    """Expected number of distinct categories in a subsample of size m without replacement."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    N = c.sum()
    if m > N:
        raise ValueError('m exceeds sample size')
    # 1 - C(N - n_i, m) / C(N, m), in logs
    lg = gammaln(N - c + 1) - gammaln(N - c - m + 1) - gammaln(N + 1) + gammaln(N - m + 1)
    lg = np.where(N - c >= m, lg, -np.inf)
    return float((1 - np.exp(lg)).sum())


def jack_se(vals):
    """Delete-one-group jackknife standard error from the leave-one-out replicates."""
    v = np.asarray(vals, float)
    return float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))


def unique_fraction(keys):
    """Fraction of items whose key occurs exactly once."""
    from collections import Counter
    keys = list(keys)
    cnt = Counter(keys)
    return sum(1 for k in keys if cnt[k] == 1) / len(keys)
