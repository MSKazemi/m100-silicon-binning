"""Estimators for the power-equivalence analysis (analysis/equivalence.py).

Kept apart from the analysis script so that tests/test_equivalence.py can check each one on
synthetic data where the right answer is known.

  slice_features       the three per-socket map summaries used in Sec. IV-L
  epoch_lookup         the configuration valid on a given day, from the clean days around it
  ols_cluster_boot     OLS plus a cluster (rack) bootstrap, via per-cluster sufficient statistics
  cluster_robust_se    CR1 sandwich standard errors, as an analytic cross-check on the bootstrap
  scales / effects     slope -> effect over the full range, per SD and per IQR of the regressor
  tost / eq_bound      two-one-sided-tests verdict, and the smallest margin it would pass at
  vif                  variance inflation factors of a set of regressors
"""
import numpy as np

FEATURES = ('meanidx', 'nadj', 'lowhalf')
MARGINS_W = (2.0, 5.0, 10.0)


# ------------------------------------------------------------------ map summaries
def active_slices(code):
    """Active slice indices 0..11 of a 24-bit active-core code (bit k = core k reports)."""
    return sorted({k // 2 for k in range(24) if (int(code) >> k) & 1})


def slice_features(code):
    """(mean active-slice index, index-adjacent active pairs, active slices with index < 6)."""
    a = active_slices(code)
    if not a:
        raise ValueError('empty map')
    nadj = sum(1 for i in range(len(a) - 1) if a[i + 1] - a[i] == 1)
    return float(np.mean(a)), nadj, sum(1 for k in a if k < 6)


# ------------------------------------------------------------------ contemporaneous map
def epoch_lookup(clean_key, clean_day, clean_code, q_key, q_day):
    """The map valid on each query (key, day), from that key's clean days.

    clean_key, clean_day, clean_code : one entry per clean socket-day; key identifies the socket
        (e.g. 2*node + socket), day is an integer day number, code the map observed that day.
    q_key, q_day : the socket-days to label.

    Returns (code, how). how = 1: a clean day, map observed that day; how = 2: inside a
    configuration epoch, i.e. between two consecutive clean days of the same socket that carry
    the identical map; how = 0: unassigned (before the first or after the last clean day of the
    socket, or inside a gap whose bracketing clean days differ -- a transition). code is -1 when
    unassigned.
    """
    ck = np.asarray(clean_key, dtype=np.int64)
    cd = np.asarray(clean_day, dtype=np.int64)
    cc = np.asarray(clean_code, dtype=np.int64)
    qk = np.asarray(q_key, dtype=np.int64)
    qd = np.asarray(q_day, dtype=np.int64)
    if cd.size and (cd.min() < 0 or cd.max() >= 1 << 24 or qd.min() < 0 or qd.max() >= 1 << 24):
        raise ValueError('day numbers must lie in [0, 2**24)')
    o = np.lexsort((cd, ck))
    ck, cd, cc = ck[o], cd[o], cc[o]
    K = (ck << 24) + cd
    if np.any(np.diff(K) == 0):
        raise ValueError('duplicate clean (key, day)')
    Q = (qk << 24) + qd
    i = np.searchsorted(K, Q, side='right') - 1
    code = np.full(Q.shape, -1, dtype=np.int64)
    how = np.zeros(Q.shape, dtype=np.int8)
    if not K.size:
        return code, how
    ii = np.clip(i, 0, None)
    same = (i >= 0) & (ck[ii] == qk)
    exact = same & (K[ii] == Q)
    code[exact], how[exact] = cc[ii[exact]], 1
    j = np.clip(i + 1, 0, len(K) - 1)
    inner = same & ~exact & (i + 1 < len(K)) & (ck[j] == qk) & (cc[ii] == cc[j])
    code[inner], how[inner] = cc[ii[inner]], 2
    return code, how


# ------------------------------------------------------------------ regression + uncertainty
def _cluster_stats(X, y, g):
    ug, inv = np.unique(g, return_inverse=True)
    k = X.shape[1]
    XtX = np.zeros((len(ug), k, k))
    Xty = np.zeros((len(ug), k))
    for c in range(len(ug)):
        s = inv == c
        Xs = X[s]
        XtX[c] = Xs.T @ Xs
        Xty[c] = Xs.T @ y[s]
    return ug, XtX, Xty


def ols_cluster_boot(X, y, g, B=4000, seed=0):
    """OLS coefficients and B cluster-bootstrap replicates.

    Resampling whole clusters with replacement and refitting is exactly equivalent to solving
    (sum_c w_c X_c'X_c) b = sum_c w_c X_c'y with multinomial multiplicities w, so the replicates
    cost one k x k solve each instead of a pass over the rows. A replicate whose normal matrix
    is singular (e.g. a resample that happens to contain no row of some month) is solved by
    least squares on the normal equations; the count of such replicates is returned.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ug, XtX, Xty = _cluster_stats(X, y, g)
    b = np.linalg.lstsq(XtX.sum(0), Xty.sum(0), rcond=None)[0]
    rng = np.random.default_rng(seed)
    G = len(ug)
    Wm = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    A = np.einsum('bc,cij->bij', Wm, XtX)
    r = Wm @ Xty
    bs = np.empty((B, X.shape[1]))
    nsing = 0
    for t in range(B):
        try:
            bs[t] = np.linalg.solve(A[t], r[t])
        except np.linalg.LinAlgError:
            bs[t] = np.linalg.lstsq(A[t], r[t], rcond=None)[0]
            nsing += 1
    return b, bs, nsing


def cluster_robust_se(X, y, g):
    """CR1 cluster-robust standard errors (Stata's small-sample factor)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.pinv(X.T @ X)
    ug, inv = np.unique(g, return_inverse=True)
    S = np.zeros((len(ug), k))
    np.add.at(S, inv, X * e[:, None])
    meat = S.T @ S
    G = len(ug)
    V = G / (G - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    return b, np.sqrt(np.diag(V))


def percentile_ci(draws, level=0.90):
    a = 100 * (1 - level) / 2
    lo, hi = np.percentile(draws, [a, 100 - a])
    return float(lo), float(hi)


# ------------------------------------------------------------------ effect scales and TOST
def scales(d):
    """Spread of a regressor on three scales: full observed range, SD, interquartile range."""
    d = np.asarray(d, float)
    q25, q75 = np.percentile(d, [25, 75])
    return {'range': float(d.max() - d.min()), 'sd': float(d.std()), 'iqr': float(q75 - q25)}


def tost(lo, hi, margin):
    """Two one-sided tests at level alpha from a (1 - 2 alpha) interval: inside +/- margin."""
    return bool(lo > -margin and hi < margin)


def eq_bound(lo, hi):
    """Smallest margin the interval fits inside; TOST passes for every margin above it."""
    return float(max(abs(lo), abs(hi)))


def wquantile(values, weights, q):
    """Weighted quantile(s), inverse-CDF definition: smallest v with cumulative weight >= q."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    idx = np.searchsorted(c, np.atleast_1d(q), side='left')
    return v[np.clip(idx, 0, len(v) - 1)]


def vif(D):
    """Variance inflation factor of each column of D (columns are regressors, no intercept)."""
    D = np.asarray(D, float)
    out = []
    for j in range(D.shape[1]):
        yj = D[:, j]
        Xo = np.column_stack([np.ones(len(D)), np.delete(D, j, axis=1)])
        bj = np.linalg.lstsq(Xo, yj, rcond=None)[0]
        r = yj - Xo @ bj
        r2 = 1 - r.var() / yj.var()
        out.append(float(1 / (1 - r2)))
    return out
