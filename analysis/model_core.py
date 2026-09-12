"""Shared, side-effect-free helpers for the harvest-map pattern statistics.

Encoding used everywhere in this file (and in harvest_maps_full.npy):

    M_s in {0,1}^12,  M[s,k] = 1  <=>  slice k of socket s is harvested (none of its two core
    sensors report),  sum_k M[s,k] = 4 for every socket analysed.

The harvested set of a socket is S = {k : M[s,k] = 1}, a 4-subset of {0,...,11}; there are
C(12,4) = 495 of them. Three groups of tools live here:

1. the pattern space (PATS, the design matrix A, the adjacency statistic ADJ, pair indicators);
2. the pairwise-interaction model
       P(S) = exp(sum_{k in S} beta_k + gamma * adj(S)) / Z(beta, gamma),   S in PATS,
   with adj(S) = #{k : k in S and k+1 in S}. Because |S| = 4 for every S, adding c to every
   beta_k multiplies every numerator by exp(4c) and cancels in Z, so beta is identified only up to
   that constant: we impose sum_k beta_k = 0 by parameterising beta = L b, b in R^11.
   Z is an exact 495-term sum, so the log-likelihood, gradient and Hessian are exact and the fit is
   a strictly concave problem, solved by Newton's method and cross-checked against L-BFGS;
3. curveball samplers for binary matrices with fixed row and column sums: the sequential trade of
   Strona et al. (2014) exactly as the paper's scripts implement it, and a vectorised "global
   trade" (random perfect matching of rows, one curveball trade per pair), whose chain is also
   symmetric, irreducible and aperiodic and so has the same uniform stationary distribution
   (Carstens, Berger & Strona 2016; Carstens et al., ESA 2018). Both are checked for exact
   uniformity on a small enumerable instance in tests/test_model.py.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

NP = 12
PATS = list(combinations(range(NP), 4))
NPAT = len(PATS)                                             # 495
PAIRS = list(combinations(range(NP), 2))                     # 66 unordered slice pairs
NPAIR = len(PAIRS)
PAIR_DIST = np.array([b - a for a, b in PAIRS])              # |dk| of each pair


def adj(S):
    """Number of unordered index-adjacent pairs {k, k+1} contained in the set S."""
    S = set(S)
    return sum(1 for k in S if k + 1 in S)


A = np.zeros((NPAT, NP))
for _i, _t in enumerate(PATS):
    A[_i, list(_t)] = 1.0
ADJ = np.array([adj(t) for t in PATS], dtype=float)
PAIRMAT = np.zeros((NPAT, NPAIR))                            # pair p contained in pattern i
_PIDX = {p: j for j, p in enumerate(PAIRS)}
for _i, _t in enumerate(PATS):
    for _p in combinations(_t, 2):
        PAIRMAT[_i, _PIDX[_p]] = 1.0
MAXGAP = np.array([max(t[i + 1] - t[i] for i in range(3)) for t in PATS], dtype=float)

_CODE2IDX = np.full(1 << NP, -1, dtype=np.int64)
for _i, _t in enumerate(PATS):
    _CODE2IDX[sum(1 << k for k in _t)] = _i
_BITS = (1 << np.arange(NP)).astype(np.int64)


def pattern_index(M):
    """Index into PATS of every row of a 0/1 map matrix (each row must have exactly four ones)."""
    idx = _CODE2IDX[np.asarray(M, dtype=np.int64) @ _BITS]
    if (idx < 0).any():
        raise ValueError('every map must harvest exactly four slices')
    return idx


def pattern_counts(M):
    return np.bincount(pattern_index(M), minlength=NPAT).astype(float)


def cooc_upper(X):
    """Co-harvest counts of the 66 slice pairs (upper triangle of X^T X, PAIRS order)."""
    W = np.asarray(X, dtype=np.int32)
    C = W.T @ W
    return C[np.triu_indices(NP, 1)]


def mean_gap(cu):
    """Mean |index distance| between harvested slices, from co-harvest counts (last axis = 66)."""
    cu = np.asarray(cu, dtype=float)
    return (cu * PAIR_DIST).sum(-1) / cu.sum(-1)


def to_matrix(v):
    """66-vector in PAIRS order -> symmetric 12x12 matrix with zero diagonal."""
    out = np.zeros((NP, NP))
    iu = np.triu_indices(NP, 1)
    out[iu] = v
    out[(iu[1], iu[0])] = v
    return out


# ---------------------------------------------------------------- pairwise-interaction model
L = np.vstack([np.eye(NP - 1), -np.ones((1, NP - 1))])      # beta = L b  =>  sum(beta) = 0


def design(gamma_free=True):
    X = A @ L
    return np.column_stack([X, ADJ]) if gamma_free else X


_X1, _X0 = design(True), design(False)


def _moments(X, theta):
    eta = X @ theta
    m = eta.max()
    w = np.exp(eta - m)
    z = w.sum()
    w /= z
    mu = X.T @ w
    cov = (X * w[:, None]).T @ X - np.outer(mu, mu)
    return w, np.log(z) + m, mu, cov


def _unpack(theta, gamma_free):
    beta = L @ theta[:NP - 1]
    return beta, (float(theta[NP - 1]) if gamma_free else 0.0)


def probs(beta, gamma):
    eta = A @ beta + gamma * ADJ
    w = np.exp(eta - eta.max())
    return w / w.sum()


def loglik(counts, beta, gamma):
    p = probs(beta, gamma)
    nz = counts > 0
    return float(counts[nz] @ np.log(p[nz]))


def fit_newton(counts, gamma_free=True, tol=1e-10, maxit=200, theta0=None):
    """Exact maximum likelihood by damped Newton. `counts` are the 495 pattern counts.

    Returns beta (sum-to-zero), gamma, the maximised log-likelihood, the final gradient norm (per
    socket), the number of iterations, the Hessian eigenvalues (per socket, in the identified
    parameterisation) and the inverse observed information of (b, gamma).
    """
    X = _X1 if gamma_free else _X0
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    T = X.T @ counts
    theta = np.zeros(X.shape[1]) if theta0 is None else np.array(theta0, dtype=float)
    w, logZ, mu, cov = _moments(X, theta)
    ll = theta @ T - N * logZ
    it = 0
    for it in range(1, maxit + 1):
        g = T - N * mu
        if np.max(np.abs(g)) / N < tol:
            break
        step = np.linalg.solve(N * cov, g)
        t = 1.0
        slack = 1e-12 * max(1.0, abs(ll))                    # rounding noise of ll itself
        while True:
            th = theta + t * step
            w2, logZ2, mu2, cov2 = _moments(X, th)
            ll2 = th @ T - N * logZ2
            if ll2 >= ll + 1e-4 * t * (g @ step) - slack or t < 1e-10:
                break
            t *= 0.5
        theta, w, logZ, mu, cov, ll = th, w2, logZ2, mu2, cov2, ll2
    g = T - N * mu
    info = N * cov
    beta, gamma = _unpack(theta, gamma_free)
    return {'beta': beta, 'gamma': gamma, 'theta': theta, 'll': float(ll),
            'grad_norm': float(np.linalg.norm(g) / N), 'iterations': it, 'N': N,
            'hess_eig': np.linalg.eigvalsh(cov), 'inv_info': np.linalg.inv(info), 'p': w,
            'mu': mu}


def fit_design(X, counts, theta0=None, tol=1e-10, maxit=200):
    """Exact MLE of the exponential family P(i) ~ exp(X[i] @ theta) over the 495 patterns, for any
    identified design X (e.g. beta via L plus several adjacency statistics). Damped Newton."""
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    T = X.T @ counts
    theta = np.zeros(X.shape[1]) if theta0 is None else np.array(theta0, dtype=float)
    w, logZ, mu, cov = _moments(X, theta)
    ll = theta @ T - N * logZ
    for _ in range(maxit):
        g = T - N * mu
        if np.max(np.abs(g)) / N < tol:
            break
        step = np.linalg.solve(N * cov, g)
        t, slack = 1.0, 1e-12 * max(1.0, abs(ll))
        while True:
            th = theta + t * step
            w2, logZ2, mu2, cov2 = _moments(X, th)
            ll2 = th @ T - N * logZ2
            if ll2 >= ll + 1e-4 * t * (g @ step) - slack or t < 1e-10:
                break
            t *= 0.5
        theta, w, logZ, mu, cov, ll = th, w2, logZ2, mu2, cov2, ll2
    g = T - N * mu
    return {'theta': theta, 'll': float(ll), 'p': w, 'mu': mu,
            'grad_norm': float(np.linalg.norm(g) / N), 'inv_info': np.linalg.inv(N * cov)}


# Within-quad and cross-quad adjacency. POWER9 groups its 24 cores into six four-core quads (EQ
# chiplets), cores 4q..4q+3 (POWER9 Processor User's Manual; pdbg's chiplet tree), so slices 2q and
# 2q+1 share a quad. adj(S) = ADJ_IN(S) + ADJ_X(S).
ADJ_IN = np.array([sum(1 for k in S if k % 2 == 0 and k + 1 in S) for S in PATS], dtype=float)
ADJ_X = np.array([sum(1 for k in S if k % 2 == 1 and k + 1 in S) for S in PATS], dtype=float)


def fit_lbfgs(counts, gamma_free=True):
    """Independent cross-check of fit_newton with scipy's L-BFGS-B on the same exact likelihood."""
    X = _X1 if gamma_free else _X0
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    T = X.T @ counts

    def f(theta):
        _, logZ, mu, _ = _moments(X, theta)
        return -(theta @ T - N * logZ) / N, -(T - N * mu) / N

    r = minimize(f, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                 options={'gtol': 1e-12, 'ftol': 1e-15, 'maxiter': 10000, 'maxcor': 30})
    beta, gamma = _unpack(r.x, gamma_free)
    return {'beta': beta, 'gamma': gamma, 'll': float(-r.fun * N),
            'grad_norm': float(np.linalg.norm(r.jac)), 'success': bool(r.success),
            'nit': int(r.nit)}


def fit_fixed_step(counts, gamma_free=True, iters=4000, lr=0.05):
    """The hand-written loop the paper's numbers originally came from, kept for comparison."""
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    n_k, s_adj = A.T @ counts, ADJ @ counts
    beta, gamma = np.zeros(NP), 0.0
    for _ in range(iters):
        p = probs(beta, gamma)
        beta = beta + lr * (n_k / N - A.T @ p)
        if gamma_free:
            gamma += lr * (s_adj / N - ADJ @ p)
        beta -= beta.mean()
    p = probs(beta, gamma)
    X = _X1 if gamma_free else _X0
    theta = np.r_[beta[:NP - 1], gamma] if gamma_free else beta[:NP - 1]
    g = X.T @ counts - N * (X.T @ p)
    return {'beta': beta, 'gamma': gamma, 'll': loglik(counts, beta, gamma),
            'grad_norm': float(np.linalg.norm(g) / N), 'theta': theta}


def se_gamma_conditional(counts, beta, gamma):
    """The i.i.d. SE previously printed by yield_model.py: 1/sqrt(N Var(adj)), which treats beta
    as known. It ignores the beta-gamma covariance and so understates the SE."""
    p = probs(beta, gamma)
    var = (ADJ ** 2) @ p - (ADJ @ p) ** 2
    return 1.0 / np.sqrt(counts.sum() * var)


def pearson_discrepancy(cnt, p):
    e = cnt.sum() * p
    return float(((cnt - e) ** 2 / e).sum())


def replicate_checks(counts, B, rng):
    """Parametric model-replicate checks of the pairwise model, plug-in and refitted.

    For b = 1..B: draw pattern counts c* ~ Multinomial(N, p_hat) (and c0* from the gamma = 0 fit),
    refit, and record the Pearson discrepancy against the refit and against p_hat, the number of
    distinct patterns, the mean largest gap, and the 66 co-harvest residuals O* - E(theta*).
    The observed residual r = O - E(theta_hat) is then standardised by the replicate distribution
    of its own cell: z_cal = (r - mean r*) / sd r*.
    """
    counts = np.asarray(counts, dtype=float)
    N = int(counts.sum())
    f1, f0 = fit_newton(counts), fit_newton(counts, gamma_free=False)
    assert f1['grad_norm'] < 1e-8 and f0['grad_norm'] < 1e-8
    p, p0 = f1['p'], f0['p']
    O = PAIRMAT.T @ counts
    E = N * (PAIRMAT.T @ p)
    rep = {k: np.empty(B) for k in ('d_plug', 'd_plug0', 'd_ref', 'd_ref0', 'u', 'mg')}
    r_ref, r_plug = np.empty((B, NPAIR)), np.empty((B, NPAIR))
    for b in range(B):
        c = rng.multinomial(N, p).astype(float)
        c0 = rng.multinomial(N, p0).astype(float)
        fr = fit_newton(c, theta0=f1['theta'])
        fr0 = fit_newton(c0, gamma_free=False, theta0=f0['theta'])
        assert fr['grad_norm'] < 1e-8 and fr0['grad_norm'] < 1e-8
        rep['d_plug'][b], rep['d_ref'][b] = pearson_discrepancy(c, p), pearson_discrepancy(c, fr['p'])
        rep['d_plug0'][b] = pearson_discrepancy(c0, p0)
        rep['d_ref0'][b] = pearson_discrepancy(c0, fr0['p'])
        rep['u'][b] = (c > 0).sum()
        rep['mg'][b] = MAXGAP @ c / N
        Oc = PAIRMAT.T @ c
        r_ref[b] = Oc - N * (PAIRMAT.T @ fr['p'])
        r_plug[b] = Oc - E
    r_obs = O - E
    mu_r, sd_r = r_ref.mean(0), r_ref.std(0, ddof=1)
    z_cal = (r_obs - mu_r) / sd_r
    z_rep = (r_ref - mu_r) / sd_r
    return {'fit': f1, 'fit0': f0, 'N': N, 'O': O, 'E': E, 'r_obs': r_obs, 'rep': rep,
            'r_ref': r_ref, 'r_plug': r_plug, 'mu_r': mu_r, 'sd_r': sd_r, 'z_cal': z_cal,
            'z_rep': z_rep, 'pearson': r_obs / np.sqrt(E),
            'd_obs': pearson_discrepancy(counts, p), 'd_obs0': pearson_discrepancy(counts, p0),
            'u_obs': int((counts > 0).sum()), 'mg_obs': float(MAXGAP @ counts / N)}


def cluster_scores(idx, clusters, fit):
    """Per-cluster score vectors sum_{s in g} (x_s - E[x]) in the identified parameterisation."""
    X = _X1
    labs = np.unique(clusters)
    S = np.zeros((len(labs), X.shape[1]))
    for j, g in enumerate(labs):
        rows = idx[clusters == g]
        S[j] = X[rows].sum(0) - len(rows) * fit['mu']
    return labs, S


# ---------------------------------------------------------------- curveball samplers
def curveball_seq(mat, n_trades, rng):
    """Sequential curveball trade (Strona et al. 2014): pick two rows, pool the columns each has
    and the other lacks, re-deal the pool at random; columns the two rows share stay in BOTH."""
    rows = [set(np.flatnonzero(r)) for r in mat]
    R = len(rows)
    for _ in range(n_trades):
        i, j = rng.integers(0, R, 2)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        ab, ba = a - b, b - a
        if not ab or not ba:
            continue
        pool = list(ab | ba)
        rng.shuffle(pool)
        k = len(ab)
        common = a & b
        rows[i], rows[j] = common | set(pool[:k]), common | set(pool[k:])
    out = np.zeros_like(mat)
    for r, s in enumerate(rows):
        out[r, list(s)] = 1
    return out


def curveball_seq_legacy(mat, n_trades, rng, stats=None):
    """The trade as implemented in binning_stats.py, full_record_structure.py, sensitivity.py,
    make_figures.py and make_figures4.py up to commit cfd73dd, kept ONLY to reproduce and diagnose
    the published numbers. Its `nb = (a | b) - na` drops the columns the two rows share from row b,
    so the size guard rejects every trade between rows that share a harvested slice: only trades
    between disjoint rows are executed. The chain is still symmetric, but it moves far more slowly
    and is not guaranteed to be irreducible (tests/test_model.py has a margin set on which it
    cannot move at all). `stats`, if a dict, receives attempted/executed trade counts."""
    rows = [set(np.flatnonzero(r)) for r in mat]
    R = len(rows)
    n_try = n_exec = 0
    for _ in range(n_trades):
        i, j = rng.integers(0, R, 2)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        ab, ba = a - b, b - a
        if not ab or not ba:
            continue
        n_try += 1
        pool = list(ab | ba)
        rng.shuffle(pool)
        na = (a & b) | set(pool[:len(a) - len(a & b)])
        nb = (a | b) - na
        if len(na) == len(a) and len(nb) == len(b):
            rows[i], rows[j] = na, nb
            n_exec += 1
    if stats is not None:
        stats['attempted'] = stats.get('attempted', 0) + n_try
        stats['executed'] = stats.get('executed', 0) + n_exec
    out = np.zeros_like(mat)
    for r, s in enumerate(rows):
        out[r, list(s)] = 1
    return out


def _strata_groups(n, strata):
    if strata is None:
        return [np.arange(n)]
    strata = np.asarray(strata)
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]


def curveball_global(mat, n_global, rng, strata=None):
    """Vectorised global curveball trades on a boolean matrix (copy returned).

    Each global trade draws a uniformly random perfect matching of the rows (within each stratum,
    if given, so each stratum keeps its own column sums) and performs one curveball trade on every
    matched pair: the two rows' disjoint columns are pooled and re-dealt uniformly at random.
    """
    X = np.array(mat, dtype=bool, copy=True)
    groups = _strata_groups(len(X), strata)
    C = X.shape[1]
    for _ in range(n_global):
        I_list, J_list = [], []
        for g in groups:
            perm = g[rng.permutation(len(g))]
            P = len(g) // 2
            I_list.append(perm[:2 * P:2])
            J_list.append(perm[1:2 * P:2])
        I, J = np.concatenate(I_list), np.concatenate(J_list)
        Ar, Br = X[I], X[J]
        diff = Ar ^ Br
        a_only = (Ar & ~Br).sum(1)
        keys = rng.random((len(I), C))
        keys[~diff] = 2.0                                     # non-pool columns sort last
        order = np.argsort(keys, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(C), order.shape), axis=1)
        chosen = diff & (ranks < a_only[:, None])
        common = Ar & Br
        X[I] = common | chosen
        X[J] = common | (diff & ~chosen)
    return X


def mc_pvalue(null_values, observed, tail='upper'):
    """Finite Monte-Carlo p-value (b + 1) / (B + 1) (Besag & Clifford 1989 convention)."""
    v = np.asarray(null_values)
    if tail == 'upper':
        b = int((v >= observed).sum())
    elif tail == 'lower':
        b = int((v <= observed).sum())
    else:
        raise ValueError(tail)
    return (b + 1) / (len(v) + 1)


def gale_ryser_start(row_sums, col_sums):
    """A deterministic, maximally structured 0/1 matrix with the given margins (greedy fill of
    each row into the columns with the largest remaining sums). Used as an alternative chain
    start for the mixing diagnostic."""
    rem = np.array(col_sums, dtype=int).copy()
    out = np.zeros((len(row_sums), len(col_sums)), dtype=bool)
    for r in np.argsort(-np.asarray(row_sums), kind='stable'):
        cols = np.argsort(-rem, kind='stable')[:row_sums[r]]
        if (rem[cols] <= 0).any():
            raise ValueError('margins not realisable greedily')
        out[r, cols] = True
        rem[cols] -= 1
    if rem.any():
        raise ValueError('margins not realisable greedily')
    return out
