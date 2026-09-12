"""Shared estimators for the socket-level thermal analysis (Sec. IV-E/F).

Everything here is a pure function of arrays, so it is unit-tested (tests/test_thermal.py) and
used by both the slow path that reads the raw 20 s series (thermal_socket_build.py) and the fast
path that runs from the committed per-socket summaries (thermal_socket_stats.py).

Conventions
-----------
Core c belongs to slice c // 2 and to quad c // 4. POWER9 arranges its SMT4 cores in groups of
four -- a Quad -- whose voltage and frequency "must track in unison" (POWER9 Processor User's
Manual, OpenPOWER v2.0, Sec. 23.4); each pair of cores in a quad shares one L2/L3 (a slice).
Three kinds of index-adjacent core pair therefore exist, and they are kept apart throughout:

    within      (2k, 2k+1)      same slice, same quad
    samequad    (4q+1, 4q+2)    different slices, same quad
    crossquad   (4q+3, 4q+4)    different slices, different quads

The floorplan score of Sec. IV-F is the Pearson correlation, over the 66 unordered slice pairs
with equal weight, between the distance of two slices under a candidate layout and their
thermal similarity (the mean of the four cross-slice core-pair correlations).
"""
from itertools import combinations

import numpy as np

NC, NP, NQ = 24, 12, 6
SLICE_PAIRS = list(combinations(range(NP), 2))          # the 66 cells, in a fixed order
IU16 = np.triu_indices(16, 1)                            # storage order of a socket's 120 pairs

# IBM's documented chip layout: POWER9 Processor Registers Specification, Vol. 1, v1.2 (29 Mar
# 2021), Fig. 2 "POWER9 Processor from a Pervasive Point of View", p. 33. Core chiplets EC00..EC23
# are the sensor index. Quad q (cores 4q..4q+3, slices 2q, 2q+1) sits in column q // 2 (left,
# middle, right) and band q % 2 (0 = top, 1 = bottom); its four cores are stacked in one column,
# EX0 (slice 2q) at the outer die edge and EX1 (slice 2q+1) next to the central nest band. The
# coordinates are the centroids of the 24 core boxes segmented from the figure's embedded
# 1227 x 891 px image (units: px). The diagram is a chip diagram, not a to-scale die photograph:
# only its topology (quads, stacking, bands, column order) is a documented fact; its proportions
# are an assumption, varied in thermal_ibm_layout.py.
IBM_COL_PX = (178.0, 664.0, 1047.0)
IBM_TOP_PX = (98.0, 158.0, 218.0, 278.0)        # EC 4q+0 .. 4q+3, q even (outer edge first)
IBM_BOT_PX = (710.0, 650.0, 590.0, 530.0)       # EC 4q+0 .. 4q+3, q odd  (outer edge first)


def ibm_core_xy(col_scale=1.0, gap_scale=1.0, pitch_scale=1.0):
    """24 x 2 core coordinates of IBM's diagram, in units of the left-to-middle column spacing.
    The three scale factors stretch the column spacing, the gap between the two bands, and the
    core pitch inside a quad relative to the diagram's own proportions (1 = as drawn)."""
    pitch = (IBM_TOP_PX[1] - IBM_TOP_PX[0]) * pitch_scale
    gap = (IBM_BOT_PX[3] - IBM_TOP_PX[3]) * gap_scale
    xs = [0.0, (IBM_COL_PX[1] - IBM_COL_PX[0]) * col_scale,
          (IBM_COL_PX[2] - IBM_COL_PX[0]) * col_scale]
    xy = np.empty((NC, 2))
    for c in range(NC):
        q, j = c // 4, c % 4
        y = j * pitch if q % 2 == 0 else 3 * pitch + gap + (3 - j) * pitch
        xy[c] = (xs[q // 2], y)
    return xy / (IBM_COL_PX[1] - IBM_COL_PX[0])


def ibm_slice_xy(**kw):
    """12 x 2 slice centres (mean of the slice's two cores) of IBM's diagram."""
    return ibm_core_xy(**kw).reshape(NP, 2, 2).mean(1)


def ibm_topology():
    """Per slice: (column 0-2, band 0 top / 1 bottom, EX 0 outer edge / 1 next to the nest)."""
    q = np.arange(NP) // 2
    return np.stack([q // 2, q % 2, np.arange(NP) % 2], 1)


# Candidate slice layouts scored in Sec. IV-F / Fig. 6 (unit spacing, Euclidean distance).
# '2 x 6 mirrored halves' has the same coordinates as '2 x 6 boustrophedon'; it is kept here so the
# committed thermal_socket_stats*.json keys stay stable, and layout_hypotheses.py draws it once.
CANDIDATES = {
    '1 x 12 linear':         [(k, 0) for k in range(NP)],
    '2 x 6 row-major':       [(k % 6, k // 6) for k in range(NP)],
    '2 x 6 boustrophedon':   [(k % 6 if k < 6 else 5 - (k - 6), k // 6) for k in range(NP)],
    '3 x 4 row-major':       [(k % 4, k // 4) for k in range(NP)],
    '4 x 3 row-major':       [(k % 3, k // 3) for k in range(NP)],
    '6 x 2 row-major':       [(k % 2, k // 2) for k in range(NP)],
    '2 x 6 mirrored halves': [(k, 0) if k < 6 else (11 - k, 1) for k in range(NP)],
}


# ----------------------------------------------------------------------------- per socket
def residual_corr(X):
    """Residual correlation of the columns of X after removing the per-row (per-timestamp)
    cross-core mean and double-centring. Identical to the paper's thermal_all_nodes.py.
    Returns (C, sd) where sd is the per-column residual standard deviation (ddof=0)."""
    R = np.asarray(X, dtype='float64')
    R = R - R.mean(1, keepdims=True)
    R = R - R.mean(0, keepdims=True)
    sd = R.std(0)
    return (R.T @ R) / len(R) / np.outer(sd, sd), sd


def pair_kind(i, j):
    """'within' | 'samequad' | 'crossquad' for index-adjacent cores, else None."""
    i, j = min(i, j), max(i, j)
    if j - i != 1:
        return None
    if i // 2 == j // 2:
        return 'within'
    return 'samequad' if i // 4 == j // 4 else 'crossquad'


def embed24(C16, idx):
    """Place a socket's 16x16 matrix into a 24x24 NaN matrix at its active positions."""
    M = np.full((NC, NC), np.nan)
    M[np.ix_(idx, idx)] = C16
    return M


def unpack(upper, idx):
    """Inverse of the storage format: 120 upper-triangle values + 16 active indices -> 24x24."""
    C = np.eye(16)
    C[IU16] = upper
    C[IU16[1], IU16[0]] = upper
    return embed24(C, idx)


def slice_similarity(C24):
    """12x12 slice similarity: mean of the four cross-slice core-pair correlations. NaN where
    either slice is absent (all four core pairs are then absent too). Diagonal is NaN."""
    B = C24.reshape(NP, 2, NP, 2)
    S = B.mean(axis=(1, 3))
    np.fill_diagonal(S, np.nan)
    return S


def cells(S):
    """The 66 upper-triangle slice cells of a 12x12 matrix, in SLICE_PAIRS order."""
    a, b = np.array(SLICE_PAIRS).T
    return S[a, b]


# ----------------------------------------------------------------------------- pooling
def pool(mats, weights=None, fisher=False):
    """Cell-wise pooled mean of a stack of 24x24 (or 12x12) matrices with NaN for unobserved
    cells. Returns (mean, support) where support is the number of matrices observing the cell.
    fisher=True averages atanh(r) and back-transforms."""
    M = np.asarray(mats, dtype='float64')
    obs = ~np.isnan(M)
    w = np.ones(len(M)) if weights is None else np.asarray(weights, dtype='float64')
    V = np.where(obs, M, 0.0)
    if fisher:
        V = np.where(obs, np.arctanh(np.clip(V, -0.999999, 0.999999)), 0.0)
    num = np.tensordot(w, V, axes=1)
    den = np.tensordot(w, obs.astype('float64'), axes=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    if fisher:
        out = np.tanh(out)
    return out, obs.sum(0)


def poststratified_pool(mats, strata, fisher=False):
    """Pool each cell so that every stratum carries its fleet share, whatever the cell's own
    composition. Cell estimate = sum_h pi_h * mean_h(cell), pi_h = share of sockets in stratum h,
    renormalised over the strata that observe the cell."""
    strata = np.asarray(strata)
    labs = np.unique(strata)
    share = np.array([(strata == h).mean() for h in labs])
    means, supp = [], []
    for h in labs:
        m, s = pool(np.asarray(mats)[strata == h], fisher=fisher)
        means.append(m); supp.append(s)
    means = np.array(means); supp = np.array(supp)
    ok = supp > 0
    w = share[:, None, None] * ok
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.nansum(np.where(ok, means, 0.0) * w, axis=0) / w.sum(0)
    return out


def twoway_cells(Y, max_iter=500, tol=1e-12):
    """Least-squares fit of y_{s,c} = u_s + theta_c on an incomplete socket x cell table (NaN =
    unobserved), by alternating means. theta is identified up to a constant; we return theta
    shifted to the mean of the pooled cell means so it sits on the correlation scale. Removing a
    socket offset guards against sockets of different overall coupling level populating
    different cells."""
    Y = np.asarray(Y, dtype='float64')
    obs = ~np.isnan(Y)
    Z = np.where(obs, Y, 0.0)
    u = np.zeros(Y.shape[0]); th = np.zeros(Y.shape[1])
    nc = obs.sum(0); ns = obs.sum(1)
    for _ in range(max_iter):
        th_new = ((Z - u[:, None]) * obs).sum(0) / np.maximum(nc, 1)
        u_new = ((Z - th_new[None, :]) * obs).sum(1) / np.maximum(ns, 1)
        done = max(np.abs(th_new - th).max(), np.abs(u_new - u).max()) < tol
        th, u = th_new, u_new
        if done:
            break
    raw = np.nanmean(np.where(obs, Y, np.nan), axis=0)
    return th - np.nanmean(th) + np.nanmean(raw)


# ----------------------------------------------------------------------------- orderings
def order_positions(order):
    """order[t] = slice at position t  ->  pos[slice] = t."""
    order = np.asarray(order)
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))
    return pos


def order_score(cellvals, order, pairs=SLICE_PAIRS):
    """Pearson corr(|pos_a - pos_b|, similarity) over the given pairs (NaN cells skipped)."""
    pos = order_positions(order)
    d = np.array([abs(pos[a] - pos[b]) for a, b in pairs], dtype='float64')
    v = np.asarray(cellvals, dtype='float64')
    m = ~np.isnan(v)
    return float(np.corrcoef(d[m], v[m])[0, 1])


def coords_score(cellvals, coords, pairs=SLICE_PAIRS):
    """Same score for an arbitrary 2-D candidate layout (Euclidean distance)."""
    c = np.asarray(coords, dtype='float64')
    d = np.array([np.linalg.norm(c[a] - c[b]) for a, b in pairs])
    v = np.asarray(cellvals, dtype='float64')
    m = ~np.isnan(v)
    return float(np.corrcoef(d[m], v[m])[0, 1])


def _cut_table(W):
    """cut[S] = sum_{a in S, b not in S} W[a,b] for every subset S of range(n) (bitmask)."""
    n = len(W)
    rs = W.sum(1) - np.diag(W)
    cut = np.zeros(1 << n)
    for S in range(1, 1 << n):
        v = (S & -S).bit_length() - 1          # lowest set bit
        T = S & ~(1 << v)
        inT = [(T >> a) & 1 for a in range(n)]
        cut[S] = cut[T] + rs[v] - 2.0 * sum(W[a, v] for a in range(n) if inT[a])
    return cut


def exact_linear_arrangement(W):
    """Exact minimum of sum_{a<b} W[a,b] * |pos_a - pos_b| over all n! orderings.

    |pos_a - pos_b| counts the gaps t (1..n-1) that separate a and b, so the objective equals the
    sum over t of the cut between the first t slices and the rest. Dynamic programming over the
    2^n prefix sets is therefore exact: f(S) = cut(S) + min_{v in S} f(S minus v). O(2^n n).
    Returns (cost, order) with order[t] the slice at position t."""
    W = np.asarray(W, dtype='float64')
    n = len(W)
    full = (1 << n) - 1
    cut = _cut_table(W)
    f = np.full(1 << n, np.inf); f[0] = 0.0
    arg = np.full(1 << n, -1, dtype=int)
    for S in range(1, 1 << n):
        best, bv = np.inf, -1
        T = S
        while T:
            v = (T & -T).bit_length() - 1
            T &= T - 1
            c = f[S & ~(1 << v)]
            if c < best:
                best, bv = c, v
        f[S] = best + (cut[S] if S != full else 0.0)
        arg[S] = bv
    order, S = [], full
    while S:
        v = arg[S]; order.append(v); S &= ~(1 << v)
    return float(f[full]), np.array(order[::-1])


def kbest_linear_arrangements(W, K):
    """The K smallest values of sum_{a<b} W[a,b]|pos_a - pos_b| over all n! orderings (each
    ordering and its reflection counted separately), with the orderings. Same DP as above,
    keeping the K best partial costs per prefix set."""
    W = np.asarray(W, dtype='float64')
    n = len(W)
    full = (1 << n) - 1
    cut = _cut_table(W)
    cost = [None] * (1 << n); pv = [None] * (1 << n); pr = [None] * (1 << n)
    cost[0] = np.zeros(1); pv[0] = np.full(1, -1); pr[0] = np.full(1, -1)
    for S in range(1, 1 << n):
        cs, vs, rs = [], [], []
        T = S
        while T:
            v = (T & -T).bit_length() - 1
            T &= T - 1
            P = S & ~(1 << v)
            cs.append(cost[P]); vs.append(np.full(len(cost[P]), v))
            rs.append(np.arange(len(cost[P])))
        c = np.concatenate(cs); v = np.concatenate(vs); r = np.concatenate(rs)
        k = min(K, len(c))
        sel = np.argpartition(c, k - 1)[:k] if k < len(c) else np.arange(len(c))
        sel = sel[np.argsort(c[sel], kind='stable')]
        add = cut[S] if S != full else 0.0
        cost[S] = c[sel] + add; pv[S] = v[sel]; pr[S] = r[sel]
    out = []
    for i in range(len(cost[full])):
        order, S, rank = [], full, i
        while S:
            v = pv[S][rank]; nr = pr[S][rank]
            order.append(v); S &= ~(1 << v); rank = nr
        out.append((float(cost[full][i]), np.array(order[::-1])))
    return out


def arrangement_cost(W, order):
    pos = order_positions(order)
    n = len(W)
    return float(sum(W[a, b] * abs(pos[a] - pos[b]) for a, b in combinations(range(n), 2)))


def best_order(Ssym):
    """Exact most-negative floorplan score over all 1-D orderings of the slices of a symmetric
    similarity matrix. For a 1-D layout the multiset of the 66 pairwise distances is the same
    for every ordering, so Pearson corr(distance, similarity) is an increasing affine function
    of sum_{a<b} S_ab |pos_a - pos_b|, and minimising the score is a weighted minimum linear
    arrangement, solved exactly by exact_linear_arrangement. Returns (score, order)."""
    S = np.array(Ssym, dtype='float64')
    np.fill_diagonal(S, 0.0)
    _, order = exact_linear_arrangement(S)
    cv = np.array([S[a, b] for a, b in combinations(range(len(S)), 2)])
    return order_score(cv, order, list(combinations(range(len(S)), 2))), order


def canonical(order):
    """An ordering and its reflection are the same layout; return the one starting smaller."""
    order = np.asarray(order)
    return order if order[0] <= order[-1] else order[::-1]


def kendall_distance_reflect(order, ref=None):
    """Number of discordant slice pairs between an ordering and the identity (or ref), taking
    the smaller of the ordering and its reflection. 0 = identical layout."""
    n = len(order)
    ref = np.arange(n) if ref is None else np.asarray(ref)
    def kd(o):
        p = order_positions(o); q = order_positions(ref)
        return sum((p[a] - p[b]) * (q[a] - q[b]) < 0 for a, b in combinations(range(n), 2))
    return int(min(kd(np.asarray(order)), kd(np.asarray(order)[::-1])))


def spearman_abs(order):
    """|Spearman rho| between an ordering's positions and the identity."""
    pos = order_positions(order)
    return float(abs(np.corrcoef(pos, np.arange(len(pos)))[0, 1]))


# ----------------------------------------------------------------------------- inference
def cluster_resample(clusters, rng):
    """Indices of units for one cluster bootstrap replicate (clusters drawn with replacement,
    every unit of a drawn cluster kept)."""
    clusters = np.asarray(clusters)
    labs, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(labs))]
    draw = rng.integers(0, len(labs), len(labs))
    return np.concatenate([members[k] for k in draw])


def mc_pvalue(null, obs, side='less'):
    """Finite Monte-Carlo p-value (b+1)/(B+1)."""
    null = np.asarray(null)
    b = (null <= obs).sum() if side == 'less' else (null >= obs).sum()
    return float((b + 1) / (len(null) + 1))


def sign_flip_test(x, rng, B=99999):
    """One-sided randomisation test of mean(x) > 0 by flipping the sign of each cluster mean.
    Returns (p, observed mean) with p = (b+1)/(B+1)."""
    x = np.asarray(x, dtype='float64')
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(B, len(x)))
    null = (signs * x).mean(1)
    return float(((null >= obs).sum() + 1) / (B + 1)), float(obs)


def ci(v, level=95):
    lo = (100 - level) / 2
    return float(np.percentile(v, lo)), float(np.percentile(v, 100 - lo))


# ----------------------------------------------------------------------------- projection
def projected_corr(Sigma, active):
    """Population residual correlation among the active cores after the analysis' common-mode
    removal, for series with covariance Sigma: P Sigma_AA P normalised, P = I - 11'/k. Used to
    show what the common-mode step can and cannot manufacture."""
    A = np.asarray(active)
    k = len(A)
    P = np.eye(k) - np.ones((k, k)) / k
    R = P @ np.asarray(Sigma)[np.ix_(A, A)] @ P
    sd = np.sqrt(np.diag(R))
    return R / np.outer(sd, sd)


def heat_source_corr(active, core_xy, lam, noise, power_cov=None):
    """Population residual correlation among the active cores under a PHYSICAL source model:
    every active core dissipates power P_j(t) (independent, unit variance, unless power_cov is
    given as a 16 x 16 covariance over the active cores), harvested cores are dark (no source),
    and sensor i reads T_i = sum_j exp(-d_ij / lam) P_j + noise. Pushed through the paper's
    common-mode step. Unlike a stationary covariance kernel sampled at the active sensors, this
    lets a harvested neighbour change the coupling of the cores around it."""
    A = np.asarray(active)
    D = np.linalg.norm(core_xy[A][:, None] - core_xy[A][None], axis=-1)
    G = np.exp(-D / lam)
    Cp = np.eye(len(A)) if power_cov is None else np.asarray(power_cov)
    Sigma = G @ Cp @ G.T + noise * np.eye(len(A))
    return projected_corr(Sigma, np.arange(len(A)))


def within_ols(y, X, groups, fe=()):
    """OLS of y on X after removing group means (socket fixed effects), optionally also a second
    set of fixed effects (e.g. slice pair) by alternating projections. Returns the coefficients."""
    Z = np.column_stack([np.asarray(X, dtype='float64'), np.asarray(y, dtype='float64')])
    sets = [np.asarray(groups)] + [np.asarray(f) for f in fe]
    for _ in range(1 if len(sets) == 1 else 60):
        for g in sets:
            _, inv = np.unique(g, return_inverse=True)
            cnt = np.bincount(inv)
            for k in range(Z.shape[1]):
                Z[:, k] -= (np.bincount(inv, Z[:, k]) / cnt)[inv]
    return np.linalg.lstsq(Z[:, :-1], Z[:, -1], rcond=None)[0]


def column_contrast(S, active_slices):
    """Geometry-versus-order test inside each die column of IBM's layout (Registers Spec v1.2,
    Fig. 2). Column c holds quad 2c (top band) and quad 2c+1 (bottom band). Slice 4c+1 (top EX1)
    sits next to the central nest band, as does slice 4c+3 (bottom EX1); slice 4c+2 (bottom EX0)
    sits at the far bottom edge. On sockets where all three are active, the index/rank order puts
    4c+2 next to 4c+1 (distance 1) and 4c+3 one step further (distance 2); the diagram puts 4c+3
    nearer. Returns the per-socket mean over columns of S[4c+1, 4c+3] - S[4c+1, 4c+2] (NaN where no
    column qualifies): positive if coupling follows the diagram, negative if it follows the order.
    S: (nsock, 12, 12) slice similarities; active_slices: (nsock, 12) bool."""
    S = np.asarray(S); SL = np.asarray(active_slices, dtype=bool)
    out = np.full((len(S), 3), np.nan)
    for c in range(3):
        t1, b0, b1 = 4 * c + 1, 4 * c + 2, 4 * c + 3
        ok = SL[:, t1] & SL[:, b0] & SL[:, b1]
        out[ok, c] = S[ok, t1, b1] - S[ok, t1, b0]
    n = np.isfinite(out).sum(1)
    tot = np.where(np.isfinite(out), out, 0.0).sum(1)
    return np.where(n > 0, tot / np.maximum(n, 1), np.nan), out


def natural_experiment(S, active_slices, ix=None):
    """Position-matched harvested-between contrast. For each slice pair (a, a+2) -- identical
    index distance and identical die positions, always in different quads -- compare the coupling
    on sockets where slice a+1 is harvested with sockets where it is active, and average the
    differences over a with weight min(n_harvested, n_active). A coupling that follows the order
    in which the surviving cores are numbered raises it (the pair becomes rank-adjacent), but so
    does a physical source model in which the harvested slice is dark (heat_source_corr): removing
    an independent heat source next to one core raises that core's correlation with distant
    cores. The statistic alone therefore does not separate physical from logical ordering.
    S: (nsock, 12, 12) slice similarities (NaN where absent); active_slices: (nsock, 12) bool."""
    S = np.asarray(S); SL = np.asarray(active_slices, dtype=bool)
    ix = np.arange(len(S)) if ix is None else np.asarray(ix)
    d, w = [], []
    for a in range(S.shape[1] - 2):
        both = SL[ix, a] & SL[ix, a + 2]
        h = both & ~SL[ix, a + 1]
        o = both & SL[ix, a + 1]
        if h.sum() == 0 or o.sum() == 0:
            continue
        d.append(np.nanmean(S[ix][h, a, a + 2]) - np.nanmean(S[ix][o, a, a + 2]))
        w.append(min(h.sum(), o.sum()))
    return float(np.average(d, weights=w))
