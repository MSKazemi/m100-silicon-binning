"""Deletion models for the sensor-presence side channel (pure functions; tested in
tests/test_side_channel.py).

A harvested core never emits, so the channel has no false positives: the map is recovered
exactly when every ACTIVE core has at least one sample surviving into the attacker's window.
Treating the active cores as deleted independently of one another,

    P(exact) = prod_i (1 - P(core i silent)),

and the deletion models below differ only in P(core i silent) for a core with n_i samples:

  iid thinning   every sample deleted independently with probability f      f ** n_i
  Poisson        surviving count ~ Poisson(mean lam), independent per core  exp(-lam)
                 (the large-n limit of iid thinning with lam = n_i (1 - f))
  decimation     the publisher keeps a fixed set of K of the S slots; the
                 n_i samples occupy uniformly random distinct slots          C(S-K, n_i) / C(S, n_i)
                 (hypergeometric; K = 1 is a single sampling interval)
  bursty         per-core two-state Markov (Gilbert) deletion over the core's
                 sample sequence, stationary deletion fraction f, mean length
                 L of a run of KEPT samples                                  f * (1 - (1-f)/(f L)) ** (n_i - 1)
                 (L = 1/f reproduces iid thinning; a longer L clumps the
                 survivors, so fewer independent chances to be seen)

Deletion that is correlated ACROSS cores (dropping whole timestamps) violates the independence
in the product and is simulated on raw timestamps instead (side_channel_raw.py), where
timestamp_deletion_bounds() brackets the simulation analytically.

The same formulas apply per SLICE instead of per core. Every map here is a union of whole
slices (cores 2j, 2j+1), so an attacker who knows that rule needs one surviving sample per
active slice, not per core: pass slice_counts(counts) instead of counts. The per-core numbers
are therefore the conservative (defender-friendly) ones. slices_revealed_iid() measures the
partial disclosure left when exact recovery fails.

The previous version of side_channel.py fed a fractional count n (1 - f) into the decimation
formula with K = S - 1. That is not a deletion model: with K = S - 1 a core is silent only if
all of its samples sit in the one excluded slot, impossible for more than one sample, so it
reported recovery 1.000 at 99.9 % deletion. The correct iid value is ~0.81 for one day.
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

SLOTS = 4320            # 20 s slots per day


# ------------------------------------------------------------------ P(core silent)
def p_silent_iid(n, f):
    """Every sample deleted independently with probability f: P(all n deleted) = f**n."""
    n = np.asarray(n, dtype=float)
    if f <= 0.0:
        return np.where(n > 0, 0.0, 1.0)
    if f >= 1.0:
        return np.ones_like(n)
    return np.exp(n * np.log(f))


def p_silent_poisson(lam):
    """Surviving count Poisson with mean lam: P(count = 0) = exp(-lam)."""
    return np.exp(-np.asarray(lam, dtype=float))


def p_silent_decimation(n, k_kept, slots=SLOTS):
    """n samples in n distinct uniformly random slots out of `slots`; the publisher keeps a
    fixed set of `k_kept` slots. P(none of the core's samples is kept) =
    C(slots - k_kept, n) / C(slots, n). n must be an integer count (0 <= n <= slots)."""
    n = np.asarray(n)
    if not np.all(np.equal(np.mod(n, 1), 0)):
        raise ValueError('decimation model needs integer sample counts')
    n = n.astype(float)
    if np.any((n < 0) | (n > slots)):
        raise ValueError('counts must lie in [0, slots]')
    K = int(k_kept)
    if K <= 0:
        return np.ones_like(n)
    if K >= slots:
        return np.where(n > 0, 0.0, 1.0)
    ok = n <= slots - K
    nn = np.where(ok, n, 0.0)
    logp = (gammaln(slots - K + 1) - gammaln(slots - K - nn + 1)
            - gammaln(slots + 1) + gammaln(slots - nn + 1))
    return np.where(ok, np.exp(logp), 0.0)


def gilbert_a(f, L):
    """Recovery (delete -> keep) transition probability of the two-state chain whose
    stationary deletion fraction is f and whose mean kept-run length is L."""
    a = (1.0 - f) / (f * L)
    if not (0.0 < a <= 1.0) or L < 1.0:
        raise ValueError(f'infeasible bursty model: f={f}, L={L} (need L >= max(1, (1-f)/f))')
    return a


def p_silent_bursty(n, f, L):
    """Two-state Markov deletion started in its stationary state: P(all n samples deleted)
    = f * (1 - a) ** (n - 1) with a = (1 - f) / (f L)."""
    n = np.asarray(n, dtype=float)
    if f >= 1.0:
        return np.ones_like(n)
    a = gilbert_a(f, L)
    out = f * np.exp((n - 1.0) * np.log1p(-a)) if a < 1.0 else np.where(n <= 1, f, 0.0)
    return np.where(n > 0, out, 1.0)


# ------------------------------------------------------------------ exact recovery
def p_exact(p_silent):
    """P(no active core silent) per socket, from an array (..., cores) of P(core silent).
    Computed in log space so that values like 5e-8 are not lost to rounding."""
    p = np.clip(np.asarray(p_silent, dtype=float), 0.0, 1.0)
    with np.errstate(divide='ignore'):
        return np.exp(np.log1p(-p).sum(axis=-1))


def poisson_curve(lam, cores=16):
    """(1 - exp(-lam)) ** cores: exact recovery when each of `cores` active cores has an
    independent Poisson(mean lam) number of surviving samples."""
    return (1.0 - np.exp(-np.asarray(lam, dtype=float))) ** cores


def mc_iid_thinning(counts, f, reps, rng):
    """Monte Carlo of iid thinning on integer counts (sockets x cores): binomially thin every
    core's count, recover the map as the cores with >= 1 survivor, return the fraction of
    (socket, replicate) draws in which all active cores survive, and its binomial SE."""
    counts = np.asarray(counts, dtype=np.int64)
    hits = 0
    for _ in range(reps):
        kept = rng.binomial(counts, 1.0 - f)
        hits += int((kept > 0).all(axis=-1).sum())
    tot = reps * counts.shape[0]
    p = hits / tot
    return p, float(np.sqrt(p * (1 - p) / tot))


def simulate_gilbert(n, f, L, reps, rng):
    """Brute-force simulation of the two-state deletion chain over n samples, `reps` times;
    returns a (reps, n) boolean array, True = sample kept. Used only to validate
    p_silent_bursty; the analysis itself uses the closed form."""
    a = gilbert_a(f, L)
    b = 1.0 / L
    kept = np.empty((reps, n), dtype=bool)
    state = rng.random(reps) >= f                     # stationary start: kept w.p. 1 - f
    kept[:, 0] = state
    for t in range(1, n):
        u = rng.random(reps)
        state = np.where(state, u >= b, u < a)        # keep->keep w.p. 1-b ; delete->keep w.p. a
        kept[:, t] = state
    return kept


def slice_counts(counts):
    """Per-slice counts from per-core counts (..., 2 m) whose columns (2j, 2j+1) are the two
    cores of one active slice (true for sorted cores of a slice-aligned map)."""
    counts = np.asarray(counts)
    return counts.reshape(*counts.shape[:-1], counts.shape[-1] // 2, 2).sum(-1)


def slices_revealed_iid(counts, f, slices=12, active=8):
    """Partial disclosure under iid deletion. counts: (sockets, 2 * active) per-core counts,
    columns (2j, 2j+1) being the two cores of one active slice. An active slice is revealed when
    >= 1 of its samples survives (no false positives). Returns, per socket, the expected number
    k of revealed active slices and the expected number of maps (active-slice sets of size
    `active` out of `slices`) still consistent with what survived, E[C(slices-k, active-k)];
    k is Poisson-binomial over the slices."""
    q = 1.0 - p_silent_iid(slice_counts(counts), f)                     # (S, active)
    dist = np.zeros((q.shape[0], active + 1))
    dist[:, 0] = 1.0
    for j in range(active):
        new = dist * (1.0 - q[:, j:j + 1])
        new[:, 1:] += dist[:, :-1] * q[:, j:j + 1]
        dist = new
    k = np.arange(active + 1)
    cand = np.array([comb(slices - kk, active - kk) for kk in k], dtype=float)
    return (dist * k).sum(1), (dist * cand).sum(1)


def decimation_all_phases(A, k):
    """Deterministic decimation measured on a presence array A (sockets x cores x slots, bool):
    the publisher keeps slots ph, ph + k, ph + 2k, ...; returns, per socket, the fraction of all
    k phases for which every core keeps >= 1 sample. With k = slots (one kept slot per day) the
    phase average is exactly the single-interval recovery rate."""
    A = np.asarray(A, dtype=bool)
    S, C, T = A.shape
    if T % k:
        raise ValueError('the number of slots must be a multiple of k')
    return A.reshape(S, C, T // k, k).any(2).all(1).mean(1)


def timestamp_deletion_bounds(A, f):
    """Whole-timestamp deletion: every slot of a socket is deleted with probability f, for all
    of its cores together, independently across slots. Exact recovery needs every core to keep
    >= 1 sample. Per socket, 1 - f ** m_all <= P(exact) <= min_i (1 - f ** n_i), where m_all is
    the number of slots in which all cores report and n_i the slots in which core i reports."""
    A = np.asarray(A, dtype=bool)
    m_all = A.all(-2).sum(-1).astype(float)
    n_i = A.sum(-1).astype(float)
    return 1.0 - p_silent_iid(m_all, f), (1.0 - p_silent_iid(n_i, f)).min(-1)


def run_lengths(mask):
    """Lengths of the runs of True along the last axis of a boolean array (rows independent)."""
    m = np.asarray(mask, dtype=bool)
    m = m.reshape(-1, m.shape[-1])
    pad = np.zeros((m.shape[0], 1), dtype=np.int8)
    dx = np.diff(np.concatenate([pad, m.astype(np.int8), pad], axis=1), axis=1).ravel()
    return np.flatnonzero(dx == -1) - np.flatnonzero(dx == 1)


def expected_unique_share(n, n_sets):
    """Expected share of n items that are unique when each draws one of n_sets equally likely
    labels independently: (1 - 1/n_sets) ** (n - 1). A uniformly random permutation of the 24
    core ids maps any fixed 16-core map to a uniformly random 16-of-24 set, so with
    n_sets = C(24, 16) this is the fingerprinting power of a FIXED per-node pseudonym."""
    return (1.0 - 1.0 / n_sets) ** (n - 1)


def deletion_for_target(counts, target, model='iid', L=None):
    """Smallest deletion fraction f at which the MEAN exact-recovery probability over the
    sockets in `counts` (sockets x cores) falls to `target`. Solved on log(1 - f)."""
    counts = np.asarray(counts, dtype=float)

    def g(logr):
        f = 1.0 - np.exp(logr)
        ps = p_silent_iid(counts, f) if model == 'iid' else p_silent_bursty(counts, f, L)
        return p_exact(ps).mean() - target
    lo, hi = np.log(1e-12), np.log(0.5)
    if g(hi) < 0:
        return 0.5
    return 1.0 - np.exp(brentq(g, lo, hi, xtol=1e-12))
