"""Shared, tested building blocks for the longitudinal (lifecycle) analyses.

Everything here is a pure function of the committed derived tables (`daily/`, `cov_all/`,
`os_all/`) or of arrays passed in, so the estimators can be unit-tested on synthetic inputs
(`tests/test_lifecycle.py`) and reused by `reboot_evidence.py`, `change_taxonomy.py`,
`exposure_rates.py` and `lifecycle_taxonomy_errors.py` without three diverging copies of the
same clean-day logic.

Conventions (identical to the paper):
  clean socket-day  exactly 16 cores report AND each reports > 90 % of 4,320 samples
  steady state      2020-06-01 onward (the commissioning window 2020-03..05 is excluded)
  interval          consecutive clean days (d0, d1] of one socket; a transition if the set differs
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2, hypergeom

EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
DAYS_PER_YEAR = 365.25
GAP_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
NS_PER_DAY = 86_400 * 10 ** 9


# ---------------------------------------------------------------------------- loading
def load_daily(root):
    df = pd.concat([pd.read_parquet(f) for f in sorted((Path(root) / 'daily').glob('daily_*.parquet'))],
                   ignore_index=True)
    df['day'] = pd.to_datetime(df['day'], utc=True)
    return df


def socket_days(df):
    """(node, socket, day, active, nfull, mask) for every socket-day in the raw table.

    `mask` is the 24-bit set of reporting cores (bit c = core c). The daily table has one row
    per (node, socket, core, day), so a sum of bits is the bitwise OR.
    """
    tmp = pd.DataFrame({'node': df.node.to_numpy(), 'socket': df.socket.to_numpy(), 'day': df.day,
                        'bit': np.left_shift(np.int64(1), df.core.to_numpy().astype(np.int64)),
                        'full': df.n.to_numpy() > 0.9 * EXPECT})
    t = (tmp.groupby(['node', 'socket', 'day'], sort=True)
            .agg(active=('bit', 'size'), nfull=('full', 'sum'), mask=('bit', 'sum')).reset_index())
    return t


def mask_to_set(m):
    m = int(m)
    return frozenset(i for i in range(24) if (m >> i) & 1)


def set_to_mask(s):
    return int(sum(1 << int(c) for c in s))


def clean_sets(df, t=None):
    """Reporting-core mask on every clean socket-day, sorted by node, socket, day."""
    t = socket_days(df) if t is None else t
    sets = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day', 'mask']]
    return sets.sort_values(['node', 'socket', 'day'], ignore_index=True)


def by_socket(sets):
    """{(node, socket): (day index array, mask array, Timestamp array)} over clean days."""
    di = _day_int(sets.day)
    mk = sets['mask'].to_numpy()
    ts = pd.DatetimeIndex(sets.day)
    return {(int(k[0]), int(k[1])): (di[idx], mk[idx], ts[idx])
            for k, idx in sets.groupby(['node', 'socket']).indices.items()}


def intervals(sets):
    """Consecutive clean-day intervals (d0, d1] of each socket, with old/new core masks."""
    s = sets.sort_values(['node', 'socket', 'day'])
    same = (s.node.values[1:] == s.node.values[:-1]) & (s.socket.values[1:] == s.socket.values[:-1])
    i1 = np.flatnonzero(same) + 1
    i0 = i1 - 1
    day = pd.DatetimeIndex(s.day)
    out = pd.DataFrame({'node': s.node.values[i1].astype(int), 'socket': s.socket.values[i1].astype(int),
                        'd0': day[i0], 'd1': day[i1],
                        'old': s['mask'].values[i0], 'new': s['mask'].values[i1]})
    out['gap'] = (out.d1 - out.d0).dt.days.astype(int)
    out['changed'] = out.old.values != out.new.values
    return out


def gap_bin(gap):
    """Index of the interval-length stratum used throughout (1, 2, 3-4, 5-8, 9+ days)."""
    gap = np.asarray(gap)
    b = np.full(gap.shape, -1, dtype=int)
    for k, (lo, hi) in enumerate(GAP_BINS):
        b[(gap >= lo) & (gap <= hi)] = k
    return b


def load_boot(root):
    """Ganglia `boottime` per node-day, with the day-flag reboot definition of the paper.

    A node rebooted on day d if two distinct boot epochs appear within d, or if the epoch differs
    from the previous *observed* day. No-coverage days are dropped BEFORE shifting.
    """
    cov = pd.concat([pd.read_parquet(f) for f in sorted((Path(root) / 'cov_all').glob('cov_*.parquet'))],
                    ignore_index=True)
    cov['day'] = pd.to_datetime(cov['day'], utc=True)
    boot = cov[['node', 'day', 'boot_min', 'boot_max', 'boot_ndistinct', 'boot_n']].dropna(subset=['boot_max'])
    boot = boot.sort_values(['node', 'day'], ignore_index=True)
    boot['prev'] = boot.groupby('node')['boot_max'].shift()
    boot['rebooted'] = ((boot.boot_ndistinct > 1) | (boot.prev.notna() & (boot.boot_max != boot.prev)))
    return boot


# ---------------------------------------------------------------------------- reboot evidence
def _day_int(x):
    """UTC day index (days since 1970-01-01) of datetimes, independent of the storage unit
    (the committed parquet files store datetime64[ms, UTC], so a raw int64 cast is not ns)."""
    s = pd.to_datetime(pd.Series(np.asarray(x) if not isinstance(x, pd.Series) else x), utc=True)
    return ((s - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)


def dayflag_reboot(iv, boot):
    """Paper's original rule. Per interval: 1 if a boot-covered day in (d0, d1] carries a reboot
    flag, 0 if covered days exist but none is flagged, NaN if no covered day lies in (d0, d1]."""
    out = np.full(len(iv), np.nan)
    d0 = _day_int(iv.d0)
    d1 = _day_int(iv.d1)
    bd = _day_int(boot.day)
    for n, idx in iv.groupby('node').indices.items():
        m = (boot.node.values == n)
        if not m.any():
            continue
        days = bd[m]
        flags = boot.rebooted.values[m].astype(int)
        cs = np.concatenate([[0], np.cumsum(flags)])
        lo = np.searchsorted(days, d0[idx], side='right')       # first covered day > d0
        hi = np.searchsorted(days, d1[idx], side='right')       # covered days <= d1
        cov = hi > lo
        r = cs[hi] - cs[lo]
        out[idx] = np.where(cov, (r > 0).astype(float), np.nan)
    return out


def node_boot_epochs(boot):
    """Every distinct boot epoch (seconds) ever reported for each node, sorted."""
    ep = {}
    for n, g in boot.groupby('node'):
        ep[int(n)] = np.unique(np.concatenate([g.boot_min.values, g.boot_max.values]))
    return ep


def epoch_reboot(iv, boot, min_obs=1200):
    """Reboot in (d0, d1] dated by the boot EPOCH rather than by the day it was first observed.

    `boottime` reports when the node last booted, so a reboot that happens during a collector
    outage is still dated exactly once collection resumes. Per interval, with W = the days
    (d0, d1]:
      1    some reported boot epoch falls on a day in W                          (boot confirmed)
      0    the node is observed after W (a day > d1, or d1 itself with >= min_obs samples)
           reporting a last-boot epoch that predates W                            (no boot)
      NaN  neither: a boot happened between the observations bracketing W but the only epoch
           reported for it lies after W, so an earlier, superseded boot inside W cannot be
           excluded, or there is no observation after W at all                    (unresolved)
    """
    out = np.full(len(iv), np.nan)
    d0 = _day_int(iv.d0)
    d1 = _day_int(iv.d1)
    bd = _day_int(boot.day)
    for n, idx in iv.groupby('node').indices.items():
        m = (boot.node.values == n)
        if not m.any():
            continue
        days = bd[m]
        bmin = boot.boot_min.values[m]
        bmax = boot.boot_max.values[m]
        nobs = boot.boot_n.values[m]
        eps = np.unique(np.concatenate([bmin, bmax]))
        epd = np.floor(eps / 86400).astype(np.int64)          # UTC day of each boot
        # confirmed: an epoch day in (d0, d1]
        lo = np.searchsorted(epd, d0[idx], side='right')
        hi = np.searchsorted(epd, d1[idx], side='right')
        yes = hi > lo
        # no-boot: first observation that closes the window reports an epoch older than W
        w_start = (d0[idx] + 1) * 86400.0                      # seconds at start of W
        j = np.searchsorted(days, d1[idx], side='left')       # first observed day >= d1
        L = len(days)
        jc = np.minimum(j, L - 1)
        same = (j < L) & (days[jc] == d1[idx])                 # d1 itself observed
        no_a = same & (nobs[jc] >= min_obs) & (bmax[jc] < w_start)
        jj = j + same.astype(int)                              # else: first observed day > d1
        jj2 = np.minimum(jj, L - 1)
        no_b = (jj < L) & (bmin[jj2] < w_start)
        no = no_a | no_b
        out[idx] = np.where(yes, 1.0, np.where(no, 0.0, np.nan))
    return out


# ---------------------------------------------------------------------------- inference
def stratified_hypergeom_test(stratum, is_t, y, B=200_000, rng=None):
    """Exact conditional test that transitions carry more reboots than controls WITHIN strata.

    Under H0 the transition labels are exchangeable among the intervals of one stratum, so the
    number of reboot-carrying transitions in stratum s is Hypergeometric(n_s, r_s, t_s),
    independently across strata. Returns the observed total, its null mean and SD, a Monte-Carlo
    p-value (b+1)/(B+1), and the exact upper tail from convolving the per-stratum pmfs.
    Strata in which either the labels or the outcomes are constant carry no information and
    contribute a fixed amount to both the observed and the null total.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    stratum = np.asarray(stratum)
    is_t = np.asarray(is_t, dtype=bool)
    y = np.asarray(y, dtype=int)
    key, inv = np.unique(stratum, return_inverse=True)
    n = np.bincount(inv)
    t = np.bincount(inv, weights=is_t).astype(int)
    r = np.bincount(inv, weights=y).astype(int)
    obs = int((y & is_t).sum())
    inf = (t > 0) & (t < n) & (r > 0) & (r < n)
    fixed = int((y[is_t] * ~inf[inv[is_t]]).sum())
    N, R, T = n[inf], r[inf], t[inf]
    mean = fixed + float((T * R / N).sum())
    var = float((T * (N - T) * R * (N - R) / (N ** 2 * (N - 1))).sum())
    # Monte Carlo
    draws = np.full(B, fixed, dtype=np.int64)
    for Ns, Rs, Ts in zip(N, R, T):
        draws += rng.hypergeometric(Rs, Ns - Rs, Ts, size=B)
    b = int((draws >= obs).sum())
    # exact tail by convolution
    pmf = np.array([1.0])
    for Ns, Rs, Ts in zip(N, R, T):
        k = np.arange(max(0, Ts - (Ns - Rs)), min(Ts, Rs) + 1)
        p = np.zeros(k.max() + 1)
        p[k] = hypergeom.pmf(k, Ns, Rs, Ts)
        pmf = np.convolve(pmf, p)
    need = obs - fixed
    p_exact = float(pmf[need:].sum()) if need < len(pmf) else 0.0
    return dict(observed=obs, null_mean=mean, null_sd=float(np.sqrt(var)),
                z=(obs - mean) / np.sqrt(var) if var > 0 else float('inf'),
                informative_strata=int(inf.sum()), informative_transitions=int(T.sum()),
                B=B, b=b, p_mc=(b + 1) / (B + 1), p_exact=p_exact)


def standardized_excess(gbin, is_t, y, nbins=len(GAP_BINS)):
    """Transition-weighted risk difference: mean over transitions of (y - control rate of the
    transition's interval-length stratum). Strata without controls are dropped."""
    gbin = np.asarray(gbin)
    is_t = np.asarray(is_t, dtype=bool)
    y = np.asarray(y, dtype=float)
    num = den = 0.0
    for g in range(nbins):
        tm = is_t & (gbin == g)
        cm = ~is_t & (gbin == g)
        if tm.sum() == 0 or cm.sum() == 0:
            continue
        num += tm.sum() * (y[tm].mean() - y[cm].mean())
        den += tm.sum()
    return num / den if den else np.nan


def cluster_bootstrap_excess(cluster, gbin, is_t, y, B=10_000, rng=None, nbins=len(GAP_BINS)):
    """Percentile CI for `standardized_excess`, resampling whole clusters with replacement."""
    rng = np.random.default_rng(1) if rng is None else rng
    cluster = np.asarray(cluster)
    gbin = np.asarray(gbin)
    is_t = np.asarray(is_t, dtype=bool)
    y = np.asarray(y, dtype=float)
    key, inv = np.unique(cluster, return_inverse=True)
    K = len(key)
    # per-cluster sufficient statistics: [bin, {T, T_y, C, C_y}]
    st = np.zeros((K, nbins, 4))
    for g in range(nbins):
        m = gbin == g
        st[:, g, 0] = np.bincount(inv[m & is_t], minlength=K)
        st[:, g, 1] = np.bincount(inv[m & is_t], weights=y[m & is_t], minlength=K)
        st[:, g, 2] = np.bincount(inv[m & ~is_t], minlength=K)
        st[:, g, 3] = np.bincount(inv[m & ~is_t], weights=y[m & ~is_t], minlength=K)

    def stat(w):
        s = np.tensordot(w, st, axes=(0, 0))                 # (nbins, 4)
        ok = (s[:, 0] > 0) & (s[:, 2] > 0)
        if not ok.any():
            return np.nan
        rd = s[ok, 1] / s[ok, 0] - s[ok, 3] / s[ok, 2]
        return float((s[ok, 0] * rd).sum() / s[ok, 0].sum())

    point = stat(np.ones(K))
    bs = np.array([stat(rng.multinomial(K, np.full(K, 1 / K))) for _ in range(B)])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(point=point, lo=float(lo), hi=float(hi), se=float(bs.std(ddof=1)), clusters=K, B=len(bs))


def poisson_ci(k, expo, a=0.05):
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2 / expo
    return lo, chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2 / expo


def cp_ci(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def cluster_bootstrap_rate(cluster_events, cluster_exposure, B=10_000, rng=None, a=0.05):
    """Percentile CI for sum(events)/sum(exposure), resampling clusters with replacement."""
    rng = np.random.default_rng(2) if rng is None else rng
    e = np.asarray(cluster_events, dtype=float)
    x = np.asarray(cluster_exposure, dtype=float)
    K = len(e)
    W = rng.multinomial(K, np.full(K, 1 / K), size=B)
    rates = (W @ e) / (W @ x)
    lo, hi = np.percentile(rates, [100 * a / 2, 100 * (1 - a / 2)])
    return dict(point=float(e.sum() / x.sum()), lo=float(lo), hi=float(hi),
                se=float(rates.std(ddof=1)), clusters=K, B=B)


# ---------------------------------------------------------------------------- taxonomy
def classify_events(sets):
    """Node-level map-change events, classified exactly as change_taxonomy.py defines them.

      RELABEL    both sockets change at the same interval and p0/p1 exchange maps exactly
      NODE-SWAP  both change at the same interval, not as a mirror
      CPU-SWAP   one socket changes; the partner is observed (>= 2 clean days spanning the
                 interval) and unchanged
      CPU-SWAP?  one socket changes; the partner is not observed across the interval
    Returns one row per event: node, day (first clean day after), kind, nsock, era, and per
    socket the (old, new, previous clean day) triple in `per`.
    """
    BS = by_socket(sets)
    ch = intervals(sets)
    ch = ch[ch.changed]
    ev = {}
    for r in ch.itertuples():
        ev.setdefault((r.node, r.d1), {})[r.socket] = (int(r.old), int(r.new), r.d0)
    rows = []
    for (n, d), per in sorted(ev.items()):
        if len(per) == 2:
            (o0, n0, _), (o1, n1, _) = per[0], per[1]
            kind = 'RELABEL' if (n0 == o1 and n1 == o0) else 'NODE-SWAP'
        else:
            s = next(iter(per))
            o, w, dprev = per[s]
            x = BS.get((n, 1 - s))
            kind = 'CPU-SWAP?'
            if x is not None:
                lo = np.searchsorted(x[0], _day_int([dprev])[0], side='left')
                hi = np.searchsorted(x[0], _day_int([d])[0], side='right')
                if hi - lo >= 2 and x[1][lo] == x[1][hi - 1]:
                    kind = 'CPU-SWAP'
        per = {s: (mask_to_set(o), mask_to_set(w), dp) for s, (o, w, dp) in per.items()}
        rows.append(dict(node=int(n), day=d, kind=kind, nsock=len(per), per=per))
    E = pd.DataFrame(rows)
    E['era'] = np.where(E.day < CUT, 'commissioning', 'steady')
    E['dies'] = E.kind.map({'RELABEL': 0, 'CPU-SWAP': 1, 'CPU-SWAP?': 1, 'NODE-SWAP': 2})
    return E


def load_os(root):
    """Ganglia `cpu_num` (the OS's own logical-CPU count) per node-day and value."""
    os_ = pd.concat([pd.read_parquet(f) for f in sorted((Path(root) / 'os_all').glob('os_*.parquet'))],
                    ignore_index=True)
    os_['day'] = pd.to_datetime(os_['day'], utc=True)
    return os_


def runs(days, max_gap=1):
    """Group sorted Timestamps into runs of consecutive days."""
    out, cur = [], [days[0]]
    for d in days[1:]:
        if (d - cur[-1]).days <= max_gap:
            cur.append(d)
        else:
            out.append(cur)
            cur = [d]
    out.append(cur)
    return out


def last_clean_before(bs, node, socket, day):
    """(core set, day) of the socket's last clean day strictly before `day`; bs = by_socket()."""
    x = bs.get((int(node), int(socket)))
    if x is None:
        return None, None
    j = np.searchsorted(x[0], _day_int([day])[0], side='left') - 1
    return (mask_to_set(x[1][j]), x[2][j]) if j >= 0 else (None, None)


def first_clean_after(bs, node, socket, day):
    """(core set, day) of the socket's first clean day strictly after `day`."""
    x = bs.get((int(node), int(socket)))
    if x is None:
        return None, None
    j = np.searchsorted(x[0], _day_int([day])[0], side='right')
    return (mask_to_set(x[1][j]), x[2][j]) if j < len(x[0]) else (None, None)


def core_loss(counts, baseline, onset=False):
    """Cores of the configuration in force (`baseline`) that are lost on one socket-day.

    counts: {core: samples that day}. A day with a core outside the baseline is a
    reconfiguration (union) day and says nothing about loss, so it returns an empty set.
      zero rule   baseline cores with no sample while >= 8 other baseline cores report
      onset rule  (onset=True) baseline cores partially sampled while every other baseline core
                  is fully sampled: the core disappeared part-way through the day at a reboot
    """
    present = set(counts)
    if baseline is None or not present or not present <= baseline:
        return set()
    lost = baseline - present
    if lost and len(baseline & present) >= 8:
        return lost
    if onset:
        part = {c for c in baseline if counts.get(c, 0) <= 0.9 * EXPECT}
        if part and len(part) <= 2 and all(counts.get(c, 0) > 0.9 * EXPECT for c in baseline - part):
            return part
    return set()


def union_relation(day_set, before, after):
    """How a socket-day's reporting cores relate to the clean maps on either side of it.

      'exact'    the day reports exactly the union of the two maps (a same-day cutover)
      'within'   every reporting core belongs to one of the two maps, but not all of them report
      'outside'  some core belongs to neither map (e.g. an intermediate die inside the gap)
      None       a neighbouring clean map is missing
    """
    if before is None or after is None:
        return None
    u = set(before) | set(after)
    s = set(day_set)
    if s == u:
        return 'exact'
    return 'within' if s <= u else 'outside'


def half_mirror(per):
    """A two-socket event in which one socket takes its partner's OLD map while the other gets a
    map seen on neither socket: a relabel combined with one new die. The taxonomy counts it as a
    NODE-SWAP (two dies); the most economical reading moves only one."""
    if len(per) != 2:
        return False
    (o0, n0, _), (o1, n1, _) = per[0], per[1]
    if n0 == o1 and n1 == o0:
        return False
    return (n0 == o1 and n1 not in (o0, o1)) or (n1 == o0 and n0 not in (o0, o1))


def slices_of(cores):
    return frozenset(c // 2 for c in cores)


def harvested_slices(cores):
    return frozenset(range(12)) - slices_of(cores)


def collision_probability(maps):
    """Unbiased probability that two DISTINCT sockets drawn from `maps` carry the same map:
    sum_p n_p (n_p - 1) / (N (N - 1)). This is the chance that a replacement die drawn from the
    same population is invisible to a map-change detector."""
    _, n = np.unique(np.asarray([tuple(sorted(m)) for m in maps]), axis=0, return_counts=True)
    N = n.sum()
    return float((n * (n - 1)).sum() / (N * (N - 1)))


def cross_collision_probability(maps_a, maps_b):
    """Probability that a socket of population A and a die drawn from population B match."""
    ca = pd.Series([tuple(sorted(m)) for m in maps_a]).value_counts(normalize=True)
    cb = pd.Series([tuple(sorted(m)) for m in maps_b]).value_counts(normalize=True)
    return float((ca * cb.reindex(ca.index, fill_value=0)).sum())


def common_slice_permutation(pairs, nslices=12):
    """Is there ONE permutation sigma of the slice indices with sigma(old_i) == new_i for every
    (old_i, new_i) pair of harvested-slice sets? A firmware sensor-ID remap applied to many
    sockets at once must satisfy this; independent new dies almost never do. Solved exactly as a
    bipartite perfect-matching problem on the candidate images of each slice. Returns the
    permutation as a list, or None."""
    cand = np.ones((nslices, nslices), dtype=bool)
    for old, new in pairs:
        o = np.zeros(nslices, bool); o[list(old)] = True
        w = np.zeros(nslices, bool); w[list(new)] = True
        cand &= np.where(o[:, None], w[None, :], ~w[None, :])
    match = [-1] * nslices                                     # target -> source

    def aug(u, seen):
        for v in np.flatnonzero(cand[u]):
            if not seen[v]:
                seen[v] = True
                if match[v] < 0 or aug(match[v], seen):
                    match[v] = u
                    return True
        return False

    for u in range(nslices):
        if not aug(u, [False] * nslices):
            return None
    sigma = [0] * nslices
    for v, u in enumerate(match):
        sigma[u] = v
    return sigma
