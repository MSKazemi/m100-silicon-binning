"""Fast path: socket-level thermal inference from the committed per-socket summaries.

Written for the external review of 2026-09-11 (THERMAL items). It re-derives the census numbers
of Sec. IV-E/F from per-socket matrices and then does what the pooled pair lists could not:

  1. uncertainty at the level of dependence: rack- and node-clustered bootstrap, rack-level
     sign-flip test, instead of resampling 15,584 pair correlations as if independent;
  2. composition: per-socket contrasts, position-matched (balanced) contrasts inside each
     socket, post-stratified and two-way (socket + cell) pooled matrices, Fisher-z pooling,
     and the support count of every cell;
  3. strata: lot, p0/p1, stable-map sockets, utilisation, socket power, map pattern, and the
     other months built by thermal_socket_build.py;
  4. the floorplan score with an EXACT optimum over all 12!/2 one-dimensional orderings, the
     exact rank of the identity ordering, and a rack bootstrap of the matrix and the gap;
  5. the quad (four-core POWER9 frequency/voltage domain) decomposition of the index-adjacent
     pairs, which the pooled within/cross contrast mixes;
  6. lagged profiles with the lag in real 20 s steps.

Inputs : analysis/thermal_socket_{meta,c5,c1,lag}_<YM>.{csv,npy}  (thermal_socket_build.py)
         daily/, cov_all/, power_all/ (committed)
Outputs: analysis/thermal_socket_stats[_<YM>].json, analysis/thermal_support_counts[_<YM>].csv
         (no suffix for 2022-08, the paper's month)
"""
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

root = Path(__file__).resolve().parent.parent
A = root / 'analysis'
YM = os.environ.get('THERM_MONTH', '22-08')
B_BOOT = int(os.environ.get('THERM_BOOT', '2000'))
rng = np.random.default_rng(20260911)
NC, NP = 24, 12
LAGS = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
OUT = {}


def load(ym):
    meta = pd.read_csv(A / f'thermal_socket_meta_{ym}.csv')
    act = np.array([list(map(int, s.split('-'))) for s in meta.active])
    c5 = np.load(A / f'thermal_socket_c5_{ym}.npy')
    c1 = np.load(A / f'thermal_socket_c1_{ym}.npy')
    lag = np.load(A / f'thermal_socket_lag_{ym}.npy')
    M5 = np.stack([tl.unpack(c5[i], act[i]) for i in range(len(meta))])
    M1 = np.stack([tl.unpack(c1[i], act[i]) for i in range(len(meta))])
    return meta, act, M5, M1, lag


meta, act, M5, M1, LAG = load(YM)
ns = len(meta)
rack = meta.rack.values
lot = np.where(rack < 22, 'A', np.where(rack <= 48, 'B', 'U'))
print(f'[{YM}] sockets {ns}, nodes {meta.node.nunique()}, racks {len(np.unique(rack))} '
      f'(Lot A {np.sum(lot == "A")}, Lot B {np.sum(lot == "B")}, undocumented {np.sum(lot == "U")})')


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# ============================================================ 0. replicate the paper's census
def pair_lists(M, act):
    W, X = [], []
    per = []
    for i in range(len(M)):
        idx = act[i]
        w = [M[i, idx[a], idx[b]] for a, b in zip(*tl.IU16)
             if idx[b] - idx[a] == 1 and idx[a] // 2 == idx[b] // 2]
        x = [M[i, idx[a], idx[b]] for a, b in zip(*tl.IU16)
             if idx[b] - idx[a] == 1 and idx[a] // 2 != idx[b] // 2]
        W += w; X += x
        per.append((np.mean(w), np.mean(x) if x else np.nan, len(w), len(x)))
    return np.array(W), np.array(X), np.array(per)


W, X, PER = pair_lists(M5, act)
pooled5, support = tl.pool(M5)
np.fill_diagonal(pooled5, 1.0)
ref = np.load(A / 'thermal_corr_allnodes.npy')
S5 = tl.slice_similarity(pooled5)
ident = tl.order_score(tl.cells(S5), np.arange(NP))
r_old = np.random.default_rng(5)          # the paper's pair-level bootstrap, reproduced exactly


def ci_pairs(v, n=4000):
    m = v[r_old.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


wo, xo = ci_pairs(W), ci_pairs(X)
comb = np.r_[W, X]
nullp = np.array([(lambda q: q[:len(W)].mean() - q[len(W):].mean())(r_old.permutation(comb))
                  for _ in range(4000)])
contrast = W.mean() - X.mean()
dsock = PER[:, 0] - PER[:, 1]
OUT['replication'] = dict(
    sockets=ns, max_abs_diff_vs_committed_matrix=float(np.nanmax(np.abs(pooled5 - ref))),
    within=r4(W.mean()), within_ci_pairs=[r4(wo[1]), r4(wo[2])], n_within=int(len(W)),
    cross=r4(X.mean()), cross_ci_pairs=[r4(xo[1]), r4(xo[2])], n_cross=int(len(X)),
    contrast=r4(contrast),
    permutation_pairs_b=int((np.abs(nullp) >= abs(contrast)).sum()), permutation_B=4000,
    sockets_within_gt_cross=int((dsock > 0).sum()), sockets_with_both=int((~np.isnan(dsock)).sum()),
    floorplan_identity=r4(ident))
print('\n=== 0. replication of the census (every 5th sample) ===')
print(f'  pooled matrix vs committed thermal_corr_allnodes.npy: max |diff| = '
      f'{OUT["replication"]["max_abs_diff_vs_committed_matrix"]:.2e}')
print(f'  within {W.mean():+.4f} [{wo[1]:+.4f},{wo[2]:+.4f}] n={len(W)}   '
      f'cross {X.mean():+.4f} [{xo[1]:+.4f},{xo[2]:+.4f}] n={len(X)}   contrast {contrast:+.4f}')
print(f'  sockets within>cross: {(dsock > 0).sum()}/{(~np.isnan(dsock)).sum()} '
      f'({100 * np.nanmean(dsock > 0):.1f}%)   1x12 fit {ident:+.3f}')

# ============================================================ helpers for clustered inference
node = meta.node.values
groups = {'rack': rack, 'node': node}
rw = np.nansum(PER[:, 0] * PER[:, 2]); rwn = PER[:, 2].sum()


def pooled_contrast(ix):
    P = PER[ix]
    ok = P[:, 3] > 0
    w = (P[:, 0] * P[:, 2]).sum() / P[:, 2].sum()
    x = (P[ok, 1] * P[ok, 3]).sum() / P[ok, 3].sum()
    return w, x, w - x


def boot(fun, clusters, B=B_BOOT, seed=1):
    r = np.random.default_rng(seed)
    return np.array([fun(tl.cluster_resample(clusters, r)) for _ in range(B)])


# ============================================================ 1. the contrast, clustered
print('\n=== 1. within- vs cross-slice contrast with clustered uncertainty ===')
res1 = {}
allix = np.arange(ns)
w0, x0, d0 = pooled_contrast(allix)
msock = np.nanmean(dsock)
for g, cl in groups.items():
    bb = boot(lambda ix: (*pooled_contrast(ix), np.nanmean(dsock[ix]),
                          np.nanmean(dsock[ix] > 0)), cl)
    res1[g] = dict(within_ci=list(map(r4, tl.ci(bb[:, 0]))), cross_ci=list(map(r4, tl.ci(bb[:, 1]))),
                   contrast_ci=list(map(r4, tl.ci(bb[:, 2]))),
                   socket_mean_contrast_ci=list(map(r4, tl.ci(bb[:, 3]))),
                   frac_positive_ci=list(map(r4, tl.ci(bb[:, 4]))),
                   se_contrast=r4(bb[:, 2].std(ddof=1)))
    print(f'  {g:>4}-bootstrap: within {tl.ci(bb[:, 0])}  cross {tl.ci(bb[:, 1])}  '
          f'contrast {d0:+.4f} {tl.ci(bb[:, 2])}  socket-mean {msock:+.4f} {tl.ci(bb[:, 3])}')
se_pair = np.sqrt(W.var(ddof=1) / len(W) + X.var(ddof=1) / len(X))
rackmean = pd.Series(dsock).groupby(rack).mean()
p_flip, _ = tl.sign_flip_test(rackmean.values, rng, B=99999)
t_rack = st.ttest_1samp(rackmean.values, 0.0, alternative='greater')
nodemean = pd.Series(dsock).groupby(node).mean()
res1.update(pooled_within=r4(w0), pooled_cross=r4(x0), pooled_contrast=r4(d0),
            socket_mean_contrast=r4(msock), socket_median_contrast=r4(np.nanmedian(dsock)),
            frac_sockets_positive=r4(np.nanmean(dsock > 0)),
            frac_nodes_positive=r4((nodemean > 0).mean()),
            racks=int(len(rackmean)), racks_positive=int((rackmean > 0).sum()),
            rack_signflip_p=p_flip, rack_signflip_B=99999,
            rack_t=r4(t_rack.statistic), rack_t_p=float(t_rack.pvalue),
            se_pairs_as_independent=r4(se_pair),
            design_effect_rack=r4((res1['rack']['se_contrast'] / se_pair) ** 2))
OUT['contrast_clustered'] = res1
print(f'  pair-level SE {se_pair:.4f} vs rack-bootstrap SE {res1["rack"]["se_contrast"]:.4f} '
      f'(design effect {res1["design_effect_rack"]:.1f})')
print(f'  racks with a positive mean socket contrast: {(rackmean > 0).sum()}/{len(rackmean)}; '
      f'sign-flip p = {p_flip:.2e}; t = {t_rack.statistic:.2f}')
print(f'  nodes positive {100 * (nodemean > 0).mean():.1f}%, sockets positive '
      f'{100 * np.nanmean(dsock > 0):.1f}%')

# ============================================================ 2. quad decomposition
print('\n=== 2. the three kinds of index-adjacent pair (slice / quad structure) ===')
kinds = ['within', 'samequad', 'crossquad']
KIND = np.full((NC, NC), '', dtype=object)
for i in range(NC - 1):
    KIND[i, i + 1] = tl.pair_kind(i, i + 1)
ksum = {k: np.zeros(ns) for k in kinds}; kcnt = {k: np.zeros(ns) for k in kinds}
for i in range(NC - 1):
    v = M5[:, i, i + 1]; ok = ~np.isnan(v); k = KIND[i, i + 1]
    ksum[k][ok] += v[ok]; kcnt[k][ok] += 1
kmean = {k: np.where(kcnt[k] > 0, ksum[k] / np.maximum(kcnt[k], 1), np.nan) for k in kinds}


def kpooled(ix):
    return [ksum[k][ix].sum() / kcnt[k][ix].sum() for k in kinds]


# balanced, position-matched contrasts inside each socket
bal_sq = np.full((ns, 6), np.nan); bal_cq = np.full((ns, 5), np.nan)
for q in range(6):
    a, b = 4 * q, 4 * q + 3
    w = (M5[:, a, a + 1] + M5[:, a + 2, a + 3]) / 2
    bal_sq[:, q] = w - M5[:, a + 1, a + 2]                        # NaN unless quad fully active
for q in range(5):
    a = 4 * q + 2
    w = (M5[:, a, a + 1] + M5[:, a + 2, a + 3]) / 2
    bal_cq[:, q] = w - M5[:, a + 1, a + 2]                        # slices 2q+1, 2q+2 active
s_sq = np.nanmean(bal_sq, 1); s_cq = np.nanmean(bal_cq, 1)
res2 = {}
kp = kpooled(allix)
bk = boot(lambda ix: (*kpooled(ix), np.nanmean(s_sq[ix]), np.nanmean(s_cq[ix])), rack)
for j, k in enumerate(kinds):
    res2[k] = dict(pooled=r4(kp[j]), rack_ci=list(map(r4, tl.ci(bk[:, j]))),
                   n_pairs=int(kcnt[k].sum()), sockets=int((kcnt[k] > 0).sum()))
    print(f'  {k:<10} pooled r = {kp[j]:+.4f} {tl.ci(bk[:, j])}  pairs {int(kcnt[k].sum()):>6} '
          f'on {(kcnt[k] > 0).sum()} sockets')
res2['balanced_within_minus_samequad'] = dict(
    mean=r4(np.nanmean(s_sq)), rack_ci=list(map(r4, tl.ci(bk[:, 3]))),
    sockets=int((~np.isnan(s_sq)).sum()), frac_positive=r4(np.nanmean(s_sq[~np.isnan(s_sq)] > 0)))
res2['balanced_within_minus_crossquad'] = dict(
    mean=r4(np.nanmean(s_cq)), rack_ci=list(map(r4, tl.ci(bk[:, 4]))),
    sockets=int((~np.isnan(s_cq)).sum()), frac_positive=r4(np.nanmean(s_cq[~np.isnan(s_cq)] > 0)))
res2['share_samequad_among_cross'] = r4(kcnt['samequad'].sum() /
                                        (kcnt['samequad'].sum() + kcnt['crossquad'].sum()))
res2['per_position'] = {
    'within': [r4(np.nanmean(M5[:, 2 * k, 2 * k + 1])) for k in range(12)],
    'samequad': [r4(np.nanmean(M5[:, 4 * q + 1, 4 * q + 2])) for q in range(6)],
    'crossquad': [r4(np.nanmean(M5[:, 4 * q + 3, 4 * q + 4])) for q in range(5)]}
OUT['quad_decomposition'] = res2
for key in ('balanced_within_minus_samequad', 'balanced_within_minus_crossquad'):
    v = res2[key]
    print(f'  {key:<32} {v["mean"]:+.4f} {v["rack_ci"]}  on {v["sockets"]} sockets, '
          f'{100 * v["frac_positive"]:.1f}% positive')
print(f'  share of same-quad pairs among the paper\'s cross-slice pairs: '
      f'{res2["share_samequad_among_cross"]:.3f}')

# ============================================================ 3. strata for the contrast
print('\n=== 3. strata ===')
# stable maps over the whole record (the paper's strict rule) and within the month
EXPECT = 4320
parts = []
for f in sorted((root / 'daily').glob('daily_*.parquet')):
    d = pd.read_parquet(f, columns=['node', 'day', 'n', 'socket', 'core'])
    d['full'] = (d.n > 0.9 * EXPECT).astype(np.int64)
    d['bit'] = np.left_shift(np.int64(1), d.core.values.astype(np.int64))
    t = d.groupby(['node', 'socket', 'day']).agg(active=('core', 'size'), nfull=('full', 'sum'),
                                                   mask=('bit', 'sum'))
    t = t[(t.active == 16) & (t.nfull == 16)].reset_index()
    t['ym'] = f.stem.split('_')[1]
    parts.append(t[['node', 'socket', 'mask', 'ym']])
clean = pd.concat(parts, ignore_index=True)
nuniq = clean.groupby(['node', 'socket'])['mask'].nunique()
nuniq_m = clean[clean.ym == YM].groupby(['node', 'socket'])['mask'].nunique()
key = list(zip(meta.node, meta.socket))
stable_rec = np.array([nuniq.get(k, 0) == 1 for k in key])
stable_mon = np.array([nuniq_m.get(k, 0) == 1 for k in key])
cov = pd.read_parquet(root / 'cov_all' / f'cov_{YM}.parquet')
util = cov.groupby('node')['cpu_user_mean'].mean()
pw = pd.read_parquet(root / 'power_all' / f'pw_{YM}.parquet').groupby(['node', 'socket'])['mean'].mean()
u = np.array([util.get(n, np.nan) for n in meta.node])
p = np.array([pw.get(k, np.nan) for k in key])
s0 = np.array([0 not in a for a in act])                      # slice 0 harvested
strata = {
    'lot': lot, 'socket': np.where(meta.socket.values == 0, 'p0', 'p1'),
    'stable_record': np.where(stable_rec, 'stable', 'changed'),
    'stable_month': np.where(stable_mon, 'stable', 'changed'),
    'utilisation': np.where(np.isnan(u), 'na', np.where(u > np.nanmedian(u), 'high', 'low')),
    'socket_power': np.where(np.isnan(p), 'na', np.where(p > np.nanmedian(p), 'high', 'low')),
    'slice0': np.where(s0, 'slice0 harvested', 'slice0 active'),
}
res3 = {}
for name, lab in strata.items():
    res3[name] = {}
    for h in sorted(set(lab)):
        if h in ('na', 'U'):
            continue
        ix = np.flatnonzero(lab == h)
        w, x, d = pooled_contrast(ix)
        bb = boot(lambda j: (pooled_contrast(ix[j])[2], np.nanmean(dsock[ix[j]]),
                             *[np.nanmean(v[ix[j]]) for v in (s_sq, s_cq)]), rack[ix], B=1000)
        kpp = kpooled(ix)
        res3[name][h] = dict(sockets=int(len(ix)), racks=int(len(np.unique(rack[ix]))),
                             within=r4(w), cross=r4(x), contrast=r4(d),
                             contrast_ci=list(map(r4, tl.ci(bb[:, 0]))),
                             frac_positive=r4(np.nanmean(dsock[ix] > 0)),
                             samequad=r4(kpp[1]), crossquad=r4(kpp[2]),
                             bal_within_minus_samequad=r4(np.nanmean(s_sq[ix])),
                             bal_within_minus_samequad_ci=list(map(r4, tl.ci(bb[:, 2]))),
                             bal_within_minus_crossquad=r4(np.nanmean(s_cq[ix])),
                             bal_within_minus_crossquad_ci=list(map(r4, tl.ci(bb[:, 3]))))
        print(f'  {name:<13} {h:<17} n={len(ix):>4}  within {w:+.4f} cross {x:+.4f} '
              f'contrast {d:+.4f} {res3[name][h]["contrast_ci"]}  +{100 * np.nanmean(dsock[ix] > 0):.0f}%'
              f'  | w-sq {np.nanmean(s_sq[ix]):+.4f}  w-cq {np.nanmean(s_cq[ix]):+.4f}')
OUT['strata_contrast'] = res3

# ============================================================ 4. Fisher z
pz, _ = tl.pool(M5, fisher=True)
Wz = np.array([np.arctanh(v) for v in W]); Xz = np.array([np.arctanh(v) for v in X])
wz, xz = np.tanh(Wz.mean()), np.tanh(Xz.mean())
Sz = tl.slice_similarity(pz)
OUT['fisher_z'] = dict(within=r4(wz), cross=r4(xz), contrast=r4(wz - xz),
                       floorplan_identity=r4(tl.order_score(tl.cells(Sz), np.arange(NP))),
                       max_abs_cell_diff_vs_direct=r4(np.nanmax(np.abs(pz - pooled5)[~np.eye(NC, dtype=bool)])))
sc, o = tl.best_order(Sz)
OUT['fisher_z'].update(exact_optimum=r4(sc), exact_order=tl.canonical(o).tolist(),
                       kendall_to_identity=tl.kendall_distance_reflect(o))
print(f'\n=== 4. Fisher-z pooling ===\n  within {wz:+.4f} cross {xz:+.4f} contrast {wz - xz:+.4f} '
      f'(direct {contrast:+.4f}); fit {OUT["fisher_z"]["floorplan_identity"]:+.3f}; '
      f'max cell change {OUT["fisher_z"]["max_abs_cell_diff_vs_direct"]:.4f}')

# ============================================================ 5. floorplan
print('\n=== 5. floorplan score (Pearson over 66 slice pairs, equal weight) ===')
RAND = np.array([rng.permutation(NP) for _ in range(4000)])
DR = np.array([[abs(tl.order_positions(o)[a] - tl.order_positions(o)[b]) for a, b in tl.SLICE_PAIRS]
               for o in RAND], dtype='float64')
DRz = (DR - DR.mean(1, keepdims=True)) / DR.std(1, keepdims=True)


def null_scores(cv):
    v = (cv - cv.mean()) / cv.std()
    return DRz @ v / len(v)


def floor_summary(S, label, verbose=True):
    cv = tl.cells(S)
    idn = tl.order_score(cv, np.arange(NP))
    nl = null_scores(cv)
    sc, o = tl.best_order(S)
    d = dict(identity=r4(idn), null_mean=r4(nl.mean()), null_sd=r4(nl.std()),
             z=r4((idn - nl.mean()) / nl.std()), p=tl.mc_pvalue(nl, idn), null_B=len(nl),
             exact_optimum=r4(sc), exact_order=tl.canonical(o).tolist(),
             gap=r4(idn - sc), kendall_to_identity=tl.kendall_distance_reflect(o),
             spearman_abs=r4(tl.spearman_abs(o)),
             candidates={k: r4(tl.coords_score(cv, c)) for k, c in tl.CANDIDATES.items()})
    if verbose:
        print(f'  {label:<34} identity {idn:+.3f}  null {nl.mean():+.3f}+/-{nl.std():.3f} '
              f'z={d["z"]:+.2f}  exact optimum {sc:+.3f} {d["exact_order"]}  KD={d["kendall_to_identity"]}')
    return d


res5 = {}
res5['census'] = floor_summary(S5, 'census (every 5th sample)')
# exact rank of the identity among all 12!/2 layouts
Wm = np.nan_to_num(S5)
kb = tl.kbest_linear_arrangements(Wm, 200)
idc = tl.arrangement_cost(Wm, np.arange(NP))
better = sorted({tuple(tl.canonical(o)) for c, o in kb if c < idc - 1e-9})
res5['census']['layouts_strictly_better_than_identity'] = len(better)
res5['census']['layouts_better_list'] = [list(map(int, b)) for b in better]
res5['census']['total_layouts'] = math.factorial(12) // 2
print(f'  identity ranks {len(better) + 1} of 239,500,800 layouts; better: {res5["census"]["layouts_better_list"]}')
res5['native_20s'] = floor_summary(tl.slice_similarity(tl.pool(M1)[0]), 'native cadence (every sample)')
res5['fisher_z'] = floor_summary(Sz, 'Fisher-z pooled')
res5['poststrat_lot_socket'] = floor_summary(
    tl.slice_similarity(tl.poststratified_pool(M5, np.char.add(lot.astype(str), meta.socket.astype(str).values))),
    'post-stratified by lot x socket')
SS = np.stack([tl.cells(tl.slice_similarity(M5[i])) for i in range(ns)])
th = tl.twoway_cells(SS)
S2w = np.full((NP, NP), np.nan)
for (a, b), v in zip(tl.SLICE_PAIRS, th):
    S2w[a, b] = S2w[b, a] = v
res5['twoway_socket_cell'] = floor_summary(S2w, 'two-way (socket + cell) fit')
# the quad (DVFS-domain) confound: drop same-quad slice pairs, or go to quad level
cvx = tl.cells(S5).copy()
sq_cells = [i for i, (a, b) in enumerate(tl.SLICE_PAIRS) if a // 2 == b // 2]
cvx[sq_cells] = np.nan
res5['excluding_samequad_slice_pairs'] = dict(identity=r4(tl.order_score(cvx, np.arange(NP))),
                                              cells=int((~np.isnan(cvx)).sum()))
Q = np.full((6, 6), np.nan)
for a in range(6):
    for b in range(6):
        if a != b:
            Q[a, b] = np.nanmean(pooled5[4 * a:4 * a + 4, 4 * b:4 * b + 4])
qp = list(combinations(range(6), 2))
qcv = np.array([Q[a, b] for a, b in qp])
qsc, qo = tl.best_order(Q)
from itertools import permutations  # noqa: E402
allq = sorted(tl.order_score(qcv, np.array(pp), qp) for pp in permutations(range(6)))
res5['quad_level'] = dict(identity=r4(tl.order_score(qcv, np.arange(6), qp)), exact_optimum=r4(qsc),
                          exact_order=tl.canonical(qo).tolist(), second_best=r4(allq[2]))
print(f'  excluding the 6 same-quad slice pairs: identity {res5["excluding_samequad_slice_pairs"]["identity"]:+.3f}'
      f';  quad level (cross-quad core pairs only): identity {res5["quad_level"]["identity"]:+.3f}, '
      f'exact optimum {qsc:+.3f} {res5["quad_level"]["exact_order"]}, next {allq[2]:+.3f}')

# rack bootstrap of the matrix, the scores and the gap
print(f'  rack bootstrap of the pooled matrix ({B_BOOT} replicates) ...', flush=True)
rb = np.random.default_rng(7)
rows = []
for _ in range(B_BOOT):
    ix = tl.cluster_resample(rack, rb)
    Pm, sup = tl.pool(M5[ix])
    Sb = tl.slice_similarity(Pm)
    if np.isnan(tl.cells(Sb)).any():
        continue
    cv = tl.cells(Sb)
    idn = tl.order_score(cv, np.arange(NP)); nl = null_scores(cv)
    sc, o = tl.best_order(Sb)
    cand = [tl.coords_score(cv, c) for c in tl.CANDIDATES.values()]
    rows.append([idn, sc, idn - sc, tl.kendall_distance_reflect(o), tl.spearman_abs(o),
                 (idn - nl.mean()) / nl.std(), *cand])
rows = np.array(rows)
names = list(tl.CANDIDATES)
lin = names.index('1 x 12 linear')
cand_ci = {k: list(map(r4, tl.ci(rows[:, 6 + j]))) for j, k in enumerate(names)}
res5['rack_bootstrap'] = dict(
    replicates=int(len(rows)), identity_ci=list(map(r4, tl.ci(rows[:, 0]))),
    optimum_ci=list(map(r4, tl.ci(rows[:, 1]))), gap_ci=list(map(r4, tl.ci(rows[:, 2]))),
    frac_identity_is_optimum=r4(np.mean(rows[:, 3] == 0)),
    frac_optimum_within_one_swap=r4(np.mean(rows[:, 3] <= 1)),
    frac_optimum_within_two=r4(np.mean(rows[:, 3] <= 2)),
    max_kendall=int(rows[:, 3].max()), spearman_abs_min=r4(rows[:, 4].min()),
    z_ci=list(map(r4, tl.ci(rows[:, 5]))), z_max=r4(rows[:, 5].max()),
    candidates_ci=cand_ci,
    frac_linear_beats_all_2d=r4(np.mean(np.all(rows[:, 6 + lin, None] < np.delete(rows[:, 6:], lin, 1), 1))))
print(f'  identity {res5["rack_bootstrap"]["identity_ci"]}, optimum {res5["rack_bootstrap"]["optimum_ci"]}, '
      f'gap {res5["rack_bootstrap"]["gap_ci"]}; optimum = identity in '
      f'{100 * res5["rack_bootstrap"]["frac_identity_is_optimum"]:.1f}%, within one swap in '
      f'{100 * res5["rack_bootstrap"]["frac_optimum_within_one_swap"]:.1f}%, max KD {int(rows[:, 3].max())}; '
      f'z {res5["rack_bootstrap"]["z_ci"]}; linear beats every 2-D candidate in '
      f'{100 * res5["rack_bootstrap"]["frac_linear_beats_all_2d"]:.1f}%')

# per-socket floorplan scores: each socket scored on its own 28 slice pairs
m = ~np.isnan(SS)
d_id = np.array([abs(a - b) for a, b in tl.SLICE_PAIRS], dtype='float64')


def socket_scores(d):
    D = np.where(m, d[None, :], np.nan); V = np.where(m, SS, np.nan)
    Dc = D - np.nanmean(D, 1, keepdims=True); Vc = V - np.nanmean(V, 1, keepdims=True)
    return np.nansum(Dc * Vc, 1) / np.sqrt(np.nansum(Dc ** 2, 1) * np.nansum(Vc ** 2, 1))


ps = socket_scores(d_id)
nullsock = np.array([np.nanmean(socket_scores(DR[i])) for i in range(1000)])
bps = boot(lambda ix: np.nanmean(ps[ix]), rack)
rackps = pd.Series(ps).groupby(rack).mean()
res5['per_socket'] = dict(mean=r4(np.nanmean(ps)), median=r4(np.nanmedian(ps)),
                          rack_ci=list(map(r4, tl.ci(bps))), frac_negative=r4(np.nanmean(ps < 0)),
                          null_mean=r4(nullsock.mean()), null_sd=r4(nullsock.std()),
                          z=r4((np.nanmean(ps) - nullsock.mean()) / nullsock.std()),
                          p=tl.mc_pvalue(nullsock, np.nanmean(ps)), null_B=1000,
                          racks_negative=int((rackps < 0).sum()), racks=int(len(rackps)))
print(f'  per-socket scores (28 pairs each): mean {np.nanmean(ps):+.3f} {res5["per_socket"]["rack_ci"]}, '
      f'{100 * np.nanmean(ps < 0):.1f}% of sockets negative, {int((rackps < 0).sum())}/{len(rackps)} racks; '
      f'random-ordering null {nullsock.mean():+.3f}+/-{nullsock.std():.3f}')

# index distance or OS order? The OS numbers the 16 surviving cores compactly, so a coupling
# produced by allocating work in logical-CPU order would follow the RANK distance among active
# slices; the index distance counts harvested slices too. NB: index distance is NOT die distance --
# IBM's chip diagram places the quads in two bands of three (thermal_lib.ibm_core_xy); the test
# with IBM's layout, and with a source model in which harvested cores are dark, is in
# thermal_ibm_layout.py.
SL = np.zeros((ns, NP), dtype=bool)
for i in range(ns):
    SL[i, np.unique(act[i] // 2)] = True
SSS = np.stack([tl.slice_similarity(M5[i]) for i in range(ns)])
rowsd = []
for i in range(ns):
    sl = np.flatnonzero(SL[i])                                    # the 8 active slices
    rk = {s_: r_ for r_, s_ in enumerate(sl)}
    for a, b in combinations(sl, 2):
        rowsd.append((i, b - a, rk[b] - rk[a], int(a // 2 == b // 2), int(rk[a] // 2 == rk[b] // 2),
                      SSS[i, a, b]))
PD = pd.DataFrame(rowsd, columns=['s', 'didx', 'drank', 'samequad', 'samelgroup', 'y'])
PD['rack'] = rack[PD.s.values]
for c in ('didx', 'drank', 'samequad', 'y'):
    PD[c + '_w'] = PD[c] - PD.groupby('s')[c].transform('mean')


def rank_ols(sub, B=1000, seed=3):
    Xw = sub[['didx_w', 'drank_w', 'samequad_w']].values; yw = sub.y_w.values
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    sidx = {r_: np.flatnonzero(sub.rack.values == r_) for r_ in np.unique(sub.rack)}
    labsr = np.array(list(sidx)); rr = np.random.default_rng(seed); bb = []
    for _ in range(B):
        ix = np.concatenate([sidx[r_] for r_ in rr.choice(labsr, len(labsr))])
        bb.append(np.linalg.lstsq(Xw[ix], yw[ix], rcond=None)[0])
    bb = np.array(bb)
    return beta, bb


natural_experiment = tl.natural_experiment


beta, bb = rank_ols(PD)
r2 = {}
yw = PD.y_w.values
for nm, cols_ in (('index', ['didx_w', 'samequad_w']), ('rank', ['drank_w', 'samequad_w']),
                  ('both', ['didx_w', 'drank_w', 'samequad_w'])):
    Z = PD[cols_].values
    e = yw - Z @ np.linalg.lstsq(Z, yw, rcond=None)[0]
    r2[nm] = r4(1 - e.var() / yw.var())
cq = PD[PD.samequad == 0]
g2 = cq.groupby(['didx', 'drank']).y.agg(['mean', 'size'])
ne = natural_experiment(SSS, SL)
rn = np.random.default_rng(4)
ne_b = np.array([natural_experiment(SSS, SL, tl.cluster_resample(rack, rn)) for _ in range(1000)])


# the same two statistics for explicit generative kernels, pushed through the paper's common-mode
# step with every socket's real active set: a stationary kernel in INDEX distance versus one in the
# RANK distance among the active cores (how the operating system numbers them), each with a
# same-quad term, fitted to the 1,948 x 120 observed correlations (Nelder-Mead on 500 sockets).
# Neither is a physical model: index distance is not die distance, and a stationary kernel ignores
# that harvested cores are dark. thermal_ibm_layout.py fits physical source models on IBM's layout.
P16 = np.eye(16) - 1.0 / 16
QUAD = np.arange(NC) // 4


def kernel_pred(theta, idx):
    wI, lI, wR, lR, q, nug = theta
    out = np.empty((len(idx), 120))
    for j, i in enumerate(idx):
        a = act[i]; pi = a.astype(float); pr = np.arange(16.0)
        K = (wI * np.exp(-np.abs(pi[:, None] - pi[None, :]) / lI)
             + wR * np.exp(-np.abs(pr[:, None] - pr[None, :]) / lR)
             + q * (QUAD[a][:, None] == QUAD[a][None, :]) + nug * np.eye(16))
        R = P16 @ K @ P16; sd = np.sqrt(np.diag(R))
        out[j] = (R / np.outer(sd, sd))[tl.IU16]
    return out


def theta_of(p, fam):
    e = np.exp(p)
    if fam == 'index':
        return (e[0], e[1], 0.0, 1.0, e[2], e[3])
    if fam == 'rank':
        return (0.0, 1.0, e[0], e[1], e[2], e[3])
    return tuple(e)


from scipy.optimize import minimize  # noqa: E402
C5u = np.stack([M5[i][np.ix_(act[i], act[i])][tl.IU16] for i in range(ns)])
subk = np.random.default_rng(0).choice(ns, 500, replace=False)
kfit = {}
for fam, p0 in (('index', np.log([1, 3, .5, 1])), ('rank', np.log([1, 3, .5, 1])),
                ('both', np.log([.5, 3, .5, 3, .5, 1]))):
    r = minimize(lambda p: ((C5u[subk] - kernel_pred(theta_of(p, fam), subk)) ** 2).sum(), p0,
                 method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-3, fatol=1e-3))
    th = theta_of(r.x, fam)
    Pk = kernel_pred(th, np.arange(ns))
    Sk = np.stack([tl.slice_similarity(tl.unpack(Pk[i], act[i])) for i in range(ns)])
    kfit[fam] = dict(r2=r4(1 - ((C5u - Pk) ** 2).sum() / ((C5u - C5u.mean()) ** 2).sum()),
                     natural_experiment=r4(natural_experiment(Sk, SL)),
                     slope_index=r4(th[0] / th[1]), slope_rank=r4(th[2] / th[3]),
                     floorplan_identity=r4(tl.order_score(tl.cells(tl.slice_similarity(tl.pool(
                         np.stack([tl.unpack(Pk[i], act[i]) for i in range(ns)]))[0])), np.arange(NP))))
res5['index_vs_rank_distance'] = dict(
    within_socket_ols=dict(beta_index=r4(beta[0]), beta_rank=r4(beta[1]), beta_samequad=r4(beta[2]),
                           ci_index=list(map(r4, tl.ci(bb[:, 0]))), ci_rank=list(map(r4, tl.ci(bb[:, 1]))),
                           ci_samequad=list(map(r4, tl.ci(bb[:, 2])))),
    within_r2=r2, pairs=int(len(PD)),
    crossquad_means={f'didx{int(a)}_drank{int(b)}': [r4(v['mean']), int(v['size'])]
                     for (a, b), v in g2.iterrows() if a <= 4},
    natural_experiment=dict(diff=r4(ne), rack_ci=list(map(r4, tl.ci(ne_b)))),
    kernel_fits=kfit)
print(f'  index vs OS-rank distance (within-socket OLS, {len(PD)} slice pairs): '
      f'beta_index {beta[0]:+.4f} {res5["index_vs_rank_distance"]["within_socket_ols"]["ci_index"]}, '
      f'beta_rank {beta[1]:+.4f} {res5["index_vs_rank_distance"]["within_socket_ols"]["ci_rank"]}; '
      f'R2 index {r2["index"]}, rank {r2["rank"]}, both {r2["both"]}')
for (a, b), v in g2.iterrows():
    if a <= 3:
        print(f'    cross-quad slice pairs, index distance {a}, rank distance {b}: '
              f'{v["mean"]:+.4f} (n={int(v["size"])})')
g3 = PD[PD.drank == 1].groupby(['samequad', 'samelgroup']).y.agg(['mean', 'size'])
res5['index_vs_rank_distance']['rank_adjacent_by_quad_and_logical_group'] = {
    f'samequad{int(a)}_samelgroup{int(b)}': [r4(v['mean']), int(v['size'])] for (a, b), v in g3.iterrows()}
for (a, b), v in g3.iterrows():
    print(f'    rank-adjacent slice pairs, same physical quad={a}, same logical 4-core group={b}: '
          f'{v["mean"]:+.4f} (n={int(v["size"])})')
print(f'  natural experiment, slices (a, a+2) with the slice between harvested minus active: '
      f'{ne:+.4f} {res5["index_vs_rank_distance"]["natural_experiment"]["rack_ci"]}')
for fam, v in kfit.items():
    print(f'    kernel fit [{fam:>5}]: R2 {v["r2"]}, predicted natural experiment {v["natural_experiment"]:+.4f}, '
          f'slope index {v["slope_index"]}, slope rank {v["slope_rank"]}, floorplan {v["floorplan_identity"]}')
res5['index_vs_rank_strata'] = {}
for name, lab in strata.items():
    res5['index_vs_rank_strata'][name] = {}
    for h in sorted(set(lab)):
        if h in ('na', 'U'):
            continue
        socks = np.flatnonzero(lab == h)
        sub = PD[np.isin(PD.s.values, socks)].copy()
        for c in ('didx', 'drank', 'samequad', 'y'):
            sub[c + '_w'] = sub[c] - sub.groupby('s')[c].transform('mean')
        bh, bbh = rank_ols(sub, B=300)
        res5['index_vs_rank_strata'][name][h] = dict(
            beta_index=r4(bh[0]), ci_index=list(map(r4, tl.ci(bbh[:, 0]))),
            beta_rank=r4(bh[1]), ci_rank=list(map(r4, tl.ci(bbh[:, 1]))),
            natural_experiment=r4(natural_experiment(SSS, SL, socks)))
        v = res5['index_vs_rank_strata'][name][h]
        print(f'    {name:<13} {h:<17} beta_index {bh[0]:+.4f} {v["ci_index"]}  beta_rank {bh[1]:+.4f} '
              f'{v["ci_rank"]}  natural exp {v["natural_experiment"]:+.4f}')

# spectrum of the slice similarity matrix (informational: one dominant ordering component)
Sg = S5.copy()
np.fill_diagonal(Sg, 0.0)
ev = np.linalg.eigvalsh(Sg)
res5['rank_one'] = dict(eigvals=[r4(x) for x in ev[::-1][:4]])
print(f'  eigenvalues of the slice similarity matrix: {res5["rank_one"]["eigvals"]}')

# strata for the floorplan
res5['strata'] = {}
for name, lab in strata.items():
    res5['strata'][name] = {}
    for h in sorted(set(lab)):
        if h in ('na', 'U'):
            continue
        ix = lab == h
        Pm, sup = tl.pool(M5[ix])
        Sh = tl.slice_similarity(Pm)
        ssup = np.array([np.sum(~np.isnan(SS[ix][:, j])) for j in range(66)])
        if np.isnan(tl.cells(Sh)).any():
            # a stratum defined by a slice's absence (e.g. slice 0 harvested) never observes it:
            # order the slices it does observe, exactly, and compare with their index order
            pres = np.flatnonzero(~np.isnan(Sh).all(1))
            Sp = Sh[np.ix_(pres, pres)]
            prs = list(combinations(range(len(pres)), 2))
            cvp = np.array([Sp[a, b] for a, b in prs])
            scp, op = tl.best_order(Sp)
            res5['strata'][name][h] = dict(sockets=int(ix.sum()), slices_observed=pres.tolist(),
                                           identity=r4(tl.order_score(cvp, np.arange(len(pres)), prs)),
                                           exact_optimum=r4(scp), exact_order=pres[tl.canonical(op)].tolist(),
                                           kendall_to_identity=tl.kendall_distance_reflect(op),
                                           per_socket_mean=r4(np.nanmean(ps[ix])))
            v = res5['strata'][name][h]
            print(f'    {name:<13} {h:<17} n={ix.sum():>4} slices {pres.min()}-{pres.max()} only  identity '
                  f'{v["identity"]:+.3f}  optimum {scp:+.3f} KD={v["kendall_to_identity"]} '
                  f'per-socket {v["per_socket_mean"]:+.3f}')
            continue
        dd = floor_summary(Sh, f'{name}={h}', verbose=False)
        dd.update(sockets=int(ix.sum()), min_cell_support=int(ssup.min()),
                  per_socket_mean=r4(np.nanmean(ps[ix])), per_socket_frac_negative=r4(np.nanmean(ps[ix] < 0)))
        res5['strata'][name][h] = dd
        print(f'    {name:<13} {h:<17} n={ix.sum():>4} min support {ssup.min():>4}  identity '
              f'{dd["identity"]:+.3f} z={dd["z"]:+.2f}  optimum {dd["exact_optimum"]:+.3f} KD={dd["kendall_to_identity"]} '
              f'per-socket {dd["per_socket_mean"]:+.3f}')
OUT['floorplan'] = res5

# ============================================================ 6. lags
print('\n=== 6. lagged profiles (Table II) ===')
nk = np.stack([kcnt[k] for k in kinds], 1)                      # pairs per socket per kind
res6 = {}
for mode, lab in ((0, 'native_20s_steps'), (1, 'decimated_rows')):
    L = LAG[:, mode]                                            # (ns, 9, 4)
    def prof(ix):
        out = []
        for li in range(len(LAGS)):
            v = L[ix, li, :3]; n = nk[ix]
            okm = ~np.isnan(v)
            s = [np.sum(np.where(okm[:, j], v[:, j] * n[:, j], 0)) / np.sum(np.where(okm[:, j], n[:, j], 0))
                 for j in range(3)]
            cross = (s[1] * n[:, 1].sum() + s[2] * n[:, 2].sum()) / (n[:, 1].sum() + n[:, 2].sum())
            out.append([s[0], cross, s[0] - cross, s[1], s[2], np.nanmean(L[ix, li, 3])])
        return np.array(out)
    P = prof(allix)
    sym = {l: (P[LAGS.index(l)] + P[LAGS.index(-l)]) / 2 for l in (1, 2, 5, 10)}
    sym[0] = P[LAGS.index(0)]
    bl = boot(lambda ix: (lambda Q: [Q[LAGS.index(0), 2], (Q[LAGS.index(10), 2] + Q[LAGS.index(-10), 2]) / 2])(prof(ix)),
              rack, B=500)
    res6[lab] = {str(l): dict(within=r4(sym[l][0]), cross=r4(sym[l][1]), excess=r4(sym[l][2]),
                              samequad=r4(sym[l][3]), crossquad=r4(sym[l][4]), autocorr=r4(sym[l][5]))
                 for l in (0, 1, 2, 5, 10)}
    res6[lab]['excess_lag0_rack_ci'] = list(map(r4, tl.ci(bl[:, 0])))
    res6[lab]['excess_lag10_rack_ci'] = list(map(r4, tl.ci(bl[:, 1])))
    res6[lab]['ratio_lag10_over_lag0'] = r4(sym[10][2] / sym[0][2])
    res6[lab]['ratio_rack_ci'] = list(map(r4, tl.ci(bl[:, 1] / bl[:, 0])))
    print(f'  {lab}:')
    print(f'    {"lag":>4} {"within":>8} {"cross":>8} {"excess":>8} {"samequad":>9} {"crossquad":>9} {"auto":>7}')
    for l in (0, 1, 2, 5, 10):
        s = sym[l]
        print(f'    {l:>4} {s[0]:>8.4f} {s[1]:>8.4f} {s[2]:>8.4f} {s[3]:>9.4f} {s[4]:>9.4f} {s[5]:>7.3f}')
    print(f'    excess ratio lag10/lag0 = {res6[lab]["ratio_lag10_over_lag0"]:.2f} {res6[lab]["ratio_rack_ci"]}')
OUT['lags'] = res6

# ============================================================ 7. support counts
sup_slice = np.zeros((NP, NP), dtype=int)
for a, b in tl.SLICE_PAIRS:
    sup_slice[a, b] = sup_slice[b, a] = int(np.sum(~np.isnan(SS[:, tl.SLICE_PAIRS.index((a, b))])))
for k in range(NP):
    sup_slice[k, k] = int(np.sum([2 * k in a for a in act]))
rows_csv = []
for a in range(NP):
    for b in range(NP):
        if a <= b:
            ix_l = {h: int(np.sum([(2 * a in act[i]) and (2 * b in act[i]) for i in np.flatnonzero(lot == h)]))
                    for h in ('A', 'B')}
            rows_csv.append(dict(slice_a=a, slice_b=b, sockets=int(sup_slice[a, b]),
                                 sockets_lotA=ix_l['A'], sockets_lotB=ix_l['B'],
                                 similarity=None if a == b else r4(S5[a, b])))
pd.DataFrame(rows_csv).to_csv(A / f'thermal_support_counts{"" if YM == "22-08" else "_" + YM}.csv',
                              index=False)
offd = sup_slice[np.triu_indices(NP, 1)]
OUT['support'] = dict(slice_pair_min=int(offd.min()), slice_pair_max=int(offd.max()),
                      slice_pair_median=float(np.median(offd)),
                      argmin=[list(map(int, p)) for p in np.array(tl.SLICE_PAIRS)[offd == offd.min()]],
                      core_pair_min=int(support[~np.eye(NC, dtype=bool)].min()),
                      per_slice=[int(sup_slice[k, k]) for k in range(NP)])
print(f'\n=== 7. support ===\n  sockets per slice pair: min {offd.min()} (pairs {OUT["support"]["argmin"]}), '
      f'median {np.median(offd):.0f}, max {offd.max()}; per slice {OUT["support"]["per_slice"]}')

# ============================================================ 8. what the common-mode step can do
# Population (noise-free) calculation with each socket's real active set: residual correlation
# after the paper's common-mode removal of a purely POSITIVE, local kernel K_ij = exp(-d_ij/l)
# plus independent sensor noise. (i) with d = index distance it produces negative values at long
# range and a strongly negative floorplan score; (ii) with the same kernel over a random physical
# arrangement of the slices the index-ordered score vanishes. The mechanical negativity rescales
# a real ordering; it cannot manufacture one.
print('\n=== 8. common-mode projection check ===')


def kernel_pool(core_pos, ell=3.0, nugget=1.0):
    K = np.exp(-np.abs(core_pos[:, None] - core_pos[None, :]) / ell) + nugget * np.eye(NC)
    mats = np.stack([tl.embed24(tl.projected_corr(K, act[i]), act[i]) for i in range(ns)])
    return tl.pool(mats)[0]


P_id = kernel_pool(np.arange(NC, dtype=float))
curve_id = [np.nanmean([P_id[i, i + d] for i in range(NC - d)]) for d in range(1, NC)]
sc_id = tl.order_score(tl.cells(tl.slice_similarity(P_id)), np.arange(NP))
rp = np.random.default_rng(8)
sc_perm = []
for _ in range(200):
    sp = rp.permutation(NP)                                         # slice -> physical slot
    core_pos = np.array([2 * sp[c // 2] + c % 2 for c in range(NC)], dtype=float)
    sc_perm.append(tl.order_score(tl.cells(tl.slice_similarity(kernel_pool(core_pos))), np.arange(NP)))
sc_perm = np.array(sc_perm)
OUT['projection_check'] = dict(
    kernel='exp(-d/3) + 1.0*I over core positions, real active sets, population residual corr',
    index_kernel_curve_first_negative_distance=int(np.argmax(np.array(curve_id) < 0) + 1),
    index_kernel_min=r4(min(curve_id)), index_kernel_floorplan=r4(sc_id),
    random_arrangement_floorplan_mean=r4(sc_perm.mean()), random_arrangement_floorplan_sd=r4(sc_perm.std()),
    random_arrangement_B=len(sc_perm))
print(f'  positive local kernel in index order -> residual curve turns negative at distance '
      f'{OUT["projection_check"]["index_kernel_curve_first_negative_distance"]} (min {min(curve_id):+.3f}), '
      f'floorplan {sc_id:+.3f}; same kernel over random slice arrangements -> index floorplan '
      f'{sc_perm.mean():+.3f} +/- {sc_perm.std():.3f}')

# ============================================================ 9. Table III, 20 s row, re-derived
# thermal_grid_sensitivity.py drew 90 nodes with default_rng(19) from the same candidate list; its
# 20 s row is a native-resolution census of those sockets, i.e. our every-sample matrices.
if YM == '22-08':
    dd_ = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
    ok_ = dd_.groupby(['node', 'socket'])['core'].nunique()
    cand = sorted(ok_[ok_ == 16].reset_index().groupby('node').size().pipe(lambda s_: s_[s_ == 2]).index)
    sel = {cand[i] for i in np.random.default_rng(19).choice(len(cand), 90, replace=False)}
    ix3 = np.flatnonzero(np.isin(meta.node.values, list(sel)))
    W3, X3, _ = pair_lists(M1[ix3], act[ix3])
    fit3 = tl.order_score(tl.cells(tl.slice_similarity(tl.pool(M1[ix3])[0])), np.arange(NP))
    OUT['table3_20s_row'] = dict(sockets=int(len(ix3)), within=r4(W3.mean()), cross=r4(X3.mean()),
                                 fit=r4(fit3), rows_per_socket_median=int(np.median(meta.rows1.values[ix3])))
    print(f'\n=== 9. Table III 20 s row from per-socket native matrices ===\n  {len(ix3)} sockets: '
          f'within {W3.mean():+.4f} cross {X3.mean():+.4f} fit {fit3:+.3f} rows/socket '
          f'{int(np.median(meta.rows1.values[ix3]))}')

json.dump(OUT, open(A / f'thermal_socket_stats{"" if YM == "22-08" else "_" + YM}.json', 'w'), indent=1)
print(f'\nwrote analysis/thermal_socket_stats{"" if YM == "22-08" else "_" + YM}.json and '
      f'thermal_support_counts{"" if YM == "22-08" else "_" + YM}.csv')
