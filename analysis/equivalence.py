"""Is the harvest map's association with socket power small enough to ignore? (Sec. IV-L, Fig. 15)

An equivalence test, not a failure to reject: a 90% interval inside +/-margin is the
two-one-sided-tests (TOST) criterion at alpha = 0.05.

DESIGN
  Outcome   dP = p0 - p1, the within-node difference in daily mean CPU-socket power (ipmi
            pX_power), steady state (from 2020-06-01). Pairing removes chassis, inlet air, PSU and
            everything else the two sockets of a node share.
  Exposure  the p0 - p1 difference in one summary of the map: mean active-slice index, number of
            index-adjacent active pairs, active slices in the low half (index < 6).
  Map       the CONTEMPORANEOUS map. Each node-day carries, for each socket, the configuration
            epoch containing it: a run of clean days with the identical map, the day itself clean
            or bracketed by clean days carrying that same map. Node-days in a transition gap,
            before a socket's first or after its last clean day, or on a day whose reporting set
            is not that map (a sub-16 guard day, a mixed reboot day) are dropped. The map and the
            power of a socket are read under the same p0/p1 tag on the same day, so a relabel can
            no longer pair one die's power with the other die's map. (The previous version gave
            every day the socket's full-record modal map.)
  Adjust    node utilisation (cpu_user), queue depth (load_one), inlet air (ambient), month.
            `cpu_speed` is the nominal clock, a constant 3800 MHz in this sample; the previous
            version listed it as a "core frequency" adjustment, but a constant adjusts for nothing
            and is dropped here. All of these are node-level: they cannot measure a p0/p1
            workload asymmetry, which is handled under WORKLOAD below.
  Uncert.   rack-clustered bootstrap (49 racks, B = 4000), percentile 90% intervals; CR1
            cluster-robust standard errors printed alongside as a check.

ONE FEATURE PER MODEL. The three summaries describe one map and are strongly collinear
(corr(d meanidx, d lowhalf) = -0.89). The previous version fitted them jointly and multiplied each
PARTIAL slope by that feature's MARGINAL range: the effect of moving mean index across its whole
observed range while holding the low-half count fixed, a contrast no pair of real maps realises.
Each summary's effect is therefore estimated alone, as Sec. IV-L describes it. The joint fit is
kept as a labelled legacy row (it reproduces the +7.9 W the paper used to print), with its
variance inflation and its whole-map contribution over realised map pairs.

EFFECT SCALES. Slope in W per unit of the feature difference; the effect over the full observed
range of that difference (the TOST quantity, and the most conservative); per SD; per IQR.

WORKLOAD. The map is set at boot, before any job runs, and the scheduler sees 16 identical cores
per socket, never the map. A socket-level workload asymmetry therefore biases the slope only through
its correlation with the map difference. Checked four ways:
  (a) SLURM placement. The job table's per-core layout gives the core-seconds allocated on each
      socket (equivalence_alloc.py), populated for jobs ending 2020-05..2020-10-13 only. On the
      days of that window whose job records are complete it is tested for correlation with
      the map difference, and the slope is re-estimated adjusting for it. The correlation is
      tiny (|r| <= 0.021) but placement moves power so much (R^2 0.40) that adjusting still
      shifts each estimate about 1 W AWAY from zero: in that window, omitting placement biases
      toward zero. It also shows that the p0 - p1 power asymmetry the paper used as a
      calibration anchor reflects scheduler placement, not socket position.
  (b) Idle node-days (cpu_user < 0.5 %, load_one < 0.5): almost no workload to distribute, hence
      almost no workload asymmetry to confound; validated against placement where both are
      recorded (allocated fraction 0.07 against 0.51 on other node-days).
  (c) Balance. Correlation of the map difference with the node-level covariates, and its mean.
  (d) I/O- and memory-subsystem power (six months, power/) were meant as negative controls, but
      they barely track CPU placement and are themselves associated with the map, so they are
      reported as co-outcomes, and never adjusted for in a headline number.
Socket temperature is a consequence of socket power and is deliberately NOT adjusted for.

MARGIN. +/-5 W per socket is the headline; 2 W and 10 W and, per feature, the smallest margin the
interval clears are reported, so a reader can apply their own.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import equivalence_core as ec

warnings.filterwarnings('ignore', message='.*PeriodArray.*')
root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
MARGIN_W = 5.0
B = 4000
SEED = 37
FEAT = list(ec.FEATURES)
NODE_COV = ['cpu_user_mean', 'cpu_speed_mean', 'load_one_mean', 'ambient_mean']
DAY0 = pd.Timestamp('2020-01-01', tz='UTC')
# socket placement: layouts exist for jobs that ENDED by 2020-10-13 05:53 UTC. A job overlapping
# day d ends on d, d+1 or d+2 (runs are <= 34.8 h), so d is usable only if the job table is
# complete on all three: at least COMPLETE_FRAC of the window's median number of job ends per day.
LAY_FROM, LAY_LAST_END = pd.Timestamp('2020-06-01', tz='UTC'), pd.Timestamp('2020-10-13', tz='UTC')
COMPLETE_FRAC = 0.2
T0 = time.time()


def dayno(s):
    return ((s - DAY0) // pd.Timedelta('1D')).astype(np.int64).to_numpy()


def say(msg=''):
    print(msg, flush=True)


# ============================================================== 1. maps
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
assert not df.duplicated(['node', 'socket', 'day', 'core']).any()
df['bit'] = np.int64(1) << df['core'].astype(np.int64)
g = df.groupby(['node', 'socket', 'day'])
t = pd.DataFrame({'active': g.size(), 'rcode': g['bit'].sum()})
t['nfull'] = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size()
t = t.fillna({'nfull': 0}).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)].rename(columns={'rcode': 'code'})
clean = clean[['node', 'socket', 'day', 'code']].reset_index(drop=True)
del df, g

FEATS = {c: ec.slice_features(c) for c in clean.code.unique()}
assert all(len(ec.active_slices(c)) == 8 for c in FEATS)
modal = (clean.groupby(['node', 'socket'])['code']
              .agg(lambda s: s.value_counts().index[0]).unstack('socket'))
nmaps = clean.groupby(['node', 'socket'])['code'].nunique().unstack('socket')
stable_nodes = set(nmaps.index[(nmaps == 1).all(axis=1)])
say(f'clean socket-days {len(clean):,}; sockets {int(nmaps.notna().sum().sum()):,}; strictly '
    f'stable sockets {int((nmaps == 1).sum().sum()):,}; nodes with both sockets stable '
    f'{len(stable_nodes)}')

# ============================================================== 2. power + node covariates
pw = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power_all').glob('*.parquet'))],
               ignore_index=True)
pw['day'] = pd.to_datetime(pw['day'], utc=True)
cov = pd.concat([pd.read_parquet(f) for f in sorted((root / 'cov_all').glob('cov_*.parquet'))],
                ignore_index=True)
cov['day'] = pd.to_datetime(cov['day'], utc=True)
cov = cov[['node', 'day'] + NODE_COV].dropna()          # sample definition as published
P = pw[pw.day >= CUT].merge(cov, on=['node', 'day'], how='inner')
Wp = P.pivot_table(index=['node', 'day'], columns='socket', values=['mean', 'count'])
Wp.columns = [f'{a}{b}' for a, b in Wp.columns]
Wp = Wp.rename(columns={'mean0': 'p0', 'mean1': 'p1', 'count0': 'c0', 'count1': 'c1'})
Wp = Wp.dropna(subset=['p0', 'p1'])
cv = P.drop_duplicates(['node', 'day']).set_index(['node', 'day'])[NODE_COV]
W = Wp.join(cv).dropna(subset=NODE_COV).reset_index()
W = W[W.node.isin(modal.dropna().index)].reset_index(drop=True)
W['dP'] = W.p0 - W.p1
W['month'] = W.day.dt.strftime('%Y-%m')
W['rack'] = W.node // 20
COV = [c for c in NODE_COV if W[c].std() > 0]
DROPPED = [c for c in NODE_COV if c not in COV]
say(f'paired node-days with power and covariates: {len(W):,} on {W.node.nunique()} nodes')
for c in DROPPED:
    say(f'  {c} is constant ({W[c].iloc[0]:g}) across the sample -> not an adjustment; dropped')

# ============================================================== 3. map per node-day
W['m0'] = modal.loc[W.node, 0].to_numpy().astype(np.int64)
W['m1'] = modal.loc[W.node, 1].to_numpy().astype(np.int64)
ckey = (2 * clean.node + clean.socket).to_numpy()
cday = dayno(clean.day)
qday = dayno(W.day)
tt = t.set_index(['node', 'socket', 'day'])
for s in (0, 1):
    code, how = ec.epoch_lookup(ckey, cday, clean.code.to_numpy(), 2 * W.node.to_numpy() + s, qday)
    # the socket's own reporting set that day, where it reported at all, must BE that map
    idx = pd.MultiIndex.from_arrays([W.node, np.full(len(W), s), W.day])
    rep = tt.reindex(idx)
    bad = rep.active.notna().to_numpy() & (
        (rep.active.to_numpy() != 16) |
        (rep.rcode.fillna(0).to_numpy().astype(np.int64) != code))
    how[bad & (how == 2)] = 0
    W[f'e{s}'], W[f'h{s}'] = code, how

# independent cross-check of the epoch join: previous/next clean day by merge_asof
chk = 0
for s in (0, 1):
    cs = clean[clean.socket == s][['node', 'day', 'code']].sort_values('day')
    q = W[['node', 'day', f'e{s}', f'h{s}']].reset_index().sort_values('day')
    prv = pd.merge_asof(q, cs, on='day', by='node', direction='backward')
    nxt = pd.merge_asof(q, cs.rename(columns={'code': 'ncode'}), on='day', by='node',
                        direction='forward')
    ok = prv[f'h{s}'] > 0
    agree = (prv.code[ok].to_numpy() == prv[f'e{s}'][ok].to_numpy()) & \
            (nxt.ncode[ok].to_numpy() == prv[f'e{s}'][ok].to_numpy())
    chk += int((~agree).sum())
say(f'epoch join cross-check (merge_asof): {chk} node-socket-days disagree')
assert chk == 0

EPOCH = (W.h0 > 0) & (W.h1 > 0)
# idle node-days: nothing to distribute between the sockets, so no workload asymmetry at all;
# validated against SLURM placement below (allocated fraction and |p0 - p1 placement|)
IDLE = (W.cpu_user_mean < 0.5) & (W.load_one_mean < 0.5)
IDLE1 = (W.cpu_user_mean < 1.0) & (W.load_one_mean < 1.0)
DESIGNS = {
    'modal': (W.index == W.index, ('m0', 'm1')),               # legacy: full-record modal map
    'epoch': (EPOCH, ('e0', 'e1')),                           # headline
    'sameday': ((W.h0 == 1) & (W.h1 == 1), ('e0', 'e1')),
    'stable': (EPOCH & W.node.isin(stable_nodes), ('e0', 'e1')),
    'epoch_cov90': (EPOCH & (W.c0 > 0.9 * EXPECT) & (W.c1 > 0.9 * EXPECT), ('e0', 'e1')),
    'idle': (EPOCH & IDLE, ('e0', 'e1')),
    'idle_1pct': (EPOCH & IDLE1, ('e0', 'e1')),
}
ep = W[EPOCH]
nonmodal = int(((ep.e0 != ep.m0) | (ep.e1 != ep.m1)).sum())
mirrored = int(((ep.e0 == ep.m1) & (ep.e1 == ep.m0) & (ep.m0 != ep.m1)).sum())
say(f'epoch design keeps {int(EPOCH.sum()):,} node-days; {nonmodal:,} of them carry a map other '
    f'than the modal one, {mirrored:,} the mirror image (a relabel) that the legacy join '
    f'sign-flipped')

FA = np.array([FEATS[c] for c in sorted(FEATS)])
FIDX = {c: i for i, c in enumerate(sorted(FEATS))}


def dfeat(sub, cols, f):
    j = FEAT.index(f)
    a = np.array([FIDX[c] for c in sub[cols[0]]])
    b = np.array([FIDX[c] for c in sub[cols[1]]])
    return FA[a, j] - FA[b, j]


def design(sub, cols, feats, adjust, extra=()):
    Xc, names = [np.ones(len(sub))], ['const']
    for f in feats:
        Xc.append(dfeat(sub, cols, f))
        names.append(f)
    for c in extra:
        Xc.append(sub[c].to_numpy(float))
        names.append(c)
    if adjust:
        for c in COV:
            v = sub[c].to_numpy(float)
            Xc.append((v - v.mean()) / v.std())
            names.append(c)
        for m in sorted(sub.month.unique())[1:]:
            Xc.append((sub.month == m).to_numpy(float))
            names.append('M' + m)
    return np.column_stack(Xc), names


REF = None      # spreads of the feature differences in the headline sample, set below


def summarise(label, model, feats, X, names, b, bs, se, n, nodes):
    """Effects on ONE fixed scale for every design: slope x the spread of the feature difference
    in the headline (epoch) sample, so that designs differ only through their slopes. The
    sample's own spread is kept as spread_own."""
    rows = []
    for f in feats:
        j = names.index(f)
        sp = REF[f] if REF is not None else ec.scales(X[:, j])
        lo_s, hi_s = ec.percentile_ci(bs[:, j])
        r = {'design': label, 'model': model, 'feature': f, 'n': int(n), 'nodes': int(nodes),
             'slope': float(b[j]), 'slope_lo': lo_s, 'slope_hi': hi_s, 'se_cr1': float(se[j]),
             'spread': sp, 'spread_own': ec.scales(X[:, j])}
        for s in ('range', 'sd', 'iqr'):
            r[s] = {'eff': float(b[j] * sp[s]), 'lo': lo_s * sp[s], 'hi': hi_s * sp[s]}
        r['eq_bound'] = ec.eq_bound(r['range']['lo'], r['range']['hi'])
        r['tost'] = {f'{m:g}': ec.tost(r['range']['lo'], r['range']['hi'], m)
                     for m in ec.MARGINS_W}
        r['equivalent'] = r['tost'][f'{MARGIN_W:g}']
        rows.append(r)
    return rows


def fit(sub, cols, feats, adjust, extra=(), seed=SEED):
    X, names = design(sub, cols, feats, adjust, extra)
    y = sub.dP.to_numpy(float)
    b, bs, nsing = ec.ols_cluster_boot(X, y, sub.rack.to_numpy(), B=B, seed=seed)
    _, se = ec.cluster_robust_se(X, y, sub.rack.to_numpy())
    return X, names, b, bs, se, nsing


REF = {f: ec.scales(dfeat(W[EPOCH], ('e0', 'e1'), f)) for f in FEAT}
say('spread of the p0 - p1 feature difference in the headline sample (the effect scales): '
    + '; '.join(f'{f} range {REF[f]["range"]:g}, sd {REF[f]["sd"]:.2f}, iqr {REF[f]["iqr"]:g}'
                for f in FEAT))
RES = {'margin': MARGIN_W, 'margins': list(ec.MARGINS_W), 'B': B, 'seed': SEED,
       'reference_spread': REF,
       'covariates': COV, 'dropped_constant_covariates': DROPPED,
       'headline': {'design': 'epoch', 'model': 'single-adjusted'},
       'rows': [], 'joint': {}, 'sample': {}}

say(f'\n=== effect of the map on the within-node power difference dP = p0 - p1 ===')
say(f'    90% rack-bootstrap intervals (B={B}); TOST verdicts at +/-2, 5, 10 W on the full-range '
    f'effect; "bound" = smallest margin the interval clears')
say(f'  {"design":<11} {"model":<17} {"feature":<8} {"n":>8} {"slope W/unit":>13} '
    f'{"range effect [90% CI]":>26} {"per SD":>7} {"per IQR":>8} {"bound":>6}  2/5/10 W')
for label, (mask, cols) in DESIGNS.items():
    sub = W[mask]
    RES['sample'][label] = {'node_days': int(len(sub)), 'nodes': int(sub.node.nunique())}
    fits = [('single-adjusted', True, [f]) for f in FEAT]
    fits += [('single-unadjusted', False, [f]) for f in FEAT]
    fits += [('joint-adjusted', True, FEAT)]
    for model, adjust, feats in fits:
        X, names, b, bs, se, nsing = fit(sub, cols, feats, adjust)
        rows = summarise(label, model, feats, X, names, b, bs, se, len(sub), sub.node.nunique())
        RES['rows'] += rows
        for r in rows:
            v = ''.join('E' if r['tost'][f'{m:g}'] else '.' for m in ec.MARGINS_W)
            say(f'  {label:<11} {model:<17} {r["feature"]:<8} {r["n"]:>8,} {r["slope"]:>+13.3f} '
                f'{r["range"]["eff"]:>+8.2f} [{r["range"]["lo"]:+6.2f},{r["range"]["hi"]:+6.2f}] '
                f'{r["sd"]["eff"]:>+7.2f} {r["iqr"]["eff"]:>+8.2f} {r["eq_bound"]:>6.2f}  {v}')
        if model == 'joint-adjusted':
            J = [names.index(f) for f in FEAT]
            D = X[:, J]
            U, cnt = np.unique(D, axis=0, return_counts=True)

            def spread(beta):
                c = U @ beta
                q05, q95 = ec.wquantile(c, cnt, [0.05, 0.95])
                return q95 - q05, c.max() - c.min()
            s0 = spread(b[J])
            sb = np.array([spread(bb[J]) for bb in bs])
            RES['joint'][label] = {
                'vif': dict(zip(FEAT, ec.vif(D))),
                'corr': {f'{FEAT[i]}~{FEAT[j]}': float(np.corrcoef(D[:, i], D[:, j])[0, 1])
                         for i in range(3) for j in range(i + 1, 3)},
                'map_p5_p95': float(s0[0]), 'map_p5_p95_hi': float(np.percentile(sb[:, 0], 95)),
                'map_range': float(s0[1]), 'map_range_hi': float(np.percentile(sb[:, 1], 95)),
                'singular_replicates': int(nsing)}
            jj = RES['joint'][label]
            vf = ', '.join(f'{k} {v:.1f}' for k, v in jj['vif'].items())
            say(f'  {label:<11} joint: VIF {vf}'
                + f'; whole-map contribution P5-P95 {jj["map_p5_p95"]:.2f} W (95% upper '
                f'{jj["map_p5_p95_hi"]:.2f}), full range {jj["map_range"]:.2f} W (upper '
                f'{jj["map_range_hi"]:.2f})')
    say()

RES['sample']['epoch'].update({'nonmodal_node_days': nonmodal, 'mirrored_vs_modal': mirrored,
                               'own_date_mismatch': chk})

# idle days weight nodes by how often they sit idle (one node has over 600); refit with one
# observation per node-epoch so that no handful of drained nodes carries the estimate
Gi = (W[EPOCH & IDLE].groupby(['node', 'e0', 'e1'])
      .agg(dP=('dP', 'mean'), rack=('rack', 'first'), ndays=('dP', 'size')).reset_index())
RES['sample']['idle_nodeweighted'] = {'node_days': int(len(Gi)), 'nodes': int(Gi.node.nunique()),
                                      'max_idle_days_one_node': int(
                                          W[EPOCH & IDLE].groupby('node').size().max())}
for f in FEAT:
    X, names, b, bs, se, _ = fit(Gi, ('e0', 'e1'), [f], False)
    r = summarise('idle_nodeweighted', 'single-unadjusted', [f], X, names, b, bs, se, len(Gi),
                  Gi.node.nunique())[0]
    RES['rows'].append(r)
    say(f'  idle, one row per node-epoch ({len(Gi):,}): {f:<8} {r["range"]["eff"]:+.2f} '
        f'[{r["range"]["lo"]:+.2f},{r["range"]["hi"]:+.2f}]  bound {r["eq_bound"]:.2f}')

# ============================================================== 4. calibration
E = W[EPOCH]
cal = {'mean_socket_W': float(0.5 * (E.p0.mean() + E.p1.mean())),
       'mean_dP_W': float(E.dP.mean()), 'sd_dP_W': float(E.dP.std())}
Ei = W[EPOCH & IDLE]
node_mean = Ei.groupby('node').dP.transform('mean')
cal.update({'idle_node_days': int(len(Ei)),
            'idle_mean_socket_W': float(0.5 * (Ei.p0 + Ei.p1).mean()),
            'idle_mean_dP_W': float(Ei.dP.mean()), 'idle_sd_dP_W': float(Ei.dP.std()),
            'idle_between_node_share': float(node_mean.var() / Ei.dP.var()),
            'idle_sd_node_mean_dP_W': float(Ei.groupby('node').dP.mean().std())})
RES['calibration'] = cal
say(f'calibration (epoch sample): mean socket power {cal["mean_socket_W"]:.1f} W; margin '
    f'{MARGIN_W:.0f} W = {100 * MARGIN_W / cal["mean_socket_W"]:.1f}% of it; mean p0-p1 '
    f'{cal["mean_dP_W"]:+.2f} W; sd of p0-p1 {cal["sd_dP_W"]:.2f} W')
say(f'  idle node-days ({cal["idle_node_days"]:,}): mean socket {cal["idle_mean_socket_W"]:.1f} W, '
    f'mean p0-p1 {cal["idle_mean_dP_W"]:+.2f} W; {100 * cal["idle_between_node_share"]:.0f}% of '
    f'the p0-p1 variance is between nodes (sd of node means {cal["idle_sd_node_mean_dP_W"]:.2f} W):'
    f' a static die/board difference, present with no workload at all')

# balance of the map difference against the node-level covariates
bal = {}
for f in FEAT:
    d = dfeat(E, ('e0', 'e1'), f)
    bal[f] = {c: float(np.corrcoef(d, E[c])[0, 1]) for c in COV}
    Xm = np.ones((len(E), 1))
    bm, bsm, _ = ec.ols_cluster_boot(Xm, d, E.rack.to_numpy(), B=B, seed=SEED)
    bal[f]['mean'] = float(bm[0])
    bal[f]['mean_lo'], bal[f]['mean_hi'] = ec.percentile_ci(bsm[:, 0])
RES['balance_node_covariates'] = bal
mx = max(abs(v) for f in FEAT for c, v in bal[f].items() if c in COV)
say(f'balance: max |corr(map difference, node covariate)| = {mx:.3f}; mean map difference '
    + ', '.join(f'{f} {bal[f]["mean"]:+.3f} [{bal[f]["mean_lo"]:+.3f},{bal[f]["mean_hi"]:+.3f}]'
                for f in FEAT))

# ============================================================== 5a. SLURM socket placement
A = pd.read_parquet(root / 'analysis' / 'equivalence_alloc.parquet')
ends = pd.Series(json.load(open(root / 'analysis' / 'equivalence_alloc_days.json')))
ends.index = pd.to_datetime(ends.index).tz_localize('UTC')
span = pd.date_range(LAY_FROM - pd.Timedelta('2D'), LAY_LAST_END + pd.Timedelta('2D'), tz='UTC')
ends = ends.reindex(span, fill_value=0)
ok_end = ends >= COMPLETE_FRAC * ends[LAY_FROM:LAY_LAST_END].median()
ok_end[ok_end.index >= LAY_LAST_END] = False                  # the table stops at 05:53 that day
say('')
usable = [d for d in span if d >= LAY_FROM and
          all(ok_end.get(d + pd.Timedelta(days=k), False) for k in (0, 1, 2))]
say(f'  job table complete for placement on {len(usable)} days; incomplete job-end days: '
    + ', '.join(str(d.date()) for d in ok_end.index[~ok_end.to_numpy()]
                if LAY_FROM - pd.Timedelta('2D') <= d < LAY_LAST_END))
L = E[E.day.isin(usable)].merge(A, on=['node', 'day'], how='left')
for c in ['core_s0', 'core_s1', 'gpu_s', 'nolay_s', 'njobs']:
    L[c] = L[c].fillna(0)
L = L[L.nolay_s == 0].copy()
cap = 16 * 86400.0
L['a0'] = (L.core_s0 / cap).clip(upper=1)
L['a1'] = (L.core_s1 / cap).clip(upper=1)
L['dA'] = L.a0 - L.a1
L['aN'] = 0.5 * (L.a0 + L.a1)
say(f'=== socket-level workload: SLURM core placement, {L.day.min().date()}..'
    f'{L.day.max().date()} ===')
say(f'  {len(L):,} epoch node-days on {L.node.nunique()} nodes; {100 * (L.njobs > 0).mean():.0f}% '
    f'ran a job; mean allocated fraction p0 {L.a0.mean():.3f}, p1 {L.a1.mean():.3f}')
Xa = np.column_stack([np.ones(len(L)), L.dA])
ba, bsa, _ = ec.ols_cluster_boot(Xa, L.dP.to_numpy(), L.rack.to_numpy(), B=B, seed=SEED)
r_pa = float(np.corrcoef(L.dP, L.dA)[0, 1])
wl = {'node_days': int(len(L)), 'nodes': int(L.node.nunique()),
      'window': [str(L.day.min().date()), str(L.day.max().date())],
      'mean_a0': float(L.a0.mean()), 'mean_a1': float(L.a1.mean()), 'mean_dA': float(L.dA.mean()),
      'mean_dP_W': float(L.dP.mean()), 'corr_dP_dA': r_pa, 'r2_dP_dA': r_pa ** 2,
      'slope_W_per_socket': float(ba[1]), 'slope_ci': ec.percentile_ci(bsa[:, 1]),
      'slot_W_at_equal_placement': float(ba[0]), 'slot_ci': ec.percentile_ci(bsa[:, 0])}
say(f'  placement explains the power asymmetry: corr(dP, dA) = {r_pa:+.3f} (R^2 {r_pa ** 2:.2f}); '
    f'dP = {ba[0]:+.2f} W [{wl["slot_ci"][0]:+.2f},{wl["slot_ci"][1]:+.2f}] + '
    f'{ba[1]:.1f} W x dA; mean dP here {L.dP.mean():+.2f} W, of which placement '
    f'{ba[1] * L.dA.mean():+.2f} W')
li = (L.cpu_user_mean < 0.5) & (L.load_one_mean < 0.5)
wl['idle_check'] = {'node_days': int(li.sum()), 'alloc_idle': float(L.aN[li].mean()),
                    'alloc_busy': float(L.aN[~li].mean()),
                    'absdA_idle': float(L.dA[li].abs().mean()),
                    'absdA_busy': float(L.dA[~li].abs().mean())}
ic = wl['idle_check']
say(f'  idle definition checked against placement: allocated fraction {ic["alloc_idle"]:.3f} on '
    f'idle vs {ic["alloc_busy"]:.3f} on other node-days; mean |dA| {ic["absdA_idle"]:.3f} vs '
    f'{ic["absdA_busy"]:.3f}')
wl['features'] = {}
for f in FEAT:
    d = dfeat(L, ('e0', 'e1'), f)
    rng_f = REF[f]['range']
    Xb = np.column_stack([np.ones(len(L)), d])
    bb, bsb, _ = ec.ols_cluster_boot(Xb, L.dA.to_numpy(), L.rack.to_numpy(), B=B, seed=SEED)
    lo_b, hi_b = ec.percentile_ci(bsb[:, 1])
    rec = {'corr_dA': float(np.corrcoef(d, L.dA)[0, 1]),
           'dA_per_unit': float(bb[1]), 'dA_per_unit_ci': [lo_b, hi_b],
           'implied_bias_range_W': float(ba[1] * bb[1] * rng_f),
           'implied_bias_range_ci_W': [float(ba[1] * lo_b * rng_f), float(ba[1] * hi_b * rng_f)]}
    for tag, extra in (('without', ()), ('with', ('dA', 'aN'))):
        X, names, b, bs, se, _ = fit(L, ('e0', 'e1'), [f], True, extra)
        r = summarise('placement_' + tag, 'single-adjusted', [f], X, names, b, bs, se, len(L),
                      L.node.nunique())[0]
        rec[tag] = {'eff': r['range']['eff'], 'lo': r['range']['lo'], 'hi': r['range']['hi'],
                    'slope': r['slope']}
        RES['rows'].append(r)
    wl['features'][f] = rec
    say(f'  {f:<8} corr(map diff, dA) {rec["corr_dA"]:+.3f}; implied bias over range '
        f'{rec["implied_bias_range_W"]:+.2f} W [{rec["implied_bias_range_ci_W"][0]:+.2f},'
        f'{rec["implied_bias_range_ci_W"][1]:+.2f}]; range effect without placement '
        f'{rec["without"]["eff"]:+.2f} [{rec["without"]["lo"]:+.2f},{rec["without"]["hi"]:+.2f}]'
        f' -> with {rec["with"]["eff"]:+.2f} [{rec["with"]["lo"]:+.2f},{rec["with"]["hi"]:+.2f}]')
RES['workload_placement'] = wl

# ============================================================== 5b. I/O and memory power
# Candidate negative controls: per-socket I/O- and memory-subsystem power respond to workload but
# have no obvious path from which cores are disabled. They turn out to be poor workload proxies
# (the I/O difference barely tracks CPU placement where both are recorded, July 2020) and are
# themselves associated with the map, so they are reported as co-outcomes, not as controls, and
# dP is NOT adjusted for them in any headline number: conditioning on a co-outcome of the die
# would remove or distort the effect being estimated.
pp = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power').glob('power_*.parquet'))],
               ignore_index=True)
pp['day'] = pd.to_datetime(pp['day'], utc=True)
pp = pp[pp.metric.isin(['pX_io_power', 'pX_mem_power']) & pp.socket.isin([0, 1])]
pv = pp.pivot_table(index=['node', 'day'], columns=['metric', 'socket'], values='mean')
pv = pd.DataFrame({'d_io': pv[('pX_io_power', 0)] - pv[('pX_io_power', 1)],
                   'd_mem': pv[('pX_mem_power', 0)] - pv[('pX_mem_power', 1)]}).dropna()
N6 = E.merge(pv.reset_index(), on=['node', 'day'], how='inner')
say(f'\n=== co-outcomes: p0-p1 I/O and memory-subsystem power (six months) ===')
say(f'  {len(N6):,} epoch node-days on {N6.node.nunique()} nodes; mean d_io '
    f'{N6.d_io.mean():+.2f} W, '
    f'd_mem {N6.d_mem.mean():+.2f} W; corr(dP, d_io) {np.corrcoef(N6.dP, N6.d_io)[0, 1]:+.2f}, '
    f'corr(dP, d_mem) {np.corrcoef(N6.dP, N6.d_mem)[0, 1]:+.2f}')
J7 = L.merge(pv.reset_index(), on=['node', 'day'], how='inner')
nc = {'node_days': int(len(N6)), 'nodes': int(N6.node.nunique()), 'features': {},
      'overlap_with_placement': {'node_days': int(len(J7)),
                                 'corr_dio_dA': float(np.corrcoef(J7.d_io, J7.dA)[0, 1]),
                                 'corr_dmem_dA': float(np.corrcoef(J7.d_mem, J7.dA)[0, 1]),
                                 'corr_dP_dA': float(np.corrcoef(J7.dP, J7.dA)[0, 1])}}
ov = nc['overlap_with_placement']
say(f'  where placement is also recorded ({ov["node_days"]:,} node-days): corr with dA is '
    f'{ov["corr_dio_dA"]:+.2f} for d_io, {ov["corr_dmem_dA"]:+.2f} for d_mem, '
    f'{ov["corr_dP_dA"]:+.2f} for dP -> I/O power is not a CPU-workload proxy')
for f in FEAT:
    rec = {}
    for outcome in ('d_io', 'd_mem', 'dP'):
        S6 = N6.assign(dP=N6[outcome])
        X, names, b, bs, se, _ = fit(S6, ('e0', 'e1'), [f], True)
        r = summarise('negctl_' + outcome, 'single-adjusted', [f], X, names, b, bs, se,
                      len(S6), S6.node.nunique())[0]
        rec[outcome] = {'eff': r['range']['eff'], 'lo': r['range']['lo'], 'hi': r['range']['hi']}
    # over-adjusted (conditions on co-outcomes); kept for transparency, never a headline
    X, names, b, bs, se, _ = fit(N6, ('e0', 'e1'), [f], True, ('d_io', 'd_mem'))
    r = summarise('coout_dP_adj_io_mem', 'single-adjusted', [f], X, names, b, bs, se,
                  len(N6), N6.node.nunique())[0]
    rec['dP_overadjusted_io_mem'] = {'eff': r['range']['eff'], 'lo': r['range']['lo'],
                                     'hi': r['range']['hi']}
    nc['features'][f] = rec
    say(f'  {f:<8} range effect on d_io {rec["d_io"]["eff"]:+.2f} [{rec["d_io"]["lo"]:+.2f},'
        f'{rec["d_io"]["hi"]:+.2f}] W, on d_mem {rec["d_mem"]["eff"]:+.2f} '
        f'[{rec["d_mem"]["lo"]:+.2f},{rec["d_mem"]["hi"]:+.2f}] W; on dP {rec["dP"]["eff"]:+.2f} '
        f'(conditioning on both co-outcomes: {rec["dP_overadjusted_io_mem"]["eff"]:+.2f} '
        f'[{rec["dP_overadjusted_io_mem"]["lo"]:+.2f},{rec["dP_overadjusted_io_mem"]["hi"]:+.2f}])')
RES['io_mem_coout'] = nc

# ============================================================== 6. robustness + save
ROB = [('epoch', 'single-adjusted'), ('sameday', 'single-adjusted'),
       ('epoch_cov90', 'single-adjusted'), ('stable', 'single-adjusted'),
       ('idle', 'single-adjusted'), ('idle_1pct', 'single-adjusted'),
       ('idle_nodeweighted', 'single-unadjusted'), ('placement_with', 'single-adjusted')]
rob = {}
for d, mdl in ROB:
    rs = [r for r in RES['rows'] if r['design'] == d and r['model'] == mdl]
    rob[d] = {r['feature']: {'eff': r['range']['eff'], 'lo': r['range']['lo'],
                             'hi': r['range']['hi'], 'eq_bound': r['eq_bound'],
                             'n': r['n'], 'nodes': r['nodes']} for r in rs}
RES['robustness'] = rob
RES['robustness_max_bound'] = max(v['eq_bound'] for d in rob.values() for v in d.values())
json.dump(RES, open(root / 'analysis' / 'equivalence_fit.json', 'w'), indent=1)

H = [r for r in RES['rows'] if r['design'] == 'epoch' and r['model'] == 'single-adjusted']
say(f'\n=== headline: contemporaneous maps, one feature per model, adjusted, rack-clustered ===')
for r in H:
    say(f'  {r["feature"]:<8} {r["range"]["eff"]:+.2f} W over the full range '
        f'[{r["range"]["lo"]:+.2f}, {r["range"]["hi"]:+.2f}]; per SD {r["sd"]["eff"]:+.2f}, '
        f'per IQR {r["iqr"]["eff"]:+.2f}; equivalent at +/-{MARGIN_W:.0f} W: '
        f'{"yes" if r["equivalent"] else "no"} (clears any margin above {r["eq_bound"]:.2f} W)')
say('\n  smallest margin cleared, by design (meanidx / nadj / lowhalf):')
for d, v in rob.items():
    say(f'    {d:<15} ' + ' / '.join(f'{v[f]["eq_bound"]:.2f}' for f in FEAT)
        + f'   n={next(iter(v.values()))["n"]:,}')
say(f'  every interval in every design lies inside +/-{RES["robustness_max_bound"]:.2f} W')
say(f'\nwrote analysis/equivalence_fit.json ({time.time() - T0:.0f} s)')
