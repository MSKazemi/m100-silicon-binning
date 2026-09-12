"""Do configuration changes really happen at reboot? Test it against a boot signal.

The paper's claim -- "changes occur only across reboots" -- was originally inferred from the fact
that every transition coincides with a GAP in reporting. A gap is consistent with a reboot but
does not demonstrate one; collectors stop for many reasons. The ganglia plugin publishes
`boottime`, the epoch at which the node last booted, so a reboot is directly observable: the
value changes. This replaces the inference with a measurement.

What is tested.

  1. What fraction of map transitions are accompanied by an actual boottime change?
  2. Is that higher than for matched intervals where no transition occurred? Without a control,
     a high rate could just mean the fleet reboots constantly.
  3. Do the guard episodes coincide with reboots? A GARD record is read by hostboot at IPL, so
     a deconfiguration should appear across a boot, and its restoration across another.
  4. The transitions WITHOUT a flagged reboot, listed one by one with their boot evidence.
  5. Reboots dated by their boot EPOCH rather than by the day a changed epoch is first seen.
     Sections 1-2 use the day-flag rule (a day is a reboot day if its epoch differs from the
     previous observed day); when the collector is down, that rule dates a reboot to the day
     collection resumes, which can lie outside the interval the reboot actually fell in.
  6. Cluster-aware inference. Control intervals repeat within nodes (and the two sockets of a
     node share one boot signal), so a permutation that shuffles reboot flags across the whole
     fleet treats dependent intervals as exchangeable. Here the unit is the node-interval, the
     permutation is exact and conditional WITHIN node x interval-length strata (and rack x
     interval-length strata), and the effect size gets a node- and a rack-cluster bootstrap CI.

`boottime` is per node, not per socket, which is the right granularity: a reboot takes the whole
chassis with it. Machine-readable results: analysis/lifecycle_reboot.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lifecycle_common as L

root = Path(__file__).resolve().parent.parent
CUT = L.CUT
rng = np.random.default_rng(31)
OUT = {}

# ------------------------------------------------------------------ boot signal per node-day
boot = L.load_boot(root)
print(f'boottime coverage: {boot.node.nunique()} nodes, {len(boot):,} node-days, '
      f'{boot.day.min().date()}..{boot.day.max().date()}')
print(f'node-days with a reboot: {boot.rebooted.sum():,} '
      f'({100 * boot.rebooted.mean():.2f}% of observed node-days)')
OUT['boottime'] = dict(nodes=int(boot.node.nunique()), node_days=int(len(boot)),
                       reboot_node_days=int(boot.rebooted.sum()),
                       first_day=str(boot.day.min().date()), last_day=str(boot.day.max().date()))

# ------------------------------------------------------------------ map transitions
df = L.load_daily(root)
sets = L.clean_sets(df)
del df
iv = L.intervals(sets)
ss = iv[iv.d1 >= CUT].reset_index(drop=True)            # an interval ending in steady state
ss['reboot'] = L.dayflag_reboot(ss, boot)                # the paper's original rule
ss['reboot_epoch'] = L.epoch_reboot(ss, boot)            # section 5

T = ss[ss.changed & ss.reboot.notna()]
C = ss[~ss.changed & ss.reboot.notna()]
print(f'\n=== 1. transitions vs the boot signal (steady state) ===')
print(f'  map transitions with boot coverage : {len(T)}')
print(f'    of which a reboot is observed    : {int(T.reboot.sum())} '
      f'({100 * T.reboot.mean():.1f}%)')
print(f'  non-transition intervals           : {len(C):,}')
print(f'    of which a reboot is observed    : {int(C.reboot.sum()):,} '
      f'({100 * C.reboot.mean():.2f}%)')
OUT['dayflag'] = dict(transitions_steady=int(ss.changed.sum()), transitions_covered=int(len(T)),
                      transitions_with_reboot=int(T.reboot.sum()), controls_covered=int(len(C)),
                      controls_with_reboot=int(C.reboot.sum()))

# ------------------------------------------------------------------ 2. gap-matched control
print('\n=== 2. matched on interval length (a longer gap has more chance to contain a boot) ===')
print('   gap (days)   transitions: reboot%    controls: reboot%')
OUT['dayflag']['by_gap'] = []
for lo, hi in L.GAP_BINS:
    tt = T[(T.gap >= lo) & (T.gap <= hi)]
    cc = C[(C.gap >= lo) & (C.gap <= hi)]
    if not len(tt):
        continue
    lab = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10 ** 5 else f'{lo}-{hi}')
    print(f'  {lab:>11}   {100 * tt.reboot.mean():8.1f}% (n={len(tt):3d})   '
          f'{100 * cc.reboot.mean():8.2f}% (n={len(cc):,})')
    OUT['dayflag']['by_gap'].append(dict(gap=lab, t_n=int(len(tt)), t_pct=100 * tt.reboot.mean(),
                                         c_n=int(len(cc)), c_pct=100 * cc.reboot.mean()))

# Gap-stratified permutation test: shuffle the reboot flag WITHIN each exact-gap stratum. With
# stratum sizes and reboot totals fixed, the difference of means is monotone in the number X of
# reboot-carrying transitions, whose null law is a sum of independent hypergeometrics. So the
# Monte-Carlo p-value is reported as (b+1)/(B+1) and the exact tail comes from convolution.
both = pd.concat([T.assign(is_t=1), C.assign(is_t=0)], ignore_index=True)
is_t = both.is_t.values.astype(bool)
y = both.reboot.values.astype(int)
nT, nC = is_t.sum(), (~is_t).sum()
res = L.stratified_hypergeom_test(both.gap.values, is_t, y, B=4000, rng=rng)
R_tot = y.sum()
to_diff = lambda x: x / nT - (R_tot - x) / nC                        # noqa: E731
obs = to_diff(res['observed'])
null_m = to_diff(res['null_mean'])
null_sd = res['null_sd'] * (1 / nT + 1 / nC)
print(f'\n  gap-stratified permutation test: observed difference {100 * obs:+.1f} pp, '
      f'null {100 * null_m:+.2f} +/- {100 * null_sd:.2f} pp, '
      f'p = {res["p_mc"]:.2e} (b={res["b"]} of B={res["B"]}; exact tail {res["p_exact"]:.1e})')
OUT['dayflag']['gap_stratified_socket_level'] = dict(
    observed_pp=100 * obs, null_mean_pp=100 * null_m, null_sd_pp=100 * null_sd,
    B=res['B'], b=res['b'], p_mc=res['p_mc'], p_exact=res['p_exact'])

# ------------------------------------------------------------------ 3. the guard episodes
print('\n=== 3. do the guard episodes sit across reboots? ===')
GUARD = [(347, 1, '2020-03-13', '2020-03-16'),
         (946, 0, '2020-08-31', '2020-09-01'),
         (411, 1, '2021-11-13', '2021-11-15')]
BD = {int(n): g for n, g in boot.groupby('node')}
for n, s, d0, d1 in GUARD:
    lo = pd.Timestamp(d0, tz='UTC') - pd.Timedelta(days=3)
    hi = pd.Timestamp(d1, tz='UTC') + pd.Timedelta(days=3)
    g = BD.get(n)
    m = g[(g.day >= lo) & (g.day <= hi)] if g is not None else g
    if m is None or not len(m):
        # ganglia boottime collection begins 2020-05-05, so the commissioning-era episode has
        # no boot telemetry at all. That is absence of coverage, not absence of a reboot.
        print(f'  node {n} p{s}  episode {d0}..{d1}  NO boottime coverage in window '
              f'(collection starts {boot.day.min().date()})')
        continue
    rb = [d.date() for d in m.day[m.rebooted]]
    print(f'  node {n} p{s}  episode {d0}..{d1}  covered, reboots in +/-3 d: '
          f'{rb if rb else "NONE"}')
print('  A GARD record is applied by hostboot at IPL, so a deconfiguration and its later')
print('  clearing should each land on a boot.')

# ------------------------------------------------------------------ 4. the non-coincident ones
print('\n=== 4. the transitions with boot coverage but no flagged reboot, one by one ===')
EP = L.node_boot_epochs(boot)
six = []
for r in T[T.reboot == 0].itertuples():
    g = BD[r.node]
    inw = g[(g.day > r.d0) & (g.day <= r.d1)]
    after = g[g.day > r.d1]
    resumed = after.day.iloc[0] if len(after) else None
    e = EP[r.node]
    ep_in = [pd.Timestamp(x, unit='s', tz='UTC') for x in e
             if r.d0 < pd.Timestamp(x, unit='s', tz='UTC').floor('D') <= r.d1]
    cov_n = [int(x) for x in inw.boot_n]
    row = dict(node=int(r.node), socket=int(r.socket), d0=str(r.d0.date()), d1=str(r.d1.date()),
               gap=int(r.gap), boot_days_in_interval=int(len(inw)), boot_samples_those_days=cov_n,
               ganglia_resumed=str(resumed.date()) if resumed is not None else None,
               boot_epoch_in_interval=[str(x) for x in ep_in])
    six.append(row)
    print(f'  node {r.node:>3} p{r.socket}  {r.d0.date()} -> {r.d1.date()} ({r.gap} d): '
          f'{len(inw)} boot-covered day(s) in the interval ({cov_n} of 1,440 samples), '
          f'boottime next seen {resumed.date() if resumed is not None else "never"}, '
          f'reporting a boot at {", ".join(str(x)[:19] for x in ep_in) or "NONE"}')
print(f'  -> {sum(bool(x["boot_epoch_in_interval"]) for x in six)} of {len(six)} carry a boot epoch '
      f'INSIDE the transition interval: the collector stopped part-way through the interval and')
print('     the day-flag rule dated the reboot to the day collection resumed.')
OUT['non_coincident_dayflag'] = six

# ------------------------------------------------------------------ 5. epoch-dated reboots
print('\n=== 5. reboots dated by their boot epoch (steady state) ===')
Te = ss[ss.changed]
Ce = ss[~ss.changed]
tv = Te.reboot_epoch
cv = Ce.reboot_epoch
print(f'  transitions: {len(Te)}  boot inside the interval {int((tv == 1).sum())}  '
      f'no boot {int((tv == 0).sum())}  unresolved {int(tv.isna().sum())}')
print(f'  controls   : {len(Ce):,}  boot inside {int((cv == 1).sum()):,}  '
      f'no boot {int((cv == 0).sum()):,}  unresolved {int(cv.isna().sum()):,}  '
      f'({100 * cv.mean():.2f}% of resolved)')
unres = Te[tv.isna()][['node', 'socket', 'd0', 'd1', 'gap']]
for r in unres.itertuples():
    print(f'    unresolved: node {r.node} p{r.socket} {r.d0.date()} -> {r.d1.date()} ({r.gap} d): '
          f'no boottime observation brackets the interval')
print('   gap (days)   transitions: boot%       controls: boot%')
OUT['epoch'] = dict(transitions=int(len(Te)), t_boot=int((tv == 1).sum()), t_no_boot=int((tv == 0).sum()),
                    t_unresolved=int(tv.isna().sum()), controls=int(len(Ce)), c_boot=int((cv == 1).sum()),
                    c_no_boot=int((cv == 0).sum()), c_unresolved=int(cv.isna().sum()),
                    unresolved=[dict(node=int(r.node), socket=int(r.socket), d0=str(r.d0.date()),
                                     d1=str(r.d1.date())) for r in unres.itertuples()], by_gap=[])
for lo, hi in L.GAP_BINS:
    tt = Te[(Te.gap >= lo) & (Te.gap <= hi)].reboot_epoch.dropna()
    cc = Ce[(Ce.gap >= lo) & (Ce.gap <= hi)].reboot_epoch.dropna()
    if not len(tt):
        continue
    lab = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10 ** 5 else f'{lo}-{hi}')
    print(f'  {lab:>11}   {100 * tt.mean():8.1f}% (n={len(tt):3d})   {100 * cc.mean():8.2f}% (n={len(cc):,})')
    OUT['epoch']['by_gap'].append(dict(gap=lab, t_n=int(len(tt)), t_pct=100 * tt.mean(),
                                       c_n=int(len(cc)), c_pct=100 * cc.mean()))

# ------------------------------------------------------------------ 6. cluster-aware inference
print('\n=== 6. cluster-aware inference (unit = node-interval; both sockets share one boot signal) ===')
OUT['cluster'] = {}
for rule, col in [('day-flag', 'reboot'), ('epoch', 'reboot_epoch')]:
    u = (ss.dropna(subset=[col])
           .groupby(['node', 'd0', 'd1'], as_index=False)
           .agg(is_t=('changed', 'max'), y=(col, 'first'), gap=('gap', 'first')))
    u['gbin'] = L.gap_bin(u.gap.values)
    u['rack'] = u.node // 20
    it = u.is_t.values.astype(bool)
    yy = u.y.values.astype(int)
    print(f'  [{rule}] node-intervals {len(u):,}: transitions {it.sum()} '
          f'(with a boot {int(yy[it].sum())}), controls {(~it).sum():,}')
    OUT['cluster'][rule] = dict(node_intervals=int(len(u)), transitions=int(it.sum()),
                                transitions_with_boot=int(yy[it].sum()))
    for lab, strata in [('gap bin', u.gbin.values),
                        ('node x gap bin', u.node.values * 10 + u.gbin.values),
                        ('rack x gap bin', u.rack.values * 10 + u.gbin.values)]:
        r = L.stratified_hypergeom_test(strata, it, yy, B=200_000, rng=rng)
        print(f'    exact conditional test within {lab:<14}: {r["observed"]} boot-carrying transitions '
              f'vs null {r["null_mean"]:.1f} +/- {r["null_sd"]:.1f} '
              f'({r["informative_transitions"]} transitions in {r["informative_strata"]} informative strata); '
              f'p_MC = {r["p_mc"]:.1e} (b={r["b"]}/{r["B"]:,}), exact tail {r["p_exact"]:.1e}')
        OUT['cluster'][rule][lab] = r
    for lab, cl in [('node', u.node.values), ('rack', u.rack.values)]:
        bs = L.cluster_bootstrap_excess(cl, u.gbin.values, it, yy, B=10_000, rng=rng)
        print(f'    gap-standardised excess boot rate, {lab}-cluster bootstrap ({bs["clusters"]} clusters): '
              f'{100 * bs["point"]:+.1f} pp, 95% CI [{100 * bs["lo"]:+.1f}, {100 * bs["hi"]:+.1f}]')
        OUT['cluster'][rule][f'excess_{lab}_bootstrap'] = bs

(root / 'analysis' / 'lifecycle_reboot.json').write_text(json.dumps(OUT, indent=1, default=float))
print('\nwrote analysis/lifecycle_reboot.json')
