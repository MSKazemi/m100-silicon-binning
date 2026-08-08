"""Do configuration changes really happen at reboot? Test it against a boot signal.

The paper's claim -- "changes occur only across reboots" -- is currently inferred from the fact
that every transition coincides with a GAP in reporting. A gap is consistent with a reboot but
does not demonstrate one; collectors stop for many reasons. The ganglia plugin publishes
`boottime`, the epoch at which the node last booted, so a reboot is directly observable: the
value changes. This replaces the inference with a measurement.

Three things are tested.

  1. What fraction of map transitions are accompanied by an actual boottime change?
  2. Is that higher than for matched intervals on the same node where no transition occurred?
     Without a control, a high rate could just mean the fleet reboots constantly.
  3. Do the guard episodes coincide with reboots? A GARD record is read by hostboot at IPL, so
     a deconfiguration should appear across a boot, and its restoration across another.

`boottime` is per node, not per socket, which is the right granularity: a reboot takes the whole
chassis with it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
rng = np.random.default_rng(31)

# ------------------------------------------------------------------ boot signal per node-day
cov = pd.concat([pd.read_parquet(f) for f in sorted((root / 'cov_all').glob('cov_*.parquet'))],
                ignore_index=True)
cov['day'] = pd.to_datetime(cov['day'], utc=True)
boot = cov[['node', 'day', 'boot_min', 'boot_max', 'boot_ndistinct']].dropna(subset=['boot_max'])
boot = boot.sort_values(['node', 'day'])
print(f'boottime coverage: {boot.node.nunique()} nodes, {len(boot):,} node-days, '
      f'{boot.day.min().date()}..{boot.day.max().date()}')

# a node rebooted on day d if the boot epoch differs from the previous observed day, or if two
# distinct boot epochs appear within the day
boot['prev'] = boot.groupby('node')['boot_max'].shift()
boot['rebooted'] = ((boot.boot_ndistinct > 1) |
                    (boot.prev.notna() & (boot.boot_max != boot.prev)))
# key on integer nanoseconds: numpy datetime64 slicing below drops tz awareness
BOOT = {(int(n), d.value): bool(r) for n, d, r in zip(boot.node, boot.day, boot.rebooted)}
BDAYS = {}
for n, g in boot.groupby('node'):
    BDAYS[int(n)] = g.day.values.astype('datetime64[ns]')
print(f'node-days with a reboot: {boot.rebooted.sum():,} '
      f'({100 * boot.rebooted.mean():.2f}% of observed node-days)')

# ------------------------------------------------------------------ map transitions
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
sets = (df.merge(clean, on=['node', 'socket', 'day'])
          .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
sets = sets.sort_values(['node', 'socket', 'day'])
S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
days_of = {k: sorted(g.day) for k, g in sets.groupby(['node', 'socket'])}


def reboot_in(node, d0, d1):
    """Did node reboot in the half-open interval (d0, d1]?"""
    ds = BDAYS.get(int(node))
    if ds is None:
        return None
    lo = np.datetime64(d0.tz_localize(None))
    hi = np.datetime64(d1.tz_localize(None))
    m = (ds > lo) & (ds <= hi)
    if not m.any():
        return None                      # no boot telemetry covering the interval
    return any(BOOT[(int(node), int(d.astype('datetime64[ns]').view('int64')))] for d in ds[m])


trans, ctrl = [], []
for (n, s), ds in days_of.items():
    for i in range(1, len(ds)):
        d0, d1 = ds[i - 1], ds[i]
        if d1 < CUT:
            continue
        changed = S[(n, s, d0)] != S[(n, s, d1)]
        r = reboot_in(n, d0, d1)
        if r is None:
            continue
        (trans if changed else ctrl).append((n, s, d0, d1, (d1 - d0).days, r))

T = pd.DataFrame(trans, columns=['node', 'socket', 'd0', 'd1', 'gap', 'reboot'])
C = pd.DataFrame(ctrl, columns=['node', 'socket', 'd0', 'd1', 'gap', 'reboot'])
print(f'\n=== 1. transitions vs the boot signal (steady state) ===')
print(f'  map transitions with boot coverage : {len(T)}')
print(f'    of which a reboot is observed    : {T.reboot.sum()} '
      f'({100 * T.reboot.mean():.1f}%)')
print(f'  non-transition intervals           : {len(C):,}')
print(f'    of which a reboot is observed    : {C.reboot.sum():,} '
      f'({100 * C.reboot.mean():.2f}%)')

# ------------------------------------------------------------------ 2. gap-matched control
print('\n=== 2. matched on interval length (a longer gap has more chance to contain a boot) ===')
print('   gap (days)   transitions: reboot%    controls: reboot%')
for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]:
    tt = T[(T.gap >= lo) & (T.gap <= hi)]
    cc = C[(C.gap >= lo) & (C.gap <= hi)]
    if not len(tt):
        continue
    lab = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10 ** 5 else f'{lo}-{hi}')
    print(f'  {lab:>11}   {100 * tt.reboot.mean():8.1f}% (n={len(tt):3d})   '
          f'{100 * cc.reboot.mean():8.2f}% (n={len(cc):,})')

# Stratified permutation test: shuffle the reboot flag WITHIN each gap stratum, so the null
# preserves the fact that a longer interval has more opportunity to contain a boot. Done in
# numpy -- concatenating the two frames leaves duplicate indices, and a pandas boolean mask
# over the result silently misaligns.
strata = pd.concat([T.assign(is_t=1), C.assign(is_t=0)], ignore_index=True)
gaps = strata.gap.values
flags = strata.reboot.values.astype(float)
is_t = strata.is_t.values.astype(bool)
groups = [np.where(gaps == g)[0] for g in np.unique(gaps)]
obs = flags[is_t].mean() - flags[~is_t].mean()
null = np.empty(4000)
for b in range(4000):
    sh = flags.copy()
    for gi in groups:
        sh[gi] = flags[rng.permutation(gi)]
    null[b] = sh[is_t].mean() - sh[~is_t].mean()
print(f'\n  gap-stratified permutation test: observed difference {100 * obs:+.1f} pp, '
      f'null {100 * null.mean():+.2f} +/- {100 * null.std():.2f} pp, '
      f'p = {max((null >= obs).mean(), 1 / len(null)):.2e}')

# ------------------------------------------------------------------ 3. the guard episodes
print('\n=== 3. do the guard episodes sit across reboots? ===')
GUARD = [(347, 1, '2020-03-13', '2020-03-16'),
         (946, 0, '2020-08-31', '2020-09-01'),
         (411, 1, '2021-11-13', '2021-11-15')]
for n, s, d0, d1 in GUARD:
    lo = pd.Timestamp(d0, tz='UTC') - pd.Timedelta(days=3)
    hi = pd.Timestamp(d1, tz='UTC') + pd.Timedelta(days=3)
    ds = BDAYS.get(n)
    if ds is None:
        print(f'  node {n} p{s}: no boottime coverage')
        continue
    m = ((ds >= np.datetime64(lo.tz_localize(None))) &
         (ds <= np.datetime64(hi.tz_localize(None))))
    if not m.any():
        # ganglia boottime collection begins 2020-05-05, so the commissioning-era episode has
        # no boot telemetry at all. That is absence of coverage, not absence of a reboot.
        print(f'  node {n} p{s}  episode {d0}..{d1}  NO boottime coverage in window '
              f'(collection starts {boot.day.min().date()})')
        continue
    rb = [pd.Timestamp(d).date() for d in ds[m]
          if BOOT[(n, int(d.astype('datetime64[ns]').view('int64')))]]
    print(f'  node {n} p{s}  episode {d0}..{d1}  covered, reboots in +/-3 d: '
          f'{rb if rb else "NONE"}')
print('  A GARD record is applied by hostboot at IPL, so a deconfiguration and its later')
print('  clearing should each land on a boot.')
