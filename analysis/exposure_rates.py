"""Rates computed over the exposure window they were actually measured in.

Two defects motivated this script.

1. The reported annualised replacement rate divided a *steady-state* numerator (96 die changes,
   counted only after the commissioning window closes on 2020-06-01) by a
   *full-record* denominator. `change_taxonomy.py` hard-coded 2.5 years and
   `rate_intervals.py` hard-coded 2.55; neither is the steady-state span, and the "932-day
   record" quoted in the paper is 2.55 x 365.25 rounded, not a figure measured from the data.
   The record is 934 calendar days (2020-03-09 to 2022-09-28) of which 858 carry data.

2. The three core-guard episodes are quoted as a single rate "over the record", but one of
   them (347 p1, 2020-03-13--16) falls inside the excluded commissioning window while the
   other two are steady-state. A rate must not mix the two eras.

Exposure is reported two ways, and the two event types take different ones, because they have
different detectability.

  *Calendar* exposure -- sockets x window length -- is correct for REPLACEMENTS. A replacement
  is detected by comparing a socket's map across consecutive clean days, and that comparison
  spans reporting gaps: indeed every transition in the record coincides with a gap. A socket is
  therefore at risk, and a replacement recoverable, on days we did not observe.

  *Observed* exposure -- clean socket-days converted to socket-years -- is correct for GUARD
  episodes. An episode is counted only on fully-sampled days, so one that fell entirely inside
  a coverage gap would be missed. Detection really is limited to days in hand.

Both are printed for both rates so the choice is visible rather than buried.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')      # end of commissioning, located in Sec. IV-H
DAYS_PER_YEAR = 365.25


def pois(k, expo, a=0.05):
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2 / expo
    return lo, chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2 / expo


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)

a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)]

print('=== the record, measured rather than assumed ===')
d0, d1 = df.day.min(), df.day.max()
print(f'  full record      {d0.date()} .. {d1.date()}')
print(f'    calendar span  {(d1 - d0).days + 1} days = {((d1 - d0).days + 1) / DAYS_PER_YEAR:.3f} y')
print(f'    days with data {df.day.nunique()} days')
print(f'    -> the paper\'s "932-day record" is 2.55 x 365.25, an assumed constant, not either '
      f'of these')

ss = clean[clean.day >= CUT]
ss_d0, ss_d1 = ss.day.min(), ss.day.max()
ss_cal = (ss_d1 - ss_d0).days + 1
print(f'\n  steady state     {ss_d0.date()} .. {ss_d1.date()}  (commissioning excluded)')
print(f'    calendar span  {ss_cal} days = {ss_cal / DAYS_PER_YEAR:.3f} y')
print(f'    days with data {ss.day.nunique()} days')

n_sock_ss = ss.groupby(['node', 'socket']).ngroups
expo_obs = len(ss) / DAYS_PER_YEAR
expo_cal = n_sock_ss * ss_cal / DAYS_PER_YEAR
print(f'\n  sockets observed in steady state : {n_sock_ss:,}')
print(f'  clean socket-days in steady state: {len(ss):,}')
print(f'  exposure, observed  : {expo_obs:8,.0f} socket-years')
print(f'  exposure, calendar  : {expo_cal:8,.0f} socket-years  '
      f'(coverage {100 * expo_obs / expo_cal:.1f}%)')

print('\n=== annualised die-change rate (sockets receiving a different die) ===')
print('  numerator: 96 dies, from the steady-state rows of the taxonomy')
print('  (68 CPU-SWAP + 6 CPU-SWAP? + 2 x 11 NODE-SWAP)')
K = 96
for lab, expo in [('calendar exposure  <- primary', expo_cal),
                  ('observed exposure', expo_obs)]:
    r = K / expo
    lo, hi = pois(K, expo)
    print(f'    {lab:<30} {100 * r:5.2f}% per socket-year   95% CI '
          f'[{100 * lo:.2f}, {100 * hi:.2f}]')
print(f'\n  Calendar is primary here: a replacement is recovered by comparing maps ACROSS a')
print('  reporting gap, so an unobserved day is still a day at risk.')
print(f'  (An earlier draft reported 1.9% = {K}/1960/2.55, a steady-state numerator over a full-record')
print('  denominator. On matched exposure the rate is higher, not lower.)')
print('  These Poisson intervals assume independent single-die events. NODE-SWAPs move two dies at once')
print('  and events cluster by rack, and 42 of the 96 are dies moved between sockets rather than new')
print('  processors: the node/rack-cluster bootstrap and the new-silicon rate are in')
print('  lifecycle_taxonomy_errors.py.')

print('\n=== core-guard episodes, separated by era ===')
GUARD = [('347 p1', '2020-03-13', 'commissioning'),
         ('946 p0', '2020-08-31', 'steady'),
         ('411 p1', '2021-11-13', 'steady')]
for s, d, era in GUARD:
    print(f'    {s}  {d}  {era}')
k_ss = sum(1 for _, _, e in GUARD if e == 'steady')
r = k_ss / expo_obs
lo, hi = pois(k_ss, expo_obs)
print(f'\n  steady-state episodes: {k_ss} over {expo_obs:,.0f} observed socket-years')
print(f'    rate = {1e4 * r:.2f} per 10,000 socket-years   95% CI '
      f'[{1e4 * lo:.2f}, {1e4 * hi:.2f}]  (a factor of {hi / max(lo, 1e-9):.0f} wide)')
print('  Reporting all three against the full record mixes an excluded-commissioning event')
print('  with two steady-state ones. Both counts are stated in the paper instead.')

# The slice-level criterion above (fully-sampled days, even count) excludes single-core
# deconfigurations by construction. lifecycle_os_crosscheck.py finds every departure of the OS
# logical-CPU count from 128 matched by a BMC loss of exactly that many cores, all in steady state:
# 3 of one slice (120) and 6 of one core (124). Their detector is the OS count, so their exposure is
# the OS-observed socket-time in steady state.
os_ = pd.concat([pd.read_parquet(f) for f in sorted((root / 'os_all').glob('os_*.parquet'))],
                ignore_index=True)
os_['day'] = pd.to_datetime(os_['day'], utc=True)
osd = os_[(os_.day >= CUT) & (os_.node < 980)].drop_duplicates(['node', 'day'])
expo_os = 2 * len(osd) / DAYS_PER_YEAR
dev = os_[(os_.day >= CUT) & os_.value.isin([120, 124])]
k_os = 0                                          # episodes = runs of consecutive deviation days
for _, g in dev.groupby('node'):
    d = pd.DatetimeIndex(sorted(g.day.unique()))
    k_os += 1 + int((np.diff(d.asi8) > 86_400 * 10 ** 9).sum())
lo, hi = pois(k_os, expo_os)
print(f'\n  OS-confirmed deconfigurations, any granularity, steady state: {k_os} over {expo_os:,.0f} '
      f'OS-observed socket-years')
print(f'    rate = {1e4 * k_os / expo_os:.1f} per 10,000 socket-years   95% CI '
      f'[{1e4 * lo:.1f}, {1e4 * hi:.1f}]')

print('\n=== proportions restated on the same footing (Clopper-Pearson) ===')
for lab, k, n in [('sockets that never change configuration', 1756, 1962),
                  ('steady-state changes that are RELABEL', 20, 105),
                  ('sockets with a fleet-unique harvest map', 97, 1962)]:
    lo, hi = cp(k, n)
    print(f'  {lab:<42} {100 * k / n:6.2f}%  95% CI [{100 * lo:.2f}, {100 * hi:.2f}]')
