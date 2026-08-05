"""Day-by-day stability of the active-core set, for every node and socket in a month.

Answers: does the active set change within a month? Does the COUNT ever leave 16?
Do transitions coincide with gaps in reporting (i.e. reboots / service events)?
Usage: daily_stability.py <extracted_dir> <year_month>
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

root = Path(sys.argv[1]); ym = sys.argv[2]
base = root / f'year_month={ym}' / 'plugin=ipmi_pub'
EXPECT = 4320  # samples/day at 20 s

parts = []
for sock in (0, 1):
    for core in range(24):
        f = base / f'metric=p{sock}_core{core}_temp' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'node'])
        if not len(d): continue
        d['day'] = d['timestamp'].dt.floor('D')
        g = d.groupby(['node', 'day']).size().reset_index(name='n')
        g['socket'] = sock; g['core'] = core
        parts.append(g)
df = pd.concat(parts, ignore_index=True)
df['node'] = df['node'].astype(int)
print(f'=== {ym} ===  rows={len(df):,}  nodes={df.node.nunique()}  days={df.day.nunique()}')

# ---- 1. active-core COUNT per (node, socket, day) ----
cnt = df.groupby(['node', 'socket', 'day']).size().rename('active').reset_index()
print('\n--- distribution of active cores per socket-day ---')
print(cnt['active'].value_counts().sort_index().to_string())
odd = cnt[cnt.active != 16]
print(f'socket-days with != 16 active: {len(odd)} / {len(cnt)} ({100*len(odd)/len(cnt):.3f}%)')

# ---- 2. partial days (core present but far below full sampling) ----
full_days = df[df.n > 0.9 * EXPECT]
print(f'\ncore-days at >90% of full sampling: {len(full_days):,} / {len(df):,} '
      f'({100*len(full_days)/len(df):.1f}%)')

# ---- 3. does the SET change within the month? ----
sets = (df.groupby(['node', 'socket', 'day'])['core']
          .apply(lambda s: frozenset(s)).rename('cores').reset_index())
chg = []
for (n, s), g in sets.groupby(['node', 'socket']):
    g = g.sort_values('day')
    uniq = g['cores'].nunique()
    if uniq > 1:
        days = g['day'].tolist(); cs = g['cores'].tolist()
        trans = [(days[i-1], days[i], sorted(cs[i-1] - cs[i]), sorted(cs[i] - cs[i-1]))
                 for i in range(1, len(cs)) if cs[i] != cs[i-1]]
        chg.append((n, s, uniq, len(g), trans))
print(f'\n--- (node, socket) with >1 distinct active set during the month: {len(chg)} '
      f'of {sets.groupby(["node","socket"]).ngroups} ---')
for n, s, u, nd, trans in sorted(chg)[:12]:
    print(f'  node {n:>3} p{s}: {u} distinct sets over {nd} reporting days')
    for d0, d1, lost, gained in trans[:4]:
        gap = (d1 - d0).days
        print(f'      {str(d0.date())} -> {str(d1.date())} (gap {gap}d)  '
              f'lost {lost}  gained {gained}')

# ---- 4. do transitions coincide with reporting gaps? ----
gaps_at_change, gaps_no_change = [], []
for (n, s), g in sets.groupby(['node', 'socket']):
    g = g.sort_values('day'); days = g['day'].tolist(); cs = g['cores'].tolist()
    for i in range(1, len(days)):
        gap = (days[i] - days[i-1]).days
        (gaps_at_change if cs[i] != cs[i-1] else gaps_no_change).append(gap)
if gaps_at_change:
    ga, gn = np.array(gaps_at_change), np.array(gaps_no_change)
    print(f'\n--- reporting gap (days) across consecutive reporting days ---')
    print(f'  where the active set CHANGED  : n={len(ga):5d}  median {np.median(ga):.0f}  '
          f'mean {ga.mean():.1f}  %gap>1: {100*(ga>1).mean():.1f}%')
    print(f'  where the active set was SAME : n={len(gn):5d}  median {np.median(gn):.0f}  '
          f'mean {gn.mean():.1f}  %gap>1: {100*(gn>1).mean():.1f}%')
