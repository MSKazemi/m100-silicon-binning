"""Detect candidate core-guard episodes: sustained periods where a socket runs with
FEWER than its normal 16 cores, at exact slice granularity, on fully-sampled days.

A guard event should look like: lose exactly one slice (a (2k,2k+1) pair) at a boot -> run
degraded for >=1 day -> then either the slice returns (GARD record cleared) or the socket comes
back with a different map (the processor was replaced). The telemetry shows the deconfiguration,
not the GARD record itself.

Scope: fully-sampled days with an EVEN count below 16 exclude single-core deconfigurations by
construction. lifecycle_os_crosscheck.py covers every granularity against the OS's CPU count.
"""
import pandas as pd, numpy as np
from pathlib import Path

import lifecycle_common as L

root = Path(__file__).resolve().parent.parent
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
EXPECT = 4320

a = df.groupby(['node', 'socket', 'day']).size().rename('active')
f = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, f], axis=1).fillna(0).reset_index()
sets = df.groupby(['node', 'socket', 'day'])['core'].apply(frozenset)
BS = L.by_socket(L.clean_sets(df))

# fully-sampled, even count, below 16  -> candidate degraded day
cand = t[(t.active == t.nfull) & (t.active < 16) & (t.active % 2 == 0) & (t.active > 0)]
print(f'total socket-days              : {len(t):,}')
print(f'fully-sampled                  : {int((t.active==t.nfull).sum()):,}')
print(f'candidate degraded socket-days : {len(cand)}')
print()

episodes = []
for (n, s), g in cand.groupby(['node', 'socket']):
    hist = sets.loc[n, s].sort_index()
    idx = list(hist.index)
    days = sorted(g['day'])
    # group consecutive candidate days into episodes
    runs, cur = [], [days[0]]
    for d in days[1:]:
        if (d - cur[-1]).days <= 1: cur.append(d)
        else: runs.append(cur); cur = [d]
    runs.append(cur)
    for run in runs:
        dset = hist[run[0]]
        i0 = idx.index(run[0]); i1 = idx.index(run[-1])
        # baseline = the ADJACENT observed day, not the socket's global modal set:
        # a socket that also transitioned elsewhere in its life has a modal set that
        # does not describe the configuration in force around this episode.
        before = hist[idx[i0 - 1]] if i0 > 0 else None
        after = hist[idx[i1 + 1]] if i1 < len(idx) - 1 else None
        base = before if (before is not None and len(before) == 16) else after
        lost = sorted(base - dset) if base is not None else None
        slices = sorted({c // 2 for c in lost}) if lost else None
        recovered = after is not None and len(after) == 16 and dset < after
        # what the socket looks like at its first CLEAN day after the episode
        nxt, nday = L.first_clean_after(BS, n, s, run[-1])
        if nxt is None:
            afterwards = 'not observed'
        elif base is not None and nxt == base:
            afterwards = f'restored, same map ({nday.date()})'
        else:
            afterwards = f'different map ({nday.date()})'
        episodes.append(dict(node=n, socket=s, start=run[0].date(), end=run[-1].date(),
                             days=len(run), ncores=len(dset), lost=lost, slices=slices,
                             slice_granular=(lost is not None and
                                             all((c ^ 1) in lost for c in lost)),
                             recovered=recovered, afterwards=afterwards))

E = pd.DataFrame(episodes)
print('=== CANDIDATE GUARD EPISODES ===')
if len(E):
    for e in E.itertuples():
        print(f'  node {e.node:>3} p{e.socket}  {e.start} .. {e.end}  ({e.days}d, {e.ncores} cores)')
        print(f'      lost cores {e.lost} = slice(s) {e.slices}  '
              f'slice-granular={e.slice_granular}  next observed day complete={e.recovered}')
        print(f'      first clean day afterwards: {e.afterwards}')
    print()
    print(f'episodes                : {len(E)}')
    print(f'distinct sockets        : {E.groupby(["node","socket"]).ngroups} of {t.groupby(["node","socket"]).ngroups}')
    print(f'all slice-granular      : {E.slice_granular.all()}')
    print(f'same map afterwards     : {int(E.afterwards.str.startswith("restored").sum())}/{len(E)}; '
          f'different map afterwards: {int(E.afterwards.str.startswith("different").sum())}/{len(E)}')
    print(f'median duration         : {E.days.median():.0f} d')
    print('\nrates on the matched exposure: exposure_rates.py; single-core episodes and the OS')
    print('cross-check: lifecycle_os_crosscheck.py')
else:
    print('  none')
