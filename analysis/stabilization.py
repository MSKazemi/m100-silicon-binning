"""When does the machine (and its monitoring) stabilise, and do the results survive excluding
the commissioning period?

A newly commissioned data centre is reconfigured constantly and its monitoring pipeline is not
yet settled, so early-period events are not evidence about steady-state behaviour. Rather than
assume a cut-off we locate it empirically from two independent signals -- the configuration
transition rate and the telemetry coverage -- then re-run every headline result on the
post-stabilisation period alone.
"""
import pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(23)
EXPECT, NP = 4320, 12

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
t['clean'] = (t.active == 16) & (t.nfull == 16)
t['month'] = t.day.dt.strftime('%y-%m')
clean = t[t.clean][['node', 'socket', 'day']]
dfc = df.merge(clean, on=['node', 'socket', 'day'])
sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()

# ---- transitions with their dates ----
tr = []
for (n, s), g in sets.groupby(['node', 'socket']):
    g = g.sort_values('day'); cs, ds = g.c.tolist(), g.day.tolist()
    for i in range(1, len(cs)):
        if cs[i] != cs[i-1]: tr.append((n, s, ds[i], (ds[i] - ds[i-1]).days))
T = pd.DataFrame(tr, columns=['node', 'socket', 'day', 'gap'])
T['month'] = T.day.dt.strftime('%y-%m')

print('=== when does the system settle? ===')
print(f'{"month":8} {"transitions":>12} {"coverage":>10} {"sockets":>9}')
cov = t.groupby('month')['clean'].mean()
socks = t.groupby('month').apply(lambda g: g.groupby(['node', 'socket']).ngroups, include_groups=False)
tc = T.groupby('month').size().reindex(sorted(cov.index), fill_value=0)
for m in sorted(cov.index):
    print(f'{m:8} {tc.get(m, 0):>12} {cov[m]:>10.3f} {socks[m]:>9}')

months = sorted(cov.index)
post = [m for m in months if m >= '20-06']
print(f'\ncommissioning window taken as 2020-03..2020-05; steady state = {post[0]}..{post[-1]} '
      f'({len(post)} months)')
print(f'  transitions in commissioning : {int(tc[[m for m in months if m < "20-06"]].sum())}')
print(f'  transitions in steady state  : {int(tc[post].sum())}')

# ---- re-run headline results on steady state only ----
print('\n=== headline results, steady state only (2020-06 onward) ===')
ss = sets[sets.day >= pd.Timestamp('2020-06-01', tz='UTC')].copy()
ss['pair'] = ss['c'].apply(lambda x: frozenset(v // 2 for v in x))
rows, keys = [], []
for (n, s), g in ss.groupby(['node', 'socket']):
    act = g['pair'].value_counts().index[0]
    v = np.ones(NP, dtype=np.int8); v[sorted(act)] = 0
    if v.sum() == 4: rows.append(v); keys.append((int(n), s))
M2 = np.array(rows); nid2 = np.array([k[0] for k in keys]); rack2 = nid2 // 20
M1 = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K1 = np.load(root / 'analysis' / 'harvest_keys_full.npy')
print(f'  sockets with a map: {len(M2)} (full record: {len(M1)})')

# are the maps themselves identical to the full-record maps?
d1 = {(n, s): tuple(np.flatnonzero(r)) for (n, s), r in zip(map(tuple, K1), M1)}
same = sum(1 for (n, s), r in zip(keys, M2) if d1.get((n, s)) == tuple(np.flatnonzero(r)))
print(f'  maps identical to full-record maps: {same}/{len(M2)} ({100*same/len(M2):.2f}%)')

print(f'  marginals %: ' + ' '.join(f'{100*x:.1f}' for x in M2.mean(0)))
print(f'  (full record): ' + ' '.join(f'{100*x:.1f}' for x in M1.mean(0)))
print(f'  distinct patterns: {len(set(map(tuple, map(np.flatnonzero, M2))))} '
      f'(full record {len(set(map(tuple, map(np.flatnonzero, M1))))})')

def welch(x, y): return (x.mean()-y.mean())/np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
A, B = M2[rack2 < 22, 0], M2[rack2 >= 22, 0]
print(f'  lot boundary: Lot A {100*A.mean():.1f}%  Lot B {100*B.mean():.1f}%  Welch t = {welch(A, B):.1f}')

# transitions during operation, steady state
Tss = T[T.day >= pd.Timestamp('2020-06-01', tz='UTC')]
print(f'  transitions: {len(Tss)}; occurring between consecutive reporting days: '
      f'{int((Tss.gap <= 1).sum())}')
print(f'  distinct dates: {Tss.day.dt.date.nunique()}, distinct nodes: {Tss.node.nunique()}')
top = Tss.groupby(Tss.day.dt.date).size().sort_values(ascending=False)
print(f'  largest single-date cluster: {top.iloc[0]} on {top.index[0]} '
      f'({100*top.iloc[0]/len(Tss):.0f}% of steady-state transitions)')
