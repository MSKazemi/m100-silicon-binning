"""Robustness checks.

OI-5  Is collection dropout random, or does it correlate with lot / harvest pattern?
      If dropout were lot-correlated, the clean-day filter could manufacture the lot boundary.
OI-8  Do the p0/p1 relabelling events show up in OTHER per-socket metrics on the same dates?
      If socket power swaps at the same instant, the relabelling is in the collector's socket
      tagging, not in the silicon.
"""
import pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(17)
EXPECT = 4320

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
t['clean'] = (t.active == 16) & (t.nfull == 16)
t['rack'] = t.node // 20
t['lot'] = np.where(t.rack < 22, 'A', 'B')

print('=== OI-5  is collection dropout lot-correlated? ===')
per = t.groupby(['node', 'socket']).agg(days=('clean', 'size'), clean=('clean', 'sum'),
                                        lot=('lot', 'first')).reset_index()
per['rate'] = per.clean / per.days
for lot, g in per.groupby('lot'):
    print(f'  Lot {lot}: n={len(g):5d} sockets   clean-day rate '
          f'mean {g.rate.mean():.4f}  median {g.rate.median():.4f}  '
          f'clean days/socket median {g.clean.median():.0f}')
A, B = per[per.lot == 'A'].rate.values, per[per.lot == 'B'].rate.values
obs = A.mean() - B.mean()
pool = np.concatenate([A, B]); nA = len(A)
null = np.array([(lambda p: p[:nA].mean() - p[nA:].mean())(rng.permutation(pool))
                 for _ in range(5000)])
print(f'  difference in clean-day rate A-B = {obs:+.4f}, permutation p = '
      f'{(np.abs(null) >= abs(obs)).mean():.4f}')

M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
H = pd.DataFrame({'node': K[:, 0], 'socket': K[:, 1], 's0': M[:, 0]})
per = per.merge(H, on=['node', 'socket'], how='inner')
r = np.corrcoef(per.s0, per.rate)[0, 1]
null2 = np.array([np.corrcoef(rng.permutation(per.s0.values), per.rate.values)[0, 1]
                  for _ in range(5000)])
print(f'  corr(slice-0 harvested, clean-day rate) = {r:+.4f}, '
      f'permutation p = {(np.abs(null2) >= abs(r)).mean():.4f}')

print('\n  lot contrast restricted to well-covered sockets:')
for thr in (0.0, 0.5, 0.8, 0.9):
    sub = per[per.rate >= thr]
    if len(sub) < 50: continue
    pa = sub[sub.lot == 'A'].s0.mean(); pb = sub[sub.lot == 'B'].s0.mean()
    print(f'    clean-day rate >= {thr:.0%}: n={len(sub):5d}  '
          f'P(slice0) Lot A {100*pa:.1f}%  Lot B {100*pb:.1f}%  gap {100*(pa-pb):+.1f} pp')

print('\n=== OI-8  do relabelling events appear in socket POWER too? ===')
P = sorted((root / 'power').glob('power_*.parquet'))
if not P:
    print('  no power tables; skipped')
else:
    pw = pd.concat([pd.read_parquet(f) for f in P], ignore_index=True)
    pw = pw[(pw.socket >= 0) & (pw.metric == 'pX_power') & (pw['count'] > 0.9 * EXPECT)]
    pw['day'] = pd.to_datetime(pw.day, utc=True)
    clean = t[t.clean][['node', 'socket', 'day']]
    dfc = df.merge(clean, on=['node', 'socket', 'day'])
    sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()
    piv = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
    covered = set(pw.day.unique())
    events = []
    for (n, s), g in sets.groupby(['node', 'socket']):
        g = g.sort_values('day'); cs, ds = g.c.tolist(), g.day.tolist()
        for i in range(1, len(cs)):
            if cs[i] == cs[i-1]: continue
            po, pn = piv.get((n, 1-s, ds[i-1])), piv.get((n, 1-s, ds[i]))
            if po is not None and pn is not None and cs[i] == po and pn == cs[i-1]:
                if ds[i-1] in covered and ds[i] in covered:
                    events.append((n, ds[i-1], ds[i]))
    events = sorted(set(events))
    print(f'  pure relabelling events with power coverage on both sides: {len(events)}')
    flips = 0; usable = 0
    for n, d0, d1 in events:
        before = pw[(pw.node == n) & (pw.day == d0)].set_index('socket')['mean']
        after = pw[(pw.node == n) & (pw.day == d1)].set_index('socket')['mean']
        if len(before) < 2 or len(after) < 2: continue
        usable += 1
        db, da = before[0] - before[1], after[0] - after[1]
        if np.sign(db) != np.sign(da) and abs(db) > 1 and abs(da) > 1: flips += 1
        print(f'    node {n} {str(d0.date())}->{str(d1.date())}: '
              f'p0-p1 power {db:+.2f} W -> {da:+.2f} W')
    if usable:
        print(f'  events where socket power difference also flipped sign: {flips}/{usable}')
