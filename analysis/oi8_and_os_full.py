"""Settle OI-8 on the full record, and extend the OS ground-truth check to all available months.

OI-8: are p0/p1 relabelling events collector-side (tags swapped) or hardware-side (dies moved)?
If the collector merely exchanged the socket tags, then EVERY per-socket metric should exchange
with them -- in particular the socket power difference p0-p1 should flip sign at the same instant.
The six-month sample gave only 4 usable events, too few to conclude. This uses all 31 months.
"""
import pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
EXPECT = 4320

# ---------------------------------------------------------------- OS ground truth, full record
print('=== OS-reported logical CPUs, all available months ===')
os_ = pd.concat([pd.read_parquet(f) for f in sorted((root / 'os_all').glob('os_*.parquet'))],
                ignore_index=True)
tot = os_.n.sum()
vc = os_.groupby('value')['n'].sum().sort_index()
for v, k in vc.items():
    print(f'  {v:>5} logical CPUs : {k:>14,} samples ({100*k/tot:8.5f}%)')
print(f'  months {len(list((root/"os_all").glob("*.parquet")))}/31 (2020-03 and 2020-04 predate '
      f'ganglia cpu_num collection), nodes {os_.node.nunique()}, samples {tot:,}')
print(f'  -> the OS confirms 128 logical CPUs = 2 x 16 cores x 4 SMT threads on '
      f'{100*vc.get(128,0)/tot:.4f}% of {tot/1e6:.0f}M samples across the whole record.')
odd = os_[os_.value != 128]
print(f'  deviating node-days: {len(odd)} on {odd.node.nunique()} nodes')

# ---------------------------------------------------------------- OI-8
print('\n=== OI-8  are relabelling events collector-side? (full record) ===')
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
sets = (df.merge(clean, on=['node', 'socket', 'day'])
          .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
days_of = {}
for (n, s), g in sets.groupby(['node', 'socket']):
    days_of[(n, s)] = sorted(g.day)

events = []
for (n, s), ds in days_of.items():
    if s != 0: continue
    for i in range(1, len(ds)):
        d0, d1 = ds[i-1], ds[i]
        a0, a1 = S[(n, 0, d0)], S[(n, 0, d1)]
        if a0 == a1: continue
        b0, b1 = S.get((n, 1, d0)), S.get((n, 1, d1))
        if b0 is None or b1 is None: continue
        if a1 == b0 and b1 == a0:                 # pure mirror -> relabelling candidate
            events.append((n, d0, d1))
print(f'  pure relabelling events in the record: {len(events)}')

pw = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power_all').glob('pw_*.parquet'))],
               ignore_index=True)
pw['day'] = pd.to_datetime(pw['day'], utc=True)
pw = pw[pw['count'] > 0.9 * EXPECT]
P = {(r.node, r.socket, r.day): r.mean for r in pw.itertuples()}

rows = []
for n, d0, d1 in events:
    b = [P.get((n, s, d0)) for s in (0, 1)]
    a_ = [P.get((n, s, d1)) for s in (0, 1)]
    if any(v is None for v in b + a_): continue
    db, da = b[0] - b[1], a_[0] - a_[1]
    rows.append((n, str(d0.date()), str(d1.date()), db, da,
                 np.sign(db) != np.sign(da) and min(abs(db), abs(da)) > 1.0))
R = pd.DataFrame(rows, columns=['node', 'before', 'after', 'dP_before', 'dP_after', 'flipped'])
print(f'  with per-socket power on both sides: {len(R)}')
if len(R):
    print(R.to_string(index=False, float_format=lambda x: f'{x:+.2f}'))
    k, nn = int(R.flipped.sum()), len(R)
    print(f'\n  power difference flips sign: {k}/{nn} ({100*k/nn:.0f}%)')
    from scipy.stats import beta
    lo = beta.ppf(.025, k, nn-k+1) if k else 0.0
    hi = beta.ppf(.975, k+1, nn-k) if k < nn else 1.0
    print(f'  95% CI [{100*lo:.0f}, {100*hi:.0f}]%')
    print('\n  A collector-side tag swap should flip EVERY per-socket metric together.')
    print('  A physical die exchange need not: power follows the die.')
