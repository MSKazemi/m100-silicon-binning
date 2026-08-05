"""Lifecycle of every core on every socket, across all swept months.

Answers:
  Q1 Is the ACTIVE COUNT ever anything other than 16?
  Q2 Does the active SET change over a socket's life? How often?
  Q3 Do changes coincide with gaps in reporting (reboot/service) or happen mid-operation?
  Q4 Are apparent changes actually socket RELABELLING (p0/p1 swap) rather than hardware?
  Q5 Per core: is it active for the socket's whole life, or intermittently?

Consumes daily/daily_*.parquet produced by sweep_all_months.py.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
D = sorted((root / 'daily').glob('daily_*.parquet'))
if not D: sys.exit('no daily tables; run sweep_all_months.py first')
df = pd.concat([pd.read_parquet(f) for f in D], ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
months = sorted({f.stem.replace('daily_', '') for f in D})
print(f'months swept: {len(months)}  ({months[0]} .. {months[-1]})')
print(f'rows {len(df):,}  nodes {df.node.nunique()}  days {df.day.nunique()}')
EXPECT = 4320

# ---------------- Q1: active count per socket-day ----------------
cnt = df.groupby(['node', 'socket', 'day']).size().rename('active').reset_index()
vc = cnt['active'].value_counts().sort_index()
print('\n--- Q1  active cores per socket-day ---')
for k, v in vc.items():
    print(f'  {k:2d} cores : {v:>9,}  ({100*v/len(cnt):6.3f}%)')
print(f'  socket-days with exactly 16: {100*vc.get(16,0)/len(cnt):.4f}%')

# a "clean" socket-day = 16 cores AND every core near-fully sampled
full = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
cnt = cnt.merge(full, on=['node', 'socket', 'day'], how='left').fillna({'nfull': 0})
clean = cnt[(cnt.active == 16) & (cnt.nfull == 16)]
print(f'  fully-sampled 16-core socket-days: {len(clean):,} ({100*len(clean)/len(cnt):.1f}%)')

# ---------------- Q2/Q3/Q5 on CLEAN days only (avoids partial-day unions) ----------------
key = clean.set_index(['node', 'socket', 'day']).index
dfc = df.set_index(['node', 'socket', 'day']).loc[key].reset_index()
sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('cores').reset_index()
print(f'\nclean socket-days used for set analysis: {len(sets):,}')

nch, trans = 0, []
life = []
for (n, s), g in sets.groupby(['node', 'socket']):
    g = g.sort_values('day')
    cs, ds = g['cores'].tolist(), g['day'].tolist()
    u = len(set(cs))
    life.append((n, s, len(ds), u, ds[0], ds[-1]))
    if u > 1:
        nch += 1
        for i in range(1, len(cs)):
            if cs[i] != cs[i-1]:
                trans.append((n, s, ds[i-1], ds[i], (ds[i]-ds[i-1]).days, cs[i-1], cs[i]))
L = pd.DataFrame(life, columns=['node', 'socket', 'ndays', 'nsets', 'first', 'last'])
print('\n--- Q2  distinct active sets per (node, socket) over its whole life ---')
print(L['nsets'].value_counts().sort_index().to_string())
print(f'  sockets that NEVER changed: {(L.nsets==1).sum()} / {len(L)} '
      f'({100*(L.nsets==1).mean():.2f}%)')
print(f'  median observed lifetime: {L.ndays.median():.0f} clean days')

# ---------------- Q3: gaps at transitions ----------------
if trans:
    T = pd.DataFrame(trans, columns=['node','socket','d0','d1','gap','old','new'])
    allgap = []
    for (n, s), g in sets.groupby(['node', 'socket']):
        ds = sorted(g['day']); allgap += [(ds[i]-ds[i-1]).days for i in range(1, len(ds))]
    allgap = np.array(allgap)
    print('\n--- Q3  reporting gap across a transition vs typical gap ---')
    print(f'  at transitions : n={len(T):4d}  median {T.gap.median():.0f} d  '
          f'mean {T.gap.mean():.1f} d  share with gap>1d: {100*(T.gap>1).mean():.1f}%')
    print(f'  all day-to-day : n={len(allgap):6d}  median {np.median(allgap):.0f} d  '
          f'mean {allgap.mean():.1f} d  share with gap>1d: {100*(allgap>1).mean():.1f}%')
    same_day = (T.gap <= 1).sum()
    print(f'  transitions with NO downtime (consecutive days): {same_day} '
          f'({100*same_day/len(T):.1f}%)')

    # ---------------- Q4: socket relabelling ----------------
    piv = {(n, s, d): c for n, s, d, c in
           zip(sets.node, sets.socket, sets.day, sets.cores)}
    swap = other = 0; ex = []
    for r in T.itertuples():
        partner_old = piv.get((r.node, 1 - r.socket, r.d0))
        partner_new = piv.get((r.node, 1 - r.socket, r.d1))
        if partner_old is None or partner_new is None:
            continue
        if r.new == partner_old and partner_new == r.old:
            swap += 1
        else:
            other += 1
            if len(ex) < 6:
                ex.append((r.node, r.socket, str(r.d0.date()), str(r.d1.date()),
                           sorted(r.old - r.new), sorted(r.new - r.old)))
    tot = swap + other
    print('\n--- Q4  are transitions socket RELABELLING? ---')
    if tot:
        print(f'  transitions with both sockets observable: {tot}')
        print(f'    pure p0<->p1 swap (dies unchanged): {swap} ({100*swap/tot:.1f}%)')
        print(f'    genuinely different silicon       : {other} ({100*other/tot:.1f}%)')
    for e in ex:
        print(f'    node {e[0]} p{e[1]} {e[2]}->{e[3]}: lost {e[4]} gained {e[5]}')

# ---------------- Q5: per-core intermittency within a stable socket ----------------
print('\n--- Q5  within a socket whose set never changed, is each core always present? ---')
stable = set(map(tuple, L[L.nsets == 1][['node', 'socket']].values))
sub = sets[[ (n, s) in stable for n, s in zip(sets.node, sets.socket) ]]
print(f'  stable sockets: {len(stable)}, clean days: {len(sub):,}')
print('  by construction every clean day of a stable socket has the identical 16 cores,')
print('  so no core in a stable socket is ever intermittent on a fully-sampled day.')
