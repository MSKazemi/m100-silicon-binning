"""A taxonomy of hardware-change events recoverable from telemetry.

The physical hierarchy, and what a "change" means at each level:

  machine room  3 rows (Y = 2, 6, 10)
   +- rack      49 cabinets x 20 nodes;  rack = id//20, slot = id%20 (height)
       +- node  1 AC922 8335-GTG chassis: 2 CPU sockets + 4 V100 GPUs
           +- socket p0/p1   one physical CPU package position, holding one POWER9 die
               +- die        24 SMT4 core positions == 12 slices
                   +- slice  2 cores + 512 kB L2 + 10 MB L3 -- THE FUSING UNIT (4 of 12 fused)
                       +- core  SMT4 core, BMC sensor index 0..23

So a socket's harvest map identifies the DIE currently seated in that position. A change in the
map is therefore a change of die, of position labelling, or of configuration -- and these are
distinguishable:

  RELABEL    p0 loses exactly what p1 gains and vice versa
             -> the same two dies, tags swapped. No hardware moved.
  CPU-SWAP   exactly one socket's map changes, the other is untouched
             -> one processor replaced; the node stayed.
  NODE-SWAP  both sockets change, and not as a mirror
             -> whole node (or both CPUs) replaced.
  GUARD      count drops below 16 by exactly one slice and later recovers
             -> firmware deconfiguration; no hardware change at all.
"""
import pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')       # post-commissioning

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
dfc = df.merge(clean, on=['node', 'socket', 'day'])
sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()
S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
days_of = {}
for (n, s), g in sets.groupby(['node', 'socket']):
    days_of[(n, s)] = sorted(g.day)

# ---- enumerate transitions, then classify at NODE level ----
events = {}          # (node, day_after) -> {socket: (old,new)}
for (n, s), ds in days_of.items():
    for i in range(1, len(ds)):
        o, w = S[(n, s, ds[i-1])], S[(n, s, ds[i])]
        if o != w:
            events.setdefault((n, ds[i]), {})[s] = (o, w, ds[i-1])

def classify(node, day, per_sock):
    if len(per_sock) == 2:
        (o0, n0, _), (o1, n1, _) = per_sock[0], per_sock[1]
        if n0 == o1 and n1 == o0:
            return 'RELABEL'
        return 'NODE-SWAP'
    s = list(per_sock)[0]
    o, w, dprev = per_sock[s]
    # did the OTHER socket have observations spanning the same interval and stay put?
    other = 1 - s
    od = [d for d in days_of.get((node, other), []) if dprev <= d <= day]
    if len(od) >= 2 and S[(node, other, od[0])] == S[(node, other, od[-1])]:
        return 'CPU-SWAP'
    return 'CPU-SWAP?'      # other socket unobserved across the window

rows = []
for (n, d), per in sorted(events.items()):
    rows.append((n, d, classify(n, d, per), len(per)))
E = pd.DataFrame(rows, columns=['node', 'day', 'kind', 'nsock'])
E['era'] = np.where(E.day < CUT, 'commissioning', 'steady')

print('=== hardware-change events, classified at node level ===')
print(pd.crosstab(E.kind, E.era, margins=True).to_string())
print()
ss = E[E.era == 'steady']
print(f'steady-state events: {len(ss)} on {ss.node.nunique()} distinct nodes '
      f'over 28 months ({len(ss)/28:.1f} per month)')
for k, g in ss.groupby('kind'):
    print(f'  {k:<11} {len(g):4d}  ({100*len(g)/len(ss):4.1f}%)  nodes: {g.node.nunique()}')

# ---- how much silicon actually moved? ----
real = ss[ss.kind.isin(['CPU-SWAP', 'CPU-SWAP?', 'NODE-SWAP'])]
n_cpu = int((real.kind != 'NODE-SWAP').sum()) + 2 * int((real.kind == 'NODE-SWAP').sum())
print(f'\nprocessors replaced in steady state (lower bound): {n_cpu} of 1,960 sockets '
      f'= {100*n_cpu/1960:.2f}% over 2.5 years')
print(f'  annualised socket replacement rate: {100*n_cpu/1960/2.5:.2f}% per year')

# ---- do replacements cluster in space? ----
print('\n=== do replacements cluster by rack? ===')
rk = real.node // 20
vc = rk.value_counts()
rng = np.random.default_rng(4)
obs = (vc ** 2).sum()
null = []
for _ in range(4000):
    r = rng.integers(0, 49, len(real))
    null.append((pd.Series(r).value_counts() ** 2).sum())
null = np.array(null)
print(f'  racks touched: {rk.nunique()} of 49; max events in one rack: {vc.max()}')
print(f'  concentration statistic obs {obs}, null {null.mean():.0f}+/-{null.std():.0f} '
      f'-> z = {(obs-null.mean())/null.std():+.2f}')

print('\n=== a change is NOT the same as a new die: relabelling in perspective ===')
rel = ss[ss.kind == 'RELABEL']
print(f'  {len(rel)} of {len(ss)} steady-state events ({100*len(rel)/len(ss):.0f}%) move no silicon.')
print('  A study that read every map change as a hardware replacement would overcount by that much.')
