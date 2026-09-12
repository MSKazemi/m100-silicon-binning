"""A taxonomy of map-change events recoverable from telemetry.

The physical hierarchy, and what a "change" means at each level:

  machine room  3 rows (Y = 2, 6, 10)
   +- rack      49 cabinets x 20 nodes;  rack = id//20, slot = id%20 (height)
       +- node  1 AC922 8335-GTG chassis: 2 CPU sockets + 4 V100 GPUs
           +- socket p0/p1   one physical CPU package position, holding one POWER9 die
               +- die        24 SMT4 core positions == 12 slices
                   +- slice  2 cores + 512 kB L2 + 10 MB L3 -- the unit that is disabled (4 of 12)
                       +- core  SMT4 core, BMC sensor index 0..23

A socket's harvest map is carried by the die in that position (lifecycle_taxonomy_errors.py
shows maps reappearing in other sockets when dies are exchanged), so a change in the map is a
change of die, of position labelling, or of configuration. A map is a low-cardinality signature,
not a serial number, so each class below is a MAP-CHANGE class whose most plausible physical
reading is the named event:

  RELABEL    p0 loses exactly what p1 gains and vice versa
             -> consistent with the same two dies, tags swapped.
  CPU-SWAP   exactly one socket's map changes, the other is observed and unchanged
             -> consistent with one processor replaced (or moved in from another socket).
  NODE-SWAP  both sockets change, and not as a mirror
             -> consistent with the node or both processors replaced (or moved).
  GUARD      the count drops below 16 at a boot by exactly one slice (or one core)
             -> a deconfiguration consistent with a GARD record; no die moves.

Error rates of these readings, the dies that moved between sockets, and the uncertainty of the
rate are in lifecycle_taxonomy_errors.py; exposure in exposure_rates.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import lifecycle_common as L

root = Path(__file__).resolve().parent.parent
df = L.load_daily(root)
sets = L.clean_sets(df)
del df
E = L.classify_events(sets)

print('=== map-change events, classified at node level ===')
print(pd.crosstab(E.kind, E.era, margins=True).to_string())
print()
ss = E[E.era == 'steady']
n_tr = int(ss.nsock.sum())
print(f'steady-state events: {len(ss)} node-level events (= {n_tr} socket-level transitions) on '
      f'{ss.node.nunique()} distinct nodes over 28 months ({len(ss)/28:.1f} per month)')
print(f'  shares below are of the {len(ss)} steady-state NODE-LEVEL events')
for k, g in ss.groupby('kind'):
    print(f'  {k:<11} {len(g):4d}  ({100*len(g)/len(ss):4.1f}%)  nodes: {g.node.nunique()}')
cs = int(ss.kind.isin(['CPU-SWAP', 'CPU-SWAP?']).sum())
print(f'  CPU-SWAP incl. CPU-SWAP?: {cs} ({100*cs/len(ss):.1f}%)')

# ---- how many sockets received a different die? ----
real = ss[ss.kind.isin(['CPU-SWAP', 'CPU-SWAP?', 'NODE-SWAP'])]
n_cpu = int(real.dies.sum())
print(f'\ndie changes in steady state: {n_cpu} (1 per CPU-SWAP[?], 2 per NODE-SWAP) = the {n_tr} '
      f'socket transitions minus the {2*int((ss.kind == "RELABEL").sum())} of RELABEL events')
print('  rate on the matched steady-state exposure: exposure_rates.py; cluster-aware interval and')
print('  the share that is dies moved between sockets: lifecycle_taxonomy_errors.py')

# ---- do die changes cluster in space? ----
print('\n=== do die-change events cluster by rack? ===')
rk = real.node // 20
vc = rk.value_counts()
rng = np.random.default_rng(4)
obs = (vc ** 2).sum()
B = 4000
null = np.array([(pd.Series(rng.integers(0, 49, len(real))).value_counts() ** 2).sum() for _ in range(B)])
b = int((null >= obs).sum())
print(f'  racks touched: {rk.nunique()} of 49; max events in one rack: {vc.max()}')
print(f'  concentration statistic obs {obs}, null {null.mean():.0f}+/-{null.std():.0f} '
      f'-> z = {(obs-null.mean())/null.std():+.2f}, Monte-Carlo p = {(b+1)/(B+1):.4f} (b={b}, B={B})')

print('\n=== a change is NOT the same as a new die: relabelling in perspective ===')
rel = ss[ss.kind == 'RELABEL']
print(f'  {len(rel)} of {len(ss)} steady-state node-level events ({100*len(rel)/len(ss):.0f}%) are mirrors,')
print('  consistent with no die moving. A study that read every map change as a hardware replacement')
print('  would overcount by that much (and, lifecycle_taxonomy_errors.py shows, by more: many of the')
print('  remaining changes are dies moved between sockets rather than new processors).')
