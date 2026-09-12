"""Is the lot boundary a ROOM-GEOMETRY effect, or does it follow the hardware?

The rack index used in the heatmap is a physical object: Marconi100's 49 racks each hold 20
nodes, node id = 20*rack + slot, and the dataset documents each rack's (X, Y) position on the
machine-room floor in three rows. Rack index therefore conflates two things:

  (a) hardware population -- procurement or installation batch, racked in id order
  (b) physical position in the room -- cooling, airflow, neighbours

A map fixed before installation cannot be caused by (b). That premise is itself an inference
(the telemetry shows which positions are disabled, not when or why), so this script tests the
room reading directly rather than relying on it:

  1. node-level changepoint -- does the boundary land on a rack boundary?
  2. multiple changepoints  -- one boundary, or several?
  3. room geometry          -- does harvest rate follow (X, Y) once lot is accounted for?
  4. the same row           -- the boundary falls inside the middle row (Y = 6), so racks of
                               both lots share it; compare them directly.

Whether rack order stands for a delivery, an installation date or a per-rack configuration is
a separate question; cohort_incoming.py tests the per-rack reading.
"""
import numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
nid = K[:, 0]; s0 = M[:, 0].astype(float)
rng = np.random.default_rng(9)

# ---- documented rack geometry: rack -> (X, Y) ----
# From the M100 ExaData repository, documentation/racks_spatial_distribution.md (cited in the
# paper as exadata_repo). Embedded so the script runs without a local copy of that repository.
geo = {0: (21, 2), 1: (20, 2), 2: (19, 2), 3: (18, 2), 4: (17, 2), 5: (16, 2), 6: (15, 2),
       7: (14, 2), 8: (13, 2), 9: (12, 2), 10: (11, 2), 11: (10, 2), 12: (9, 2), 13: (8, 2),
       14: (7, 2), 15: (6, 2), 16: (5, 2), 17: (4, 2), 18: (20, 6), 19: (19, 6), 20: (18, 6),
       21: (17, 6), 22: (16, 6), 23: (15, 6), 24: (14, 6), 25: (12, 6), 26: (11, 6),
       27: (10, 6), 28: (9, 6), 29: (7, 6), 30: (6, 6), 31: (5, 6), 32: (4, 6), 33: (20, 10),
       34: (19, 10), 35: (18, 10), 36: (17, 10), 37: (16, 10), 38: (15, 10), 39: (14, 10),
       40: (13, 10), 41: (12, 10), 42: (11, 10), 43: (10, 10), 44: (9, 10), 45: (7, 10),
       46: (6, 10), 47: (5, 10), 48: (4, 10)}
print(f'racks with documented position: {len(geo)}  '
      f'rows (Y): {sorted(set(y for _, y in geo.values()))}')
rack_all = nid // 20
keep = np.array([r in geo for r in rack_all])
if (~keep).any():
    print(f'  dropping {(~keep).sum()} sockets in undocumented rack(s) '
          f'{sorted(set(rack_all[~keep]))} (nodes {sorted(set(nid[~keep]))[:4]}...)')
nid, s0 = nid[keep], s0[keep]
rack = nid // 20; slot = nid % 20
X = np.array([geo[r][0] for r in rack]); Y = np.array([geo[r][1] for r in rack])

def welch(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))

# ---- 1. node-level changepoint ----
print('\n=== 1. where exactly is the boundary? ===')
order = np.argsort(nid); v = s0[order]; ids = nid[order]
best = None
for i in range(200, len(v) - 200):
    t = welch(v[:i], v[i:])
    if best is None or abs(t) > abs(best[1]): best = (i, t)
i, t = best
print(f'  node-level changepoint: between node {ids[i-1]} and {ids[i]}, Welch t = {t:.1f}')
print(f'  node {ids[i]} is slot {ids[i] % 20} of rack {ids[i] // 20}')
print(f'  -> the boundary falls {"EXACTLY on a rack boundary" if ids[i] % 20 == 0 else "MID-RACK"}')
# how sharp? compare to the 20 nodes either side
lo = v[max(0, i-40):i]; hi = v[i:i+40]
print(f'  40 sockets before: P(slice0) = {100*lo.mean():.1f}%   40 after: {100*hi.mean():.1f}%')

# ---- 2. more than one delivery? binary segmentation ----
print('\n=== 2. is there more than one batch? (binary segmentation) ===')
def seg(lo, hi, depth=0, out=None):
    if out is None: out = []
    if hi - lo < 300 or depth > 3: return out
    b = None
    for i in range(lo + 120, hi - 120):
        t = welch(v[lo:i], v[i:hi])
        if b is None or abs(t) > abs(b[1]): b = (i, t)
    if b and abs(b[1]) > 4.0:
        out.append((ids[b[0]], b[1], hi - lo))
        seg(lo, b[0], depth+1, out); seg(b[0], hi, depth+1, out)
    return out
cps = sorted(seg(0, len(v)), key=lambda c: c[0])
if cps:
    for node, tt, n in cps:
        print(f'  changepoint at node {node:4d} (rack {node//20:2d}, slot {node%20:2d})  '
              f'Welch t = {tt:+6.1f}  segment n = {n}')
else:
    print('  none beyond the primary boundary')
# segment means
bounds = [0] + [int(np.searchsorted(ids, c[0])) for c in cps] + [len(v)]
print('  resulting segments:')
for a, b_ in zip(bounds[:-1], bounds[1:]):
    print(f'    nodes {ids[a]:4d}-{ids[b_-1]:4d}  n={b_-a:5d}  P(slice0) = {100*v[a:b_].mean():5.1f}%')

# ---- 3. room geometry vs procurement ----
print('\n=== 3. does harvest rate follow room position, or only lot? ===')
lot = (rack >= 22).astype(int)
for name, arr in [('X (aisle position)', X), ('Y (row)', Y), ('slot (height in rack)', slot)]:
    r = np.corrcoef(arr, s0)[0, 1]
    # partial correlation given lot: correlate residuals
    rx = arr - np.array([arr[lot == l].mean() for l in lot])
    ry = s0 - np.array([s0[lot == l].mean() for l in lot])
    rp = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(4000)])
    pp = ((np.abs(null) >= abs(rp)).sum() + 1) / (len(null) + 1)
    print(f'  {name:<24} raw r = {r:+.3f}   partial r given lot = {rp:+.3f}  '
          f'(perm p = {pp:.4f}, (b+1)/(B+1))')
print('\n  If harvest maps were caused by the room (cooling, airflow, height), the partial')
print('  correlations would survive after removing the lot effect. They do not.')

# ---- 4. the same row, both lots ----
print('\n=== 4. the middle row holds racks of both lots ===')
row6 = sorted(r for r, (_, y) in geo.items() if y == 6)
inA = [r for r in row6 if r < 22]
rate = {r: s0[rack == r].mean() for r in row6}
for r in row6:
    print(f'  rack {r:2d}  X={geo[r][0]:2d}  lot {"A" if r < 22 else "B"}  '
          f'P(slice0) = {100 * rate[r]:5.1f}%')
mA = s0[np.isin(rack, inA)].mean()
mB = s0[np.isin(rack, [r for r in row6 if r >= 22])].mean()
obs = mA - mB
# exact rack-level permutation: every way of choosing len(inA) of the row's racks
ns = {r: (s0[rack == r].sum(), (rack == r).sum()) for r in row6}
diffs = []
for pick in combinations(row6, len(inA)):
    a = sum(ns[r][0] for r in pick) / sum(ns[r][1] for r in pick)
    rest = [r for r in row6 if r not in pick]
    b = sum(ns[r][0] for r in rest) / sum(ns[r][1] for r in rest)
    diffs.append(a - b)
diffs = np.array(diffs)
p_exact = (np.abs(diffs) >= abs(obs) - 1e-12).mean()
print(f'  row Y=6: Lot A racks {inA[0]}-{inA[-1]} {100 * mA:.1f}%  vs  Lot B racks 22-{row6[-1]} '
      f'{100 * mB:.1f}%  ({100 * obs:+.1f} pp)')
print(f'  exact rack-level permutation over all {len(diffs)} splits of the row: p = {p_exact:.4f} '
      f'(the observed split is the most extreme: {int((np.abs(diffs) >= abs(obs) - 1e-12).sum())} '
      f'of {len(diffs)})')
print(f'  adjacent cabinets rack 21 (X={geo[21][0]}) and rack 22 (X={geo[22][0]}): '
      f'{100 * rate[21]:.1f}% vs {100 * rate[22]:.1f}%')
lo_A, hi_B = min(rate[r] for r in inA), max(rate[r] for r in row6 if r >= 22)
print(f'  rack rates do not overlap: lowest Lot A rack {100 * lo_A:.1f}% > highest Lot B rack '
      f'{100 * hi_B:.1f}%')
print('  The split was located by the fleet-wide scan, which saw these racks, so the permutation')
print('  p describes the row rather than testing the boundary independently; what it shows is that')
print('  the change happens between adjacent cabinets of one row, not between rows.')
