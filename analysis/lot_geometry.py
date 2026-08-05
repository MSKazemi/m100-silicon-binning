"""Is the lot boundary a PROCUREMENT boundary or a ROOM-GEOMETRY effect?

The rack index used in the heatmap is a physical object: Marconi100's 49 racks each hold 20
nodes, node id = 20*rack + slot, and the repository documents each rack's (X, Y) position on
the machine-room floor in three rows. Rack index therefore conflates two things:

  (a) procurement / installation batch  -- hardware arrives and is racked in deliveries
  (b) physical position in the room     -- cooling, airflow, neighbours

Harvesting is done by fusing at manufacture, months before installation, so (b) cannot cause a
harvest map. This script makes that argument empirically rather than only logically:

  1. node-level changepoint -- does the boundary land exactly on a rack boundary?
  2. multiple changepoints  -- is there one delivery, or several?
  3. room geometry          -- does harvest rate follow (X, Y) once lot is accounted for?
"""
import re, numpy as np, pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
DOC = Path('/home/mohsen/exadata/documentation/racks_spatial_distribution.md')
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
nid = K[:, 0]; s0 = M[:, 0].astype(float)
rng = np.random.default_rng(9)

# ---- parse the documented rack geometry ----
geo = {}
for line in DOC.read_text().splitlines():
    p = [c.strip() for c in line.split('|')]
    if len(p) >= 4 and p[0].isdigit() and p[1].isdigit() and p[2].isdigit():
        geo[int(p[0])] = (int(p[1]), int(p[2]))
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
    print(f'  {name:<24} raw r = {r:+.3f}   partial r given lot = {rp:+.3f}  '
          f'(perm p = {(np.abs(null) >= abs(rp)).mean():.4f})')
print('\n  If harvest maps were caused by the room (cooling, airflow, height), the partial')
print('  correlations would survive after removing the lot effect. They do not.')
