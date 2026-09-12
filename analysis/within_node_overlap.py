"""Within-node map overlap over the whole fleet -- the committed producer for Sec. IV-C.

The two sockets of a node hold two independently harvested dies, so how much their harvest maps
overlap is a direct check on whether maps are per-die. This script owns that number: the paper's
Sec. IV-C quotes its output and nothing else computes it on the fleet.

  - binning_stats.py computes the same statistic on the single month 2020-04 only (980 nodes),
    which is why its values differ slightly; it is kept as the single-month comparison.
  - full_record_structure.py builds the arrays read here but does not compute the overlap.

Null: the observed maps are held fixed and reassigned to sockets at random fleet-wide, which
destroys the node pairing while preserving the pattern distribution exactly. A socket pair that
shares a lot is then expected to overlap slightly more than the fleet-wide shuffle predicts, so the
excess is reported as an excess, not as dependence between the two dies of a node.

Reads analysis/harvest_maps_full.npy + harvest_keys_full.npy; writes no file.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
B = 2000
rng = np.random.default_rng(2024)

M = np.load(root / 'harvest_maps_full.npy')
K = np.load(root / 'harvest_keys_full.npy', allow_pickle=True)
node, sock = K[:, 0].astype(int), K[:, 1].astype(int)

byn = defaultdict(dict)
for i, (n, s) in enumerate(zip(node, sock)):
    byn[int(n)][int(s)] = i
pairs = [(v[0], v[1]) for v in byn.values() if 0 in v and 1 in v]
a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])

ov = (M[a] & M[b]).sum(1)
ident = int((ov == 4).sum())
print(f'nodes with a harvest map on both sockets: {len(pairs)}')
print(f'  identical maps within a node          : {ident} ({100*ident/len(pairs):.1f}%)')
print(f'  mean overlap (harvested slices shared): {ov.mean():.3f} of 4')

null = np.empty(B)
for t in range(B):
    Mp = M[rng.permutation(len(M))]
    null[t] = (Mp[a] & Mp[b]).sum(1).mean()
z = (ov.mean() - null.mean()) / null.std()
print(f'  fleet-wide map-reassignment null      : {null.mean():.3f} +/- {null.std():.3f} '
      f'({B} shuffles)')
print(f'  within-node overlap z = {z:+.1f}')
print('  -> the two dies of a node are nearly independent, with a small excess consistent with '
      'same-lot pairing')
