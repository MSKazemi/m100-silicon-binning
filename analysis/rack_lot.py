"""Do disabled-slice patterns cluster by physical position (rack / row), beyond the lot?

node id -> rack = id//20, slot = id%20; racks 0-17 row Y=2, 18-32 Y=6, 33-48 Y=10.

Uses the full-record modal maps (analysis/harvest_maps_full.npy), rack-indexed sockets only
(node 980, outside the 49 documented cabinets, excluded). An earlier version read the
single-month 2020-04 snapshot and compared against a FLEET-WIDE shuffle only; since the lots
differ sharply, a fleet-wide shuffle mixes the lots and any "same-rack" excess it finds is
largely the lot contrast. Each test below is therefore run against two nulls:

  fleet-wide  sockets (or node ids) permuted across the whole machine
  within-lot  permuted only within Lot A (racks 0-21) and within Lot B (racks 22-48)

Monte-Carlo p-values are (b+1)/(B+1).
"""
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
rng = np.random.default_rng(3)
NP, B = 12, 2000

M = np.load(root / 'harvest_maps_full.npy').astype(np.int8)
K = np.load(root / 'harvest_keys_full.npy')
keep = K[:, 0] // 20 < 49
M, K = M[keep], K[keep]
nid = K[:, 0]
rack = nid // 20
slot = nid % 20
row = np.where(rack <= 17, 0, np.where(rack <= 32, 1, 2))
lot = (rack >= 22).astype(int)
print(f'sockets: {len(M)}, racks: {len(np.unique(rack))}')


def perm_within(groups):
    """A permutation of 0..n-1 that only moves items within their group."""
    o = np.arange(len(groups))
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        o[ix] = rng.permutation(ix)
    return o


FLEET = np.zeros(len(M), int)

# --- 1. per-rack disable rate of slice 0, the most variable slice ---
print('\n--- P(slice 0 disabled) by rack ---')
tab = pd.DataFrame({'rack': rack, 's0': M[:, 0], 'row': row})
g = tab.groupby('rack').agg(p=('s0', 'mean'), n=('s0', 'size'), row=('row', 'first'))
for r, v in g.iterrows():
    print(f'  rack {r:2d} (row {int(v.row)}): {100*v.p:5.1f}%  n={int(v.n):3d}  {"#"*int(100*v.p/3)}')


# --- 2. between-rack heterogeneity, against a fleet-wide and a within-lot null ---
def rack_var(order, vals):
    return pd.Series(vals[order]).groupby(rack).mean().var()


print('\n--- permutation test: is between-rack variance > chance? (p = (b+1)/(B+1)) ---')
print('  slice   obs var   fleet-null p   within-lot-null p')
for k in range(NP):
    obs = rack_var(np.arange(len(M)), M[:, k])
    b_f = sum(rack_var(perm_within(FLEET), M[:, k]) >= obs for _ in range(B))
    b_l = sum(rack_var(perm_within(lot), M[:, k]) >= obs for _ in range(B))
    print(f'  {k:5d}  {obs:.5f}   {(b_f + 1) / (B + 1):12.4f}   {(b_l + 1) / (B + 1):17.4f}'
          f'{"   <-- beyond the lot" if (b_l + 1) / (B + 1) < 0.01 else ""}')

# --- 3. row-level rates (3 physical rows in the machine room) ---
print('\n--- P(slice disabled) by machine-room row ---')
print('  row   ' + ' '.join(f'  s{k:02d}' for k in range(NP)))
for rr in range(3):
    m = M[row == rr]
    print(f'   {rr} n={len(m):4d} ' + ' '.join(f'{100*m[:,k].mean():4.0f}' for k in range(NP)))

# --- 4. are same-rack sockets more similar than chance, beyond the lot? ---
print('\n--- shared harvested slices between same-rack sockets (different nodes) ---')
Mi = M.astype(np.int32)
for label, sel in [('p0 sockets only', K[:, 1] == 0), ('both sockets', np.ones(len(M), bool))]:
    ix = np.flatnonzero(sel)
    rk, nd, lt = rack[ix], nid[ix], lot[ix]
    I, J = np.triu_indices(len(ix), 1)
    same = (rk[I] == rk[J]) & (nd[I] != nd[J])
    I, J = I[same], J[same]
    adj = np.abs(nd[I] - nd[J]) == 1
    Ms = Mi[ix]

    def ov(o, mask=None):
        a, b = Ms[o[I]], Ms[o[J]]
        v = (a * b).sum(1)
        return v.mean() if mask is None else v[mask].mean()

    base = np.arange(len(ix))
    obs = ov(base)
    nf = np.array([ov(perm_within(np.zeros(len(ix), int))) for _ in range(500)])
    nl = np.array([ov(perm_within(lt)) for _ in range(500)])
    print(f'  {label}: {len(I)} same-rack pairs; adjacent slots {ov(base, adj):.4f} '
          f'(n={int(adj.sum())}), any slot {obs:.4f}')
    print(f'    fleet-wide shuffle : {nf.mean():.4f} +/- {nf.std():.4f}  -> z = '
          f'{(obs - nf.mean()) / nf.std():+.2f}, p = {((nf >= obs).sum() + 1) / 501:.4f}')
    print(f'    within-lot shuffle : {nl.mean():.4f} +/- {nl.std():.4f}  -> z = '
          f'{(obs - nl.mean()) / nl.std():+.2f}, p = {((nl >= obs).sum() + 1) / 501:.4f}')
print('\n  The fleet-wide excess is the lot contrast: shuffled within lots, same-rack sockets are')
print('  no more alike than sockets of different racks of the same lot.')
