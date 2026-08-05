"""Do disabled-slice patterns cluster by physical position (rack / row)?
node id -> rack = id//20, slot = id%20; racks 0-17 row Y=2, 18-32 Y=6, 33-48 Y=10."""
import pandas as pd, numpy as np
from pathlib import Path
root = Path(__file__).resolve().parent
rng = np.random.default_rng(3)
NP = 12

df = pd.read_parquet(root / 'counts_20-04.parquet')
df['pair'] = df['core'] // 2
rows, keys = [], []
for (node, sock), g in df.groupby(['node', 'socket']):
    v = np.ones(NP, dtype=np.int8); v[sorted(set(g['pair']))] = 0
    if v.sum() == 4: rows.append(v); keys.append((int(node), sock))
M = np.array(rows); nid = np.array([k[0] for k in keys])
rack = nid // 20; slot = nid % 20
row = np.where(rack <= 17, 0, np.where(rack <= 32, 1, 2))
print(f'sockets: {len(M)}, racks: {len(np.unique(rack))}')

# --- 1. per-rack disable rate of slice 0, the most variable slice ---
print('\n--- P(slice 0 disabled) by rack ---')
tab = pd.DataFrame({'rack': rack, 's0': M[:, 0], 'row': row})
g = tab.groupby('rack').agg(p=('s0', 'mean'), n=('s0', 'size'), row=('row', 'first'))
for r, v in g.iterrows():
    print(f'  rack {r:2d} (row {int(v.row)}): {100*v.p:5.1f}%  n={int(v.n):3d}  {"#"*int(100*v.p/3)}')

# --- 2. between-rack heterogeneity test (permutation) ---
def rack_var(labels, vals):
    d = pd.DataFrame({'r': labels, 'v': vals}).groupby('r')['v'].mean()
    return d.var()
print('\n--- permutation test: is between-rack variance > chance? ---')
for k in [0, 1, 2, 5, 11]:
    obs = rack_var(rack, M[:, k])
    null = np.array([rack_var(rack, rng.permutation(M[:, k])) for _ in range(2000)])
    p = (null >= obs).mean()
    print(f'  slice {k:2d}: obs var {obs:.5f}, null {null.mean():.5f}  p = {p:.4f}'
          f'  {"<-- clustered" if p < 0.01 else ""}')

# --- 3. row-level rates (3 physical rows in the machine room) ---
print('\n--- P(slice disabled) by machine-room row ---')
hdr = '        ' + ' '.join(f's{k:02d}' % () for k in range(NP))
print('  row   ' + ' '.join(f'  s{k:02d}' for k in range(NP)))
for rr in range(3):
    m = M[row == rr]
    print(f'   {rr} n={len(m):4d} ' + ' '.join(f'{100*m[:,k].mean():4.0f}' for k in range(NP)))

# --- 4. are neighbouring nodes in a rack more similar than chance? ---
print('\n--- similarity of disabled sets vs physical distance (socket p0 only) ---')
p0 = {n: set(np.flatnonzero(M[i])) for i, (n, s) in enumerate(keys) if s == 0}
ns = sorted(p0)
def mean_overlap(pairs):
    return np.mean([len(p0[a] & p0[b]) for a, b in pairs])
same_rack = [(a, b) for a in ns for b in ns if a < b and a//20 == b//20]
adjacent  = [(a, b) for a, b in same_rack if abs(a - b) == 1]
diff_rack = [(a, b) for a in ns[::3] for b in ns[::3] if a < b and a//20 != b//20]
print(f'  adjacent slots in a rack : overlap {mean_overlap(adjacent):.4f} (n={len(adjacent)})')
print(f'  same rack, any slot      : overlap {mean_overlap(same_rack):.4f} (n={len(same_rack)})')
print(f'  different racks          : overlap {mean_overlap(diff_rack):.4f} (n={len(diff_rack)})')
null = []
for _ in range(500):
    perm = dict(zip(ns, rng.permutation(ns)))
    null.append(np.mean([len(p0[perm[a]] & p0[perm[b]]) for a, b in same_rack]))
null = np.array(null)
print(f'  null (node ids shuffled) : overlap {null.mean():.4f} +/- {null.std():.4f}'
      f'  -> z = {(mean_overlap(same_rack)-null.mean())/null.std():+.2f}')
