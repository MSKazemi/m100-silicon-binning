"""Does the IPMI core index map to PHYSICAL die position?

Method: on a socket, all active cores share workload + ambient (strong common mode).
Remove it (subtract the per-timestamp cross-core mean), then measure residual
correlation as a function of index distance. If index ~ physical position, thermal
coupling -> correlation decays with distance.
"""
import os, pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
base = Path(os.environ.get('M100_IPMI_DIR', root / 'ipmi')) / 'year_month=20-04' / 'plugin=ipmi_pub'
rng = np.random.default_rng(7)

cnt = pd.read_parquet(root / 'counts_20-04.parquet')
cnt['pair'] = cnt['core'] // 2
# nodes with a full, well-sampled socket 0
good = (cnt[(cnt.socket == 0) & (cnt.n > 4000)].groupby('node')['core'].count())
nodes = sorted(good[good == 16].index, key=int)
sel = [nodes[i] for i in rng.choice(len(nodes), 40, replace=False)]
print(f'{len(nodes)} nodes with 16 well-sampled p0 cores; using {len(sel)}')

# load once, filter to selected nodes
frames = {}
for c in range(24):
    f = base / f'metric=p0_core{c}_temp' / 'a_0.parquet'
    d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
    d = d[d.node.isin(sel)]
    if len(d): frames[c] = d
print('cores loaded:', len(frames))

gap_r, gap_n = {}, {}
per_node = []
WITHIN, CROSS1 = [], []   # |gap|=1 split by whether the two cores share a slice
for node in sel:
    cols = {}
    for c, d in frames.items():
        s = d[d.node == node]
        if len(s) > 4000:
            cols[c] = s.set_index('timestamp')['value'].astype('float32')
    if len(cols) != 16: continue
    X = pd.DataFrame(cols).dropna()
    if len(X) < 3000: continue
    idx = np.array(sorted(cols))
    R = X.values
    R = R - R.mean(axis=1, keepdims=True)      # strip socket-wide common mode
    R = R - R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0)
    C = (R.T @ R) / len(R) / np.outer(sd, sd)
    for a in range(16):
        for b in range(a + 1, 16):
            g = abs(idx[a] - idx[b])
            gap_r.setdefault(g, []).append(C[a, b])
            if g == 1:
                (WITHIN if idx[a] // 2 == idx[b] // 2 else CROSS1).append(C[a, b])
    # per-node: mean corr of index-adjacent (gap<=1 within a slice, gap 2 across slices)
    per_node.append((node, np.mean([C[a, b] for a in range(16) for b in range(a+1, 16)
                                    if abs(idx[a]-idx[b]) <= 2]),
                           np.mean([C[a, b] for a in range(16) for b in range(a+1, 16)
                                    if abs(idx[a]-idx[b]) >= 12])))

print(f'\nnodes analysed: {len(per_node)}')
print('\n--- residual correlation vs |core index distance| (pooled) ---')
for g in sorted(gap_r):
    v = np.array(gap_r[g])
    print(f'  gap {g:2d}: r = {v.mean():+.4f}  (n={len(v):5d}, sem={v.std()/np.sqrt(len(v)):.4f})')

near = np.array([p[1] for p in per_node]); far = np.array([p[2] for p in per_node])
d = near - far
print(f'\nper-node near(gap<=2) minus far(gap>=12): mean {d.mean():+.4f} +/- {d.std():.4f}')
print(f'  nodes where near > far: {(d>0).sum()}/{len(d)}')
print(f'  paired t = {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}')

print('\n--- within-slice sibling vs cross-slice neighbour (both are |gap|=1) ---')
print(f'  within-slice  (cores 2k,2k+1, share L2/L3): r = {np.mean(WITHIN):+.4f} (n={len(WITHIN)})')
print(f'  cross-slice   (cores 2k+1,2k+2, gap=1)   : r = {np.mean(CROSS1):+.4f} (n={len(CROSS1)})')
