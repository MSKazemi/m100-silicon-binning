"""A1 -- does the harvest map predict anything operational?

Design: WITHIN-NODE PAIRED contrast. The two sockets of a node share chassis, airflow, inlet
air, power supply and (to first order) workload, so the difference p0-p1 removes the node-level
confounds that make cross-node power comparisons unreliable. We ask whether that difference is
predicted by how the two sockets' harvest maps differ.

Predictors per socket, from its modal harvest map:
  mean_idx  mean index of the ACTIVE slices   (where on the die the live silicon sits)
  spread    SD of active-slice indices        (clustered vs dispersed)
  adj       # adjacent active slice pairs     (thermal-neighbour count)
  lowhalf   # active slices with k < 6
"""
import pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations

root = Path(__file__).resolve().parent.parent
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')          # 1 = harvested
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')          # (node, socket)
P = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power').glob('power_*.parquet'))],
              ignore_index=True)
print(f'power rows {len(P):,}  months {P.day.dt.strftime("%y-%m").nunique()}  nodes {P.node.nunique()}')

def feats(v):
    act = np.flatnonzero(v == 0)
    return dict(mean_idx=act.mean(), spread=act.std(),
                adj=sum(1 for a, b in combinations(act, 2) if abs(a - b) == 1),
                lowhalf=int((act < 6).sum()))
F = pd.DataFrame([feats(v) for v in M])
F['node'] = K[:, 0]; F['socket'] = K[:, 1]

# per (node, socket, month): mean socket power, requiring well-sampled days
P['month'] = P.day.dt.strftime('%y-%m')
sock = P[(P.socket >= 0) & (P['count'] > 0.9 * 4320)]
agg = (sock.pivot_table(index=['node', 'socket', 'month'], columns='metric',
                        values='mean', aggfunc='mean').reset_index())
node_amb = (P[(P.socket == -1) & (P['count'] > 0.9 * 4320)]
            .pivot_table(index=['node', 'month'], columns='metric', values='mean')
            .reset_index())
agg = agg.merge(node_amb, on=['node', 'month'], how='left')
agg = agg.merge(F, on=['node', 'socket'], how='inner')
print(f'socket-months with power + harvest map: {len(agg):,}')

# ---------------- paired within-node contrast ----------------
w = agg.pivot_table(index=['node', 'month'],
                    columns='socket',
                    values=['pX_power', 'pX_vdd_temp', 'mean_idx', 'spread', 'adj', 'lowhalf'])
w = w.dropna()
print(f'node-months with BOTH sockets measured: {len(w):,}')

out = []
for y in ['pX_power', 'pX_vdd_temp']:
    dy = w[(y, 0)] - w[(y, 1)]
    print(f'\n=== paired contrast, {y}:  mean(p0-p1) = {dy.mean():+.3f}, SD = {dy.std():.3f} ===')
    # is there ANY systematic p0/p1 offset (a socket-position effect)?
    tstat = dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy)))
    print(f'  socket-position offset: t = {tstat:+.1f} (n={len(dy)})')
    for x in ['mean_idx', 'spread', 'adj', 'lowhalf']:
        dx = w[(x, 0)] - w[(x, 1)]
        ok = dx.notna() & dy.notna()
        xv, yv = dx[ok].values, dy[ok].values
        if xv.std() == 0:
            print(f'  d{x}: zero variance, skipped'); continue
        r = np.corrcoef(xv, yv)[0, 1]
        # permutation p-value
        rng = np.random.default_rng(0)
        null = np.array([np.corrcoef(rng.permutation(xv), yv)[0, 1] for _ in range(2000)])
        p = (np.abs(null) >= abs(r)).mean()
        # slope in physical units
        b = np.polyfit(xv, yv, 1)[0]
        out.append((y, x, r, p, b))
        print(f'  d{x:<9} vs d{y:<12} r = {r:+.4f}   perm p = {p:.4f}   slope = {b:+.4f}')

R = pd.DataFrame(out, columns=['response', 'predictor', 'r', 'p', 'slope'])
R.to_csv(root / 'analysis' / 'a1_impact_results.csv', index=False)

# ---------------- effect-size context ----------------
pw = agg['pX_power'].dropna()
print(f'\ncontext: socket power mean {pw.mean():.1f} W, SD {pw.std():.1f} W, '
      f'IQR {pw.quantile(.25):.1f}-{pw.quantile(.75):.1f} W')
best = R.reindex(R.r.abs().sort_values(ascending=False).index).iloc[0]
print(f'strongest association: d{best.predictor} vs d{best.response}, '
      f'r = {best.r:+.4f} (p = {best.p:.4f}), explaining {100*best.r**2:.2f}% of variance')

# ---------------- equivalence framing ----------------
for y in ['pX_power']:
    dy = (w[(y, 0)] - w[(y, 1)])
    for x in ['mean_idx']:
        dx = w[(x, 0)] - w[(x, 1)]
        ok = dx.notna() & dy.notna(); dx, dy2 = dx[ok].values, dy[ok].values
        cf = np.polyfit(dx, dy2, 1); b = cf[0]
        se = np.sqrt(np.sum((dy2 - np.polyval(cf, dx)) ** 2) /
                     (len(dy2) - 2) / np.sum((dx - dx.mean()) ** 2))
        print(f'\nequivalence: slope of d{y} on d{x} = {b:+.3f} W per slice-index unit, '
              f'95% CI [{b-1.96*se:+.3f}, {b+1.96*se:+.3f}]')
        print(f'  full-range effect across observed d{x} '
              f'({dx.min():+.1f}..{dx.max():+.1f}): {abs(b)*(dx.max()-dx.min()):.2f} W '
              f'against a {pw.std():.1f} W socket-to-socket SD')
