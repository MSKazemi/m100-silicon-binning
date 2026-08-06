"""F11 -- operational impact of the harvest map (A1). Effect-size comparison, not a p-value."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

root = Path(__file__).resolve().parent.parent
FIG = root / 'paper' / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#4a5568'

M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
P = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power').glob('power_*.parquet'))],
              ignore_index=True)
def feats(v):
    act = np.flatnonzero(v == 0)
    return dict(mean_idx=act.mean(), spread=act.std(),
                adj=sum(1 for a, b in combinations(act, 2) if abs(a - b) == 1))
F = pd.DataFrame([feats(v) for v in M]); F['node'] = K[:, 0]; F['socket'] = K[:, 1]
P['month'] = P.day.dt.strftime('%y-%m')
sock = P[(P.socket >= 0) & (P['count'] > 0.9 * 4320)]
agg = sock.pivot_table(index=['node', 'socket', 'month'], columns='metric',
                       values='mean', aggfunc='mean').reset_index().merge(F, on=['node', 'socket'])
w = agg.pivot_table(index=['node', 'month'], columns='socket',
                    values=['pX_power', 'mean_idx']).dropna()
dx = (w[('mean_idx', 0)] - w[('mean_idx', 1)]).values
dy = (w[('pX_power', 0)] - w[('pX_power', 1)]).values
cf = np.polyfit(dx, dy, 1)
r = np.corrcoef(dx, dy)[0, 1]
sd_socket = agg['pX_power'].std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={'width_ratios': [1.15, 1]})
ax1.plot(dx, dy, '.', ms=1.6, alpha=.18, color=BLUE, rasterized=True)
xs = np.linspace(dx.min(), dx.max(), 50)
ax1.plot(xs, np.polyval(cf, xs), '-', color=RED, lw=1.8,
         label=f'slope {cf[0]:+.2f} W/unit\n$r$={r:+.3f}, $r^2$={100*r**2:.2f}\\%')
ax1.axhline(0, color='k', lw=.6)
ax1.set_xlabel('$\\Delta$ mean active-slice index (p0 $-$ p1)')
ax1.set_ylabel('$\\Delta$ socket power [W]')
ax1.legend(fontsize=6.5, frameon=False, loc='upper left')
ax1.set_title(f'within-node paired contrast, n={len(dx):,} node-months', fontsize=7.5)

labels = ['harvest map\n(full observed range)', 'socket position\n(p0 vs p1 offset)',
          'socket-to-socket\nspread (1 SD)']
vals = [abs(cf[0]) * (dx.max() - dx.min()), abs(dy.mean()), sd_socket]
cols = [BLUE, '#dd6b20', GREY]
ax2.barh(range(3), vals, color=cols)
for i, v in enumerate(vals):
    ax2.text(v + .5, i, f'{v:.1f} W', va='center', fontsize=7.5, fontweight='bold')
    ax2.text(.4, i - .40, labels[i].replace('\n', ' '), va='bottom', ha='left',
             fontsize=6.0, color='#1a202c',
             bbox=dict(fc='white', ec='none', alpha=.75, pad=.8))
ax2.set_yticks([])
ax2.set_xlabel('effect on socket power [W]'); ax2.set_xlim(0, max(vals) * 1.30)
ax2.invert_yaxis()
ax2.set_title('the harvest map is the smallest effect present', fontsize=7.5)
fig.savefig(FIG / 'f11_operational_impact.pdf'); plt.close(fig)
print(f'F11 written. slope={cf[0]:+.3f} r={r:+.4f} range_effect={vals[0]:.2f}W '
      f'position={vals[1]:.2f}W sd={vals[2]:.2f}W')
