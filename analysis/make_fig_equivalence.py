"""Fig. 12 replacement: an equivalence forest plot.

The figure it replaces showed a scatter of the within-node power contrast and asserted a null.
A forest plot with a shaded indifference band shows the thing the claim actually depends on --
whether each interval fits inside the margin -- and makes an inconclusive result look
inconclusive rather than negative.
"""
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

root = Path(__file__).resolve().parent
FIG = root.parent / 'paper' / 'figures'
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))
BLUE, ORANGE, GREEN = '#2b6cb0', '#dd6b20', '#2f855a'
GREY, INK = '#a0aec0', '#2d3748'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25,
                     'grid.linewidth': .5, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.edgecolor': '#cbd5e0', 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})

R = json.load(open(root / 'equivalence_fit.json'))
margin = R['margin']
NICE = {'meanidx': 'mean active-slice index', 'nadj': 'adjacent active pairs',
        'lowhalf': 'active slices in low half'}

rows = [r for r in R['rows'] if r['model'] == 'adjusted']
rows += [r for r in R['rows'] if r['model'] == 'unadjusted']

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.axvspan(-margin, margin, color=GREEN, alpha=.16, lw=0)
ax.axvline(0, color=INK, lw=.9, zorder=1)
for x in (-margin, margin):
    ax.axvline(x, color=GREEN, lw=.9, ls='--', zorder=1)

y = np.arange(len(rows))
for i, r in enumerate(rows):
    col = GREEN if r['equivalent'] else ORANGE
    ax.errorbar(r['eff'], i, xerr=[[r['eff'] - r['lo']], [r['hi'] - r['eff']]],
                fmt='o', color=col, ms=5, lw=1.5, capsize=2.5, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([f"{NICE[r['feature']]}" for r in rows], fontsize=6.8)
ax.set_ylim(len(rows) - .3, -1.15)          # headroom for the group labels; y runs downward
n_adj = sum(1 for r in R['rows'] if r['model'] == 'adjusted')
ax.axhline(n_adj - .5, color=GREY, lw=.7)
# rows 0..n_adj-1 are the adjusted fits and sit at the TOP, since y increases downward
ax.text(0.02, 0.985, 'covariate-adjusted', transform=ax.transAxes, fontsize=6.2,
        color=INK, ha='left', va='top', style='italic')
ax.text(0.02, (1 - (n_adj + .05) / len(rows)) * 0.86, 'unadjusted', transform=ax.transAxes,
        fontsize=6.2, color=INK, ha='left', va='top', style='italic')
ax.annotate(f'$\\pm${margin:.0f} W indifference band', (0, -1.05), fontsize=6.2,
            color=GREEN, ha='center', va='bottom')
ax.set_xlabel('effect on socket power over the feature range (W), 90% CI')
fig.savefig(FIG / 'f11_operational_impact.pdf')
fig.savefig(SCRATCH / 'f11.png', dpi=150)
plt.close(fig)
print('wrote f11_operational_impact.pdf (equivalence forest)')
for r in rows:
    print(f"  {r['model']:<11} {r['feature']:<9} {r['eff']:+7.2f} "
          f"[{r['lo']:+.2f},{r['hi']:+.2f}]  {'EQUIV' if r['equivalent'] else 'no'}")
