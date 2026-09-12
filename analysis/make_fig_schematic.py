"""Fig. 2 -- POWER9 slice schematic annotated with the measured per-slice harvest rate
(paper/figures/f1_slice_schematic.pdf).

Split out of make_figures.py (block F1), which also needs the raw IPMI archive; this script
needs only the committed full-record modal maps (harvest_maps_full.npy), so it is deterministic
and runs in CI. Slices are drawn in sensor-index order, as one row; the brackets above the row
mark the six quads (EQ0..EQ5 = slices {2q, 2q+1}), which IBM's chip diagram places in two bands
of three (Registers Spec v1.2, Fig. 2, p. 33). The physical layout is documented, not inferred.

    python analysis/make_fig_schematic.py [--no-quads] [--out DIR]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--no-quads', action='store_true', help='omit the quad brackets (legacy drawing)')
ap.add_argument('--out', default=str(root.parent / 'paper' / 'figures'))
args = ap.parse_args()
FIG = Path(args.out); FIG.mkdir(parents=True, exist_ok=True)
NP = 12
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#4a5568'

M = np.load(root / 'harvest_maps_full.npy')
print(f'sockets {len(M)}')

fig, ax = plt.subplots(figsize=(7.2, 2.5))
rate = M.mean(0)
for k in range(NP):
    x = k * 1.0
    col = plt.cm.Reds(0.15 + 1.6 * (rate[k] - rate.min()))
    ax.add_patch(Rectangle((x, 1.05), .92, .55, fc=col, ec='k', lw=.7))
    ax.text(x + .23, 1.32, f'{2*k}', ha='center', va='center', fontsize=7)
    ax.text(x + .69, 1.32, f'{2*k+1}', ha='center', va='center', fontsize=7)
    ax.plot([x + .46, x + .46], [1.07, 1.58], color='k', lw=.4, ls=':')
    # L2 sits between the cores and L3; each row is labelled on the left, units in the boxes
    ax.add_patch(Rectangle((x, .83), .92, .17, fc='#e2e8f0', ec='k', lw=.5))
    ax.text(x + .46, .915, '512 kB', ha='center', va='center', fontsize=6)
    ax.add_patch(Rectangle((x, .57), .92, .22, fc='#c6f6d5', ec='k', lw=.7))
    ax.text(x + .46, .68, '10 MB', ha='center', va='center', fontsize=6)
    ax.annotate('', (x + .46, .55), (x + .46, .30), arrowprops=dict(arrowstyle='<->', lw=.8, color='#38a169'))
    ax.text(x + .46, 1.68, f'{100*rate[k]:.0f}%', ha='center', fontsize=7,
            color=RED if rate[k] > .45 else GREY, fontweight='bold' if rate[k] > .45 else 'normal')
    ax.text(x + .46, 1.90, f'S{k}', ha='center', fontsize=7, color=BLUE)
ax.add_patch(Rectangle((0, .12), NP * 1.0 - .08, .18, fc='#2d3748', ec='k'))
ax.text(NP / 2, .21, 'on-chip fabric:  7 TB/s switch,  256 GB/s $\\times$ 12',
        ha='center', va='center', color='w', fontsize=7)
ax.text(-.15, 1.90, 'slice', ha='right', fontsize=7, color=BLUE)
ax.text(-.15, 1.68, 'harvested', ha='right', fontsize=7, color=RED)
ax.text(-.15, 1.32, 'SMT4 cores', ha='right', fontsize=7)
ax.text(-.15, .915, 'L2', ha='right', va='center', fontsize=7)
ax.text(-.15, .68, 'L3', ha='right', va='center', fontsize=7)
ymax = 2.05
if not args.no_quads:
    # quad brackets: slices 2q and 2q+1 share a quad (one EQ, one V/F domain); the label row
    # sits above the slice labels, and the sentence in the caption says the row is schematic
    yb = 2.09
    for q in range(6):
        x0, x1 = 2 * q + .04, 2 * q + 1.88
        ax.plot([x0, x0, x1, x1], [yb, yb + .07, yb + .07, yb], color=GREY, lw=.7)
        ax.text((x0 + x1) / 2, yb + .12, f'quad {q}', ha='center', va='bottom', fontsize=6.5, color=GREY)
    ax.text(-.15, yb + .12, 'quad', ha='right', va='bottom', fontsize=7, color=GREY)
    ymax = 2.40
ax.set_xlim(-2.2, NP + .2); ax.set_ylim(.05, ymax); ax.axis('off'); ax.grid(False)
fig.savefig(FIG / 'f1_slice_schematic.pdf'); plt.close(fig)
print('wrote', FIG / 'f1_slice_schematic.pdf', '(quads omitted)' if args.no_quads else '(with quad brackets)')
