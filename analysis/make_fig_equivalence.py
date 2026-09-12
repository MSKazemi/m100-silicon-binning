"""Fig. 15: equivalence forest for the harvest map's association with socket power.

Reads analysis/equivalence_fit.json (written by equivalence.py). For each map summary it shows the
effect over the summary's full observed range (the TOST quantity, one common scale for every
design) with its rack-clustered 90% interval, under four designs:

  all node-days     contemporaneous maps, 28 steady-state months (the headline)
  idle node-days    cpu_user < 0.5 %, load_one < 0.5: almost no workload to split between sockets
  placement-adj.    2020-06..10, adjusted for SLURM core placement on each socket
  maps never change the nodes whose two sockets carry one map over the whole record

The +/-5 W band is the headline margin; dotted lines mark +/-2 and +/-10 W so the verdict at other
margins can be read off. The previous version's estimate (joint fit, partial slope times marginal
range) is drawn in grey on the headline row for comparison.
"""
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

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
NICE = {'meanidx': 'mean active-\nslice index', 'nadj': 'adjacent\nactive pairs',
        'lowhalf': 'active slices\nin low half'}
ORDER = ['meanidx', 'lowhalf', 'nadj']
DESIGNS = [('epoch', 'single-adjusted', 'o', 'all node-days (headline)'),
           ('idle', 'single-adjusted', 's', 'idle node-days'),
           ('placement_with', 'single-adjusted', '^', 'adjusted for SLURM placement'),
           ('stable', 'single-adjusted', 'D', 'nodes whose maps never change')]
OFF = np.linspace(-0.3, 0.3, len(DESIGNS))


def row(design, model, feature):
    for r in R['rows']:
        if r['design'] == design and r['model'] == model and r['feature'] == feature:
            return r
    raise KeyError((design, model, feature))


fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.axvspan(-margin, margin, color=GREEN, alpha=.14, lw=0, zorder=0)
for x in (-margin, margin):
    ax.axvline(x, color=GREEN, lw=.9, ls='--', zorder=1)
for x in (-10, -2, 2, 10):          # the +/-2 W guides stop short of the margin label
    ax.axvline(x, color=GREY, lw=.7, ls=':', zorder=1, ymax=.9 if abs(x) == 2 else 1)
ax.axvline(0, color=INK, lw=.8, zorder=1)

for gi, f in enumerate(ORDER):
    for k, (d, m, mk, _) in enumerate(DESIGNS):
        r = row(d, m, f)
        y = gi + OFF[k]
        e, lo, hi = r['range']['eff'], r['range']['lo'], r['range']['hi']
        col = GREEN if r['equivalent'] else ORANGE
        big = d == 'epoch'
        cont = ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt=mk, color=col,
                           ms=5.2 if big else 3.6, lw=1.6 if big else 1.0,
                           capsize=2 if big else 1.4,
                           mec=col, mfc=col if big else 'white', zorder=4)
        # Colour is not the only cue for the verdict: an interval that leaves the margin is drawn
        # dashed, and the legend names both states in words.
        if not r['equivalent']:
            for seg in cont[2]:
                seg.set_linestyle((0, (2.2, 1.4)))
    old = row('modal', 'joint-adjusted', f)
    ax.plot(old['range']['eff'], gi + OFF[0], marker='x', color=GREY, ms=4.5, mew=1.2, zorder=3)
    if gi < len(ORDER) - 1:
        ax.axhline(gi + .5, color='#e2e8f0', lw=.6, zorder=0)

ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([NICE[f] for f in ORDER], fontsize=6.8)
ax.set_ylim(len(ORDER) - .45, -0.95)
# symmetric, so that both +/-10 W guides named in the caption are inside the axes
ax.set_xlim(-11, 11)
ax.set_xticks([-10, -5, -2, 0, 2, 5, 10])
ax.set_xticklabels(['$-$10', '$-$5', '$-$2', '0', '2', '5', '10'])
ax.grid(axis='y', visible=False)
ax.text(0, -0.72, f'$\\pm${margin:.0f} W margin', fontsize=6.2, color=GREEN, ha='center',
        va='center')
ax.set_xlabel('association with p0$-$p1 socket power over the full observed\n'
              'range of the map-summary difference (W), 90% CI', fontsize=7)
hand = [Line2D([], [], marker=mk, ls='', color=INK, mfc=INK if d == 'epoch' else 'white',
               ms=4 if d == 'epoch' else 3.4, label=lab) for d, _, mk, lab in DESIGNS]
hand.append(Line2D([], [], marker='x', ls='', color=GREY, ms=4, mew=1.2,
                   label='previous version (joint fit)'))
hand.append(Line2D([], [], ls='-', color=GREEN, lw=1.4,
                   label=f'inside the $\\pm${margin:.0f} W margin (equivalent)'))
hand.append(Line2D([], [], ls=(0, (2.2, 1.4)), color=ORANGE, lw=1.4,
                   label='outside it (not equivalent)'))
ax.legend(handles=hand, loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=5.8,
          frameon=False, handletextpad=.3, columnspacing=.8)
fig.savefig(FIG / 'f11_operational_impact.pdf')
fig.savefig(SCRATCH / 'f11.png', dpi=150)
plt.close(fig)
print('wrote f11_operational_impact.pdf (equivalence forest)')
for f in ORDER:
    for d, m, _, lab in DESIGNS:
        r = row(d, m, f)
        print(f"  {f:<8} {lab:<31} {r['range']['eff']:+6.2f} [{r['range']['lo']:+.2f},"
              f"{r['range']['hi']:+.2f}]  {'EQUIV' if r['equivalent'] else 'no'}")
    o = row('modal', 'joint-adjusted', f)
    print(f"  {f:<8} {'previous version (joint fit)':<31} {o['range']['eff']:+6.2f}")
