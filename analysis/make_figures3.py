"""Figures for the analyses added in response to review.

Palette note. The paper's earlier house colours paired red #c53030 with orange #dd6b20 to
distinguish the two thermal series -- the single most important contrast in the paper. Those two
sit at Delta-E 13.1 for normal vision, below the legibility floor of 15, and closer still under
simulated deuteranopia. They are replaced here by blue/orange, which clears every check
(worst adjacent pair Delta-E 31.2 normal, 22.8 protan). Grey is used for non-data only --
null envelopes, reference lines, baselines -- never as a categorical series, since it fails the
chroma floor by construction.
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
FIG.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))

BLUE, ORANGE, GREEN = '#2b6cb0', '#dd6b20', '#2f855a'
GREY, INK = '#a0aec0', '#2d3748'                     # non-data only
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25,
                     'grid.linewidth': .5, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.edgecolor': '#cbd5e0', 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})


# ------------------------------------------------------------------ 1. changepoint profile
prof = np.load(root / 'changepoint_profile.npy')
null = np.load(root / 'changepoint_null.npy')
cands, tvals = prof[0], prof[1]
env = np.percentile(null, 95)
env99 = np.percentile(null, 99)

fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.axhspan(-env, env, color=GREY, alpha=.35, lw=0,
           label='rack-permutation null, 95%')
ax.axhline(env99, color=GREY, lw=.8, ls=':')
ax.axhline(-env99, color=GREY, lw=.8, ls=':', label='99%')
ax.plot(cands, tvals, '-', color=BLUE, lw=1.6, label='observed Welch $t$')
k = int(np.argmax(np.abs(tvals)))
ax.plot([cands[k]], [tvals[k]], 'o', color=ORANGE, ms=6, zorder=5,
        markeredgecolor='white', markeredgewidth=1.2)
ax.annotate(f'rack {int(cands[k])}\n$t={tvals[k]:.1f}$', (cands[k], tvals[k]),
            textcoords='offset points', xytext=(9, -4), fontsize=7, color=ORANGE, weight='bold')
ax.set_xlabel('candidate boundary (rack index)')
ax.set_ylabel('Welch $t$, $P$(slice 0)')
ax.legend(fontsize=6, frameon=False, loc='lower left')
fig.savefig(FIG / 'f12_changepoint.pdf'); fig.savefig(SCRATCH / 'f12.png', dpi=150)
plt.close(fig)
print('wrote f12_changepoint.pdf')


# ------------------------------------------------------------------ 2. PPC residual heatmap
resid = np.load(root / 'ppc_resid_fleet.npy')
lim = np.abs(resid).max()
fig, ax = plt.subplots(figsize=(3.0, 2.5))
# diverging: two hues, neutral midpoint -- polarity data (observed above/below model)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'div', [BLUE, '#f7f7f5', ORANGE])
im = ax.imshow(resid, cmap=cmap, vmin=-lim, vmax=lim)
ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_xlabel('slice')
ax.set_ylabel('slice')
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
cb.set_label('Pearson residual\n(observed $-$ model)', fontsize=6.5)
cb.ax.tick_params(labelsize=6)
fig.savefig(FIG / 'f13_model_check.pdf'); fig.savefig(SCRATCH / 'f13.png', dpi=150)
plt.close(fig)
print('wrote f13_model_check.pdf')


# ------------------------------------------------------------------ 3. two null results
fit = json.load(open(root / 'survival_fit.json'))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.3))

# --- left: hazard-ratio forest
names = {'s0': 'slice 0 harvested', 'nadj': 'adjacent harvested pairs',
         'spread': 'index dispersion', 'rarity': 'pattern rarity', 'lot': 'lot B'}
fs = fit['feats']
hr = [np.exp(fit['beta'][j + 1]) for j in range(len(fs))]
lo = [np.exp(fit['lo'][j + 1]) for j in range(len(fs))]
hi = [np.exp(fit['hi'][j + 1]) for j in range(len(fs))]
y = np.arange(len(fs))
axL.axvspan(1 / 1.15, 1.15, color=GREY, alpha=.30, lw=0)
axL.axvline(1.0, color=INK, lw=.9)
axL.errorbar(hr, y, xerr=[np.array(hr) - np.array(lo), np.array(hi) - np.array(hr)],
             fmt='o', color=BLUE, ms=5, lw=1.4, capsize=2.5)
axL.set_yticks(y)
axL.set_yticklabels([names[f] for f in fs], fontsize=7)
axL.set_xscale('log')
axL.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0])
axL.set_xticks([], minor=True)
axL.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
axL.set_xlim(0.45, 2.2)
axL.set_xlabel('replacement hazard ratio per SD  (95% CI)')
axL.invert_yaxis()
axL.set_title(f'no map feature predicts replacement\n'
              f'{fit["n_events"]} events, {fit["n_months"]:,} socket-months',
              fontsize=7.5, color=INK)

# --- right: side-channel recovery curve.
# The axis runs all the way to the published operating point, so the margin between "enough to
# recover the map" and "what the dataset actually ships" is visible rather than asserted.
lam = np.logspace(-1, 4, 400)
axR.plot(lam, (1 - np.exp(-lam)) ** 16, '-', color=BLUE, lw=1.8,
         label='modelled, 16 cores')
arrow = dict(arrowstyle='-', lw=.7, shrinkA=0, shrinkB=3)
axR.plot([7], [(1 - np.exp(-7)) ** 16], 'o', color=ORANGE, ms=5.5,
         markeredgecolor='white', markeredgewidth=1)
axR.annotate('99% recovered after\n7 samples ($\\approx$140 s)', (7, (1 - np.exp(-7)) ** 16),
             xytext=(0.13, 0.62), textcoords='data', fontsize=6.5, color=ORANGE,
             ha='left', va='center', arrowprops=dict(color=ORANGE, **arrow))
axR.plot([4320], [0.997], 's', color=GREEN, ms=5.5,
         markeredgecolor='white', markeredgewidth=1)
axR.annotate('as published:\n4,320 samples/core/day', (4320, 0.997),
             xytext=(120, 0.30), textcoords='data', fontsize=6.5, color=GREEN,
             ha='left', va='center', arrowprops=dict(color=GREEN, **arrow))
axR.set_xscale('log')
axR.set_xlabel('samples per core in the observation window')
axR.set_ylabel('P(exact map recovered)')
axR.set_ylim(-.05, 1.12)
axR.legend(fontsize=6, frameon=False, loc='upper left')
axR.set_title('the disclosure costs about two minutes', fontsize=7.5, color=INK)

fig.tight_layout()
fig.savefig(FIG / 'f14_null_results.pdf'); fig.savefig(SCRATCH / 'f14.png', dpi=150)
plt.close(fig)
print('wrote f14_null_results.pdf')
