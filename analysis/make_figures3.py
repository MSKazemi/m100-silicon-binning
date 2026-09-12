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


# ------------------------------------------------------------------ 2. model-replicate residual heatmap
# Simulation-calibrated residuals (ppc.py): each cell's observed-minus-expected co-harvest count
# standardised by its spread over 10,000 refitted parametric replicates. Pearson residuals
# (ppc_resid_fleet.npy) understate the misfit, because the replicate SD of a cell is only
# 0.40-0.81 sqrt(E). Dots mark cells outside the simultaneous 95% band (max-|z| bound).
resid = np.load(root / 'ppc_resid_fleet_simcal.npy')
crit = json.load(open(root / 'model_ppc.json'))['Fleet']['residual_fwer_crit_abs_z']
lim = np.abs(resid).max()
fig, ax = plt.subplots(figsize=(3.3, 2.9))
# diverging: two hues, neutral midpoint -- polarity data (observed above/below model)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'div', [BLUE, '#f7f7f5', ORANGE])
im = ax.imshow(resid, cmap=cmap, vmin=-lim, vmax=lim)
iy, ix = np.where((np.abs(resid) > crit) & ~np.eye(12, dtype=bool))
ax.plot(ix, iy, 'o', color=INK, ms=2.2, lw=0)
ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_xlabel('slice')
ax.set_ylabel('slice')
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
cb.set_label('calibrated residual $z$ (observed $-$ model)', fontsize=7)
cb.ax.tick_params(labelsize=6)
print(f'f13: {len(ix) // 2} off-diagonal cells beyond |z| > {crit:.2f}; colour limits +/-{lim:.2f}')
fig.savefig(FIG / 'f13_model_check.pdf'); fig.savefig(SCRATCH / 'f13.png', dpi=150)
plt.close(fig)
print('wrote f13_model_check.pdf')


# ------------------------------------------------------------------ 3. two null results
fit = json.load(open(root / 'survival_fit.json'))
cix = json.load(open(root / 'survival_cindex.json'))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.3))

# --- left: hazard-ratio forest (binary features 0->1, continuous per SD); survival.py v2:
# die-tracked endpoint, grouped proportional hazards with a monthly baseline, rack bootstrap.
names = {'s0': 'slice 0 harvested (0→1)', 'nadj': 'adjacent harvested pairs (SD)',
         'spread': 'index dispersion (SD)', 'rarity': 'pattern rarity (SD)',
         'lot': 'lot B vs lot A (0→1)'}
fs = fit['feats']
hr = np.array([fit['hr'][f] for f in fs])
lo = np.array([fit['hr_lo'][f] for f in fs])
hi = np.array([fit['hr_hi'][f] for f in fs])
y = np.arange(len(fs))
axL.axvspan(1 / 1.15, 1.15, color=GREY, alpha=.30, lw=0)
axL.axvline(1.0, color=INK, lw=.9)
axL.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt='o', color=BLUE, ms=5, lw=1.4, capsize=2.5)
axL.set_yticks(y)
axL.set_yticklabels([names[f] for f in fs], fontsize=7)
axL.set_xscale('log')
axL.set_xticks([0.33, 0.5, 1.0, 2.0, 3.0])
axL.set_xticks([], minor=True)
axL.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
axL.set_xlim(0.28, 3.8)
axL.set_xlabel('replacement hazard ratio (95% rack-bootstrap CI)')
axL.invert_yaxis()
c, (clo, chi) = cix['cindex'], cix['cindex_ci95']
axL.set_title(f'no map feature significantly predicts replacement\n'
              f'{fit["n_events"]} events, {fit["n_months"]:,} die-months; '
              f'held-out C = {c:.2f} [{clo:.2f}, {chi:.2f}]', fontsize=7.5, color=INK)

# --- right: side-channel recovery against the mean number of samples per core surviving into
# the window (side_channel.py / side_channel_raw.py). Curves: independent (Poisson) deletion
# with a surviving sample needed on every active core, or on every active slice; circles:
# independent deletion of the empirical per-core counts; squares: whole timestamps deleted
# (simulated on the raw 20 s presence); diamonds: no deletion, measured.
sc = json.load(open(root / 'side_channel_results.json'))
lam = np.logspace(-1, 5, 500)
axR.plot(lam, (1 - np.exp(-lam)) ** 16, '-', color=BLUE, lw=1.8,
         label='independent deletion, per-core rule')
axR.plot(lam, (1 - np.exp(-2 * lam)) ** 8, '--', color=BLUE, lw=1.0,
         label='independent deletion, slice rule')
F = sc['iid']['f']
pts = [(1, 0.999), (2, 0.999), (7, 0.9999), (31, 0.9999), (31, 0.99999)]
for D, f in pts:
    w = sc['iid']['windows'][str(D)]
    j = F.index(f)
    axR.plot(w['mean_surviving_per_core'][j], w['p_exact'][j], 'o', color=BLUE, ms=4.5,
             mfc='white', mew=1.2, zorder=3)
axR.plot([], [], 'o', color=BLUE, ms=4.5, mfc='white', mew=1.2,
         label='empirical counts, 1 d to 1 month')
n1 = sc['iid']['windows']['1']['mean_samples_per_core']
for r in sc['raw']['timestamp_deletion']:
    if r['f'] >= 0.999:
        axR.plot(n1 * (1 - r['f']), r['measured'], 's', color=ORANGE, ms=4.5,
                 mec='white', mew=.8, zorder=4)
axR.plot([], [], 's', color=ORANGE, ms=4.5, label='whole timestamps deleted')
axR.plot([], [], 'D', color=GREEN, ms=4.5, label='no deletion, measured')
one = F.index(0.999)
x1, y1 = (sc['iid']['windows']['1']['mean_surviving_per_core'][one],
          sc['iid']['windows']['1']['p_exact'][one])
arrow = dict(arrowstyle='-', lw=.7, shrinkA=0, shrinkB=3)
axR.annotate(f'99.9% deleted\n1 day: {y1:.2f}', (x1, y1), xytext=(0.034, 0.60),
             textcoords='data', fontsize=6.3, color=BLUE, ha='left', va='center',
             arrowprops=dict(color=BLUE, **arrow))
m31 = sc['iid']['windows']['31']
xm, ym = m31['mean_surviving_per_core'][F.index(0.9999)], m31['p_exact'][F.index(0.9999)]
axR.annotate(f'99.99% deleted\n1 month: {ym:.2f}', (xm, ym), xytext=(30, 0.60),
             textcoords='data', fontsize=6.3, color=BLUE, ha='left', va='center',
             arrowprops=dict(color=BLUE, **arrow))
pub = sc['raw']['window'][0]['p_exact']                    # one 20 s interval, measured
axR.plot([n1 / 4320], [pub], 'D', color=GREEN, ms=5, mec='white', mew=1, zorder=4)
axR.annotate('one 20 s\ninterval', (n1 / 4320, pub),
             xytext=(0.034, 0.86), textcoords='data', fontsize=6.3, color=GREEN, ha='left',
             va='center', arrowprops=dict(color=GREEN, **arrow))
med = sc['single_interval_model']['median_samples_per_core']
day1 = sc['measured_days'][0]['exact']
axR.plot([med], [day1], 'D', color=GREEN, ms=5, mec='white', mew=1, zorder=4)
axR.annotate(f'as published:\n{med:,.0f}/core/day', (med, day1),
             xytext=(700, 0.80), textcoords='data', fontsize=6.3, color=GREEN, ha='left',
             va='center', arrowprops=dict(color=GREEN, **arrow))
axR.set_xscale('log')
axR.set_xlim(0.028, 2e5)
axR.set_xlabel('mean surviving samples per core in the window')
axR.set_ylabel('P(exact map recovered)')
axR.set_ylim(-.05, 1.12)
axR.legend(fontsize=5.4, frameon=False, loc='lower right', handletextpad=.3, borderaxespad=.2,
           labelspacing=.25)
axR.set_title('pooling days undoes random deletion', fontsize=7.5, color=INK)
print(f'f14 left: {fit["n_events"]} events, {fit["n_months"]} die-months, C {c:.3f} '
      f'[{clo:.3f}, {chi:.3f}]; right: 99.9% 1 d {y1:.4f}, 99.99% 31 d {ym:.5f}, '
      f'one interval {pub:.5f}, one day {day1:.5f}')

fig.tight_layout()
fig.savefig(FIG / 'f14_null_results.pdf'); fig.savefig(SCRATCH / 'f14.png', dpi=150)
plt.close(fig)
print('wrote f14_null_results.pdf')
