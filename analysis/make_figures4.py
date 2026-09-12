"""Upgrades to two figures whose statistic, not whose styling, was the problem.

Fig. 3 (marginals by lot) plotted bare bars. Sockets in a rack are not independent -- the paper
measures that dependence directly -- so the bars needed rack-clustered uncertainty to be read
as estimates rather than as counts.

Fig. 4 (co-harvest structure) plotted z-scores against the curveball null. A z is a signal-to-
noise ratio: it grows with the square root of the sample size and so mostly reports how many
sockets a lot contains. The effect size is the log-ratio of observed to expected co-harvest,
which is what this version plots. The grey bars are the simultaneous 95% band of the curveball
null (the spread of the null, not a confidence interval for the effect), and filled points are
family-wise significant by the max-|z| test, both read from analysis/model_curveball.json.

Usage: make_figures4.py [fig3] [fig4]   (default: both; `fig4` alone leaves Fig. 3 untouched)
"""
import json
import os
import sys
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

M = np.load(root / 'harvest_maps_full.npy')
K = np.load(root / 'harvest_keys_full.npy')
NP = 12
rack = K[:, 0] // 20
# Fig. 3 uses the rack-indexed population of Sec. IV-A: the 49 documented cabinets (racks 0-48).
# Nodes 980-983 (rack index 49, 2 mapped sockets) have no documented floor position; earlier
# versions resampled them as a 28th Lot B cluster.
DOC = rack < 49
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ Fig 3: marginals + rack CI
def rack_boot(Msub, rk, B=2000):
    """Rack-clustered bootstrap of the 12 per-slice harvest rates."""
    racks = np.unique(rk)
    idx = {r: np.where(rk == r)[0] for r in racks}
    out = np.empty((B, NP))
    for b in range(B):
        pick = rng.choice(racks, len(racks), replace=True)
        sel = np.concatenate([idx[r] for r in pick])
        out[b] = Msub[sel].mean(0)
    return out


def fig3():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(NP)
    for (lab, m, rk, col, off, hatch) in [
            ('Lot A (racks 0\u201321)', M[rack < 22], rack[rack < 22], BLUE, -0.19, None),
            ('Lot B (racks 22\u201348)', M[(rack >= 22) & DOC], rack[(rack >= 22) & DOC], ORANGE,
             +0.19, '////')]:
        p = m.mean(0)
        bt = rack_boot(m, rk)
        lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
        ax.bar(x + off, p, width=.36, color=col, label=f'{lab}, {len(m):,} sockets', zorder=2,
               hatch=hatch, edgecolor='white' if hatch else None, linewidth=0)
        ax.errorbar(x + off, p, yerr=[p - lo, hi - p], fmt='none', ecolor=INK,
                    lw=.9, capsize=1.6, zorder=3)
    ax.axhline(1 / 3, color=GREY, lw=1.0, ls='--', zorder=1)
    ax.annotate('uniform (4 of 12)', (11.4, 1 / 3), fontsize=6, color=INK,
                ha='right', va='bottom')
    ax.set_xticks(x)
    ax.set_xlabel('slice (sensor index)')
    ax.set_ylabel('$P$(slice harvested)')
    ax.legend(fontsize=6.2, frameon=False, loc='upper right')
    fig.savefig(FIG / 'f2_marginals.pdf', metadata={'CreationDate': None})
    fig.savefig(SCRATCH / 'f2.png', dpi=150)
    plt.close(fig)
    print('wrote f2_marginals.pdf (rack-clustered bootstrap CIs)')


# ------------------------------------------------------------------ Fig 4: effect size, not z
# The null is no longer re-simulated here. analysis/curveball_null.py draws 100,000 exchangeable
# curveball null matrices (Besag-Clifford parallel construction, corrected trade) and stores, per
# pair, the observed count, the null mean and SD, the simultaneous max-|z| band and the family-wise
# decision. This figure only plots them, so it cannot drift from the numbers in the text.
def fig4():
    cb = json.load(open(root / 'model_curveball.json'))['main']
    P = cb['pairs']
    vals = np.array([q['log2_ratio'] for q in P])
    dist = np.array([q['dk'] for q in P])
    lo = np.array([q['simult_log2_lo'] for q in P])
    hi = np.array([q['simult_log2_hi'] for q in P])
    sig = np.array([q['sig_maxT'] for q in P])
    pairs = [tuple(q['pair']) for q in P]
    jr = np.random.default_rng(7)

    # The claim this figure supports is that the co-harvest excess is confined to slices that are
    # CLOSE in index, so distance is the natural x-axis. Ranking 66 pairs on a categorical axis
    # leaves ~3.6 pt per row at column width, which no font size or labelling scheme rescues.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(0, color=INK, lw=.9, zorder=3)
    jit = (jr.random(len(dist)) - .5) * .28
    ax.vlines(dist + jit, lo, hi, color=GREY, lw=.8, alpha=.55, zorder=2,
              label='null spread (simultaneous 95%)')
    # Significant excesses and significant DEFICITS are both filled, in two colours, so each one
    # needs its own legend entry: a single entry would label 11 of the 20 significant points with
    # the other sign's swatch.
    for mask, col, lab in ((sig & (vals > 0), ORANGE, 'family-wise significant (excess)'),
                           (sig & (vals <= 0), BLUE, 'family-wise significant (deficit)')):
        ax.scatter(dist[mask] + jit[mask], vals[mask], s=17, c=col, zorder=4,
                   edgecolors='white', linewidths=.6, label=lab)
    ax.scatter(dist[~sig] + jit[~sig], vals[~sig], s=13, facecolors='none',
               edgecolors=GREY, linewidths=.8, zorder=4, label='not significant')
    # the six slice pairs (2m, 2m+1) that share one POWER9 quad (IBM EQ m; Sec. IV-D/F)
    quad = np.array([a // 2 == b // 2 for a, b in pairs])
    ax.scatter(dist[quad] + jit[quad], vals[quad], s=34, marker='s', facecolors='none',
               edgecolors=INK, linewidths=.6, zorder=5, label='same quad (slices $2m, 2m{+}1$)')
    mean_by_d = [vals[dist == k].mean() for k in range(1, NP)]
    ax.plot(range(1, NP), mean_by_d, '-', color=INK, lw=1.3, zorder=5, label='mean by distance')
    top = np.argsort(-vals)[:3]
    for i, dy in zip(top[np.argsort(-vals[top])], (1, 4, -7)):
        ax.annotate(f'{pairs[i][0]}–{pairs[i][1]}', (dist[i] + jit[i], vals[i]),
                    textcoords='offset points', xytext=(6, dy), fontsize=6, color=ORANGE)
    ax.set_xticks(range(1, NP))
    ax.set_xlabel('slice index distance $|\\Delta k|$')
    ax.set_ylabel('$\\log_2\\dfrac{\\mathrm{observed}}{\\mathrm{expected}}$')
    ax.legend(fontsize=5.6, frameon=False, loc='upper right', ncol=1)
    fig.savefig(FIG / 'f3_cooccurrence.pdf', metadata={'CreationDate': None})
    fig.savefig(SCRATCH / 'f3.png', dpi=150)
    plt.close(fig)
    print(f'wrote f3_cooccurrence.pdf (log-ratio effect size; {int(sig.sum())} family-wise '
          f'significant pairs of 66 against B = {cb["B"]:,} exchangeable null draws)')
    print(f'  mean log2 ratio, |dk|<=2 pairs : {vals[dist <= 2].mean():+.3f}')
    print(f'  mean log2 ratio, distant pairs : {vals[dist > 2].mean():+.3f}')


if __name__ == '__main__':
    which = set(sys.argv[1:]) or {'fig3', 'fig4'}
    if 'fig3' in which:
        fig3()
    if 'fig4' in which:
        fig4()
