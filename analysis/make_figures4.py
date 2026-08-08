"""Upgrades to two figures whose statistic, not whose styling, was the problem.

Fig. 3 (marginals by lot) plotted bare bars. Sockets in a rack are not independent -- the paper
measures that dependence directly -- so the bars needed rack-clustered uncertainty to be read
as estimates rather than as counts.

Fig. 4 (co-harvest structure) plotted z-scores against the curveball null. A z is a signal-to-
noise ratio: it grows with the square root of the sample size and so mostly reports how many
sockets a lot contains. The effect size is the log-ratio of observed to expected co-harvest,
which is what this version plots, with the null spread shown alongside so significance is still
legible.
"""
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

M = np.load(root / 'harvest_maps_full.npy')
K = np.load(root / 'harvest_keys_full.npy')
NP = 12
rack = K[:, 0] // 20
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


fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = np.arange(NP)
for (lab, m, rk, col, off) in [
        ('Lot A (racks 0--21)', M[rack < 22], rack[rack < 22], BLUE, -0.19),
        ('Lot B (racks 22--48)', M[rack >= 22], rack[rack >= 22], ORANGE, +0.19)]:
    p = m.mean(0)
    bt = rack_boot(m, rk)
    lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
    ax.bar(x + off, p, width=.36, color=col, label=lab, zorder=2)
    ax.errorbar(x + off, p, yerr=[p - lo, hi - p], fmt='none', ecolor=INK,
                lw=.9, capsize=1.6, zorder=3)
ax.axhline(1 / 3, color=GREY, lw=1.0, ls='--', zorder=1)
ax.annotate('uniform (4 of 12)', (11.4, 1 / 3), fontsize=6, color=INK,
            ha='right', va='bottom')
ax.set_xticks(x)
ax.set_xlabel('slice index (= position on the die)')
ax.set_ylabel('$P$(slice harvested)')
ax.legend(fontsize=6.2, frameon=False, loc='upper right')
fig.savefig(FIG / 'f2_marginals.pdf')
fig.savefig(SCRATCH / 'f2.png', dpi=150)
plt.close(fig)
print('wrote f2_marginals.pdf (rack-clustered bootstrap CIs)')


# ------------------------------------------------------------------ Fig 4: effect size, not z
def curveball(mat, n_swaps):
    m = mat.copy()
    rows = [set(np.flatnonzero(r)) for r in m]
    R = len(rows)
    for _ in range(n_swaps):
        i, j = rng.integers(0, R, 2)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        ab, ba = a - b, b - a
        if not ab or not ba:
            continue
        pool = list(ab | ba)
        rng.shuffle(pool)
        na = (a & b) | set(pool[:len(a) - len(a & b)])
        nb = (a | b) - na
        if len(na) == len(a) and len(nb) == len(b):
            rows[i], rows[j] = na, nb
    out = np.zeros_like(mat)
    for r, s in enumerate(rows):
        out[r, list(s)] = 1
    return out


def cooc(m):
    w = m.astype(np.int32)          # int8 matmul overflows at 127 -- must widen
    return w.T @ w


NSIM = 200
obs = cooc(M).astype(float)
null = np.zeros((NSIM, NP, NP))
cur = M.copy()
for s in range(NSIM):
    cur = curveball(cur, 20000)
    null[s] = cooc(cur)
exp = null.mean(0)
with np.errstate(divide='ignore', invalid='ignore'):
    lr = np.log2(np.maximum(obs, .5) / np.maximum(exp, .5))
np.fill_diagonal(lr, np.nan)

iu = np.triu_indices(NP, 1)
vals = lr[iu]
sd_null = np.log2(np.maximum(null, .5) / np.maximum(exp, .5)).std(0)[iu]
dist = np.abs(iu[0] - iu[1])
sig = np.abs(vals) > 2 * sd_null

# The claim this figure supports is that the co-harvest excess is confined to slices that are
# CLOSE on the die, so distance is the natural x-axis. Ranking 66 pairs on a categorical axis
# leaves ~3.6 pt per row at column width, which no font size or labelling scheme rescues.
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.axhline(0, color=INK, lw=.9, zorder=3)
jit = (rng.random(len(dist)) - .5) * .28
ax.errorbar(dist + jit, vals, yerr=2 * sd_null, fmt='none', ecolor=GREY, lw=.6,
            alpha=.8, zorder=2)
cols = np.where(vals > 0, ORANGE, BLUE)
ax.scatter(dist[sig] + jit[sig], vals[sig], s=17, c=cols[sig], zorder=4,
           edgecolors='white', linewidths=.6, label='$|$effect$|>2\\sigma_{\\mathrm{null}}$')
ax.scatter(dist[~sig] + jit[~sig], vals[~sig], s=13, facecolors='none',
           edgecolors=GREY, linewidths=.8, zorder=4, label='not significant')
mean_by_d = [vals[dist == k].mean() for k in range(1, NP)]
ax.plot(range(1, NP), mean_by_d, '-', color=INK, lw=1.3, zorder=5, label='mean by distance')
for i in np.argsort(-vals)[:3]:
    ax.annotate(f'{iu[0][i]}–{iu[1][i]}', (dist[i] + jit[i], vals[i]),
                textcoords='offset points', xytext=(5, 1), fontsize=6, color=ORANGE)
ax.set_xticks(range(1, NP))
ax.set_xlabel('slice index distance $|\\Delta k|$  (= physical distance, Sec. IV-F)')
ax.set_ylabel('$\\log_2\\dfrac{\\mathrm{observed}}{\\mathrm{expected}}$')
ax.legend(fontsize=5.8, frameon=False, loc='upper right', ncol=1)
fig.savefig(FIG / 'f3_cooccurrence.pdf')
fig.savefig(SCRATCH / 'f3.png', dpi=150)
plt.close(fig)
print('wrote f3_cooccurrence.pdf (log-ratio effect size, not z)')

adj = [(a, b) for a, b in zip(*iu) if abs(a - b) <= 2]
print(f'  mean log2 ratio, |dk|<=2 pairs : {np.nanmean([lr[a, b] for a, b in adj]):+.3f}')
far = [(a, b) for a, b in zip(*iu) if abs(a - b) > 2]
print(f'  mean log2 ratio, distant pairs : {np.nanmean([lr[a, b] for a, b in far]):+.3f}')
