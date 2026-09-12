"""Fig. 5: residual thermal coupling over the 1,948-socket census, with socket-level uncertainty.

Rebuilt for the external review of 2026-09-11 from the committed per-socket summaries
(thermal_socket_build.py), which the earlier pooled-matrix-only version could not use:

  (a) coupling against core index distance. Band: 95 % rack-bootstrap interval of the pooled
      mean (the 49 cabinets resampled, every socket of a drawn cabinet kept). At distance 1 the
      three kinds of index-adjacent pair are shown separately -- same slice, different slices of
      the same quad, different quads -- with rack-bootstrap intervals; the schematic states the
      inclusion rule (a pair counts only when both of its cores are active on that socket).
  (b) the per-socket, position-matched contrasts, by lot: each socket compares its own
      within-slice pairs with the cross-slice pair between the same two slices.
  (c) how many sockets support each slice-pair cell of the pooled matrix.
"""
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

root = Path(__file__).resolve().parent
FIG = root.parent / 'paper' / 'figures'
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))
B = int(os.environ.get('THERM_BOOT', '1000'))
BLUE, ORANGE, PURPLE, GREEN = '#2b6cb0', '#dd6b20', '#6b46c1', '#2f855a'
GREY, INK = '#a0aec0', '#2d3748'
plt.rcParams.update({'font.size': 7, 'axes.grid': True, 'grid.alpha': .25,
                     'grid.linewidth': .5, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.edgecolor': '#cbd5e0', 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})

meta = pd.read_csv(root / 'thermal_socket_meta_22-08.csv')
act = np.array([list(map(int, s.split('-'))) for s in meta.active])
c5 = np.load(root / 'thermal_socket_c5_22-08.npy')
M = np.stack([tl.unpack(c5[i], act[i]) for i in range(len(meta))])
rack = meta.rack.values
lot = np.where(rack < 22, 'A', np.where(rack <= 48, 'B', 'U'))
NC = 24
obs = ~np.isnan(M)
V = np.where(obs, M, 0.0)
GAP = [(d, [(i, i + d) for i in range(NC - d)]) for d in range(1, NC)]
KP = {k: [(i, i + 1) for i in range(NC - 1) if tl.pair_kind(i, i + 1) == k]
      for k in ('within', 'samequad', 'crossquad')}


def stats_of(ix):
    """pooled cell means over sockets ix -> curve over distance + three pair-kind means"""
    num = V[ix].sum(0); den = obs[ix].sum(0)
    P = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    curve = [np.nanmean([P[i, j] for i, j in pr]) for _, pr in GAP]
    kinds = [sum(num[i, j] for i, j in pr) / sum(den[i, j] for i, j in pr) for pr in KP.values()]
    cross_all = (sum(num[i, j] for i, j in KP['samequad'] + KP['crossquad']) /
                 sum(den[i, j] for i, j in KP['samequad'] + KP['crossquad']))
    return np.array(curve + kinds + [cross_all])


allix = np.arange(len(meta))
est = stats_of(allix)
rng = np.random.default_rng(55)
bt = np.array([stats_of(tl.cluster_resample(rack, rng)) for _ in range(B)])
lo, hi = np.percentile(bt, 2.5, 0), np.percentile(bt, 97.5, 0)
nd = len(GAP)

# per-socket position-matched contrasts
w_sq = np.full((len(M), 6), np.nan); w_cq = np.full((len(M), 5), np.nan)
for q in range(6):
    a = 4 * q
    w_sq[:, q] = (M[:, a, a + 1] + M[:, a + 2, a + 3]) / 2 - M[:, a + 1, a + 2]
for q in range(5):
    a = 4 * q + 2
    w_cq[:, q] = (M[:, a, a + 1] + M[:, a + 2, a + 3]) / 2 - M[:, a + 1, a + 2]
s_sq = np.nanmean(w_sq, 1); s_cq = np.nanmean(w_cq, 1)
within_s = np.array([np.nanmean([M[i, 2 * k, 2 * k + 1] for k in range(12)]) for i in range(len(M))])
cross_s = np.array([np.nanmean([M[i, a, b] for a, b in KP['samequad'] + KP['crossquad']])
                    for i in range(len(M))])
paper_s = within_s - cross_s

fig = plt.figure(figsize=(3.45, 4.4))

# ---- schematic of the inclusion rule
axs = fig.add_axes([0.02, 0.905, 0.96, 0.085]); axs.axis('off')
axs.set_xlim(-0.6, 11.6); axs.set_ylim(-1.05, 1.05)
for c in range(8):
    axs.add_patch(Rectangle((c - 0.36, -0.25), 0.72, 0.5, fc='#edf2f7', ec=INK, lw=.6))
    axs.text(c, 0, f'{c}', ha='center', va='center', fontsize=5.5)
for k in range(4):
    axs.plot([2 * k - 0.36, 2 * k + 1.36], [-0.42, -0.42], color=GREY, lw=1.0)
    axs.text(2 * k + 0.5, -0.8, f'slice {k}', ha='center', va='center', fontsize=5)
for q in range(2):
    axs.plot([4 * q - 0.4, 4 * q + 3.4], [0.42, 0.42], color=INK, lw=.8)
    axs.text(4 * q + 1.5, 0.78, f'quad {q}', ha='center', va='center', fontsize=5)
for (a, b), col in (((0, 1), ORANGE), ((1, 2), PURPLE), ((3, 4), GREEN)):
    axs.plot([a + 0.36, b - 0.36], [0, 0], color=col, lw=3.2, solid_capstyle='butt')
axs.text(7.9, 0.0, 'all three: index distance 1;\ncounted only on sockets\nwhere both cores are active',
         ha='left', va='center', fontsize=4.9, color=INK)

# ---- (a) decay curve
ax = fig.add_axes([0.14, 0.525, 0.84, 0.32])
gaps = np.arange(1, nd + 1)
ax.fill_between(gaps, lo[:nd], hi[:nd], color=BLUE, alpha=.22, lw=0,
                label='pooled mean, 95% rack-bootstrap band')
ax.plot(gaps, est[:nd], 'o-', color=BLUE, ms=2.6, lw=1.1)
ax.axhline(0, color=INK, lw=.6)
off = {'within': -0.35, 'samequad': 0.0, 'crossquad': 0.35}
lab = {'within': 'same slice', 'samequad': 'other slice, same quad',
       'crossquad': 'other quad'}
for j, (k, col, mk) in enumerate((('within', ORANGE, 'v'), ('samequad', PURPLE, 'D'),
                                  ('crossquad', GREEN, '^'))):
    e = est[nd + j]
    ax.errorbar([1 + off[k]], [e], yerr=[[e - lo[nd + j]], [hi[nd + j] - e]], fmt=mk, color=col,
                ms=4.5, capsize=1.8, lw=1.1, zorder=5, markeredgecolor='white', markeredgewidth=.5,
                label=f'{lab[k]}  $r={e:+.3f}$')
ax.set_xlabel('core index distance $|i-j|$ (sensor index)')
ax.set_ylabel('residual correlation $r$')
ax.set_xticks([1, 4, 8, 12, 16, 20, 23]); ax.set_xlim(0.3, 23.5)
ax.legend(fontsize=5.2, frameon=False, loc='upper right', handlelength=1.4)
ax.set_title('(a) 1,948 sockets, 2022-08, every 5th 20 s sample', fontsize=6.5, loc='left')

# ---- (b) per-socket position-matched contrasts by lot
axb = fig.add_axes([0.25, 0.06, 0.36, 0.33])
rows = [('same slice $-$\nother quad', s_cq, GREEN), ('same slice $-$\nsame quad', s_sq, PURPLE),
        ('same slice $-$\nboth kinds', paper_s, BLUE)]
ypos, data, cols, tick = [], [], [], []
for r_, (name, v, col) in enumerate(rows):
    for j, L in enumerate(('A', 'B')):
        x = v[(lot == L) & ~np.isnan(v)]
        ypos.append(r_ * 1.0 + (j - 0.5) * 0.36); data.append(x); cols.append(col)
    tick.append(r_ * 1.0)
bp = axb.boxplot(data, positions=ypos, vert=False, widths=0.3, showfliers=False,
                 patch_artist=True, medianprops=dict(color='k', lw=.8),
                 whiskerprops=dict(lw=.6), capprops=dict(lw=.6))
for j, (patch, col) in enumerate(zip(bp['boxes'], cols)):
    patch.set_facecolor(col); patch.set_alpha(.35 if j % 2 == 0 else .75); patch.set_lw(.5)
for r_ in range(3):
    axb.text(0.56, r_ - 0.18, 'A', fontsize=5, va='center')
    axb.text(0.56, r_ + 0.18, 'B', fontsize=5, va='center')
axb.invert_yaxis()
axb.axvline(0, color=INK, lw=.6)
axb.set_yticks(tick); axb.set_yticklabels([r[0] for r in rows], fontsize=5.3)
axb.set_xlabel('per-socket contrast in $r$', fontsize=6)
axb.set_xlim(-0.45, 0.65)
axb.set_title('(b) per socket, Lot A / B', fontsize=6.5, loc='left')

# ---- (c) support counts
axc = fig.add_axes([0.70, 0.06, 0.22, 0.33])
Ss = np.stack([tl.slice_similarity(M[i]) for i in range(len(M))])
sup = (~np.isnan(Ss)).sum(0).astype(float)
for k in range(12):
    sup[k, k] = np.nan
sup[np.triu_indices(12, 1)] = np.nan
im = axc.imshow(sup, cmap='Blues', vmin=0, vmax=np.nanmax(sup), origin='upper')
axc.set_xticks([0, 4, 8, 11]); axc.set_yticks([0, 4, 8, 11])
axc.set_xlabel('slice', fontsize=6); axc.set_ylabel('slice', fontsize=6)
axc.grid(False)
cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=5); cb.set_label('sockets', fontsize=5.5)
mn = int(np.nanmin(sup)); a_, b_ = np.unravel_index(np.nanargmin(sup), sup.shape)
axc.set_title(f'(c) support, min {mn}', fontsize=6.5, loc='left')

fig.savefig(FIG / 'f4_thermal_decay.pdf')
fig.savefig(SCRATCH / 'f4.png', dpi=150)
plt.close(fig)
print('wrote f4_thermal_decay.pdf from the per-socket census')
print(f'  within-slice (pooled pairs)   : {est[nd]:+.4f} [{lo[nd]:+.4f}, {hi[nd]:+.4f}]')
print(f'  same-quad cross-slice         : {est[nd + 1]:+.4f} [{lo[nd + 1]:+.4f}, {hi[nd + 1]:+.4f}]')
print(f'  cross-quad cross-slice        : {est[nd + 2]:+.4f} [{lo[nd + 2]:+.4f}, {hi[nd + 2]:+.4f}]')
print(f'  all cross-slice (paper)       : {est[nd + 3]:+.4f} [{lo[nd + 3]:+.4f}, {hi[nd + 3]:+.4f}]')
print(f'  slice-pair support: min {mn} (slices {b_},{a_}), max {int(np.nanmax(sup))}')
