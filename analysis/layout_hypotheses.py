"""Which arrangement of the 12 slices best explains the thermal coupling?

We score explicit candidate layouts: the floorplan score is the Pearson correlation, over the 66
unordered slice pairs with equal weight, between a layout's slice distance (Euclidean) and the
slices' thermal similarity (mean of the four cross-slice core-pair correlations of the pooled
census matrix). Only pairs of cores in DIFFERENT slices enter, so shared L2/L3 cannot contribute.
More negative = closer slices are more coupled = better fit.

Candidates: the paper's unit-spacing grids (thermal_lib.CANDIDATES), IBM's documented layout
(POWER9 Processor Registers Specification Vol. 1 v1.2, Fig. 2, p. 33: six quads in two bands of
three, each quad's cores stacked; thermal_lib.ibm_core_xy, at the diagram's own proportions, which
are not guaranteed to scale -- thermal_ibm_layout.py varies them), and the best of all 12!/2
one-dimensional orderings.

The best 1-D ordering is found EXACTLY. For any 1-D ordering the multiset of the 66 pairwise
distances is the same, so the score is an increasing affine function of
sum_{a<b} S_ab |pos_a - pos_b|; minimising it is a weighted minimum linear arrangement, which
dynamic programming over the 2^12 prefix sets solves exactly (thermal_lib.exact_linear_arrangement,
checked against brute force in tests/test_thermal.py). Earlier versions used a 200-restart hill
climb, which cannot certify optimality; it happened to find the same optimum.

If the committed per-socket summaries are present (thermal_socket_build.py), the figure also
carries 95 % rack-bootstrap intervals (49 cabinets resampled) for every candidate's score.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

root = Path(__file__).resolve().parent
# default is the fleet census the paper reports; thermal_corr_24x24.npy is the old 3-day sample
CORR = np.load(root / os.environ.get('THERM_MATRIX', 'thermal_corr_allnodes.npy'))
NC, NP = 24, 12
rng = np.random.default_rng(5)
NLAYOUTS = math.factorial(NP) // 2
IBM_LAB = 'IBM layout (diagram proportions)'
IBM_XY = tl.ibm_slice_xy()

S = tl.slice_similarity(CORR)
obs = tl.cells(S)
# the same score over the 60 slice pairs in DIFFERENT quads (the six same-quad cells masked):
# beyond the quad the coupling follows the core numbering (thermal_column_test.py), not the layout
SQ = np.array([a // 2 == b // 2 for a, b in tl.SLICE_PAIRS])
obs60 = np.where(SQ, np.nan, obs)

# '2 x 6 mirrored halves' has exactly the coordinates of '2 x 6 boustrophedon' (both put slice k
# at (11 - k, 1) for k >= 6); earlier versions scored and drew it twice. Keep one.
CANDS = {}
for lab, c in tl.CANDIDATES.items():
    if not any(np.array_equal(np.asarray(c), np.asarray(v)) for v in CANDS.values()):
        CANDS[lab] = c
print('Candidate floorplans (more negative = better: closer slices are hotter together)')
res, res60 = {}, {}
for lab, c in CANDS.items():
    res[lab] = tl.coords_score(obs, c)
    res60[lab] = tl.coords_score(obs60, c)
    print(f'  {lab:<34} corr(distance, thermal r) = {res[lab]:+.3f}   (60 cross-quad pairs: '
          f'{res60[lab]:+.3f})')
print('  (2 x 6 mirrored halves: identical coordinates to 2 x 6 boustrophedon, not repeated)')
best = min(res, key=res.get)
print(f'\nbest candidate: {best} ({res[best]:+.3f}) among the unit-spacing grids')
r_ibm = tl.coords_score(obs, IBM_XY)
r_ibm60 = tl.coords_score(obs60, IBM_XY)
print(f"IBM's documented layout (Registers Spec v1.2 Fig. 2, diagram proportions): {r_ibm:+.3f}   "
      f'(60 cross-quad pairs: {r_ibm60:+.3f})')
print("  IBM's layout wins on all 66 pairs only through the six same-quad pairs: over the 60 pairs in "
      f"different quads the 1 x 12 index order fits better ({res60['1 x 12 linear']:+.3f} vs {r_ibm60:+.3f})")

# random-ordering null for 1x12 (same draws as every earlier version, so z is unchanged)
lin = [(k, 0) for k in range(NP)]
null = []
for _ in range(4000):
    p = rng.permutation(NP)
    coords = [lin[p[k]] for k in range(NP)]
    null.append(tl.coords_score(obs, coords))
null = np.array(null)
r_lin = res['1 x 12 linear']
print(f'\nrandom-permutation null for 1x12: mean {null.mean():+.3f}, sd {null.std():.3f}')
print(f'  identity ordering r = {r_lin:+.3f} -> z = {(r_lin - null.mean()) / null.std():+.2f}, '
      f'Monte-Carlo p = {tl.mc_pvalue(null, r_lin):.2e} ((b+1)/(B+1), B = {len(null)})')
# relabelling null for IBM's layout: the 12 slices assigned at random to its 12 positions
null_ibm = np.array([tl.coords_score(obs, IBM_XY[rng.permutation(NP)]) for _ in range(4000)])
print(f'  IBM layout r = {r_ibm:+.3f} vs relabelling null {null_ibm.mean():+.3f} +/- {null_ibm.std():.3f}'
      f' -> z = {(r_ibm - null_ibm.mean()) / null_ibm.std():+.2f}, Monte-Carlo p = '
      f'{tl.mc_pvalue(null_ibm, r_ibm):.2e}')

# exact optimum over all 12!/2 orderings, and the exact rank of the identity
bestv, bestp = tl.best_order(S)
best60 = tl.order_score(obs60, bestp)
W = np.nan_to_num(S)
kb = tl.kbest_linear_arrangements(W, 200)
idc = tl.arrangement_cost(W, np.arange(NP))
better = sorted({tuple(int(x) for x in tl.canonical(o)) for c, o in kb if c < idc - 1e-9})
rank = len(better) + 1
print(f'\nexact optimum over all 12!/2 = {NLAYOUTS:,} one-dimensional orderings: r = {bestv:+.3f}')
print(f'  slice order = {tl.canonical(bestp).tolist()}')
print(f'  identity    = {list(range(NP))}  (r = {r_lin:+.3f})')
print(f'  identity ranks {rank} of {NLAYOUTS:,}; layouts scoring better: {better}')
print(f'  exact permutation p (random ordering scores <= identity) = {rank}/{NLAYOUTS:,} '
      f'= {rank / NLAYOUTS:.1e}')
print(f'  Kendall distance to identity (up to reflection) = {tl.kendall_distance_reflect(bestp)}, '
      f'|Spearman| = {tl.spearman_abs(bestp):.3f}')
print(f"  IBM's documented 2-D layout {'beats' if r_ibm < bestv else 'does not beat'} every "
      f'1-D ordering ({r_ibm:+.3f} vs {bestv:+.3f})')

# ------------------------------------------------------------------ rack bootstrap (optional)
boot = None
meta_f = root / 'thermal_socket_meta_22-08.csv'
if meta_f.exists() and os.environ.get('THERM_MATRIX', 'thermal_corr_allnodes.npy') == 'thermal_corr_allnodes.npy':
    import pandas as pd
    meta = pd.read_csv(meta_f)
    act = np.array([list(map(int, s.split('-'))) for s in meta.active])
    c5 = np.load(root / 'thermal_socket_c5_22-08.npy')
    M = np.stack([tl.unpack(c5[i], act[i]) for i in range(len(meta))])
    rb = np.random.default_rng(11)
    rows = []
    for _ in range(int(os.environ.get('THERM_BOOT', '500'))):
        ix = tl.cluster_resample(meta.rack.values, rb)
        Sb = tl.slice_similarity(tl.pool(M[ix])[0])
        cv = tl.cells(Sb)
        sc, _o = tl.best_order(Sb)
        rows.append([tl.coords_score(cv, c) for c in CANDS.values()]
                    + [tl.coords_score(cv, IBM_XY), sc])
    boot = np.array(rows)
    print(f'\nrack bootstrap ({len(boot)} replicates of the 49 cabinets), 95% intervals'
          f' -- a diagnostic; the paper quotes thermal_ibm_layout.py\'s 1,000-replicate intervals'
          f' for the index order and IBM\'s layout, and so does the figure below:')
    for j, lab in enumerate(list(CANDS) + [IBM_LAB, 'exact 1-D optimum']):
        lo, hi = tl.ci(boot[:, j])
        print(f'  {lab:<38} [{lo:+.3f}, {hi:+.3f}]')
    print(f'  gap identity - optimum: [{tl.ci(boot[:, 0] - boot[:, -1])[0]:+.4f}, '
          f'{tl.ci(boot[:, 0] - boot[:, -1])[1]:+.4f}]')
    d = boot[:, -2] - boot[:, -1]
    print(f'  IBM layout - best 1-D ordering: [{tl.ci(d)[0]:+.4f}, {tl.ci(d)[1]:+.4f}], IBM better in '
          f'{100 * np.mean(d < 0):.1f}% of replicates')

# ------------------------------------------------------------------ figure
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
FIG = root.parent / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(4.2, 2.7))
# inset: IBM's layout, schematic -- slices at the diagram's positions, the index path through them
axs = ax.inset_axes([0.6, 0.015, 0.395, 0.82])
axs.set_facecolor('w')
xy = IBM_XY.copy()
w, h = 0.42, 0.21
axs.add_patch(Rectangle((-0.3, 0.44), 2.4, 0.4, color='#e2e8f0', lw=0, zorder=0))
axs.text(0.9, 0.64, 'nest', ha='center', va='center', fontsize=7, color='#4a5568')
for q in range(6):
    s0, s1 = xy[2 * q], xy[2 * q + 1]
    y0 = min(s0[1], s1[1]) - h / 2 - 0.03
    axs.add_patch(Rectangle((s0[0] - w / 2 - 0.04, y0), w + 0.08, abs(s0[1] - s1[1]) + h + 0.06,
                            fill=False, ls=':', lw=0.7, color='#4a5568', zorder=1))
axs.plot(xy[:, 0], xy[:, 1], '-', color='#c53030', lw=0.8, alpha=.8, zorder=2)
for k in range(NP):
    axs.add_patch(Rectangle((xy[k, 0] - w / 2, xy[k, 1] - h / 2), w, h, color='#2b6cb0', zorder=3))
    axs.text(xy[k, 0], xy[k, 1], str(k), ha='center', va='center', color='w', fontsize=7.5, zorder=4)
axs.set_xlim(-0.35, 2.15); axs.set_ylim(1.66, -0.32)
axs.set_aspect('equal'); axs.set_xticks([]); axs.set_yticks([]); axs.grid(False)
for sp in axs.spines.values():
    sp.set_color('#a0aec0'); sp.set_linewidth(0.6)
axs.text(0.9, -0.27, "IBM's layout (schematic)", ha='center', va='top', fontsize=7)
axs.text(0.9, 1.63, 'dotted: quads; red: index', ha='center', va='bottom', fontsize=6.3,
         color='#4a5568')

# scores
labs = list(res) + [IBM_LAB, 'best of all 12!/2 1-D orderings']
vals = [res[lab] for lab in res] + [r_ibm, bestv]
cols = ['#2b6cb0'] * len(res) + ['#2f855a', '#718096']
cols[list(res).index('1 x 12 linear')] = '#c53030'
labs[list(res).index('1 x 12 linear')] = '1 x 12 line (index order)'
y = np.arange(len(labs))
ax.barh(y, vals, color=cols)
if boot is not None:
    lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
    # The index-order and IBM intervals are also quoted in the paper, where they come from
    # thermal_ibm_layout.py's 1,000-replicate bootstrap. Read those from its committed JSON rather
    # than re-deriving them here at 500 replicates, so the figure and the text agree to the last
    # digit instead of to 0.001. The grid candidates, whose intervals the text does not quote,
    # keep this script's own bootstrap.
    ibmj = root / 'thermal_ibm_layout.json'
    if ibmj.exists():
        rb_ci = json.load(open(ibmj))['months']['22-08']['rack_bootstrap']
        for lab, key in (('1 x 12 line (index order)', 'linear_ci'), (IBM_LAB, 'ibm_ci'),
                         ('best of all 12!/2 1-D orderings', 'best_1d_ci')):
            if lab in labs and key in rb_ci:
                j = labs.index(lab)
                lo[j], hi[j] = rb_ci[key]
        print(f"  figure error bars for the index order, IBM's layout and the 1-D optimum are "
              f"thermal_ibm_layout.json's {rb_ci['replicates']}-replicate intervals, which are the "
              f"ones the paper quotes")
    ax.errorbar(vals, y, xerr=[np.array(vals) - lo, hi - np.array(vals)], fmt='none',
                ecolor='k', elinewidth=.8, capsize=2, label='95% rack-bootstrap interval')
vals60 = [res60[lab] for lab in res] + [r_ibm60, best60]
ax.plot(vals60, y, 'D', ms=3.6, mfc='w', mec='k', mew=.8, zorder=5,
        label='60 slice pairs in different quads only')
ax.axvspan(null.mean() - 2 * null.std(), null.mean() + 2 * null.std(), color='k', alpha=.12,
           label='random 1-D ordering null ($\\pm2\\sigma$)')
ax.axvline(null.mean(), color='k', lw=.8)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5)
ax.set_xlabel('corr(layout distance, thermal coupling)\n(more negative = better fit)', fontsize=8)
ax.invert_yaxis()
ax.set_xlim(min(vals) * 1.18, 1.1)
ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.3, -0.24), ncol=1)
fig.savefig(FIG / 'f5_layout_hypotheses.pdf'); plt.close(fig)
old = FIG / 'f5_mds_layout.pdf'
if old.exists():
    old.unlink()
print('\nwrote', FIG / 'f5_layout_hypotheses.pdf')
