"""Figs. 7 and 8 of the paper: the rack x slice harvest heatmap and the per-lot pattern
rank-frequency plot.

Split out of make_figures.py (blocks F6 and F7), which also rebuilds a superseded thermal matrix
from raw IPMI data and would overwrite analysis/thermal_corr_24x24.npy and Figs. 3-4 if run
without it. This script needs only the committed full-record harvest maps.

Population: the rack-indexed sockets of Sec. IV-A, i.e. the 49 documented cabinets (racks 0-48,
1,960 sockets). Nodes 980-983 (rack index 49) have no documented floor position and are left out,
as in every rack-indexed result; earlier versions drew them as a 50th heatmap row and counted
them in Lot B of Fig. 8.

Usage: make_fig_lots.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

root = Path(__file__).resolve().parent
FIG = root.parent / 'paper' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
NP = 12
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED = '#2b6cb0', '#c53030'

M = np.load(root / 'harvest_maps_full.npy')
K = np.load(root / 'harvest_keys_full.npy')
rack_all = K[:, 0] // 20
keep = rack_all < 49                       # the 49 documented cabinets (Sec. IV-A)
M, rack = M[keep], rack_all[keep]
lotA = rack < 22
print(f'sockets {len(M)} in racks 0-48 (dropped {int((~keep).sum())} on nodes 980-983); '
      f'Lot A {int(lotA.sum())}, Lot B {int((~lotA).sum())}')

# ---------------------------------------------------------------- Fig. 7: rack x slice heatmap
racks = np.arange(49)
tab = np.array([[M[rack == r, k].mean() for k in range(NP)] for r in racks])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
im = ax.imshow(100 * tab, aspect='auto', cmap='magma_r', vmin=0, vmax=100)
ax.axhline(21.5, color='#38b2ac', lw=2.2)
ax.text(6.0, 19.6, 'lot boundary (racks 21 | 22)', color='#0d7377', fontsize=6.8,
        ha='center', va='center', fontweight='bold',
        bbox=dict(fc='white', ec='#38b2ac', lw=.6, pad=1.6, alpha=.92))
ax.set_xlabel('slice index $k$'); ax.set_ylabel('rack'); ax.grid(False)
ax.set_xticks(range(NP))
fig.colorbar(im, ax=ax, label='P(slice harvested) [%]', shrink=.9)
fig.savefig(FIG / 'f6_rack_heatmap.pdf', metadata={'CreationDate': None}); plt.close(fig)
print(f'wrote f6_rack_heatmap.pdf ({len(racks)} rack rows)')

# ---------------------------------------------------------------- Fig. 8: pattern rank-frequency
fig, ax = plt.subplots(figsize=(5.0, 2.3))
for m, lab, col, ls in [(M[lotA], 'Lot A', RED, '-'), (M[~lotA], 'Lot B', BLUE, '--')]:
    v = pd.Series([tuple(np.flatnonzero(r)) for r in m]).value_counts().values
    ax.plot(np.arange(1, len(v) + 1), 100 * v / v.sum(), ls, color=col, lw=1.4,
            label=f'{lab} ({len(m):,} sockets): {len(v)} distinct patterns')
    print(f'  {lab}: {len(m)} sockets, {len(v)} distinct patterns, most common {100 * v[0] / v.sum():.1f}%')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('pattern rank'); ax.set_ylabel('share of sockets [%]')
ax.legend(fontsize=6.5, frameon=False)
fig.savefig(FIG / 'f7_pattern_rank.pdf', metadata={'CreationDate': None}); plt.close(fig)
print('wrote f7_pattern_rank.pdf')
