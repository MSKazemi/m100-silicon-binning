"""Per-node harvest map — Fig. 7 at full resolution, with no aggregation.

Fig. 7 (`f6_rack_heatmap.pdf`) shows P(slice harvested) averaged over each rack's 40 sockets: 49
rows. That is the right form for seeing the procurement boundary, but it hides the object the
paper is actually about — the individual die. This renders every socket as its own row.

Two views, because they answer different questions:

  f15_node_heatmap.pdf   every node, both sockets side by side (p0 | p1), 12 slices each.
                         Nothing is averaged away: each row is one physical node and you can read
                         the two dies in it independently. Split into four columns so each node
                         gets enough pixels to actually be a row rather than an alias.

  f16_node_pairmap.pdf   the same 981 nodes in one 12-column panel, colouring how many of the
                         node's two dies fused each slice (neither / one / both). More compact,
                         and it makes the near-independence of the two sockets in a node visible:
                         if the dies were matched, "one" would be rare.

Rack boundaries are ruled every 20 nodes and the procurement boundary at node 440 is marked, so
the structure Fig. 7 shows can still be located in the unaggregated data.
"""
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

root = Path(__file__).resolve().parent.parent
FIG = root / 'paper' / 'figures'
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))

BLUE, ORANGE, INK, PALE = '#2b6cb0', '#dd6b20', '#2d3748', '#e2e8f0'
ABSENT = '#f7fafc'
NSLICE, RACK_W, LOT_NODE = 12, 20, 440

plt.rcParams.update({'font.size': 8, 'savefig.bbox': 'tight', 'figure.dpi': 200,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})

M = np.load(root / 'analysis' / 'harvest_maps_full.npy')      # 1 = harvested
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')      # (node, socket)
node, sock = K[:, 0], K[:, 1]
NODES = np.arange(node.max() + 1)

# -1 = socket absent from the record, 0 = active, 1 = harvested
grid = np.full((len(NODES), 2, NSLICE), -1.0)
for r in range(len(M)):
    grid[node[r], sock[r]] = M[r]
present = (grid[:, :, 0] >= 0).any(1)
print(f'nodes with at least one mapped socket: {present.sum()}  of {len(NODES)} ids')

CMAP = ListedColormap([ABSENT, PALE, BLUE])
NORM = BoundaryNorm([-1.5, -.5, .5, 1.5], CMAP.N)


def rule_racks(ax, lo, hi):
    for n in range(lo - lo % RACK_W, hi + 1, RACK_W):
        if lo <= n <= hi:
            ax.axhline(n - .5, color='white', lw=.5, alpha=.9)
    if lo <= LOT_NODE <= hi:
        ax.axhline(LOT_NODE - .5, color=ORANGE, lw=2.0)


# ---------------------------------------------------------------- view 1: p0 | p1 per node
NPANEL = 4
per = int(np.ceil(len(NODES) / NPANEL))
fig, axes = plt.subplots(1, NPANEL, figsize=(11.5, 10.5))
for i, ax in enumerate(axes):
    lo, hi = i * per, min((i + 1) * per, len(NODES)) - 1
    block = grid[lo:hi + 1]                                   # (n, 2, 12)
    # one row per node: p0's twelve slices, a one-column gutter, then p1's twelve
    img = np.full((block.shape[0], NSLICE * 2 + 1), -1.0)
    img[:, :NSLICE] = block[:, 0]
    img[:, NSLICE + 1:] = block[:, 1]
    ax.imshow(img, aspect='auto', interpolation='nearest', cmap=CMAP, norm=NORM,
              extent=[-.5, NSLICE * 2 + .5, hi + .5, lo - .5])
    rule_racks(ax, lo, hi)
    ax.axvline(NSLICE, color='white', lw=2.5)
    # label the slice index under every other column; without it a reader cannot tell which
    # column is slice 0, which is the whole point of the left-hand density
    ticks = list(range(0, NSLICE, 2)) + [NSLICE + 1 + k for k in range(0, NSLICE, 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k) for k in range(0, NSLICE, 2)] * 2, fontsize=6)
    ax.tick_params(axis='x', length=2, pad=1.5)
    for xc, lab in ((NSLICE / 2 - .5, 'p0'), (NSLICE * 1.5 + .5, 'p1')):
        ax.annotate(lab, (xc, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -13), textcoords='offset points',
                    ha='center', va='top', fontsize=8.5, weight='bold', color=INK,
                    annotation_clip=False)
    ax.set_yticks(np.arange(lo - lo % 100 + (100 if lo % 100 else 0), hi + 1, 100))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlabel('slice', fontsize=7.5, labelpad=1)
    if i == 0:
        ax.set_ylabel('node id   (thin rules = rack boundaries, every 20 nodes)')
    ax.set_title(f'nodes {lo}–{hi}', fontsize=8.5, pad=16)
fig.suptitle('Recovered harvest map of every individual die on Marconi100\n'
             '1,962 sockets on 981 nodes — blue = slice fused off at manufacture; '
             'orange rule = procurement boundary at node 440',
             fontsize=10.5, y=.985)
fig.legend(handles=[Patch(fc=BLUE, label='slice fused off (harvested)'),
                    Patch(fc=PALE, label='slice active'),
                    Patch(fc=ABSENT, ec='#cbd5e0', lw=.4, label='socket not in the record'),
                    Patch(fc=ORANGE, label='procurement boundary')],
           fontsize=7.5, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.012))
fig.tight_layout(rect=[0, .015, 1, .95])
fig.savefig(FIG / 'f15_node_heatmap.pdf')
fig.savefig(SCRATCH / 'f15.png', dpi=140)
plt.close(fig)
print('wrote f15_node_heatmap.pdf')

# ---------------------------------------------------------------- view 2: how many of the 2 dies
both = np.where((grid >= 0).all(1)[:, None] & True, grid[:, 0] + grid[:, 1], np.nan)
both = np.where((grid[:, 0, :] >= 0) & (grid[:, 1, :] >= 0),
                grid[:, 0, :] + grid[:, 1, :], -1.0)
CM2 = ListedColormap([ABSENT, PALE, '#90cdf4', BLUE])
NM2 = BoundaryNorm([-1.5, -.5, .5, 1.5, 2.5], CM2.N)

NP2 = 3
per2 = int(np.ceil(len(NODES) / NP2))
fig, axes = plt.subplots(1, NP2, figsize=(8.4, 9.5))
for i, ax in enumerate(axes):
    lo, hi = i * per2, min((i + 1) * per2, len(NODES)) - 1
    ax.imshow(both[lo:hi + 1], aspect='auto', interpolation='nearest', cmap=CM2, norm=NM2,
              extent=[-.5, NSLICE - .5, hi + .5, lo - .5])
    rule_racks(ax, lo, hi)
    ax.set_xticks(range(0, NSLICE, 2))
    ax.set_yticks(np.arange(lo - lo % 100 + (100 if lo % 100 else 0), hi + 1, 100))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlabel('slice')
    if i == 0:
        ax.set_ylabel('node id')
    ax.set_title(f'nodes {lo}–{hi}', fontsize=8.5)
fig.suptitle('How many of a node\'s two dies fused each slice\n'
             'of the slices fused at all in a node, 74% are fused on only ONE of its two dies — '
             'the two sockets are near-independent parts',
             fontsize=10, y=.98)
fig.legend(handles=[Patch(fc=BLUE, label='both dies fused this slice'),
                    Patch(fc='#90cdf4', label='one of the two'),
                    Patch(fc=PALE, label='neither'),
                    Patch(fc=ABSENT, ec='#cbd5e0', lw=.4, label='a socket missing')],
           fontsize=7.5, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.015))
fig.tight_layout(rect=[0, .015, 1, .94])
fig.savefig(FIG / 'f16_node_pairmap.pdf')
fig.savefig(SCRATCH / 'f16.png', dpi=140)
plt.close(fig)
print('wrote f16_node_pairmap.pdf')

# ---------------------------------------------------------------- the claim view 2 makes
ok = (grid[:, 0, 0] >= 0) & (grid[:, 1, 0] >= 0)
pair = grid[ok][:, 0, :] + grid[ok][:, 1, :]
n = ok.sum()
print(f'\nnodes with both sockets mapped: {n}')
print(f'  slice-instances where both dies fused it : {(pair == 2).sum():6d} '
      f'({100 * (pair == 2).mean():.1f}%)')
print(f'  exactly one of the two                   : {(pair == 1).sum():6d} '
      f'({100 * (pair == 1).mean():.1f}%)')
print(f'  neither                                  : {(pair == 0).sum():6d} '
      f'({100 * (pair == 0).mean():.1f}%)')
