"""The image a visitor sees before they scroll: the recovered maps themselves.

A repository whose first screen is prose asks the reader to take the finding on trust. This
renders the actual recovered data — every socket's harvest map, ordered by rack — so the two
procurement lots are visible as a band rather than asserted as a statistic.

Written as `docs/assets/hero.png` (README, docs home) and `docs/assets/social-preview.png`
(1280x640, the GitHub social card every share renders).
"""
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402

root = Path(__file__).resolve().parent.parent
OUT = root / 'docs' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))

BLUE, ORANGE, INK, PALE = '#2b6cb0', '#dd6b20', '#2d3748', '#e2e8f0'
plt.rcParams.update({'font.size': 9, 'savefig.bbox': 'tight',
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})

M = np.load(root / 'analysis' / 'harvest_maps_full.npy')   # 1 = harvested
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
rack = K[:, 0] // 20
BOUNDARY = 22

# One row per socket is ~1 px at any sensible figure size, so the lot boundary aliases away.
# Aggregating to P(slice harvested) per rack keeps every socket in the estimate while making
# the boundary a crisp band, which is the whole point of the image.
RACKS = np.arange(rack.max() + 1)
H = np.vstack([M[rack == r].mean(0) if (rack == r).any() else np.full(12, np.nan)
               for r in RACKS])


def draw(ax, show_labels=True):
    im = ax.imshow(H, aspect='auto', interpolation='nearest',
                   cmap='Blues', vmin=0, vmax=1)
    # plot() rather than axhline() so the rule stops at the data instead of running out
    # across the annotation margin
    ax.plot([-.5, 11.5], [BOUNDARY - .5] * 2, color=ORANGE, lw=2.6,
            solid_capstyle='butt', zorder=5)
    ax.set_xticks(range(12))
    ax.set_yticks([0, 10, 21, 30, 40, 48])
    if show_labels:
        ax.set_xlabel('slice position on the die  (0 – 11)')
        ax.set_ylabel('rack cabinet  (20 nodes each)')
        ax.text(12.3, 8, 'Lot A\nslice 0 fused on\n88.2% of dies',
                ha='left', va='center', fontsize=9, color=INK)
        ax.annotate('procurement\nboundary — rack 22', xy=(11.5, BOUNDARY - .5),
                    xytext=(12.3, BOUNDARY - .5), ha='left', va='center',
                    fontsize=9, color=ORANGE, weight='bold',
                    arrowprops=dict(arrowstyle='-', color=ORANGE, lw=1.4))
        ax.text(12.3, 36, 'Lot B\n39.3%', ha='left', va='center',
                fontsize=9, color=INK)
        ax.set_xlim(-.5, 18.5)
    return im


# ---------------------------------------------------------------- hero (README / docs)
fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
im = draw(ax)
cb = fig.colorbar(im, ax=ax, fraction=.033, pad=.30)
cb.set_label('P(slice fused off)', fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
ax.set_title('Which slices the vendor fused off, across every POWER9 die on Marconi100\n'
             '1,962 sockets recovered from public BMC telemetry',
             fontsize=11, color=INK, pad=12)
fig.savefig(OUT / 'hero.png', facecolor='white')
fig.savefig(SCRATCH / 'hero.png', facecolor='white')
plt.close(fig)

# ---------------------------------------------------------------- social preview (1280x640)
# GitHub wants exactly 1280x640; a tight bbox crops to the ink and breaks that, and passing
# bbox_inches=None to savefig falls back to the rcParam rather than disabling it.
plt.rcParams['savefig.bbox'] = 'standard'
fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=.08,
                      left=.10, right=.97, top=.86, bottom=.10)
ax = fig.add_subplot(gs[0, 0])
draw(ax, show_labels=False)
ax.set_xlim(-.5, 11.5)
ax.set_xlabel('slice position on the die', fontsize=10)
ax.set_ylabel('49 rack cabinets  (1,962 sockets)', fontsize=10)
ax.set_yticks([])

txt = fig.add_subplot(gs[0, 1])
txt.axis('off')
txt.text(0, .96, 'Which 16 of 24?', fontsize=30, weight='bold', color=INK, va='top')
txt.text(0, .78, 'Recovering per-die CPU core-harvest maps\nfrom out-of-band BMC telemetry',
         fontsize=15, color=INK, va='top', linespacing=1.5)
txt.text(0, .555,
         'Vendors fuse off defective cores and never say which.\n'
         'A BMC exposes one sensor per physical core position\n'
         'and does not renumber survivors — so the sensors that\n'
         'report ARE the harvest map.',
         fontsize=11.5, color='#4a5568', va='top', linespacing=1.65)
for i, line in enumerate([
        'slice-granular fusing — 1,962 / 1,962, zero exceptions',
        'per-die, not per-SKU — 443 of 495 patterns',
        'two procurement lots — unseen racks placed 95.9% right',
        'recovery costs ~2 minutes of ordinary telemetry']):
    txt.text(.012, .305 - i * .064, '—', fontsize=11, color=ORANGE, va='top', weight='bold')
    txt.text(.055, .305 - i * .064, line, fontsize=11, color=INK, va='top')
txt.text(0, .015, 'github.com/MSKazemi/m100-silicon-binning',
         fontsize=10.5, color=BLUE, va='bottom')
fig.savefig(OUT / 'social-preview.png', facecolor='white')
fig.savefig(SCRATCH / 'social.png', facecolor='white')
plt.close(fig)

print(f'wrote {OUT/"hero.png"}')
print(f'wrote {OUT/"social-preview.png"}  (1280x640)')
