"""F0 -- the physical hierarchy, and what a "change" means at each level.

room -> rack -> node -> socket -> die -> slice -> core, with the real counts, and the four
kinds of change that telemetry can tell apart.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
plt.rcParams.update({'font.size': 7.5, 'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY, GREEN = '#2b6cb0', '#c53030', '#4a5568', '#2f855a'

fig, (axT, axB) = plt.subplots(2, 1, figsize=(7.15, 3.9),
                               gridspec_kw={'height_ratios': [2.35, 1], 'hspace': -0.06})
for ax in (axT, axB): ax.axis('off')
axT.set_xlim(0, 100); axT.set_ylim(6, 40)
axB.set_xlim(0, 100); axB.set_ylim(0, 12)

def box(ax, x, y, w, h, fc, ec='k', lw=.7, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))

# ---------------- 1. machine room ----------------
axT.text(9, 37.5, 'machine room', ha='center', fontweight='bold', fontsize=8.2)
for r, (n, y) in enumerate([(18, 27), (15, 22), (16, 17)]):
    for i in range(n):
        box(axT, 1.3 + i * .88, y, .62, 3.4, BLUE if r == 0 else '#90cdf4', lw=.35)
axT.text(9, 13.0, '49 racks, 3 rows\n$(X,Y)$ documented', ha='center', va='top',
         fontsize=6.2, color=GREY)

# ---------------- 2. rack ----------------
axT.text(28, 37.5, 'rack', ha='center', fontweight='bold', fontsize=8.2)
for i in range(20):
    box(axT, 24.5, 16.5 + i * .92, 7, .70, '#bee3f8', lw=.35)
axT.text(28, 13.0, '20 nodes\nslot = height', ha='center', va='top',
         fontsize=6.2, color=GREY)
axT.annotate('', (24.2, 26), (10.4, 24), arrowprops=dict(arrowstyle='->', color=GREY, lw=.9))

# ---------------- 3. node ----------------
axT.text(48, 37.5, 'node (AC922 8335-GTG)', ha='center', fontweight='bold', fontsize=8.2)
box(axT, 38.5, 19.5, 19, 13.5, '#f7fafc')
box(axT, 40, 27, 7.5, 4.6, '#2c5282'); axT.text(43.75, 29.3, 'socket p0', ha='center', color='w', fontsize=6.5)
box(axT, 48.5, 27, 7.5, 4.6, '#2c5282'); axT.text(52.25, 29.3, 'socket p1', ha='center', color='w', fontsize=6.5)
for i in range(4):
    box(axT, 40 + i * 4.1, 21.3, 3.4, 3.4, '#c6f6d5', lw=.4)
    axT.text(41.7 + i * 4.1, 23, 'V100', ha='center', fontsize=5.3)
axT.text(48, 13.0, '2 sockets + 4 GPUs\n$\\mathrm{id}=20\\,\\mathrm{rack}+\\mathrm{slot}$',
         ha='center', va='top', fontsize=6.2, color=GREY)
axT.annotate('', (38.2, 28), (31.8, 26), arrowprops=dict(arrowstyle='->', color=GREY, lw=.9))

# ---------------- 4. die ----------------
axT.text(80, 37.5, 'the die in one socket', ha='center', fontweight='bold', fontsize=8.2)
harv = {0, 1, 4, 9}
for k in range(12):
    x = 62 + k * 3.15
    box(axT, x, 27, 2.8, 4.6, '#e2e8f0' if k in harv else '#2c5282', lw=.5)
    axT.plot([x + 1.4, x + 1.4], [27.15, 31.45], color='k' if k in harv else 'w', lw=.4, ls=':')
    box(axT, x, 25.3, 2.8, 1.4, '#c6f6d5', lw=.4)
    if k in harv:
        axT.text(x + 1.4, 29.3, '$\\times$', ha='center', va='center', color=RED, fontsize=8)
axT.text(80, 33.4, '12 slices  ·  4 harvested ($\\times$)  ·  16 of 24 cores active',
         ha='center', fontsize=6.6)
axT.annotate('', (61.7, 29), (57.8, 29), arrowprops=dict(arrowstyle='->', color=GREY, lw=.9))

# zoom on one slice
box(axT, 66, 13.6, 22, 7.6, 'white', ec=BLUE, lw=1.0)
box(axT, 67.4, 16.6, 8.8, 2.9, '#2c5282'); axT.text(71.8, 18.05, 'core $2k$', ha='center', color='w', fontsize=6.2)
box(axT, 77.4, 16.6, 9.2, 2.9, '#2c5282'); axT.text(82.0, 18.05, 'core $2k{+}1$', ha='center', color='w', fontsize=6.2)
box(axT, 67.4, 14.4, 19.2, 1.8, '#c6f6d5'); axT.text(77.0, 15.3, '512 kB L2 + 10 MB L3', ha='center', fontsize=5.9)
axT.text(77, 22.4, 'one slice = the unit that is fused', ha='center', fontsize=6.6,
         color=BLUE, fontweight='bold', zorder=6,
         bbox=dict(fc='white', ec='none', pad=1.2, alpha=1.0))
axT.plot([62.9, 67.5], [25.2, 21.3], color=BLUE, lw=.6, ls='--', zorder=1)
axT.plot([65.7, 87.5], [25.2, 21.3], color=BLUE, lw=.6, ls='--', zorder=1)

# ---------------- taxonomy strip ----------------
axB.text(50, 10.6, 'what a change in the recovered map means, and how often (steady state)',
         ha='center', fontsize=7.3, fontweight='bold')
kinds = [('RELABEL', '19%', 'p0/p1 tags swap;\nno silicon moves', GREY),
         ('CPU-SWAP', '70%', 'one socket changes;\nprocessor replaced', BLUE),
         ('NODE-SWAP', '11%', 'both change, no mirror;\nwhole node replaced', RED),
         ('GUARD', '3 events', 'one slice deconfigured;\nfirmware, reversible', GREEN)]
for i, (name, share, desc, col) in enumerate(kinds):
    x = 2.0 + i * 24.6
    box(axB, x, .6, 22.4, 8.4, 'white', ec=col, lw=1.1)
    axB.text(x + 11.2, 6.9, name, ha='center', fontsize=7.0, color=col, fontweight='bold')
    axB.text(x + 11.2, 4.9, share, ha='center', fontsize=7.4, color=col)
    axB.text(x + 11.2, 2.2, desc, ha='center', fontsize=6.1, color=GREY, linespacing=1.45)
fig.savefig(FIG / 'f0_hierarchy.pdf')
print('wrote f0_hierarchy.pdf')
