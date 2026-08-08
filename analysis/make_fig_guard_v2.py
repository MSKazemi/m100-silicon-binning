"""Fig. 10, rebuilt to show all three collectors that corroborate the guard episodes.

The shipped version plotted only the BMC per-core grid. But the paper's claim about these
episodes is not "a sensor went quiet" -- it is that *three independent collectors agree*: the BMC
loses exactly one slice, the operating system's own logical-CPU count drops by exactly 8
(= one slice x 4 SMT threads), and the episode sits across a reboot, which is what a GARD record
applied by hostboot at IPL requires. Only the first of those was visible.

Each panel now carries the core grid plus two strips underneath: the OS core count and the boot
signal. A reader can see the three lines of evidence coincide without taking it on trust.

Node 347's episode predates ganglia collection (both cpu_num and boottime begin later), so its
strips are drawn as "no coverage" rather than silently blank -- absence of evidence, marked as
such.
"""
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap   # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

root = Path(__file__).resolve().parent.parent
FIG = root / 'paper' / 'figures'
SCRATCH = Path(os.environ.get('FIGURE_PREVIEW_DIR', tempfile.gettempdir()))
EXPECT = 4320

BLUE, ORANGE, GREEN = '#2b6cb0', '#dd6b20', '#2f855a'
GREY, INK, PALE = '#a0aec0', '#2d3748', '#e2e8f0'
plt.rcParams.update({'font.size': 8, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.edgecolor': '#cbd5e0', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})

# 0 harvested · 1 configured+full · 2 configured+partial · 3 configured but SILENT
CMAP = ListedColormap([PALE, BLUE, '#90cdf4', ORANGE])
NORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], CMAP.N)

EPISODES = [(347, 1, '2020-03-13', '2020-03-16', (8, 9)),
            (946, 0, '2020-08-31', '2020-09-01', (16, 17)),
            (411, 1, '2021-11-13', '2021-11-15', (12, 13))]

daily = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
                  ignore_index=True)
daily['day'] = pd.to_datetime(daily['day'], utc=True)
os_ = pd.concat([pd.read_parquet(f) for f in sorted((root / 'os_all').glob('os_*.parquet'))],
                ignore_index=True)
os_['day'] = pd.to_datetime(os_['day'], utc=True)
cov = pd.concat([pd.read_parquet(f) for f in sorted((root / 'cov_all').glob('cov_*.parquet'))],
                ignore_index=True)
cov['day'] = pd.to_datetime(cov['day'], utc=True)
# Drop no-coverage days BEFORE shifting, exactly as reboot_evidence.py does. Shifting first makes
# each day's predecessor a NaN row rather than the last day actually observed, which silently
# turns a genuine reboot into a non-event whenever it follows a coverage gap -- and node 946's
# 2020-08-30 boot is precisely such a case.
cov = cov.dropna(subset=['boot_max']).sort_values(['node', 'day'])
cov['prev'] = cov.groupby('node')['boot_max'].shift()
cov['rebooted'] = (cov.boot_ndistinct > 1) | (cov.prev.notna() & (cov.boot_max != cov.prev))

fig, axes = plt.subplots(3, 1, figsize=(6.9, 5.9),
                         gridspec_kw={'hspace': .95})

for ax, (n, s, d0, d1, lost) in zip(axes, EPISODES):
    lo = pd.Timestamp(d0, tz='UTC') - pd.Timedelta(days=3)
    hi = pd.Timestamp(d1, tz='UTC') + pd.Timedelta(days=3)
    g = daily[(daily.node == n) & (daily.socket == s) &
              (daily.day >= lo) & (daily.day <= hi)]
    days = sorted(g.day.unique())
    pre = g[g.day < pd.Timestamp(d0, tz='UTC')]
    conf = set(pre[pre.n > .9 * EXPECT].core.unique()) | set(lost)

    grid = np.zeros((24, len(days)), dtype=int)
    for j, d in enumerate(days):
        gg = g[g.day == d].set_index('core')['n'].to_dict()
        for c in range(24):
            if c not in conf:
                grid[c, j] = 0
            elif gg.get(c, 0) == 0:
                grid[c, j] = 3
            elif gg.get(c, 0) > .9 * EXPECT:
                grid[c, j] = 1
            else:
                grid[c, j] = 2

    ax.pcolormesh(np.arange(len(days) + 1), np.arange(25), grid, cmap=CMAP, norm=NORM,
                  edgecolors='white', linewidth=.3)
    ax.set_ylim(30.5, 0)                       # room for the two strips below the grid
    ax.set_yticks([x + .5 for x in (0, 8, 16, 23)])
    ax.set_yticklabels([0, 8, 16, 23], fontsize=6)
    ax.set_ylabel('BMC core index', fontsize=7)

    # ---- strip 1: the OS's own logical-CPU count
    o = os_[(os_.node == n)].set_index('day')['value'].to_dict()
    for j, d in enumerate(days):
        v = o.get(pd.Timestamp(d))
        if v is None:
            col, txt = '#f7fafc', '·'
        elif v == 128:
            col, txt = BLUE, ''
        else:
            col, txt = ORANGE, str(int(v))
        ax.add_patch(plt.Rectangle((j, 25.2), 1, 2, fc=col, ec='white', lw=.3))
        if txt:
            ax.text(j + .5, 26.2, txt, ha='center', va='center', fontsize=5,
                    color='white' if txt != '·' else GREY)

    # ---- strip 2: the independent boot signal
    cv = cov[cov.node == n].set_index('day')['rebooted'].to_dict()
    for j, d in enumerate(days):
        v = cv.get(pd.Timestamp(d))
        col = '#f7fafc' if v is None else (GREEN if v else PALE)
        ax.add_patch(plt.Rectangle((j, 27.6), 1, 2, fc=col, ec='white', lw=.3))
        if v:
            ax.text(j + .5, 28.6, 'boot', ha='center', va='center', fontsize=4.6,
                    color='white')

    # labels go to the RIGHT of the grid; on the left they collide with the core tick labels
    ax.text(len(days) + .25, 26.2, 'OS CPUs', ha='left', va='center', fontsize=6)
    ax.text(len(days) + .25, 28.6, 'reboot', ha='left', va='center', fontsize=6)
    ax.set_xlim(0, len(days) + 2.6)
    ax.set_xticks(np.arange(len(days)) + .5)
    ax.set_xticklabels([str(pd.Timestamp(d).date())[5:] for d in days],
                       fontsize=5.5, rotation=90)
    covered = any(o.get(pd.Timestamp(d)) is not None for d in days)
    note = '' if covered else '   (ganglia collection had not yet begun — no OS or boot coverage)'
    ax.set_title(f'node {n} p{s}: slice {lost[0] // 2} (cores {lost[0]},{lost[1]}) silent '
                 f'{d0[5:]}–{d1[5:]}{note}', fontsize=7)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

axes[0].legend(handles=[
    Patch(fc=PALE, ec=GREY, lw=.3, label='harvested at manufacture'),
    Patch(fc=BLUE, label='configured, fully sampled'),
    Patch(fc='#90cdf4', label='configured, partial'),
    Patch(fc=ORANGE, label='configured but SILENT / OS count $\\neq$ 128'),
    Patch(fc=GREEN, label='reboot observed'),
    Patch(fc='#f7fafc', ec=GREY, lw=.3, label='no coverage')],
    fontsize=5.8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.72))

fig.savefig(FIG / 'f9_guard_episodes.pdf')
fig.savefig(SCRATCH / 'f9.png', dpi=150)
plt.close(fig)
print('wrote f9_guard_episodes.pdf with OS and boot strips')
