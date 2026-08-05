"""New figures from the reviewer's plan:
  F8  per-month stability of the harvest-map marginals  (answers M3 visually)
  F9  guard-episode timeline                            (answers M4 visually)
  F10 transition-date histogram over the full record    (answers F3 in the plan)
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
FIG = root / 'paper' / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#4a5568'
NP, EXPECT = 12, 4320

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
dfc = df.merge(clean, on=['node', 'socket', 'day'])
sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()

# ---------------------------------------------------------------- F8 stability
PM = pd.read_csv(root / 'analysis' / 'per_month_marginals.csv')
fig, ax = plt.subplots(figsize=(6.0, 2.6))
x = np.arange(len(PM))
cmap = plt.cm.viridis(np.linspace(0, .92, NP))
for k in range(NP):
    ax.plot(x, 100 * PM[f's{k}'], '-', color=cmap[k], lw=1.3,
            label=f'S{k}' if k in (0, 1, 2, 11) else None)
ax.set_xticks(x[::3]); ax.set_xticklabels(PM.month[::3], rotation=45, ha='right')
ax.set_ylabel('P(slice harvested) [%]'); ax.set_xlabel('month')
sd = PM[[f's{k}' for k in range(NP)]].std(axis=0, ddof=1).max()
ax.set_title(f'harvest-map marginals are static: max across-month SD = {100*sd:.2f} pp',
             fontsize=8)
ax.legend(fontsize=6.5, frameon=False, ncol=4, loc='upper right')
fig.savefig(FIG / 'f8_month_stability.pdf'); plt.close(fig)

# ---------------------------------------------------------------- F9 guard episodes
EP = [(347, 1, '2020-03-13', '2020-03-16', [8, 9]),
      (411, 1, '2021-11-13', '2021-11-15', [12, 13]),
      (946, 0, '2020-08-31', '2020-09-01', [16, 17])]
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
# states: 0 harvested at manufacture, 1 configured & fully sampled,
#         2 configured & partially sampled, 3 configured but SILENT (the guard event)
CMAP = ListedColormap(['#e2e8f0', '#2c5282', '#90cdf4', '#e53e3e'])
NORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], CMAP.N)
fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.0))
for ax, (n, s, d0, d1, lost) in zip(axes, EP):
    g = df[(df.node == n) & (df.socket == s)]
    lo = pd.Timestamp(d0, tz='UTC') - pd.Timedelta(days=4)
    hi = pd.Timestamp(d1, tz='UTC') + pd.Timedelta(days=3)
    g = g[(g.day >= lo) & (g.day <= hi)]
    days = sorted(g.day.unique())
    # the configuration in force just BEFORE the episode defines which cores exist
    pre = g[g.day < pd.Timestamp(d0, tz='UTC')]
    conf = set(pre[pre.n > .9 * EXPECT].core.unique()) | set(lost)
    grid = np.zeros((24, len(days)), dtype=int)
    for j, d in enumerate(days):
        gg = g[g.day == d].set_index('core')['n'].to_dict()
        for c in range(24):
            if c not in conf: grid[c, j] = 0
            elif gg.get(c, 0) == 0: grid[c, j] = 3
            elif gg.get(c, 0) > .9 * EXPECT: grid[c, j] = 1
            else: grid[c, j] = 2
    ax.pcolormesh(np.arange(len(days) + 1), np.arange(25), grid, cmap=CMAP, norm=NORM,
                  edgecolors='white', linewidth=.3)
    ax.invert_yaxis()
    ax.set_yticks([x + .5 for x in (0, 8, 16, 23)]); ax.set_yticklabels([0, 8, 16, 23], fontsize=6)
    ax.set_ylabel('core', fontsize=7)
    ax.set_xticks(np.arange(len(days)) + .5)
    ax.set_xticklabels([str(pd.Timestamp(d).date())[5:] for d in days], fontsize=5.5, rotation=90)
    ax.set_title(f'node {n} p{s}: slice {lost[0]//2} (cores {lost[0]},{lost[1]}) silent '
                 f'{d0[5:]}–{d1[5:]} while every other node sensor stays at full cadence',
                 fontsize=7)
    ax.grid(False)
axes[0].legend(handles=[Patch(fc='#e2e8f0', ec='k', lw=.3, label='harvested at manufacture'),
                        Patch(fc='#2c5282', label='configured, fully sampled'),
                        Patch(fc='#90cdf4', label='configured, partial'),
                        Patch(fc='#e53e3e', label='configured but SILENT (guard)')],
               fontsize=6, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.5, 1.62))
fig.tight_layout()
fig.savefig(FIG / 'f9_guard_episodes.pdf'); plt.close(fig)

# ---------------------------------------------------------------- F10 transition dates
tr = []
for (n, s), g in sets.groupby(['node', 'socket']):
    g = g.sort_values('day'); cs, ds = g.c.tolist(), g.day.tolist()
    for i in range(1, len(cs)):
        if cs[i] != cs[i-1]: tr.append(ds[i])
T = pd.Series(tr)
fig, ax = plt.subplots(figsize=(6.0, 2.3))
vc = T.dt.date.value_counts().sort_index()
ax.bar(pd.to_datetime(vc.index), vc.values, width=6, color=BLUE)
top = vc.idxmax()
ax.annotate(f'{vc.max()} transitions\non {top}', (pd.Timestamp(top), vc.max()),
            xytext=(28, -6), textcoords='offset points', fontsize=6.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=.8))
ax.set_ylabel('configuration transitions'); ax.set_xlabel('date')
ax.set_title(f'{len(T)} transitions over 2.5 years, on {T.dt.date.nunique()} distinct dates',
             fontsize=8)
fig.savefig(FIG / 'f10_transition_dates.pdf'); plt.close(fig)
print('wrote f8, f9, f10;', len(T), 'transitions')
