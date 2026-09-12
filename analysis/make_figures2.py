"""Fig. 12 (per-month per-slice harvest rates) and, on request, Fig. 14 (transition dates).

Usage:  make_figures2.py         writes f8_month_stability.pdf (Fig. 12) only
        make_figures2.py f10     also writes f10_transition_dates.pdf (Fig. 14; reads daily/)

Fig. 12 shows POPULATION marginals: in each month, the share of clean socket-days on which each
slice is harvested. Because the contributing sockets change from month to month, the figure also
overlays a fixed cohort -- the 1,900 sockets with a clean day in every post-commissioning month,
one vote each per month (analysis/balanced_panel.py) -- so a flat line cannot be an artefact of
composition. Per-socket staticity is a separate statement, made in the text from the same cohort
(share of socket-months whose map differs from the month before).

The superseded guard-episode figure (an old f9 block) used to be written from here and would
overwrite Fig. 13; it was removed. Fig. 13 is analysis/make_fig_guard_v2.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

root = Path(__file__).resolve().parent.parent
FIG = root / 'paper' / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#4a5568'
NP, EXPECT = 12, 4320
PDF_META = {'CreationDate': None}


def fig12():
    PM = pd.read_csv(root / 'analysis' / 'per_month_marginals.csv')
    BP = json.load(open(root / 'analysis' / 'model_balanced_panel.json'))
    bal = pd.DataFrame(BP['balanced']).set_index('month')
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(PM))
    pos = {m: i for i, m in enumerate(PM.month)}
    xb = np.array([pos[m] for m in bal.index])
    ax.axvspan(-.5, pos['20-05'] + .5, color='#a0aec0', alpha=.25, lw=0)
    ax.annotate('commis-\nsioning', (pos['20-05'] + .45, 52), fontsize=5.8, color=GREY, ha='right')
    cmap = plt.cm.viridis(np.linspace(0, .92, NP))
    for k in range(NP):
        # Every slice is named in the legend rather than four of twelve, and the colour is
        # viridis by slice index: a sequential, lightness-monotone encoding, so the ordering
        # survives greyscale printing and colour-vision deficiency.
        ax.plot(x, 100 * PM[f's{k}'], '-', color=cmap[k], lw=1.3, label=f'S{k}')
        ax.plot(xb, 100 * bal[f's{k}'], ':', color=cmap[k], lw=1.1)
    ax.plot([], [], '-', color=GREY, lw=1.3, label='all clean socket-days')
    ax.plot([], [], ':', color=GREY, lw=1.1, label=f'fixed cohort ({BP["cohort_sockets"]:,} sockets)')
    ax.set_xticks(x[::3])
    ax.set_xticklabels(PM.month[::3], rotation=45, ha='right')
    ax.set_ylabel('P(slice harvested) [%]')
    ax.set_xlabel('month')
    ax.set_title('per-slice harvest rate by month (population marginals): max across-month SD '
                 f'{BP["pop_max_sd_pp_all_months"]:.2f} pp, fixed cohort {BP["bal_max_sd_pp"]:.2f} pp',
                 fontsize=8)
    ax.set_ylim(23, 67.5)
    ax.legend(fontsize=6.0, frameon=False, ncol=7, loc='upper right', handlelength=1.4,
              columnspacing=.8)
    fig.savefig(FIG / 'f8_month_stability.pdf', metadata=PDF_META)
    plt.close(fig)
    print('wrote f8_month_stability.pdf (population marginals + fixed cohort)')


def fig14():
    df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
                   ignore_index=True)
    df['day'] = pd.to_datetime(df['day'], utc=True)
    a = df.groupby(['node', 'socket', 'day']).size().rename('active')
    fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
    t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
    clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
    dfc = df.merge(clean, on=['node', 'socket', 'day'])
    sets = dfc.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()
    tr = []
    for (n, s), g in sets.groupby(['node', 'socket']):
        g = g.sort_values('day')
        cs, ds = g.c.tolist(), g.day.tolist()
        for i in range(1, len(cs)):
            if cs[i] != cs[i - 1]:
                tr.append(ds[i])
    T = pd.Series(tr)
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    vc = T.dt.date.value_counts().sort_index()
    ax.bar(pd.to_datetime(vc.index), vc.values, width=6, color=BLUE)
    top = vc.idxmax()
    ax.annotate(f'{vc.max()} transitions\non {top}', (pd.Timestamp(top), vc.max()),
                xytext=(28, -6), textcoords='offset points', fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=.8))
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), color='#a0aec0', alpha=.30,
               lw=0, label='commissioning (excluded)')
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    ax.set_ylabel('configuration transitions')
    ax.set_xlabel('date')
    ax.set_title(f'{len(T)} transitions over 2.5 years, on {T.dt.date.nunique()} distinct dates',
                 fontsize=8)
    fig.savefig(FIG / 'f10_transition_dates.pdf')
    plt.close(fig)
    print('wrote f10_transition_dates.pdf;', len(T), 'transitions')


if __name__ == '__main__':
    fig12()
    if 'f10' in sys.argv[1:]:
        fig14()
