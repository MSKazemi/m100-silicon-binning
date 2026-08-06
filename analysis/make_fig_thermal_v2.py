"""F4 v2 -- thermal decay with bootstrap CI, from the full-month both-socket matrix."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(__file__).resolve().parent
FIG = root.parent / 'paper' / 'figures'
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED = '#2b6cb0', '#c53030'
G = pd.read_csv(root / 'thermal_gap_v2.csv')
fig, ax = plt.subplots(figsize=(5.2, 2.5))
ax.fill_between(G.gap, G.lo, G.hi, color=BLUE, alpha=.20, lw=0, label='95% bootstrap CI')
ax.plot(G.gap, G.r, 'o-', color=BLUE, ms=3.2, lw=1.2, label='pooled mean')
ax.axhline(0, color='k', lw=.6)
ax.errorbar([1], [0.3767], yerr=[[0.3767-0.3661],[0.3865-0.3767]], fmt='v', color=RED, ms=7,
            capsize=2, label='within-slice sibling $r$=+0.377')
ax.errorbar([1], [0.3075], yerr=[[0.3075-0.2910],[0.3243-0.3075]], fmt='^', color='#dd6b20',
            ms=7, capsize=2, label='cross-slice neighbour $r$=+0.308')
ax.set_xlabel('core index distance $|i-j|$'); ax.set_ylabel('residual correlation $r$')
ax.legend(fontsize=6.2, frameon=False)
ax.set_title('240 sockets, both p0 and p1, full month 2022-08, native 20 s cadence',
             fontsize=7.5)
fig.savefig(FIG / 'f4_thermal_decay.pdf'); plt.close(fig)
print('F4 regenerated with CI band')
