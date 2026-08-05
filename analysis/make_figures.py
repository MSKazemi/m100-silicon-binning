"""Build the pooled 24x24 thermal correlation matrix, recover die topology by classical
MDS, and emit every figure used in the paper (vector PDF for LaTeX)."""
import os, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent
FIG = root.parent / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
base = Path(os.environ.get('M100_IPMI_DIR', root / 'ipmi')) / 'year_month=20-04' / 'plugin=ipmi_pub'
rng = np.random.default_rng(11)
NP, NC = 12, 24
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 140, 'savefig.bbox': 'tight'})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#4a5568'

# ---------------------------------------------------------------- disable maps
# full-record modal harvest maps (built by full_record_structure.py) so that every figure
# uses the same population as the text; the single-month snapshot is no longer used here.
M = np.load(root / 'harvest_maps_full.npy')
K = np.load(root / 'harvest_keys_full.npy')
nid = K[:, 0]; rack = nid // 20
cnt = pd.read_parquet(root / 'counts_20-04.parquet'); cnt['pair'] = cnt['core'] // 2
lotA = rack < 22
print(f'sockets {len(M)}  lotA {lotA.sum()}  lotB {(~lotA).sum()}')

# ---------------------------------------------------------------- curveball null
def curveball(mat, n):
    rowsets = [set(np.flatnonzero(r)) for r in mat]; R = len(rowsets)
    for _ in range(n):
        i, j = rng.integers(0, R, 2)
        if i == j: continue
        a, b = rowsets[i], rowsets[j]
        if not (a - b) or not (b - a): continue
        pool = list((a - b) | (b - a)); rng.shuffle(pool)
        na = (a & b) | set(pool[:len(a) - len(a & b)]); nb = (a | b) - na
        if len(na) == len(a) and len(nb) == len(b): rowsets[i], rowsets[j] = na, nb
    out = np.zeros_like(mat)
    for r, s in enumerate(rowsets): out[r, list(s)] = 1
    return out

def cooc(m): mm = m.astype(np.int32); return mm.T @ mm
obs_c = cooc(M); null = np.zeros((200, NP, NP)); cur = M.copy()
for k in range(200):
    cur = curveball(cur, 20000)
    assert (cur.sum(0) == M.sum(0)).all() and (cur.sum(1) == 4).all()
    null[k] = cooc(cur)
Z = (obs_c - null.mean(0)) / (null.std(0) + 1e-9); np.fill_diagonal(Z, np.nan)

# ---------------------------------------------------------------- thermal matrix
good = cnt[(cnt.socket == 0) & (cnt.n > 4000)].groupby('node')['core'].count()
nodes = sorted(good[good == 16].index, key=int)
sel = [nodes[i] for i in rng.choice(len(nodes), 120, replace=False)]
frames = {}
for c in range(NC):
    f = base / f'metric=p0_core{c}_temp' / 'a_0.parquet'
    if f.exists():
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        frames[c] = d[d.node.isin(sel)]
Csum = np.zeros((NC, NC)); Cn = np.zeros((NC, NC))
for node in sel:
    cols = {c: d[d.node == node].set_index('timestamp')['value'].astype('float32')
            for c, d in frames.items()}
    cols = {c: s for c, s in cols.items() if len(s) > 4000}
    if len(cols) != 16: continue
    X = pd.DataFrame(cols).dropna()
    if len(X) < 3000: continue
    idx = np.array(sorted(cols)); R = X.values
    R = R - R.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
    C = (R.T @ R) / len(R) / np.outer(R.std(0), R.std(0))
    for a in range(16):
        for b in range(16):
            Csum[idx[a], idx[b]] += C[a, b]; Cn[idx[a], idx[b]] += 1
CORR = Csum / np.maximum(Cn, 1); np.fill_diagonal(CORR, 1.0)
print(f'thermal matrix built; min pair coverage = {Cn[np.triu_indices(NC,1)].min():.0f} nodes')

# ---------------------------------------------------------------- classical MDS
D = np.sqrt(np.maximum(2 * (1 - CORR), 0)); J = np.eye(NC) - np.ones((NC, NC)) / NC
B = -0.5 * J @ (D ** 2) @ J
w, V = np.linalg.eigh(B); order = np.argsort(w)[::-1]
w, V = w[order], V[:, order]
emb = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
var = w[:4] / np.sum(w[w > 0])
print('MDS explained variance (first 4 axes):', np.round(var, 3))
# orient so axis-1 increases with core index
if np.corrcoef(emb[:, 0], np.arange(NC))[0, 1] < 0: emb[:, 0] *= -1
rho = np.corrcoef(emb[:, 0], np.arange(NC))[0, 1]
print(f'Spearman-like corr(MDS axis 1, core index) = {rho:+.3f}')

# ================================================================= FIGURES
# F1 -- slice schematic annotated with harvest rate
fig, ax = plt.subplots(figsize=(7.2, 2.5))
rate = M.mean(0)
for k in range(NP):
    x = k * 1.0
    col = plt.cm.Reds(0.15 + 1.6 * (rate[k] - rate.min()))
    ax.add_patch(Rectangle((x, 1.05), .92, .55, fc=col, ec='k', lw=.7))
    ax.text(x + .23, 1.32, f'{2*k}', ha='center', va='center', fontsize=7)
    ax.text(x + .69, 1.32, f'{2*k+1}', ha='center', va='center', fontsize=7)
    ax.plot([x + .46, x + .46], [1.07, 1.58], color='k', lw=.4, ls=':')
    ax.add_patch(Rectangle((x, .72), .92, .28, fc='#c6f6d5', ec='k', lw=.7))
    ax.text(x + .46, .86, '10M L3', ha='center', va='center', fontsize=6)
    ax.add_patch(Rectangle((x, .50), .92, .16, fc='#e2e8f0', ec='k', lw=.5))
    ax.text(x + .46, .58, '512k L2', ha='center', va='center', fontsize=5)
    ax.annotate('', (x + .46, .48), (x + .46, .30), arrowprops=dict(arrowstyle='<->', lw=.8, color='#38a169'))
    ax.text(x + .46, 1.68, f'{100*rate[k]:.0f}%', ha='center', fontsize=7,
            color=RED if rate[k] > .45 else GREY, fontweight='bold' if rate[k] > .45 else 'normal')
    ax.text(x + .46, 1.90, f'S{k}', ha='center', fontsize=7, color=BLUE)
ax.add_patch(Rectangle((0, .12), NP * 1.0 - .08, .18, fc='#2d3748', ec='k'))
ax.text(NP / 2, .21, 'on-chip fabric  —  7 TB/s switch,  256 GB/s $\\times$ 12',
        ha='center', va='center', color='w', fontsize=7)
ax.text(-.15, 1.90, 'slice', ha='right', fontsize=7, color=BLUE)
ax.text(-.15, 1.68, 'harvested', ha='right', fontsize=7, color=RED)
ax.text(-.15, 1.32, 'SMT4 cores', ha='right', fontsize=7)
ax.set_xlim(-2.2, NP + .2); ax.set_ylim(.05, 2.05); ax.axis('off'); ax.grid(False)
fig.savefig(FIG / 'f1_slice_schematic.pdf'); plt.close(fig)

# F2 -- marginals, lot A vs lot B
fig, ax = plt.subplots(figsize=(5.0, 2.4))
x = np.arange(NP)
def binci(v):
    n = len(v); pm = v.mean()
    return 100 * pm, 196 * np.sqrt(pm * (1 - pm) / n)     # 1.96*100*SE
mA = np.array([binci(M[lotA][:, k]) for k in range(NP)])
mB = np.array([binci(M[~lotA][:, k]) for k in range(NP)])
ax.bar(x - .21, mA[:, 0], .42, yerr=mA[:, 1], capsize=1.6, error_kw=dict(lw=.7),
       label=f'Lot A, racks 0–21 (n={lotA.sum()})', color=RED)
ax.bar(x + .21, mB[:, 0], .42, yerr=mB[:, 1], capsize=1.6, error_kw=dict(lw=.7),
       label=f'Lot B, racks 22–48 (n={(~lotA).sum()})', color=BLUE)
ax.axhline(100 / 3, ls='--', lw=.9, color='k', label='uniform (4/12)')
ax.set_xticks(x); ax.set_xlabel('slice index $k$'); ax.set_ylabel('P(slice harvested) [%]')
ax.legend(fontsize=6.5, frameon=False); fig.savefig(FIG / 'f2_marginals.pdf'); plt.close(fig)

# F3 -- co-disable z heatmap
fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.imshow(Z, cmap='RdBu_r', vmin=-20, vmax=20)
ax.set_xlabel('slice $j$'); ax.set_ylabel('slice $i$')
ax.set_xticks(range(NP)); ax.set_yticks(range(NP)); ax.grid(False)
fig.colorbar(im, ax=ax, label='co-harvest $z$ vs curveball null', shrink=.85)
fig.savefig(FIG / 'f3_cooccurrence.pdf'); plt.close(fig)

# F4 is produced by make_fig_thermal_v2.py from the full-month, both-socket matrix.
# It was previously generated here from the three-day 2020-04 sample, which overstated the
# within/cross-slice contrast threefold -- do not regenerate it from `CORR` above.

# F5 -- MDS recovered layout
fig, ax = plt.subplots(figsize=(5.0, 2.6))
sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.arange(NC), cmap='viridis', s=52, zorder=3, ec='k', lw=.4)
for k in range(NP):
    ax.plot(emb[[2*k, 2*k+1], 0], emb[[2*k, 2*k+1], 1], '-', color=RED, lw=1.6, zorder=2)
for c in range(NC):
    ax.annotate(str(c), emb[c], fontsize=5.5, ha='center', va='center', color='w', zorder=4)
ax.set_xlabel(f'MDS axis 1 ({100*var[0]:.0f}% var)  —  corr with core index = {rho:+.2f}')
ax.set_ylabel(f'axis 2 ({100*var[1]:.0f}%)')
fig.colorbar(sc, ax=ax, label='IPMI core index', shrink=.9)
fig.savefig(FIG / 'f5_mds_layout.pdf'); plt.close(fig)

# F6 -- rack x slice heatmap
tab = np.array([[M[(rack == r), k].mean() for k in range(NP)] for r in sorted(set(rack))])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
im = ax.imshow(100 * tab, aspect='auto', cmap='magma_r', vmin=0, vmax=100)
ax.axhline(21.5, color='#38b2ac', lw=2.2)
ax.text(6.0, 19.6, 'lot boundary (node 440)', color='#0d7377', fontsize=6.8,
        ha='center', va='center', fontweight='bold',
        bbox=dict(fc='white', ec='#38b2ac', lw=.6, pad=1.6, alpha=.92))
ax.set_xlabel('slice index $k$'); ax.set_ylabel('rack'); ax.grid(False)
ax.set_xticks(range(NP))
fig.colorbar(im, ax=ax, label='P(slice harvested) [%]', shrink=.9)
fig.savefig(FIG / 'f6_rack_heatmap.pdf'); plt.close(fig)

# F7 -- pattern rank-frequency, lot A vs B
fig, ax = plt.subplots(figsize=(5.0, 2.3))
for m, lab, col in [(M[lotA], 'Lot A', RED), (M[~lotA], 'Lot B', BLUE)]:
    v = pd.Series([tuple(np.flatnonzero(r)) for r in m]).value_counts().values
    ax.plot(np.arange(1, len(v)+1), 100*v/v.sum(), '-', color=col, lw=1.4,
            label=f'{lab}: {len(v)} distinct patterns')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('pattern rank'); ax.set_ylabel('share of sockets [%]')
ax.legend(fontsize=6.5, frameon=False); fig.savefig(FIG / 'f7_pattern_rank.pdf'); plt.close(fig)

np.save(root / 'thermal_corr_24x24.npy', CORR)
print('figures written to', FIG)
for f in sorted(FIG.glob('*.pdf')): print(' ', f.name)
