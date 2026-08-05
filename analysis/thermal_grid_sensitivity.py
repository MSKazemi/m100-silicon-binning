"""S4 -- does the thermal result depend on the alignment grid? (review finding R2-Mo2)

Sec. IV-E aligns all core streams on a common one-minute grid before computing residual
correlations. That choice was asserted, never tested. Raw 20 s timestamps do not coincide across
cores, so SOME alignment is required -- the question is whether the answer depends on how coarse
it is. We compare 20 s (nearest-sample join), 1 min (the paper) and 5 min, on the same sockets,
and report both load-bearing quantities: the within/cross-slice contrast, and the floorplan fit.
"""
import os, pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('THERM_MONTH', '22-08')
base = Path(os.environ.get('THERM_DIR', root / '.therm')) / f'year_month={YM}' / 'plugin=ipmi_pub'
rng = np.random.default_rng(19)
NC, NP, NSEL = 24, 12, 90

dd = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
ok = dd.groupby(['node', 'socket'])['core'].nunique()
full = ok[ok == 16].reset_index().groupby('node').size()
cand = sorted(full[full == 2].index)
sel = set(str(cand[i]) for i in rng.choice(len(cand), min(NSEL, len(cand)), replace=False))
print(f'nodes sampled: {len(sel)} of {len(cand)} with both sockets fully configured')
del dd, ok, full

RAW = {}
for s in (0, 1):
    for c in range(NC):
        f = base / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        d = d[d.node.isin(sel)]
        RAW[(s, c)] = {n: v[['timestamp', 'value']] for n, v in d.groupby('node', observed=True)}
        del d
print(f'metrics loaded: {len(RAW)}')

def run(grid_label, floor):
    """floor=None -> nearest-20s join on the exact timestamp; else pandas floor freq."""
    W, X, per = [], [], []
    Csum = np.zeros((NC, NC)); Cn = np.zeros((NC, NC))
    for node in sorted(sel, key=int):
        for s in (0, 1):
            cols = {}
            for c in range(NC):
                g = RAW.get((s, c), {}).get(node)
                if g is None or len(g) < 20000: continue
                if floor is None:
                    t = g['timestamp']
                else:
                    t = g['timestamp'].dt.floor(floor)
                ser = g.assign(t=t).groupby('t')['value'].mean().astype('float32')
                cols[c] = ser
            if len(cols) != 16: continue
            Xd = pd.DataFrame(cols).dropna()
            if len(Xd) < 2000: continue
            idx = np.array(sorted(cols)); R = Xd.values
            R = R - R.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
            sd = R.std(0)
            if (sd == 0).any(): continue
            C = (R.T @ R) / len(R) / np.outer(sd, sd)
            per.append(len(Xd))
            for a in range(16):
                for b in range(16):
                    Csum[idx[a], idx[b]] += C[a, b]; Cn[idx[a], idx[b]] += 1
            for a in range(16):
                for b in range(a + 1, 16):
                    if abs(idx[a] - idx[b]) == 1:
                        (W if idx[a] // 2 == idx[b] // 2 else X).append(C[a, b])
    CORR = Csum / np.maximum(Cn, 1); np.fill_diagonal(CORR, 1.0)
    # floorplan fit: slice-level distance vs coupling, identity 1x12 ordering
    S = np.zeros((NP, NP))
    for a in range(NP):
        for b in range(NP):
            if a != b:
                S[a, b] = np.mean([CORR[2*a+i, 2*b+j] for i in (0, 1) for j in (0, 1)])
    obs = np.array([S[a, b] for a, b in combinations(range(NP), 2)])
    dist = np.array([abs(a - b) for a, b in combinations(range(NP), 2)])
    fit = np.corrcoef(dist, obs)[0, 1]
    print(f'{grid_label:>8} | sockets {len(per):>4} | rows/socket {int(np.median(per)):>6} | '
          f'within {np.mean(W):+.4f} | cross {np.mean(X):+.4f} | '
          f'gap {np.mean(W)-np.mean(X):+.4f} | 1x12 fit {fit:+.3f}')
    return np.mean(W), np.mean(X), fit

print(f'\n{"grid":>8} | {"n":>12} | {"samples":>13} | {"within-slice":>13} | '
      f'{"cross-slice":>12} | {"contrast":>9} | {"floorplan":>10}')
print('-' * 104)
for lab, fl in [('20 s', None), ('1 min', '1min'), ('5 min', '5min'), ('15 min', '15min')]:
    run(lab, fl)
print('\nThe paper uses the 1 min grid. A result that moved materially across these rows would')
print('mean the alignment choice, not the silicon, was driving it.')
