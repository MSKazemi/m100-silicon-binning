"""R3-Mo1: is the within-slice thermal excess ABUTMENT or SHARED ACTIVITY?

Two cores of a slice are (a) physically adjacent and (b) sharing an L2/L3 block. Either would
produce excess correlation. The paper currently attributes it to (a). This tests what can
actually be distinguished at a 20 s sampling cadence:

  1. LAG PROFILE. Cross-correlate every pair at lags 0, +/-1, +/-2, +/-5, +/-10 samples.
     Heat diffusion between adjacent cores has a sub-second time constant, so at 20 s BOTH
     mechanisms peak at lag 0 -- the test is therefore about the SHAPE of the profile, not the
     location of its peak. A broader profile indicates a low-pass (thermal) path; a profile as
     narrow as the workload's own autocorrelation indicates activity tracking.

  2. THE DECISIVE STRUCTURAL POINT. The floorplan result of Sec. IV-F is computed from
     CROSS-slice pairs only, and different slices share no cache. So whatever explains the
     within-slice excess, it cannot explain the floorplan fit. We verify that the cross-slice-only
     distance decay is present and quantify it separately.
"""
import os, pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('THERM_MONTH', '22-08')
base = Path(os.environ.get('THERM_DIR', root / '.therm')) / f'year_month={YM}' / 'plugin=ipmi_pub'
rng = np.random.default_rng(41)
NC, NP, NSEL = 24, 12, 60
LAGS = [0, 1, 2, 5, 10]

dd = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
ok = dd.groupby(['node', 'socket'])['core'].nunique()
full = ok[ok == 16].reset_index().groupby('node').size()
cand = sorted(full[full == 2].index)
sel = set(str(cand[i]) for i in rng.choice(len(cand), NSEL, replace=False))
del dd, ok, full
RAW = {}
for s in (0, 1):
    for c in range(NC):
        f = base / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        d = d[d.node.isin(sel)]
        RAW[(s, c)] = {n: v.set_index('timestamp')['value'].astype('float32')
                       for n, v in d.groupby('node', observed=True)}
        del d
print(f'{len(sel)} nodes, {len(RAW)} metrics loaded')

def lagcorr(x, y, L):
    """corr(x_t, y_{t+L}) on already-standardised columns."""
    if L == 0: return float(np.mean(x * y))
    if L > 0:  return float(np.mean(x[:-L] * y[L:]))
    return float(np.mean(x[-L:] * y[:L]))

W = {l: [] for l in LAGS + [-l for l in LAGS if l]}
X = {l: [] for l in W}
auto = {l: [] for l in W}                 # per-core autocorrelation, the reference shape
crossdist = {}                            # slice-distance -> lag-0 corr, cross-slice pairs only
nsock = 0
for node in sorted(sel, key=int):
    for s in (0, 1):
        cols = {c: RAW.get((s, c), {}).get(node) for c in range(NC)}
        cols = {c: v for c, v in cols.items() if v is not None and len(v) > 100000}
        if len(cols) != 16: continue
        D = pd.DataFrame(cols).dropna()
        if len(D) < 100000: continue
        idx = np.array(sorted(cols)); R = D.values.astype('float64')
        R = R - R.mean(1, keepdims=True)          # strip socket-wide common mode
        R = R - R.mean(0, keepdims=True)
        sd = R.std(0)
        if (sd == 0).any(): continue
        R = R / sd
        nsock += 1
        for a in range(16):
            for L in W:
                auto[L].append(lagcorr(R[:, a], R[:, a], L))
            for b in range(a + 1, 16):
                ga, gb = idx[a], idx[b]
                same = ga // 2 == gb // 2
                if abs(ga - gb) == 1:
                    tgt = W if same else X
                    for L in tgt:
                        tgt[L].append(lagcorr(R[:, a], R[:, b], L))
                if not same:
                    dsl = abs(ga // 2 - gb // 2)
                    crossdist.setdefault(dsl, []).append(lagcorr(R[:, a], R[:, b], 0))
print(f'sockets analysed: {nsock}\n')

print('=== 1. lag profile (20 s per sample), common mode removed ===')
print(f'{"lag":>5} {"within-slice":>13} {"cross-slice":>12} {"excess":>9} {"self (autocorr)":>16}')
for L in sorted(W):
    w, x, a = np.mean(W[L]), np.mean(X[L]), np.mean(auto[L])
    print(f'{L:>5} {w:>13.4f} {x:>12.4f} {w-x:>9.4f} {a:>16.4f}')

w0, x0 = np.mean(W[0]), np.mean(X[0])
w1 = (np.mean(W[1]) + np.mean(W[-1])) / 2
x1 = (np.mean(X[1]) + np.mean(X[-1])) / 2
a0, a1 = np.mean(auto[0]), (np.mean(auto[1]) + np.mean(auto[-1])) / 2
print(f'\n  retention at +/-1 sample (20 s):')
print(f'    within-slice excess : {(w1-x1)/(w0-x0):.3f} of its lag-0 value')
print(f'    core autocorrelation: {a1/a0:.3f} of its lag-0 value')
print('  If the excess decays much faster than the signal itself, it tracks something')
print('  instantaneous (shared activity). If it decays like the signal, a low-pass')
print('  (thermal) path is at least as good an explanation.')

print('\n=== 2. the floorplan signal uses cross-slice pairs only ===')
print(f'{"slice distance":>15} {"lag-0 corr":>11} {"pairs":>9}')
for d in sorted(crossdist):
    print(f'{d:>15} {np.mean(crossdist[d]):>11.4f} {len(crossdist[d]):>9}')
ds = np.array(sorted(crossdist)); vs = np.array([np.mean(crossdist[d]) for d in ds])
print(f'\n  corr(slice distance, coupling) over CROSS-slice pairs only = '
      f'{np.corrcoef(ds, vs)[0,1]:+.3f}')
print('  Different slices share no L2/L3, so this decay cannot be cache-sharing.')
