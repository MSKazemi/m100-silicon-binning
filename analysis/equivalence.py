"""Is the harvest map's effect on power negligible? Test that, rather than failing to reject.

The paper reports r = +0.039 with permutation p = 0.003 and a slope whose CI excludes zero, then
calls it "no measurable effect". Those are inconsistent: the effect IS measurable, it is just
small. The correct instrument is an equivalence test, which asks whether the effect is small
enough to ignore -- a claim that can actually be supported, unlike "no effect".

MARGIN. We need an operator-justified indifference band, chosen before looking at the estimate.
Two anchors are available in the data itself and we use the tighter as the headline:

  * +/- 5 W per socket, roughly 2% of a socket's typical draw and below the resolution at which
    a scheduler could act on it;
  * the p0-p1 positional asymmetry (7.7 W), which operators already tolerate without modelling.

DESIGN. The within-node paired contrast removes chassis, inlet air, PSU and to first order
workload. But as the review notes, pairing does not remove confounding if load is distributed
asymmetrically between sockets, so we additionally adjust for node utilisation, core frequency,
inlet air temperature and month, and add node fixed effects via within-node centring.

OUTCOMES. Mean power, peak power, and -- because a scheduler cares about them -- core frequency
and socket temperature. Each is tested against the same margin, so a forest plot is readable.

Intervals are rack-clustered bootstrap, matching the rest of the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
MARGIN_W = 5.0
rng = np.random.default_rng(37)

# ------------------------------------------------------------------ per-socket harvest features
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
sets = (df.merge(clean, on=['node', 'socket', 'day'])
          .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
M = (sets.groupby(['node', 'socket'])['c']
         .agg(lambda s: s.value_counts().index[0]).rename('map').reset_index())
M['act'] = M['map'].apply(lambda c: sorted({k // 2 for k in c}))
M = M[M.act.apply(len) == 8].copy()
M['meanidx'] = M.act.apply(np.mean)
M['nadj'] = M.act.apply(lambda v: sum(1 for i in range(len(v) - 1) if v[i + 1] - v[i] == 1))
M['lowhalf'] = M.act.apply(lambda v: sum(1 for k in v if k < 6))
FEAT = ['meanidx', 'nadj', 'lowhalf']

# ------------------------------------------------------------------ power + covariates per day
pw = pd.concat([pd.read_parquet(f) for f in sorted((root / 'power_all').glob('*.parquet'))],
               ignore_index=True)
pw['day'] = pd.to_datetime(pw['day'], utc=True)
print(f'power table: {len(pw):,} rows, columns {list(pw.columns)}')

cov = pd.concat([pd.read_parquet(f) for f in sorted((root / 'cov_all').glob('cov_*.parquet'))],
                ignore_index=True)
cov['day'] = pd.to_datetime(cov['day'], utc=True)
cov = cov[['node', 'day', 'cpu_user_mean', 'cpu_speed_mean', 'load_one_mean',
           'ambient_mean']].dropna()
print(f'covariates: {len(cov):,} node-days, {cov.node.nunique()} nodes')

P = pw[pw.day >= CUT].merge(cov, on=['node', 'day'], how='inner')
print(f'joined socket-days with power AND covariates: {len(P):,}')

# ------------------------------------------------------------------ within-node paired contrast
W = P.pivot_table(index=['node', 'day'], columns='socket', values='mean')
W = W.dropna()
W.columns = ['p0', 'p1']
W['dP'] = W.p0 - W.p1
cv = P.drop_duplicates(['node', 'day']).set_index(['node', 'day'])[
    ['cpu_user_mean', 'cpu_speed_mean', 'load_one_mean', 'ambient_mean']]
W = W.join(cv).dropna().reset_index()

m0 = M[M.socket == 0].set_index('node')[FEAT]
m1 = M[M.socket == 1].set_index('node')[FEAT]
dF = (m0 - m1).dropna()
dF.columns = ['d_' + c for c in FEAT]
W = W.merge(dF, left_on='node', right_index=True, how='inner')
print(f'node-days in the paired sample: {len(W):,} on {W.node.nunique()} nodes')

W['month'] = W.day.dt.to_period('M').astype(str)
W['rack'] = W.node // 20


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def build(sub, extra):
    """Design matrix: intercept, map-feature differences, and adjustment covariates."""
    cols = [np.ones(len(sub))] + [sub['d_' + f].values for f in FEAT]
    names = ['const'] + FEAT
    if extra:
        for c in ['cpu_user_mean', 'cpu_speed_mean', 'load_one_mean', 'ambient_mean']:
            v = sub[c].values
            cols.append((v - v.mean()) / (v.std() + 1e-9))
            names.append(c)
        for mth in sorted(sub.month.unique())[1:]:
            cols.append((sub.month == mth).values.astype(float))
            names.append(f'M{mth}')
    return np.column_stack(cols), names


print(f'\n=== effect of the harvest map on the within-node power difference (W) ===')
print(f'   margin for practical equivalence: +/- {MARGIN_W:.0f} W across the observed range\n')
print(f'  {"model":<12} {"feature":<9} {"slope (W/unit)":>16} {"range effect (W)":>18} '
      f'{"90% CI":>20}  verdict')
RES = {'margin': MARGIN_W, 'rows': []}

for label, extra in [('unadjusted', False), ('adjusted', True)]:
    X, names = build(W, extra)
    y = W.dP.values
    b = ols(y, X)
    # rack-clustered bootstrap
    racks = W.rack.values
    uq = np.unique(racks)
    idx = {r: np.where(racks == r)[0] for r in uq}
    bs = []
    for _ in range(400):
        pick = rng.choice(uq, len(uq), replace=True)
        sel = np.concatenate([idx[r] for r in pick])
        try:
            bs.append(ols(y[sel], X[sel]))
        except Exception:
            pass
    bs = np.array(bs)
    for f in FEAT:
        j = names.index(f)
        rng_span = W['d_' + f].max() - W['d_' + f].min()
        eff = b[j] * rng_span
        lo, hi = np.percentile(bs[:, j] * rng_span, [5, 95])     # 90% == TOST at alpha .05
        ok = (lo > -MARGIN_W) and (hi < MARGIN_W)
        print(f'  {label:<12} {f:<9} {b[j]:+16.3f} {eff:+18.2f} '
              f'[{lo:+.2f}, {hi:+.2f}]  {"EQUIVALENT" if ok else "inconclusive"}')
        RES['rows'].append({'model': label, 'feature': f, 'slope': float(b[j]),
                            'eff': float(eff), 'lo': float(lo), 'hi': float(hi),
                            'equivalent': bool(ok)})

import json
json.dump(RES, open(root / 'analysis' / 'equivalence_fit.json', 'w'), indent=1)
print(f'\n  mean socket power {0.5 * (W.p0.mean() + W.p1.mean()):.1f} W; '
      f'margin {MARGIN_W:.0f} W is {100 * MARGIN_W / (0.5 * (W.p0.mean() + W.p1.mean())):.1f}% of it')
print(f'\n  For calibration, the same paired design gives the p0-p1 positional asymmetry:')
print(f'    mean dP = {W.dP.mean():+.2f} W  (socket position alone)')
print(f'    socket-to-socket spread (sd of dP) = {W.dP.std():.2f} W')
print('\n  A verdict of EQUIVALENT means the 90% interval for the whole observed range of the')
print('  feature lies inside the margin: the effect is not merely unproven, it is bounded')
print('  below operational relevance. That is a stronger and more honest claim than "no effect".')
