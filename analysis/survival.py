"""Does a die's manufacturing map predict whether it later gets replaced?

This is the question the paper's power analysis gestures at but never asks. The harvest map is a
fingerprint of how a die came out of the fab; replacement is what the field eventually thought of
it. If dies whose harvested slices sit in a tight spatial cluster are the ones that fail, that
connects manufacturing variability to operational reliability far more directly than a power
correlation does.

DESIGN. Discrete-time (monthly) survival. One row per socket-month at risk, outcome = "this
socket's die was replaced this month", where replacement means a CPU-SWAP or NODE-SWAP from the
taxonomy -- RELABEL is excluded, because no silicon moves. Sockets are censored at their last
observed clean day. Only the post-commissioning window is used, matching every other longitudinal
result. Baseline covariates are taken from the socket's FIRST steady-state map, never from a map
observed after the event, which would leak the outcome.

MODEL. Logistic hazard fitted by maximum likelihood. Coefficients carry rack-clustered bootstrap
intervals, because sockets in a rack are not independent.

VALIDATION. Grouped 7-fold cross-validation over racks: fit on ~42 racks, predict the held-out
ones, and report a concordance index and a calibration table. In-sample significance on 96 events
would prove very little; held-out discrimination is the claim worth making. A null result is
still worth reporting, and is reported as one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
rng = np.random.default_rng(23)

# ------------------------------------------------------------------ rebuild maps and transitions
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
sets = (df.merge(clean, on=['node', 'socket', 'day'])
          .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
sets = sets[sets.day >= CUT].sort_values(['node', 'socket', 'day'])

S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
days_of = {k: sorted(g.day) for k, g in sets.groupby(['node', 'socket'])}

# node-level classification, so RELABEL can be told from a genuine swap
events = {}
for (n, s), ds in days_of.items():
    for i in range(1, len(ds)):
        if S[(n, s, ds[i - 1])] != S[(n, s, ds[i])]:
            events.setdefault((n, ds[i]), {})[s] = (S[(n, s, ds[i - 1])], S[(n, s, ds[i])])

swap = {}          # (node, socket) -> first date silicon actually moved
for (n, d), per in sorted(events.items()):
    if len(per) == 2:
        (o0, n0), (o1, n1) = per[0], per[1]
        if n0 == o1 and n1 == o0:
            continue                                   # RELABEL: no silicon moved
        for s in (0, 1):
            swap.setdefault((n, s), d)
    else:
        s = next(iter(per))
        swap.setdefault((n, s), d)
print(f'sockets with >=1 silicon-moving event in steady state: {len(swap)}')


# ------------------------------------------------------------------ baseline features
def feats(active):
    """Features of a harvest map, from the set of ACTIVE slices."""
    h = sorted(set(range(12)) - {k // 2 for k in active})     # harvested slice indices
    nadj = sum(1 for i in range(len(h) - 1) if h[i + 1] - h[i] == 1)
    return h, nadj


rows = []
for (n, s), ds in days_of.items():
    base = S[(n, s, ds[0])]
    h, nadj = feats(base)
    if len(h) != 4:
        continue
    rows.append(dict(node=n, socket=s, rack=n // 20, key=tuple(h),
                     s0=int(0 in h), nadj=nadj, spread=float(np.std(h)),
                     lot=int(n // 20 >= 22),
                     first=ds[0], last=ds[-1], event_day=swap.get((n, s))))
B = pd.DataFrame(rows)
freq = B.key.value_counts(normalize=True)
B['rarity'] = -np.log(B.key.map(freq).values)
B = B[B.rack < 49].reset_index(drop=True)
print(f'sockets in the survival sample: {len(B):,}   '
      f'events: {B.event_day.notna().sum()}')

# ------------------------------------------------------------------ person-period expansion
per = []
for r in B.itertuples():
    m0 = r.first.to_period('M')
    end = (r.event_day if pd.notna(r.event_day) else r.last).to_period('M')
    m = m0
    while m <= end:
        ev = int(pd.notna(r.event_day) and m == end)
        per.append((r.Index, r.rack, m.ordinal, ev, r.s0, r.nadj, r.spread, r.rarity, r.lot))
        m += 1
P = pd.DataFrame(per, columns=['i', 'rack', 'month', 'y', 's0', 'nadj', 'spread', 'rarity', 'lot'])
print(f'socket-months at risk: {len(P):,}   events: {P.y.sum()}   '
      f'monthly hazard {100 * P.y.mean():.3f}%')

FEATS = ['s0', 'nadj', 'spread', 'rarity', 'lot']
Z = P[FEATS].values.astype(float)
Z = (Z - Z.mean(0)) / Z.std(0)                    # standardise, so coefficients compare
y = P.y.values.astype(float)


def nll(b, Z, y):
    eta = b[0] + Z @ b[1:]
    return float(np.logaddexp(0, eta).sum() - (y * eta).sum())


def fit(Z, y):
    b0 = np.zeros(Z.shape[1] + 1)
    b0[0] = np.log(max(y.mean(), 1e-6) / (1 - y.mean()))
    return minimize(nll, b0, args=(Z, y), method='BFGS').x


beta = fit(Z, y)
print('\n=== logistic hazard, rack-clustered bootstrap intervals ===')
racks = P.rack.values
uniq = np.unique(racks)
boot = []
for _ in range(400):
    pick = rng.choice(uniq, len(uniq), replace=True)
    idx = np.concatenate([np.where(racks == r)[0] for r in pick])
    try:
        boot.append(fit(Z[idx], y[idx]))
    except Exception:
        pass
boot = np.array(boot)
print(f'  {"term":<10} {"coef":>8} {"95% CI":>22}   hazard ratio per SD')
for j, f in enumerate(['(intercept)'] + FEATS):
    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    hr = '' if j == 0 else f'{np.exp(beta[j]):.2f}  [{np.exp(lo):.2f}, {np.exp(hi):.2f}]'
    print(f'  {f:<10} {beta[j]:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {hr}')


# ------------------------------------------------------------------ held-out validation
def cindex(risk, y):
    pos, neg = risk[y == 1], risk[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    s = 0.0
    for p in pos:
        s += (p > neg).mean() + 0.5 * (p == neg).mean()
    return s / len(pos)


import json
json.dump({'feats': FEATS,
           'beta': beta.tolist(),
           'lo': np.percentile(boot, 2.5, axis=0).tolist(),
           'hi': np.percentile(boot, 97.5, axis=0).tolist(),
           'n_events': int(P.y.sum()), 'n_months': int(len(P))},
          open(root / 'analysis' / 'survival_fit.json', 'w'), indent=1)

print('\n=== grouped cross-validation over racks (7 folds) ===')
folds = {r: i % 7 for i, r in enumerate(rng.permutation(uniq))}
fold = np.array([folds[r] for r in racks])
oof = np.zeros(len(y))
for f in range(7):
    tr, te = fold != f, fold == f
    b = fit(Z[tr], y[tr])
    oof[te] = b[0] + Z[te] @ b[1:]
c = cindex(oof, y)
print(f'  out-of-fold concordance index : {c:.4f}   (0.50 = no discrimination)')
json.dump({'cindex': float(c)}, open(root / 'analysis' / 'survival_cindex.json', 'w'))
b_perm = [cindex(rng.permutation(oof), y) for _ in range(2000)]
print(f'  permutation null              : {np.mean(b_perm):.4f} +/- {np.std(b_perm):.4f}   '
      f'p = {(np.array(b_perm) >= c).mean():.3f}')

print('\n  calibration by predicted-risk decile (out of fold)')
q = pd.qcut(oof, 10, labels=False, duplicates='drop')
cal = pd.DataFrame({'q': q, 'y': y, 'p': 1 / (1 + np.exp(-oof))}).groupby('q').agg(
    n=('y', 'size'), observed=('y', 'mean'), predicted=('p', 'mean'))
cal['observed'] *= 100
cal['predicted'] *= 100
print(cal.to_string(float_format=lambda v: f'{v:.3f}'))

print('\n=== verdict ===')
if c < 0.55 and (np.array(b_perm) >= c).mean() > 0.05:
    print('  No held-out discrimination: the baseline harvest map does NOT predict which dies')
    print('  are later replaced. Reported as a null result -- it bounds how far the map can be')
    print('  pushed as a reliability covariate, which is itself useful to an operator.')
else:
    print('  The map carries held-out signal about replacement; see the coefficients above.')
