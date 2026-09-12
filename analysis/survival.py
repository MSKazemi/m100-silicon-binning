"""Does a die's harvest map predict whether it is later pulled from service?

v2 (2026-09, in response to external review). What changed from v1, and why:

  * DIES, NOT LABELS. v1 charged every event to the socket label. A label can change die without
    silicon leaving the node: after a RELABEL the next swap on that label removes the OTHER die,
    and three "CPU-SWAP?" pairs and four "NODE-SWAP"s are one die leaving plus the partner die
    re-appearing on the other socket. Each die is now followed by its map (survival_core.
    track_dies); a change is an event for the die whose map actually left.
  * BASELINE HAZARD. v1 had an intercept only (constant monthly hazard). Monthly event counts
    range 0-15 and March 2021 carries two days of telemetry, so its events surface in April. The
    model now has one baseline effect per calendar month (shrunk towards their mean, SD 1 on the
    log scale). 1,958 of 1,960 dies enter in June 2020, so calendar month and time since entry
    coincide.
  * GROUPED PROPORTIONAL HAZARDS. Complementary log-log link (Prentice & Gloeckler 1978), so
    exp(beta) is a hazard ratio; logit (v1) kept as a sensitivity. Binary covariates are reported
    0 -> 1, continuous ones per SD.
  * NO LEAKAGE. Pattern rarity and standardisation are learned on training racks only.
  * A SURVIVAL METRIC. v1's "cindex" was a pooled person-month AUC. Reported now: risk-set
    concordance (each event month's cases against dies still at risk that month, within held-out
    folds), cumulative/dynamic AUC at 12 and 28 months, and the out-of-fold survival
    log-likelihood and integrated Brier score against a baseline-only model fitted in the same
    folds. Nulls come from re-running the whole pipeline under rack-block permutations of the
    covariates, p = (b+1)/(B+1); intervals from a rack-clustered bootstrap.
  * ENDPOINT UNCERTAINTY. Events are replacement-consistent map changes, not FRU records. The
    analysis is repeated for the v1 label endpoint, for dies that left the fleet, and for
    CPU-SWAP-like, NODE-SWAP-like and any-change endpoints. The any-change endpoint, the only one
    near a multiplicity threshold, gets a 4,999-permutation null; lot-only and map-only sub-models
    show which covariates carry the held-out ranking.

Outputs: analysis/survival_fit.json, survival_cindex.json, survival_sensitivity.json (schema in
each file's "schema" key). The v1 statistic is re-derived and diagnosed by survival_legacy_check.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_core as sc  # noqa: E402

root = Path(__file__).resolve().parent.parent
OUT = root / 'analysis'
T = 28                                            # June 2020 .. September 2022
K = 7                                             # rack folds
B_BOOT, B_PERM, B_PERM_SENS, B_BOOT_SENS, B_METRIC = 1000, 499, 499, 500, 2000
R_PART = 10
LAM_ALPHA = 1.0
PRIMARY = 'pulled'

# ------------------------------------------------------------------ data and dies
sets = sc.clean_sets(root)
tr = sc.transitions(sets)
first = sets.groupby(['node', 'socket']).agg(c=('c', 'first'), first=('day', 'min'),
                                             last=('day', 'max'))
initial = {k: (r.c, r['first']) for k, r in first.iterrows()}
U = first.reset_index()
U['key'] = U.c.map(sc.harvested)
U['rack'] = U.node // 20
U = U[(U.rack < 49) & (U.key.map(len) == 4)].sort_values(['rack', 'node', 'socket'])
U = U.reset_index(drop=True)
assert (U.groupby('rack').size() == 40).all(), 'rack-block permutation assumes 40 dies per rack'
U['lot'] = (U.rack >= 22).astype(int)
uniq = sorted(set(U.key))
kid = {k: i for i, k in enumerate(uniq)}
ids = U.key.map(kid).values
F = sc.map_features(uniq)
lot = U.lot.values
rack = U.rack.values
a = np.array([sc.month_index(x) for x in U['first']])
last_m = np.array([sc.month_index(x) for x in U['last']])
assert (last_m == T - 1).all(), 'every die is followed to the end of the record'
print(f'dies in the survival sample: {len(U):,} on {U.rack.nunique()} racks; '
      f'{(a == 0).sum():,} enter in June 2020, {(a > 0).sum()} later')

# ------------------------------------------------------------------ endpoint audit
events, log = sc.track_dies(tr, initial, slack_days=3)
legacy = sc.legacy_first_events(sets)


def endpoint(which):
    if which == 'label':
        e = pd.Series([legacy.get((n, s), pd.NaT) for n, s in zip(U.node, U.socket)])
        c = pd.Series([pd.NaT] * len(U))
    else:
        e, c = sc.endpoint_table(events, U, which)
    d = e.notna().values.astype(int)
    b = np.full(len(U), T - 1)
    b[d == 1] = [sc.month_index(x) for x in e[d == 1]]
    cm = (d == 0) & c.notna().values
    b[cm] = [sc.month_index(x) for x in c[cm]]
    return b, d


ENDPOINTS = ['pulled', 'retired', 'label', 'single', 'double', 'any']
EP = {w: endpoint(w) for w in ENDPOINTS}

# how many cross-node matches would identical maps produce by chance?
pk = U.key.value_counts(normalize=True)
slack = pd.Timedelta(days=3)
exp_chance = 0.0
for r in tr.itertuples():
    other = tr[(tr.node != r.node) & (tr.d0 - slack <= r.d1) & (r.d0 - slack <= tr.d1)]
    exp_chance += len(other) * pk.get(sc.harvested(r.old), 0.0)
recip = log['reciprocal_cross_node']      # moves that are one half of a two-way node exchange

print('\n=== endpoint audit: following each die by its map ===')
print(f'  steady-state map changes: {len(tr)} in {log["episodes"]} maintenance episodes')
print(f'  within-node moves (relabel / die on partner socket): {log["within_node_moves"]}')
print(f'  dies re-found on ANOTHER node in the same episode : {log["cross_node_moves"]} '
      f'(chance expectation from map frequencies: {exp_chance:.1f}); {recip} of them in '
      f'{recip // 2} two-way exchanges between the same two nodes')
print(f'  dies that left the fleet (= spares that arrived)   : {log["retired"]}')
for w in ENDPOINTS:
    b_, d_ = EP[w]
    print(f'  endpoint {w:<8} events {d_.sum():3d}   die-months at risk {(b_ - a + 1).sum():,}')
# the two v1 defects: events that were not departures, and events charged to the wrong die
pl = dict(zip(zip(U.node, U.socket), sc.endpoint_table(events, U, 'pulled')[0]))
not_departures, reattributed = 0, 0
for (n, s), ev_ in legacy.items():
    if (n, s) not in pl or pl[(n, s)] == ev_:
        continue
    other = (n, 1 - s)
    if pl.get(other) == ev_ and legacy.get(other) != ev_:
        reattributed += 1                    # the event belongs to the partner die
    elif pd.isna(pl[(n, s)]):
        not_departures += 1                  # the die was re-found on the partner socket
moved_within = not_departures
print(f'  v1 label events in which the die was re-found on the partner socket: {not_departures}')
print(f'  v1 label events charged to the other die of the node (swap after a relabel): '
      f'{reattributed}')
fk = pd.Series([next((x['kind'] for x in events[lab] if x['kind'] != 'relabel'), None)
                for lab in zip(U.node, U.socket)]).value_counts().to_dict()
print(f'  first departures of the {EP["pulled"][1].sum()} dies pulled from their node: '
      f'{fk.get("retired", 0)} left the fleet, {fk.get("moved", 0)} were re-found on another node')
for w in ENDPOINTS:
    d_ = EP[w][1]
    print(f'  {w:<8} events by lot: A {d_[lot == 0].sum():3d} of {(lot == 0).sum()}, '
          f'B {d_[lot == 1].sum():3d} of {(lot == 1).sum()}')

# ------------------------------------------------------------------ in-sample model
def spec_of(name):
    if name == 'fe_pen':
        return dict(pen_base=sc.fe_penalty(T, LAM_ALPHA))
    if name == 'fe_pen01':
        return dict(pen_base=sc.fe_penalty(T, 0.1))
    if name == 'const':
        return dict(basis=np.ones((T, 1)))
    if name == 'rcs':
        q = np.percentile(np.arange(T), [5, 27.5, 50, 72.5, 95])
        return dict(basis=sc.rcs_basis(T, q))
    raise ValueError(name)


def full_design(ids_, lot_):
    X, _, mu, sd = sc.design_ids(ids_, ids_, lot_, lot_, F)
    return X, sd


X_all, SD_all = full_design(ids, lot)
racks = np.unique(rack)
rack_idx = {r: np.where(rack == r)[0] for r in racks}


def fit_hr(b_, d_, base='fe_pen', link='cloglog', nboot=B_BOOT, X=X_all, seed=1):
    kw = spec_of(base) if base != 'fe_free' else {}
    if base == 'fe_free':
        kw['month_mask'] = np.bincount(b_[d_ == 1], minlength=T) > 0
    f = sc.fit_dth(X, a, b_, d_, T, link=link, **kw)
    bs = []
    r2 = np.random.default_rng(seed)
    for _ in range(nboot):
        idx = np.concatenate([rack_idx[r] for r in r2.choice(racks, len(racks), replace=True)])
        kb = dict(kw)
        if base == 'fe_free':
            kb['month_mask'] = np.bincount(b_[idx][d_[idx] == 1], minlength=T) > 0
        bs.append(sc.fit_dth(X[idx], a[idx], b_[idx], d_[idx], T, link=link, **kb)['beta'])
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {'beta': f['beta'], 'lo': lo, 'hi': hi, 'alpha': f['alpha'], 'fit': f,
            'n_events': int(d_.sum()), 'n_months': int((b_ - a + 1).sum())}


def hr_line(res):
    out = []
    for j, name in enumerate(sc.FEATS):
        out.append(f'{np.exp(res["beta"][j]):.2f} [{np.exp(res["lo"][j]):.2f}, '
                   f'{np.exp(res["hi"][j]):.2f}]')
    return out


SCALE = {'s0': '0->1', 'nadj': 'per SD', 'spread': 'per SD', 'rarity': 'per SD', 'lot': '0->1'}
b0, d0 = EP[PRIMARY]
prim = fit_hr(b0, d0)
n_ev = int(d0.sum())
print('\n=== grouped-time proportional hazards (cloglog), one baseline effect per calendar month ===')
print(f'  endpoint: die pulled from its node; {n_ev} events, {prim["n_months"]:,} die-months, '
      f'{n_ev / len(sc.FEATS):.0f} events per covariate')
print(f'  monthly events: {np.bincount(b0[d0 == 1], minlength=T).tolist()}')
print(f'  {"term":<8} {"scale":<14} hazard ratio [95% rack-bootstrap CI]')
for j, name in enumerate(sc.FEATS):
    sdtxt = '' if SCALE[name] == '0->1' else f' (SD {SD_all[j]:.2f})'
    print(f'  {name:<8} {SCALE[name] + sdtxt:<14} {hr_line(prim)[j]}')

# ------------------------------------------------------------------ held-out validation
def folds_of(seed):
    r = np.random.default_rng(seed).permutation(racks)
    fmap = {rk: i % K for i, rk in enumerate(r)}
    return np.array([fmap[x] for x in rack])


PARTS = [folds_of(s) for s in range(100, 100 + R_PART)]    # repeated rack partitions
FOLD = PARTS[0]


def baseline_only(b_, d_, fold, spec):
    h0 = np.empty((len(a), T))
    for f in range(K):
        trn, te = fold != f, fold == f
        f0 = sc.fit_dth(np.zeros((trn.sum(), 0)), a[trn], b_[trn], d_[trn], T, **spec)
        h0[te] = sc.inv_link(f0['alpha'], 'cloglog')[None, :]
    return h0


def cv_scores(ids_, lot_, b_, d_, fold, spec, lam_beta=0.0):
    """Out-of-fold covariate scores and hazards; rarity and scaling learned on training racks."""
    score = np.empty(len(a))
    raw = np.empty(len(a))
    h = np.empty((len(a), T))
    betas = []
    for f in range(K):
        trn, te = fold != f, fold == f
        Xt, Xe, mu, sd = sc.design_ids(ids_[trn], ids_[te], lot_[trn], lot_[te], F)
        fit = sc.fit_dth(Xt, a[trn], b_[trn], d_[trn], T, lam_beta=lam_beta, **spec)
        s = Xe @ fit['beta']
        score[te] = s
        raw[te] = (Xe * sd + mu) @ (fit['beta'] / sd)       # raw scale: comparable across folds
        h[te] = sc.inv_link(fit['alpha'][None, :] + s[:, None], 'cloglog')
        betas.append(fit['beta'])
    return score, raw, h, np.array(betas)


def ibs_of(h, b_, d_):
    err, valid = sc.brier_curve(h, a, b_, d_)
    return float(np.mean((err * valid).sum(0) / valid.sum(0)))


def metrics(score, raw, h, h0, b_, d_, fold):
    ll = sc.loglik_rows(h, a, b_, d_)
    ll0 = sc.loglik_rows(h0, a, b_, d_)
    i1, i0 = ibs_of(h, b_, d_), ibs_of(h0, b_, d_)
    return {'C': sc.risk_set_concordance(score, a, b_, d_, fold),
            'Cpool': sc.risk_set_concordance(raw, a, b_, d_, None),
            'auc12': sc.cd_auc(score, a, b_, d_, 11, fold),
            'auc28': sc.cd_auc(score, a, b_, d_, T - 1, fold),
            'dll': float((ll - ll0).sum()),
            'ibs': i1, 'ibs0': i0, 'dibs_pct': 100 * (i1 - i0) / i0}


def cv_all(ids_, lot_, b_, d_, parts, h0s, spec):
    """Every metric on every partition; the reported value is the mean over partitions."""
    ms, outs = [], []
    for fo, h0 in zip(parts, h0s):
        s, raw, h, betas = cv_scores(ids_, lot_, b_, d_, fo, spec)
        ms.append(metrics(s, raw, h, h0, b_, d_, fo))
        outs.append((s, raw, h, betas))
    return pd.DataFrame(ms), outs


def rack_block(r2):
    """Move each rack's covariate block (keys and lot) to another rack, shuffled inside."""
    pi = r2.permutation(racks)
    nid, nlot = ids.copy(), lot.copy()
    for rk, src in zip(racks, pi):
        dst, s_ = rack_idx[rk], rack_idx[src][r2.permutation(40)]
        nid[dst], nlot[dst] = ids[s_], lot[s_]
    return nid, nlot


def perm_null(b_, d_, parts, h0s, B, seed, spec):
    r2 = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        nid, nlot = rack_block(r2)
        out.append(cv_all(nid, nlot, b_, d_, parts, h0s, spec)[0].mean())
    return pd.DataFrame(out)


def pval(null, obs, larger_is_better=True):
    b = int((null >= obs).sum()) if larger_is_better else int((null <= obs).sum())
    return (b + 1) / (len(null) + 1)


def _pairs(score, b_, d_, fold, tau=None):
    return sc.pair_parts(score, a, b_, d_, rack, racks, group=fold, tau=tau)


def boot_prep(score, raw, h, h0, b_, d_, fold):
    e1, v1 = sc.brier_curve(h, a, b_, d_)
    e0, _ = sc.brier_curve(h0, a, b_, d_)
    dl = sc.loglik_rows(h, a, b_, d_) - sc.loglik_rows(h0, a, b_, d_)
    return {'pairs': {'C': _pairs(score, b_, d_, fold), 'Cpool': _pairs(raw, b_, d_, None),
                      'auc12': _pairs(score, b_, d_, fold, 11),
                      'auc28': _pairs(score, b_, d_, fold, T - 1)},
            'dl': np.array([dl[rack_idx[r]].sum() for r in racks]),
            'E1': np.array([(e1 * v1)[rack_idx[r]].sum(0) for r in racks]),
            'E0': np.array([(e0 * v1)[rack_idx[r]].sum(0) for r in racks]),
            'NV': np.array([v1[rack_idx[r]].sum(0) for r in racks])}


def boot_eval(P, w):
    out = {k: sc.weighted_pairs(parts, w) for k, parts in P['pairs'].items()}
    out['dll'] = float(w @ P['dl'])
    n = w @ P['NV']
    i1, i0 = np.mean((w @ P['E1']) / n), np.mean((w @ P['E0']) / n)
    out['dibs_pct'] = 100 * (i1 - i0) / i0
    return out


def boot_ci(preps, nboot, seed):
    """Rack-clustered bootstrap of the partition-averaged metrics (predictions held fixed)."""
    r2 = np.random.default_rng(seed)
    rows = []
    for _ in range(nboot):
        w = np.bincount(r2.integers(0, len(racks), len(racks)), minlength=len(racks)).astype(float)
        rows.append(pd.DataFrame([boot_eval(P, w) for P in preps]).mean())
    rows = pd.DataFrame(rows)
    return {k: np.percentile(rows[k], [2.5, 97.5]).tolist() for k in rows}


SPEC = spec_of('fe_pen')
h0s = [baseline_only(b0, d0, fo, SPEC) for fo in PARTS]
per_part, outs = cv_all(ids, lot, b0, d0, PARTS, h0s, SPEC)
m_obs = per_part.mean()
preps = [boot_prep(o[0], o[1], o[2], h0, b0, d0, fo) for o, h0, fo in zip(outs, h0s, PARTS)]
for P_, (_, m_) in zip(preps, per_part.iterrows()):
    chk = boot_eval(P_, np.ones(len(racks)))
    assert all(abs(chk[k] - m_[k]) < 1e-9 for k in chk), 'bootstrap decomposition mismatch'
ci = boot_ci(preps, B_METRIC, 11)
null = perm_null(b0, d0, PARTS, h0s, B_PERM, 12, SPEC)
p = {k: pval(null[k], m_obs[k]) for k in ['C', 'Cpool', 'auc12', 'auc28', 'dll']}
p['dibs_pct'] = pval(null.dibs_pct, m_obs.dibs_pct, larger_is_better=False)
s_obs, raw_obs, h_obs, cv_betas = outs[0]

print(f'\n=== held-out validation: {K}-fold CV grouped by rack, {R_PART} rack partitions; '
      f'rarity and scaling learned in-fold ===')


def row(label, k, fmt='.3f'):
    lo, hi = ci[k]
    print(f'  {label:<34}: {m_obs[k]:{fmt}}  95% CI [{lo:{fmt}}, {hi:{fmt}}]   '
          f'null {null[k].mean():{fmt}} +/- {null[k].std():{fmt}}   p = {p[k]:.3f}'
          f'   partitions [{per_part[k].min():{fmt}}, {per_part[k].max():{fmt}}]')


row('risk-set concordance', 'C')
row('  same, pooled across folds', 'Cpool')
row('cumulative/dynamic AUC, 12 months', 'auc12')
row('cumulative/dynamic AUC, 28 months', 'auc28')
row('log-lik vs baseline-only (nats)', 'dll', '+.2f')
row('integrated Brier vs baseline (%)', 'dibs_pct', '+.2f')
print(f'  integrated Brier score {m_obs.ibs:.5f} with the map, {m_obs.ibs0:.5f} baseline-only')
print(f'  (null: {B_PERM} rack-block permutations of the covariates, whole pipeline re-run on the '
      f'same partitions; p = (b+1)/(B+1); CI: rack-clustered bootstrap, {B_METRIC} draws)')

# calibration: quintiles of the out-of-fold covariate score
qq = pd.qcut(s_obs, 5, labels=False)
cal = []
for g in range(5):
    m = qq == g
    R = sc.risk_grid(a[m], b0[m], d0[m], T)
    R[np.where(d0[m] == 1)[0], b0[m][d0[m] == 1]] = 1
    cal.append({'quintile': g + 1, 'dies': int(m.sum()), 'events': int(d0[m].sum()),
                'observed_pct_per_month': 100 * d0[m].sum() / R.sum(),
                'predicted_pct_per_month': 100 * (h_obs[m] * R).sum() / R.sum()})
cal = pd.DataFrame(cal)
print('  calibration by out-of-fold risk quintile, partition 1 (% per die-month):')
print('  ' + cal.to_string(index=False, float_format=lambda v: f'{v:.3f}').replace('\n', '\n  '))

# ------------------------------------------------------------------ sensitivity
print('\n=== sensitivity: baseline hazard and link (primary endpoint) ===')
sens_base = {}
for name, base, link in [('constant hazard (v1)', 'const', 'cloglog'),
                         ('restricted cubic spline, 4 df', 'rcs', 'cloglog'),
                         ('monthly, unpenalised', 'fe_free', 'cloglog'),
                         ('monthly, shrinkage SD 3.2', 'fe_pen01', 'cloglog'),
                         ('monthly, logit link', 'fe_pen', 'logit')]:
    r_ = fit_hr(b0, d0, base=base, link=link, nboot=B_BOOT_SENS)
    sens_base[name] = r_
    print(f'  {name:<32} ' + '  '.join(f'{n}={x}' for n, x in zip(sc.FEATS, hr_line(r_))))

print('\n=== sensitivity: endpoint definition (replacement-consistent events) ===')
print(f'  (one rack partition, {B_PERM_SENS} rack-block permutations each; HR 95% rack-bootstrap)')
sens_ep = {}
for w in ENDPOINTS:
    b_, d_ = EP[w]
    r_ = prim if w == PRIMARY else fit_hr(b_, d_, nboot=B_BOOT_SENS)
    h0_ = [h0s[0]] if w == PRIMARY else [baseline_only(b_, d_, FOLD, SPEC)]
    mdf, o_ = cv_all(ids, lot, b_, d_, [FOLD], h0_, SPEC)
    mm = mdf.mean()
    cc = boot_ci([boot_prep(o_[0][0], o_[0][1], o_[0][2], h0_[0], b_, d_, FOLD)], 1000, 14)
    nl = perm_null(b_, d_, [FOLD], h0_, B_PERM_SENS, 13, SPEC)
    pp = {k: pval(nl[k], mm[k]) for k in ['C', 'Cpool', 'auc28', 'dll']}
    sens_ep[w] = {'hr': r_, 'm': mm.to_dict(), 'p': pp, 'ci95': cc}
    print(f'  {w:<8} events {d_.sum():3d}  C {mm["C"]:.3f} [{cc["C"][0]:.2f}, {cc["C"][1]:.2f}] '
          f'(p {pp["C"]:.3f})  '
          f'dLL {mm["dll"]:+.2f} (p {pp["dll"]:.3f})  ' +
          '  '.join(f'{n}={x}' for n, x in zip(sc.FEATS, hr_line(r_))))

print('\n=== which covariates carry the held-out ranking (one rack partition) ===')


def cv_C(ids_, lot_, b_, d_, fold, cols=None):
    """Held-out risk-set concordance only (fast path for sub-models and long permutation runs)."""
    s = np.empty(len(a))
    for f in range(K):
        trn, te = fold != f, fold == f
        Xt, Xe, _, _ = sc.design_ids(ids_[trn], ids_[te], lot_[trn], lot_[te], F)
        if cols is not None:
            Xt, Xe = Xt[:, cols], Xe[:, cols]
        s[te] = Xe @ sc.fit_dth(Xt, a[trn], b_[trn], d_[trn], T, **SPEC)['beta']
    return sc.risk_set_concordance(s, a, b_, d_, fold)


sub_C = {}
for w in [PRIMARY, 'any']:
    b_, d_ = EP[w]
    sub_C[w] = {'all five': cv_C(ids, lot, b_, d_, FOLD),
                'lot only': cv_C(ids, lot, b_, d_, FOLD, [4]),
                'map features only (no lot)': cv_C(ids, lot, b_, d_, FOLD, [0, 1, 2, 3])}
    print(f'  {w:<8} ' + '   '.join(f'{k}: C {v:.3f}' for k, v in sub_C[w].items()))

# the broadest endpoint is the only one near a multiplicity threshold: resolve its p more finely
B_ANY = 4999
b_, d_ = EP['any']
r_any = np.random.default_rng(15)
null_any = np.array([cv_C(*rack_block(r_any), b_, d_, FOLD) for _ in range(B_ANY)])
c_any = sub_C['any']['all five']
p_any = pval(null_any, c_any)
c_any_parts = [cv_C(ids, lot, b_, d_, fo) for fo in PARTS]
print(f'  any-change endpoint, {B_ANY} rack-block permutations: C {c_any:.3f}, null '
      f'{null_any.mean():.3f} +/- {null_any.std():.3f}, p = {p_any:.4f} '
      f'(Monte Carlo SE {np.sqrt(p_any * (1 - p_any) / B_ANY):.4f}); Bonferroni threshold for '
      f'{len(ENDPOINTS)} endpoints {0.05 / len(ENDPOINTS):.4f}; mean C over the {R_PART} '
      f'partitions {np.mean(c_any_parts):.3f}')

print('\n=== sensitivity: ridge-penalised covariates, nested rack CV ===')
GRID = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
s_r, h_r, lam_sel = np.empty(len(a)), np.empty((len(a), T)), []
for f in range(K):
    trn, te = np.where(FOLD != f)[0], np.where(FOLD == f)[0]
    in_racks = np.unique(rack[trn])
    inner = {rk: i % 6 for i, rk in enumerate(np.random.default_rng(f).permutation(in_racks))}
    ifold = np.array([inner[x] for x in rack[trn]])
    best, best_ll = None, -np.inf
    for lam in GRID:
        ll = 0.0
        for g in range(6):
            it, iv = trn[ifold != g], trn[ifold == g]
            Xt, Xv, _, _ = sc.design_ids(ids[it], ids[iv], lot[it], lot[iv], F)
            sb, mb = Xt[:, [0, 4]].std(0), Xt[:, [0, 4]].mean(0)
            sb[sb == 0] = 1
            Xt[:, [0, 4]] = (Xt[:, [0, 4]] - mb) / sb
            Xv[:, [0, 4]] = (Xv[:, [0, 4]] - mb) / sb
            fit = sc.fit_dth(Xt, a[it], b0[it], d0[it], T, lam_beta=lam, **SPEC)
            hv = sc.inv_link(fit['alpha'][None, :] + (Xv @ fit['beta'])[:, None], 'cloglog')
            ll += sc.loglik_rows(hv, a[iv], b0[iv], d0[iv]).sum()
        if ll > best_ll:
            best, best_ll = lam, ll
    lam_sel.append(best)
    Xt, Xe, _, _ = sc.design_ids(ids[trn], ids[te], lot[trn], lot[te], F)
    sb, mb = Xt[:, [0, 4]].std(0), Xt[:, [0, 4]].mean(0)
    sb[sb == 0] = 1
    Xt[:, [0, 4]] = (Xt[:, [0, 4]] - mb) / sb
    Xe[:, [0, 4]] = (Xe[:, [0, 4]] - mb) / sb
    fit = sc.fit_dth(Xt, a[trn], b0[trn], d0[trn], T, lam_beta=best, **SPEC)
    s_r[te] = Xe @ fit['beta']
    h_r[te] = sc.inv_link(fit['alpha'][None, :] + s_r[te][:, None], 'cloglog')
m_ridge = metrics(s_r, s_r, h_r, h0s[0], b0, d0, FOLD)
print(f'  selected ridge penalty per outer fold: {lam_sel}')
print(f'  concordance {m_ridge["C"]:.3f}, log-lik vs baseline-only {m_ridge["dll"]:+.2f} nats, '
      f'IBS {m_ridge["dibs_pct"]:+.2f} %')

# ------------------------------------------------------------------ what the intervals exclude
print('\n=== what the data exclude ===')
lo_all = np.exp(prim['lo'])
hi_all = np.exp(prim['hi'])
cont = [1, 2, 3]
binr = [0, 4]
print(f'  no interval excludes 1: {bool(((lo_all <= 1) & (hi_all >= 1)).all())}')
print(f'  continuous features (per SD): every interval inside [{lo_all[cont].min():.2f}, '
      f'{hi_all[cont].max():.2f}] -> a hazard increase above {100 * (hi_all[cont].max() - 1):.0f}% '
      f'per SD is excluded; a protective effect down to {lo_all[cont].min():.2f} is not')
print(f'  binary contrasts (0->1): slice 0 [{lo_all[0]:.2f}, {hi_all[0]:.2f}], lot B '
      f'[{lo_all[4]:.2f}, {hi_all[4]:.2f}] -> effects up to about '
      f'{max(hi_all[0], 1 / lo_all[4]):.1f}-fold are NOT excluded')
noise = {}
for phi in (0.25, 0.5):
    up = (hi_all - phi) / (1 - phi)
    dn = np.clip((lo_all - phi) / (1 - phi), 0, None)
    # noise cases are ranked at random, so C_obs = phi/2 + (1 - phi) C_true
    c_up = 0.5 + (ci['C'][1] - 0.5) / (1 - phi)
    noise[phi] = (dn, up, c_up)
    print(f'  if {int(phi * 100)}% of events were unrelated to the die (non-differential noise), '
          f'the continuous-feature bound widens to [{dn[cont].min():.2f}, {up[cont].max():.2f}] '
          f'per SD and the concordance upper bound for the die-related events to {c_up:.2f}')
print(f'  a replacement by a spare with an identical map is invisible: probability '
      f'{100 * (pk ** 2).sum():.2f}% on average, {100 * pk.max():.1f}% for the most common map')
# a die with map k loses a fraction p_k of its events, so log(observed hazard) gains log(1 - p_k):
# regressing that offset on the standardised design gives the multiplicative bias per HR unit
p_die = U.key.map(pk).values
miss_off = np.log1p(-p_die)
A_ = np.column_stack([np.ones(len(U)), X_all])
miss_bias = np.exp(np.linalg.lstsq(A_, miss_off, rcond=None)[0][1:])
print(f'  resulting bias in the hazard ratios (multiplicative, per unit of scale): '
      + '  '.join(f'{n} {v:.3f}' for n, v in zip(sc.FEATS, miss_bias))
      + f'; commonest vs never-seen map {1 / (1 - pk.max()):.3f}')

# ------------------------------------------------------------------ outputs
ep_counts = {w: {'events': int(EP[w][1].sum()), 'die_months': int((EP[w][0] - a + 1).sum()),
                 'events_lot_A': int(EP[w][1][lot == 0].sum()),
                 'events_lot_B': int(EP[w][1][lot == 1].sum())} for w in ENDPOINTS}
hr = {n: float(np.exp(prim['beta'][j])) for j, n in enumerate(sc.FEATS)}
hr_lo = {n: float(np.exp(prim['lo'][j])) for j, n in enumerate(sc.FEATS)}
hr_hi = {n: float(np.exp(prim['hi'][j])) for j, n in enumerate(sc.FEATS)}
alpha_bar = float(prim['alpha'].mean())
json.dump({
    'schema': ('v2 2026-09. Primary model: grouped-time proportional hazards (cloglog link), one '
               'baseline effect per calendar month shrunk to their mean (Gaussian, SD 1), endpoint '
               '"pulled" = die-tracked first departure of the die from its node. beta/lo/hi keep '
               'the v1 layout: beta[0] = mean monthly baseline log-hazard (NOT a constant-hazard '
               'intercept), beta[1:] = log hazard ratios for feats, on the scale in "scale" '
               '(binary 0->1, continuous per SD of the 1,960 dies). lo/hi: 95% rack-clustered '
               'percentile bootstrap. hr/hr_lo/hr_hi are the same on the HR scale. The v1 '
               'numbers are kept under "legacy_v1".'),
    'feats': sc.FEATS,
    'scale': SCALE,
    'sd': {n: float(SD_all[j]) for j, n in enumerate(sc.FEATS)},
    'beta': [alpha_bar] + prim['beta'].tolist(),
    'lo': [None] + prim['lo'].tolist(),              # no interval for the baseline summary
    'hi': [None] + prim['hi'].tolist(),
    'hr': hr, 'hr_lo': hr_lo, 'hr_hi': hr_hi,
    'n_events': n_ev, 'n_months': prim['n_months'], 'n_dies': int(len(U)),
    'n_racks': int(len(racks)), 'events_per_covariate': n_ev / len(sc.FEATS),
    'endpoint': PRIMARY, 'n_boot': B_BOOT, 'link': 'cloglog',
    'baseline_log_hazard_by_month': prim['alpha'].tolist(),
    'events_by_month': np.bincount(b0[d0 == 1], minlength=T).tolist(),
    'legacy_v1': {'beta': [-6.4106956659962115, -0.0031851960482449817, -0.22622123427659765,
                           -0.04016903169431786, -0.24212972426494825, -0.1773225674308788],
                  'lo': [-6.758249195254479, -0.4254314305207855, -0.5275663642466216,
                         -0.2803496923245366, -0.6061846034667365, -0.49496863346454695],
                  'hi': [-6.181307560699927, 0.41221248649981324, 0.01914973093516472,
                         0.2105138599054951, 0.06260728707434728, 0.1766096173592948],
                  'n_events': 92, 'n_months': 53597,
                  'note': 'v1: label endpoint, constant hazard, logit, every feature per row-SD'},
}, open(OUT / 'survival_fit.json', 'w'), indent=1)

json.dump({
    'schema': ('v2 2026-09. "cindex" is now the risk-set concordance (Harrell-type, discrete '
               'time; Heagerty-Zheng incident/dynamic summary): each event month\'s cases vs dies '
               'still at risk and event-free that month, within held-out rack folds, ties 1/2. '
               'Also: cumulative/dynamic AUC at 12 and 28 months; out-of-fold survival '
               'log-likelihood and integrated Brier score vs a baseline-only model fitted in the '
               'same folds. CIs: rack-clustered bootstrap of out-of-fold predictions. p: '
               'rack-block permutation of covariates with the whole pipeline re-run, (b+1)/(B+1). '
               'legacy_v1 holds the old value, which was a pooled person-month AUC.'),
    'cindex': float(m_obs['C']), 'cindex_ci95': ci['C'],
    'cindex_null_mean': float(null.C.mean()), 'cindex_null_sd': float(null.C.std()),
    'cindex_p': p['C'], 'n_perm': B_PERM, 'n_partitions': R_PART,
    'cindex_pooled_across_folds': float(m_obs['Cpool']), 'cindex_pooled_ci95': ci['Cpool'],
    'cindex_pooled_null_mean': float(null.Cpool.mean()), 'cindex_pooled_p': p['Cpool'],
    'cd_auc_12m': float(m_obs['auc12']), 'cd_auc_12m_ci95': ci['auc12'],
    'cd_auc_12m_p': p['auc12'],
    'cd_auc_28m': float(m_obs['auc28']), 'cd_auc_28m_ci95': ci['auc28'],
    'cd_auc_28m_p': p['auc28'],
    'delta_loglik_nats': float(m_obs['dll']), 'delta_loglik_ci95': ci['dll'],
    'delta_loglik_p': p['dll'],
    'ibs_model': float(m_obs['ibs']), 'ibs_baseline_only': float(m_obs['ibs0']),
    'delta_ibs_pct': float(m_obs['dibs_pct']), 'delta_ibs_pct_ci95': ci['dibs_pct'],
    'delta_ibs_p': p['dibs_pct'],
    'per_partition': per_part.to_dict(orient='list'),
    'calibration_quintiles': cal.to_dict(orient='records'),
    'folds': K, 'endpoint': PRIMARY,
    'legacy_v1': {'cindex': 0.44835031671156317, 'null_mean': 0.4991, 'null_sd': 0.0310,
                  'p': 0.949, 'what_it_was': 'pooled person-month AUC over all event vs '
                  'non-event rows, with fold intercepts; not a survival concordance'},
}, open(OUT / 'survival_cindex.json', 'w'), indent=1)


def pack(r_):
    return {'hr': np.exp(r_['beta']).tolist(), 'lo': np.exp(r_['lo']).tolist(),
            'hi': np.exp(r_['hi']).tolist(), 'n_events': r_['n_events'],
            'n_months': r_['n_months']}


json.dump({
    'schema': 'v2 2026-09. HR lists follow feats order in survival_fit.json.',
    'feats': sc.FEATS,
    'endpoint_audit': {'transitions': int(len(tr)), **{k: int(v) for k, v in log.items()},
                       'chance_cross_node_matches_expected': exp_chance,
                       'v1_label_events_that_were_partner_socket_moves': int(moved_within),
                       'v1_label_events_charged_to_partner_die': int(reattributed),
                       'first_departure_kinds': {k: int(v) for k, v in fk.items()},
                       'counts': ep_counts,
                       'identical_map_miss_prob_mean': float((pk ** 2).sum()),
                       'identical_map_miss_prob_max': float(pk.max()),
                       'identical_map_miss_hr_bias': dict(zip(sc.FEATS, miss_bias.tolist())),
                       'reciprocal_note': ('reciprocal_cross_node = cross-node moves whose label '
                                           'received, in the same episode, the die that left '
                                           'the destination label (two per two-way exchange)')},
    'submodels_C_one_partition': sub_C,
    'any_endpoint_fine_null': {'C': c_any, 'n_perm': B_ANY, 'p': p_any,
                               'null_mean': float(null_any.mean()),
                               'null_sd': float(null_any.std()),
                               'C_mean_over_partitions': float(np.mean(c_any_parts)),
                               'bonferroni_threshold': 0.05 / len(ENDPOINTS)},
    'baselines': {k: pack(v) for k, v in sens_base.items()},
    'endpoints': {w: {**pack(v['hr']), 'metrics': v['m'], 'p': v['p'], 'ci95': v['ci95']}
                  for w, v in sens_ep.items()},
    'ridge_nested': {'grid': GRID, 'selected': lam_sel, 'metrics': m_ridge},
    'noise_deattenuation': {str(k): {'lo': v[0].tolist(), 'hi': v[1].tolist(),
                                     'cindex_upper': float(v[2])}
                            for k, v in noise.items()},
    'cv_fold_betas': cv_betas.tolist(),
}, open(OUT / 'survival_sensitivity.json', 'w'), indent=1, default=float)

print('\n=== verdict ===')
null_result = (p['C'] > 0.05) and (p['dll'] > 0.05) and not (ci['C'][0] > 0.5)
if null_result:
    print('  No held-out discrimination distinguishable from chance: the baseline harvest map does')
    print(f'  not rank the dies later pulled significantly above chance (risk-set concordance '
          f'{m_obs["C"]:.3f}, 95% CI upper bound {ci["C"][1]:.3f}, p = {p["C"]:.3f}), and adding '
          f'the map '
          + ('lowers' if m_obs['dll'] < 0 else 'does not significantly raise')
          + ' the held-out survival log-likelihood.')
    print('  A bounded null: the intervals above say which hazard ratios remain possible.')
else:
    print('  The map carries held-out signal about replacement; see the coefficients above.')
