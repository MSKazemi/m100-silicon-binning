"""Re-derive the v1 reliability statistic (0.448) and take it apart, one reviewer claim at a time.

v1 `survival.py` (commit cfd73dd) reported an "out-of-fold concordance index" of 0.448 against a
permutation null of 0.499 +/- 0.031. This script reproduces that number exactly from the committed
data and then changes ONE thing at a time:

  (a) standardisation learned inside the training folds       -> reviewer: leakage (SV-03)
  (b) pattern rarity learned inside the training folds        -> reviewer: leakage (SV-02)
  (c) the fold intercept dropped from the held-out score      -> pooled-CV artefact?
  (d) the same model scored with a risk-set concordance       -> reviewer: wrong metric (SV-04)
  (e) the v1 statistic's null with the whole CV re-run under rack-block permutations of the
      covariates, instead of v1's shuffle of fixed out-of-fold scores across rows
  (f) the same v1 pipeline on 50 other random rack partitions: how typical was v1's partition?

Output: analysis/survival_legacy_check.json. Runtime ~2 min.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_core as sc  # noqa: E402

root = Path(__file__).resolve().parent.parent
T = 28
B = 499

sets = sc.clean_sets(root)
leg = sc.legacy_first_events(sets)
S, days_of = sc.label_histories(sets)
rows = []
for (n, s), ds in days_of.items():
    h = sc.harvested(S[(n, s, ds[0])])
    if len(h) != 4:
        continue
    rows.append(dict(node=n, socket=s, rack=n // 20, key=h, first=ds[0], last=ds[-1],
                     ev=leg.get((n, s))))
Bt = pd.DataFrame(rows)
freq = Bt.key.value_counts(normalize=True)              # v1: whole fleet, before the rack filter
Bt['rarity_v1'] = -np.log(Bt.key.map(freq).values)
Bt = Bt[Bt.rack < 49].reset_index(drop=True)
Fm = sc.map_features(list(Bt.key))
Bt['s0'], Bt['nadj'], Bt['spread'] = Fm[:, 0], Fm[:, 1], Fm[:, 2]
Bt['lot'] = (Bt.rack >= 22).astype(int)
a = np.array([sc.month_index(x) for x in Bt['first']])
end = [e if pd.notna(e) else lst for e, lst in zip(Bt.ev, Bt['last'])]
b = np.array([sc.month_index(x) for x in end])
d = Bt.ev.notna().values.astype(int)
rack = Bt.rack.values
w_rows = b - a + 1
print(f'v1 sample: {len(Bt):,} sockets, {d.sum()} events, {w_rows.sum():,} socket-months')


def row_standardise(X, w):
    mu = (X * w[:, None]).sum(0) / w.sum()
    sd = np.sqrt(((X - mu) ** 2 * w[:, None]).sum(0) / w.sum())
    return mu, sd


def pooled_auc(score):
    """v1 `cindex`: every event row against every non-event row (person-months), ties 1/2.
    With static scores this is a unit-level AUC with controls weighted by non-event months."""
    wn = (w_rows - d).astype(float)
    ev = np.where(d == 1)[0]
    tot = 0.0
    for i in ev:
        tot += (wn * ((score[i] > score) + 0.5 * (score[i] == score))).sum() / wn.sum()
    return tot / len(ev)


# v1 fold assignment, recovered by replaying its random stream (seed 23, 400 bootstrap draws)
rng = np.random.default_rng(23)
uniq = np.unique(rack)
for _ in range(400):
    rng.choice(uniq, len(uniq), replace=True)
folds = {r: i % 7 for i, r in enumerate(rng.permutation(uniq))}
fold = np.array([folds[r] for r in rack])
ONE = np.ones((T, 1))
FEATS = ['s0', 'nadj', 'spread', 'rarity_v1', 'lot']


def v1_cv(Xraw, fold, infold_std=False, infold_rarity=False, keys=None, drop_intercept=False):
    """v1 pipeline: constant hazard, logit, rows standardised; returns out-of-fold scores."""
    oof = np.empty(len(a))
    mu_all, sd_all = row_standardise(Xraw, w_rows)
    for f in range(7):
        trn, te = fold != f, fold == f
        X = Xraw.copy()
        if infold_rarity:
            tk = [keys[i] for i in np.where(trn)[0]]
            X[:, 3] = sc.rarity(tk, keys)
        if infold_std or infold_rarity:
            mu, sd = row_standardise(X[trn], w_rows[trn])
        else:
            mu, sd = mu_all, sd_all
        Z = (X - mu) / sd
        fit = sc.fit_dth(Z[trn], a[trn], b[trn], d[trn], T, link='logit', basis=ONE)
        oof[te] = (0 if drop_intercept else fit['g'][0]) + Z[te] @ fit['beta']
    return oof


X0 = Bt[FEATS].values.astype(float)
keys = list(Bt.key)
res = {}
oof = v1_cv(X0, fold)
res['v1_reproduced'] = pooled_auc(oof)
print(f'\nv1 pooled person-month AUC, reproduced          : {res["v1_reproduced"]:.4f}  '
      f'(paper: 0.448)')
oof_a = v1_cv(X0, fold, infold_std=True)
res['a_infold_standardisation'] = pooled_auc(oof_a)
res['a_max_abs_score_change'] = float(np.abs(oof_a - oof).max())
print(f'(a) + standardisation learned in-fold           : {res["a_infold_standardisation"]:.4f}  '
      f'(max |change in score| {res["a_max_abs_score_change"]:.1e}: affine-invariant MLE)')
oof_b = v1_cv(X0, fold, infold_rarity=True, keys=keys)
res['b_infold_rarity'] = pooled_auc(oof_b)
print(f'(b) + rarity learned in-fold (and scaling)       : {res["b_infold_rarity"]:.4f}')
oof_c = v1_cv(X0, fold, drop_intercept=True)
res['c_no_fold_intercept'] = pooled_auc(oof_c)
print(f'(c) fold intercept removed from the score        : {res["c_no_fold_intercept"]:.4f}')
res['d_risk_set_concordance_within_fold'] = sc.risk_set_concordance(oof_c, a, b, d, fold)
res['d_risk_set_concordance_pooled'] = sc.risk_set_concordance(oof_c, a, b, d, None)
print(f'(d) same model, risk-set concordance within fold : '
      f'{res["d_risk_set_concordance_within_fold"]:.4f}  (pooled {res["d_risk_set_concordance_pooled"]:.4f})')

# event rate per held-out fold vs its training intercept: the mechanism behind (c)
per_fold = []
for f in range(7):
    trn, te = fold != f, fold == f
    Z = (X0 - row_standardise(X0, w_rows)[0]) / row_standardise(X0, w_rows)[1]
    fit = sc.fit_dth(Z[trn], a[trn], b[trn], d[trn], T, link='logit', basis=ONE)
    per_fold.append({'fold': f, 'test_events': int(d[te].sum()),
                     'test_rate_pct': 100 * d[te].sum() / w_rows[te].sum(),
                     'train_intercept': float(fit['g'][0])})
per_fold = pd.DataFrame(per_fold)
res['fold_rate_vs_intercept_corr'] = float(np.corrcoef(per_fold.test_rate_pct,
                                                      per_fold.train_intercept)[0, 1])
print('    held-out event rate vs training intercept, per fold: r = '
      f'{res["fold_rate_vs_intercept_corr"]:+.2f}')
print('    ' + per_fold.to_string(index=False, float_format=lambda v: f'{v:.4f}')
      .replace('\n', '\n    '))

# (e) nulls: v1's own (shuffle fixed scores over rows) and a full-pipeline rack-block null
r2 = np.random.default_rng(5)
racks = np.unique(rack)
rack_idx = {r: np.where(rack == r)[0] for r in racks}
assert all(len(v) == 40 for v in rack_idx.values())
null_full, null_c = [], []
for _ in range(B):
    pi = r2.permutation(racks)
    Xp = X0.copy()
    for rk, src in zip(racks, pi):
        Xp[rack_idx[rk]] = X0[rack_idx[src][r2.permutation(40)]]
    null_full.append(pooled_auc(v1_cv(Xp, fold)))
null_full = np.array(null_full)
res['e_full_pipeline_null_mean'] = float(null_full.mean())
res['e_full_pipeline_null_sd'] = float(null_full.std())
res['e_full_pipeline_p_upper'] = float(((null_full >= res['v1_reproduced']).sum() + 1) / (B + 1))
res['e_full_pipeline_p_lower'] = float(((null_full <= res['v1_reproduced']).sum() + 1) / (B + 1))
res['n_perm'] = B
print(f'(e) v1 statistic under {B} rack-block permutations, whole CV re-run: '
      f'{null_full.mean():.4f} +/- {null_full.std():.4f}; observed 0.448 has '
      f'one-sided p(above) = {res["e_full_pipeline_p_upper"]:.3f}, '
      f'p(below) = {res["e_full_pipeline_p_lower"]:.3f}')

# (f) how typical was v1's rack partition? The same v1 pipeline on 50 other random partitions
N_PART = 50
alt = []
for s in range(1000, 1000 + N_PART):
    fm = {r: i % 7 for i, r in enumerate(np.random.default_rng(s).permutation(uniq))}
    fo = np.array([fm[r] for r in rack])
    alt.append((pooled_auc(v1_cv(X0, fo)),
                sc.risk_set_concordance(v1_cv(X0, fo, drop_intercept=True), a, b, d, fo)))
alt = np.array(alt)
res['f_other_partitions_pooled_auc_mean'] = float(alt[:, 0].mean())
res['f_other_partitions_pooled_auc_sd'] = float(alt[:, 0].std())
res['f_v1_partition_pooled_auc_rank'] = int((alt[:, 0] < res['v1_reproduced']).sum())
res['f_other_partitions_risk_set_C_mean'] = float(alt[:, 1].mean())
res['f_v1_partition_risk_set_C_rank'] = int(
    (alt[:, 1] < res['d_risk_set_concordance_within_fold']).sum())
res['f_n_partitions'] = N_PART
print(f'(f) v1 pipeline on {N_PART} other random rack partitions: pooled AUC '
      f'{alt[:, 0].mean():.4f} +/- {alt[:, 0].std():.4f} (v1 partition 0.4484 is above '
      f'{res["f_v1_partition_pooled_auc_rank"]} of them); risk-set C {alt[:, 1].mean():.4f} '
      f'(v1 partition {res["d_risk_set_concordance_within_fold"]:.4f} is above '
      f'{res["f_v1_partition_risk_set_C_rank"]} of them)')
res['per_fold'] = per_fold.to_dict(orient='list')
json.dump(res, open(root / 'analysis' / 'survival_legacy_check.json', 'w'), indent=1)
print('\nwrote analysis/survival_legacy_check.json')
