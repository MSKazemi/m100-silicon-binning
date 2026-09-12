"""Estimators and data plumbing for the replacement-survival analysis (`survival.py`).

Kept separate from the driver so every formula can be unit-tested (`tests/test_survival.py`)
without loading the 31-month daily table.

Contents
--------
clean_sets              clean steady-state socket-days and their active-core sets (as before)
legacy_first_events     the v1 endpoint: first CPU-SWAP/NODE-SWAP on a socket *label*
track_dies              the v2 endpoint: follow each die by its map through relabelling and moves
map_features, rarity    baseline covariates; rarity is estimated from TRAINING keys only
fit_dth                 discrete-time hazard, grouped proportional hazards (cloglog) or logit
risk_set_concordance    Harrell-type concordance over monthly risk sets, within held-out groups
cd_auc                  cumulative/dynamic AUC at a horizon
loglik_rows, brier_curve  proper scores for the discrete hazard

Notation (matches the paper). Calendar month t = 0..T-1 (June 2020 = 0). Die i is at risk from
its entry month a_i to its exit month b_i; d_i = 1 if b_i is an event month. The interval hazard
is h_i(t) = P(T_i = t | T_i >= t, x_i); with the complementary log-log link
h_i(t) = 1 - exp(-exp(alpha_t + x_i' beta)), which is the grouped-time form of a continuous-time
proportional-hazards model (Prentice & Gloeckler 1978), so exp(beta) is a hazard ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')
N_PATTERNS = 495                                  # C(12, 4): every possible 4-of-12 map


# ====================================================================== data
def clean_sets(root: Path) -> pd.DataFrame:
    """Clean steady-state socket-days (16 active cores, all at >90 % cadence) and their core sets.

    Identical to the construction in v1 `survival.py` and `change_taxonomy.py`.
    """
    df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
                   ignore_index=True)
    df['day'] = pd.to_datetime(df['day'], utc=True)
    a = df.groupby(['node', 'socket', 'day']).size().rename('active')
    fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
    t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
    clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
    sets = (df.merge(clean, on=['node', 'socket', 'day'])
              .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
    return sets[sets.day >= CUT].sort_values(['node', 'socket', 'day']).reset_index(drop=True)


def harvested(active) -> tuple:
    """Harvested slice indices of a map given its ACTIVE core set."""
    return tuple(sorted(set(range(12)) - {k // 2 for k in active}))


def month_index(day, origin=CUT) -> int:
    """Calendar month index, origin month = 0."""
    return (day.year - origin.year) * 12 + (day.month - origin.month)


def label_histories(sets: pd.DataFrame):
    S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
    days_of = {k: sorted(g.day) for k, g in sets.groupby(['node', 'socket'])}
    return S, days_of


def transitions(sets: pd.DataFrame) -> pd.DataFrame:
    """Every map change on a socket label between consecutive clean days."""
    S, days_of = label_histories(sets)
    rows = []
    for (n, s), ds in days_of.items():
        for i in range(1, len(ds)):
            o, w = S[(n, s, ds[i - 1])], S[(n, s, ds[i])]
            if o != w:
                rows.append((n, s, ds[i - 1], ds[i], o, w))
    return pd.DataFrame(rows, columns=['node', 'sock', 'd0', 'd1', 'old', 'new'])


# ====================================================================== endpoints
def legacy_first_events(sets: pd.DataFrame) -> dict:
    """v1 endpoint, reproduced exactly: first silicon-moving event on each socket LABEL.

    Changes are grouped by (node, first clean day after the change); a mirror is a RELABEL and
    moves no silicon; any other change marks every changed label. No die is followed through a
    relabel, so a later swap on a relabelled socket is charged to the die that used to sit there.
    """
    S, days_of = label_histories(sets)
    events = {}
    for (n, s), ds in days_of.items():
        for i in range(1, len(ds)):
            if S[(n, s, ds[i - 1])] != S[(n, s, ds[i])]:
                events.setdefault((n, ds[i]), {})[s] = (S[(n, s, ds[i - 1])], S[(n, s, ds[i])])
    swap = {}
    for (n, d), per in sorted(events.items()):
        if len(per) == 2:
            (o0, n0), (o1, n1) = per[0], per[1]
            if n0 == o1 and n1 == o0:
                continue
            for s in (0, 1):
                swap.setdefault((n, s), d)
        else:
            swap.setdefault((n, next(iter(per))), d)
    return swap


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def _overlap(a0, a1, b0, b1, slack):
    return (a0 - slack <= b1) and (b0 - slack <= a1)


def track_dies(tr: pd.DataFrame, initial: dict, slack_days: int = 3):
    """Follow dies by their maps through relabelling and physical moves.

    Parameters
    ----------
    tr       transitions() output (every node, so cross-node moves can be matched)
    initial  {(node, sock): (first_map, first_day)} -- the dies present at entry
    slack    days of tolerance when deciding that two changes may be the same maintenance act

    A change on a label is a DEPARTURE of the old map and an ARRIVAL of the new one. Changes whose
    observation windows [last day with old map, first day with new map] overlap (within `slack`)
    and that share a node or a map are one EPISODE. Inside an episode a departing map is matched to
    an arriving identical map -- on the same node first (relabel, or a die moved to the partner
    socket), then on another node (a die moved between nodes). Unmatched departures are dies that
    left the fleet; unmatched arrivals are new (spare) dies.

    Returns (events, log): events[(node, sock)] is the chronological list of what happened to the
    ORIGINAL die first seen on that label: dicts with day, kind in {'relabel', 'moved', 'retired'}
    and n_out (dies leaving that node in the episode: 1 = CPU-SWAP-like, 2 = NODE-SWAP-like).
    `log` counts episode outcomes for reporting; 'reciprocal_cross_node' counts cross-node moves
    whose label received, in the same episode, the die that left the destination label.
    """
    slack = pd.Timedelta(days=slack_days)
    tr = tr.sort_values('d1').reset_index(drop=True)
    n = len(tr)
    uf = _UF(n)
    for i in range(n):
        ri = tr.iloc[i]
        for j in range(i + 1, n):
            rj = tr.iloc[j]
            if (ri.node, ri.sock) == (rj.node, rj.sock):
                continue          # consecutive changes on one label are never one act
            if not _overlap(ri.d0, ri.d1, rj.d0, rj.d1, slack):
                continue
            if ri.node == rj.node or ri.old == rj.new or rj.old == ri.new:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    # holders: label -> die id; dies: id -> {'orig': (node, sock) or None, 'map': ...}
    holder, dies = {}, {}
    for lab, (m, _) in initial.items():
        dies[lab] = {'orig': lab, 'map': m}
        holder[lab] = lab
    spare = [0]
    events = {lab: [] for lab in initial}
    log = {'episodes': 0, 'within_node_moves': 0, 'cross_node_moves': 0, 'retired': 0,
           'spares_in': 0, 'holder_mismatch': 0, 'reciprocal_cross_node': 0}

    def new_die(m):
        spare[0] += 1
        sid = ('spare', spare[0])
        dies[sid] = {'orig': None, 'map': m}
        return sid

    order = sorted(groups.values(), key=lambda g: min(tr.d1[i] for i in g))
    for g in order:
        log['episodes'] += 1
        g = sorted(g, key=lambda i: (tr.d1[i], tr.d0[i]))
        lab_of = {i: (tr.node[i], tr.sock[i]) for i in g}
        # a label may change more than once inside an episode: its k-th departure is the die that
        # arrived with its (k-1)-th change -- a chain, not a match
        prev_on_label = {}
        last = {}
        for i in g:
            prev_on_label[i] = last.get(lab_of[i])
            last[lab_of[i]] = i
        used, match = set(), {}
        for same_node in (True, False):
            for i in g:
                if i in match:
                    continue
                lab, m = lab_of[i], tr.old[i]
                cands = [(abs((tr.d1[i] - tr.d1[j]).days), j)
                         for j in g
                         if j not in used and tr.new[j] == m and lab_of[j] != lab
                         and (lab_of[j][0] == lab[0]) == same_node
                         and _overlap(tr.d0[i], tr.d1[i], tr.d0[j], tr.d1[j], slack)]
                if cands:
                    _, j = min(cands)
                    used.add(j)
                    match[i] = j
        # a cross-node move is RECIPROCAL when the die arriving on its label in the same episode
        # is the one that left the label it went to (a two-way exchange between two nodes)
        log['reciprocal_cross_node'] += sum(1 for i, j in match.items()
                                            if lab_of[j][0] != lab_of[i][0] and match.get(j) == i)
        # resolve die identities to a fixed point: departure <- holder or chain predecessor's
        # arrival; arrival <- its matched departure, else a new (spare) die
        dep_die, arr_die = {}, {}
        for _ in range(len(g) + 2):
            for i in g:
                p = prev_on_label[i]
                dep_die[i] = holder.get(lab_of[i]) if p is None else arr_die.get(p)
            src = {j: i for i, j in match.items()}
            for j in g:
                if j in src:
                    arr_die[j] = dep_die.get(src[j])
                elif j not in arr_die or arr_die[j] is None:
                    arr_die[j] = new_die(tr.new[j])
                    log['spares_in'] += 1
        for i in g:
            if dep_die[i] is None:
                dep_die[i] = new_die(tr.old[i])
            if dies[dep_die[i]]['map'] != tr.old[i]:
                log['holder_mismatch'] += 1
        # which pre-episode dies left each node (moved off-node or retired)
        nodes = {lab_of[i][0] for i in g}
        pre = {nd: {holder.get((nd, s)) for s in (0, 1)} - {None} for nd in nodes}
        new_holder = dict(holder)
        for lab in {lab_of[i] for i in g}:
            new_holder[lab] = arr_die[last[lab]]
        post = {nd: {new_holder.get((nd, s)) for s in (0, 1)} - {None} for nd in nodes}
        n_out = {nd: len(pre[nd] - post[nd]) for nd in nodes}
        for i in g:
            did = dep_die[i]
            if i in match:
                kind = 'relabel' if lab_of[match[i]][0] == lab_of[i][0] else 'moved'
                log['within_node_moves' if kind == 'relabel' else 'cross_node_moves'] += 1
            else:
                kind = 'retired'
                log['retired'] += 1
            if dies[did]['orig'] is not None:
                events[dies[did]['orig']].append(
                    {'day': tr.d1[i], 'kind': kind, 'n_out': n_out[lab_of[i][0]],
                     'node': lab_of[i][0]})
        holder = new_holder
    return events, log


def endpoint_table(events: dict, units: pd.DataFrame, which: str) -> pd.Series:
    """First event day per unit under an endpoint definition (NaT = none).

    'pulled'  the die's map left its node (CPU-SWAP- or NODE-SWAP-like, incl. moves to other nodes)
    'retired' the die's map left the fleet (not re-found on any node in the same episode)
    'single'  pulled while the partner die stayed (CPU-SWAP-like); NODE-SWAP-like -> censored
    'double'  pulled together with the partner (NODE-SWAP-like); CPU-SWAP-like -> censored
    'any'     any change touching the die, relabelling included
    Returns (event_day, censor_day) where censor_day marks a competing-event censoring (or NaT).
    """
    ev, cen = [], []
    for lab in zip(units.node, units.socket):
        e = c = pd.NaT
        for x in events.get(lab, []):
            if which == 'any':
                e = x['day']
                break
            if which == 'retired':
                if x['kind'] == 'retired':
                    e = x['day']
                    break
                continue
            if x['kind'] == 'relabel':
                continue
            # the die left its node
            if which == 'pulled':
                e = x['day']
                break
            like = 'single' if x['n_out'] == 1 else 'double'
            if like == which:
                e = x['day']
            else:
                c = x['day']
            break
        ev.append(e)
        cen.append(c)
    return pd.Series(ev, index=units.index), pd.Series(cen, index=units.index)


# ====================================================================== covariates
def map_features(keys) -> np.ndarray:
    """(s0, nadj, spread) for each harvested-slice tuple."""
    out = np.empty((len(keys), 3))
    for i, h in enumerate(keys):
        out[i, 0] = float(0 in h)
        out[i, 1] = sum(1 for k in range(len(h) - 1) if h[k + 1] - h[k] == 1)
        out[i, 2] = float(np.std(h))
    return out


def rarity(train_keys, keys, a: float = 0.5, k: int = N_PATTERNS) -> np.ndarray:
    """-log of the smoothed frequency of each key among TRAIN keys only.

    Additive (Krichevsky-Trofimov, a = 1/2) smoothing over the k possible patterns gives a finite
    value to a pattern never seen in training. Held-out keys never enter the estimate.
    """
    vc = pd.Series(list(train_keys)).value_counts()
    n = len(train_keys)
    cnt = np.array([vc.get(h, 0) for h in keys], dtype=float)
    return -np.log((cnt + a) / (n + a * k))


def design(keys_train, keys_eval, lot_train, lot_eval, standardise=True):
    """Covariate matrices with every learned quantity (rarity, mean, SD) from the TRAIN rows.

    Columns: s0 (0/1), nadj, spread, rarity (per training SD when standardised), lot (0/1).
    Binary columns are left on their 0/1 scale so exp(beta) is a 0->1 hazard ratio.
    """
    ft, fe = map_features(keys_train), map_features(keys_eval)
    rt, re_ = rarity(keys_train, keys_train), rarity(keys_train, keys_eval)
    Xt = np.column_stack([ft[:, 0], ft[:, 1], ft[:, 2], rt, lot_train]).astype(float)
    Xe = np.column_stack([fe[:, 0], fe[:, 1], fe[:, 2], re_, lot_eval]).astype(float)
    mu, sd = np.zeros(5), np.ones(5)
    if standardise:
        cont = [1, 2, 3]
        mu[cont] = Xt[:, cont].mean(0)
        sd[cont] = Xt[:, cont].std(0)
    return (Xt - mu) / sd, (Xe - mu) / sd, mu, sd


def design_ids(ids_train, ids_eval, lot_train, lot_eval, F, a: float = 0.5, k: int = N_PATTERNS,
               standardise=True):
    """`design` on integer key ids with a per-id feature table F (n_ids, 3) -- same arithmetic,
    fast enough to sit inside a permutation loop."""
    cnt = np.bincount(ids_train, minlength=F.shape[0]).astype(float)
    lp = -np.log((cnt + a) / (len(ids_train) + a * k))
    Xt = np.column_stack([F[ids_train], lp[ids_train], lot_train]).astype(float)
    Xe = np.column_stack([F[ids_eval], lp[ids_eval], lot_eval]).astype(float)
    mu, sd = np.zeros(5), np.ones(5)
    if standardise:
        cont = [1, 2, 3]
        mu[cont] = Xt[:, cont].mean(0)
        sd[cont] = Xt[:, cont].std(0)
    return (Xt - mu) / sd, (Xe - mu) / sd, mu, sd


FEATS = ['s0', 'nadj', 'spread', 'rarity', 'lot']


# ====================================================================== the hazard model
def _terms(eta, link, event):
    """log-likelihood contribution and its first two eta-derivatives, elementwise."""
    if link == 'cloglog':
        eta = np.minimum(eta, 6.0)
        mu = np.exp(eta)
        if not event:                                  # log(1 - h) = -mu
            return -mu, -mu, -mu
        em1 = np.expm1(mu)
        l = np.log(-np.expm1(-mu))
        small = mu < 1e-4
        d1 = np.where(small, 1 - mu / 2 + mu ** 2 / 12, mu / np.where(small, 1, em1))
        d2 = np.where(small, -mu / 2 + mu ** 2 / 6,
                      mu * (em1 - mu * np.exp(mu)) / np.where(small, 1, em1) ** 2)
        return l, d1, d2
    if link == 'logit':
        s = 1 / (1 + np.exp(-eta))
        if not event:
            return -np.logaddexp(0, eta), -s, -s * (1 - s)
        return -np.logaddexp(0, -eta), 1 - s, -s * (1 - s)
    raise ValueError(link)


def inv_link(eta, link):
    return -np.expm1(-np.exp(np.minimum(eta, 6.0))) if link == 'cloglog' else 1 / (1 + np.exp(-eta))


def risk_grid(a, b, d, T):
    """(R, event months): R[i, t] = 1 for at-risk NON-event months of unit i."""
    t = np.arange(T)
    R = (t[None, :] >= a[:, None]) & (t[None, :] <= b[:, None])
    ev = d.astype(bool)
    R[np.where(ev)[0], b[ev]] = False
    return R.astype(float)


def fe_penalty(T, lam):
    """Gaussian shrinkage of monthly effects towards their mean: lam * (I - J/T)."""
    return lam * (np.eye(T) - np.ones((T, T)) / T)


def fit_dth(X, a, b, d, T, link='cloglog', basis=None, pen_base=None, lam_beta=0.0,
            month_mask=None, max_iter=200, tol=1e-8):
    """Maximum (penalised) likelihood for a discrete-time hazard with baseline alpha = basis @ g.

    X (n, p) static covariates; a, b, d entry month, exit month, event flag. `pen_base` is a
    penalty matrix on g; `lam_beta` a ridge on beta. `month_mask` (bool, T) drops months from the
    likelihood -- used for the unpenalised fixed-effects fit, where a month with no event has an
    MLE of alpha = -inf and carries no information about beta.
    Returns dict(g, beta, alpha, loglik, penalised, converged, iters, cov).
    """
    X = np.asarray(X, float)
    n, p = X.shape
    basis = np.eye(T) if basis is None else np.asarray(basis, float)
    q = basis.shape[1]
    R = risk_grid(a, b, d, T)
    if month_mask is not None:
        R[:, ~month_mask] = 0.0
    ev = np.where(d.astype(bool))[0]
    et = b[ev]
    P = np.zeros((q + p, q + p))
    if pen_base is not None:
        P[:q, :q] = pen_base
    P[q:, q:] = lam_beta * np.eye(p)
    rate = max(len(ev), 0.5) / max(R.sum() + len(ev), 1.0)
    a0 = np.log(-np.log1p(-rate)) if link == 'cloglog' else np.log(rate / (1 - rate))
    g0 = np.linalg.lstsq(basis, np.full(T, a0), rcond=None)[0]
    theta = np.concatenate([g0, np.zeros(p)])

    def evaluate(th, need_grad=True):
        g, beta = th[:q], th[q:]
        alpha = basis @ g
        xb = X @ beta
        ee = alpha[et] + xb[ev]
        l1, d11, d21 = _terms(ee, link, event=True)
        if link == 'cloglog':
            # log(1 - h) = -exp(alpha_t) exp(x_i beta) factorises: every non-event sum is a
            # product with the at-risk matrix, O(n T) with no elementwise exp over the grid
            u, v = np.exp(xb), np.exp(alpha)
            Rv = R @ v
            Ru = R.T @ u
            ll = float(-u @ Rv + l1.sum())
        else:
            eta = alpha[None, :] + xb[:, None]
            l0, d10, d20 = _terms(eta, link, event=False)
            ll = float((l0 * R).sum() + l1.sum())
        llp = ll - 0.5 * th @ P @ th
        if not need_grad:
            return llp, ll
        if link == 'cloglog':
            ga = -v * Ru + np.bincount(et, d11, T)
            gb = -X.T @ (u * Rv) + X[ev].T @ d11
            Haa = np.diag(-v * Ru + np.bincount(et, d21, T))
            Hab = -v[:, None] * (R.T @ (u[:, None] * X))
            Hbb = -X.T @ (X * (u * Rv)[:, None])
        else:
            w1, w2 = d10 * R, d20 * R
            ga = w1.sum(0) + np.bincount(et, d11, T)
            gb = X.T @ w1.sum(1) + X[ev].T @ d11
            Haa = np.diag(w2.sum(0) + np.bincount(et, d21, T))
            Hab = w2.T @ X
            Hbb = X.T @ (X * w2.sum(1)[:, None])
        np.add.at(Hab, et, d21[:, None] * X[ev])
        Hbb = Hbb + (X[ev] * d21[:, None]).T @ X[ev]
        grad = np.concatenate([basis.T @ ga, gb]) - P @ th
        H = np.block([[basis.T @ Haa @ basis, basis.T @ Hab], [Hab.T @ basis, Hbb]]) - P
        return llp, ll, grad, H

    llp, ll, grad, H = evaluate(theta)
    converged = False
    for it in range(max_iter):
        try:
            step = np.linalg.solve(-H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-H, grad, rcond=None)[0]
        s = 1.0
        while True:
            new = theta + s * step
            llp_new, _ = evaluate(new, need_grad=False)
            if np.isfinite(llp_new) and llp_new >= llp - 1e-12:
                break
            s /= 2
            if s < 1e-10:
                break
        theta = new
        llp_old = llp
        llp, ll, grad, H = evaluate(theta)
        if np.max(np.abs(grad)) < tol or abs(llp - llp_old) < 1e-12:
            converged = True
            break
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan)
    return {'g': theta[:q], 'beta': theta[q:], 'alpha': basis @ theta[:q], 'loglik': ll,
            'penalised': llp, 'converged': converged, 'iters': it + 1, 'cov': cov}


def rcs_basis(T, knots):
    """Restricted cubic spline basis in t = 0..T-1 (Harrell), intercept included."""
    t = np.arange(T, dtype=float)
    k = np.asarray(knots, float)
    K = len(k)
    cols = [np.ones(T), t]
    scale = (k[-1] - k[0]) ** 2

    def pos3(x):
        return np.clip(x, 0, None) ** 3
    for j in range(K - 2):
        cols.append((pos3(t - k[j]) - pos3(t - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + pos3(t - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / scale)
    return np.column_stack(cols)


# ====================================================================== validation metrics
def risk_set_concordance(score, a, b, d, group=None, return_parts=False):
    """Concordance over monthly risk sets (Harrell-type C for discrete time).

    For every event month t and every case i with an event in t, each unit j of the same `group`
    still at risk in t and event-free in t is a comparable control; the pair is concordant when
    score_i > score_j (ties count 1/2). Tied event times are not compared (neither is a control
    for the other). Pooled over months and groups this is the Heagerty-Zheng incident/dynamic
    summary concordance. Censoring enters only through the risk sets, which is valid under
    independent censoring (no IPCW needed).
    """
    score = np.asarray(score, float)
    group = np.zeros(len(score), int) if group is None else np.asarray(group)
    cases = np.where(d.astype(bool))[0]
    conc = pairs = 0.0
    parts = []
    for i in cases:
        t = b[i]
        ctrl = (group == group[i]) & (a <= t) & ((b > t) | ((b == t) & (d == 0)))
        s = score[ctrl]
        c = (score[i] > s).sum() + 0.5 * (score[i] == s).sum()
        conc += c
        pairs += ctrl.sum()
        if return_parts:
            parts.append((i, np.where(ctrl)[0]))
    C = conc / pairs if pairs else np.nan
    return (C, parts) if return_parts else C


def cd_auc(score, a, b, d, tau, group=None):
    """Cumulative/dynamic AUC at horizon tau: cases have an event in months <= tau, controls are
    event-free through tau. Units censored before tau are neither (none are, in this record, for
    the primary endpoint: every die is followed to the last month unless it has the event)."""
    score = np.asarray(score, float)
    group = np.zeros(len(score), int) if group is None else np.asarray(group)
    case = (d == 1) & (b <= tau) & (a <= tau)
    ctrl = (a <= tau) & ((b > tau) | ((b == tau) & (d == 0)))
    conc = pairs = 0.0
    for gv in np.unique(group):
        cs, ct = score[case & (group == gv)], score[ctrl & (group == gv)]
        if len(cs) and len(ct):
            conc += (cs[:, None] > ct[None, :]).sum() + 0.5 * (cs[:, None] == ct[None, :]).sum()
            pairs += len(cs) * len(ct)
    return conc / pairs if pairs else np.nan


def pair_parts(score, a, b, d, cluster, clusters, group=None, tau=None):
    """Concordance numerators/denominators per case x control-cluster, for a cluster bootstrap.

    tau=None: risk-set concordance; tau given: cumulative/dynamic AUC at tau. Returns
    (case cluster positions, num, den) so that for cluster multiplicities w the resampled
    statistic is sum_c w[c] (num[c] @ w) / sum_c w[c] (den[c] @ w); w = 1 reproduces
    risk_set_concordance / cd_auc exactly.
    """
    pos = {r: j for j, r in enumerate(clusters)}
    idx = np.array([pos[x] for x in cluster])
    cases = np.where((d == 1) & ((b <= tau) if tau is not None else True))[0]
    num = np.zeros((len(cases), len(clusters)))
    den = np.zeros((len(cases), len(clusters)))
    for k, i in enumerate(cases):
        t = b[i] if tau is None else tau
        ctrl = (a <= t) & ((b > t) | ((b == t) & (d == 0)))
        if group is not None:
            ctrl &= group == group[i]
        c = (score[i] > score).astype(float) + 0.5 * (score[i] == score)
        num[k] = np.bincount(idx[ctrl], c[ctrl], minlength=len(clusters))
        den[k] = np.bincount(idx[ctrl], minlength=len(clusters))
    return idx[cases], num, den


def weighted_pairs(parts, w):
    cp, num, den = parts
    wc = w[cp]
    return (wc * (num @ w)).sum() / max((wc * (den @ w)).sum(), 1e-12)


def loglik_rows(h, a, b, d):
    """Per-unit discrete-time survival log-likelihood from an (n, T) hazard grid."""
    T = h.shape[1]
    R = risk_grid(a, b, d, T)
    hc = np.clip(h, 1e-15, 1 - 1e-15)
    ll = (R * np.log1p(-hc)).sum(1)
    ev = np.where(d.astype(bool))[0]
    ll[ev] += np.log(hc[ev, b[ev]])
    return ll


def brier_curve(h, a, b, d):
    """Per-unit squared error (1{event-free at t} - S_i(t))^2 and validity mask, t = 0..T-1.

    S_i(t) = prod_{u=a_i..t} (1 - h_i(u)). A unit contributes from its entry month; a unit
    censored before t (competing-event censoring only) stops contributing after its exit.
    """
    n, T = h.shape
    t = np.arange(T)
    atrisk = t[None, :] >= a[:, None]
    logS = np.cumsum(np.where(atrisk, np.log1p(-np.clip(h, 0, 1 - 1e-15)), 0.0), axis=1)
    S = np.exp(logS)
    alive = ~((d[:, None] == 1) & (t[None, :] >= b[:, None]))
    valid = atrisk & ((t[None, :] <= b[:, None]) | (d[:, None] == 1))
    err = (alive.astype(float) - S) ** 2
    return err, valid
