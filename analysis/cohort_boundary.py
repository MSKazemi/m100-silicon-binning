"""The rack-22 boundary: what its uncertainty is, how many populations the maps support, and
whether held-out classification survives when the boundary is re-found inside every fold.

Responds to an external review of Sec. IV-G. Everything runs off the committed modal maps
(analysis/harvest_maps_full.npy, the same 1,960 rack-indexed sockets changepoint.py derives,
verified identical), so no daily table is read.

  1. BOUNDARY UNCERTAINTY. changepoint.py resamples sockets within each rack with every rack
     held in place and called the result a "95% CI". That is a conditional stability check.
     Here it is reported as such, beside three analyses that respect the rack clustering:
       a. sockets resampled within rack                          (the original check)
       b. NODES resampled within rack, both sockets together      (keeps the node pairing)
       c. whole RACKS resampled with replacement within each side of the fitted boundary and
          put back in order (a cluster bootstrap conditional on the two-segment model)
       d. a parametric rack-level beta-binomial bootstrap: rack counts simulated with the
          fitted between-rack overdispersion, boundary re-found by the same scan
       e. the rack-level beta-binomial profile likelihood of every admissible boundary.
  2. HOW MANY POPULATIONS, UNDER A VALID MAP LIKELIHOOD. The paper chose k with a product of
     twelve independent Bernoullis, which puts mass on maps that do not have four harvested
     slices. k = 1..5 is re-selected with three models defined on the 495 valid maps only:
     conditional Bernoulli (exactly four), the pairwise model of Sec. IV-J, and a Dirichlet-
     smoothed multinomial over the 495 maps; rack-held-out likelihood decides.
  3. NESTED HELD-OUT CLASSIFICATION. For every held-out rack the boundary is re-estimated
     from the other 48 racks only (max-|t| scan, and the k=2 segmentation), cohort models
     are fitted on those 48, and the held-out rack is classified from its maps.
  4. STEP OR GRADIENT. Slice-0 status against rack index: a step, a logistic trend, and a
     step with within-segment trends, compared by rack-held-out likelihood.
  5. WITHOUT THE RACK ORDER. A two-component mixture over maps, fitted with no rack
     information at all, and the rack profile of its components.

Writes analysis/cohort_boundary.json and analysis/changepoint_boot_hist.npy.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import betabinom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort_models as cm  # noqa: E402

root = Path(__file__).resolve().parent.parent
OUT = root / 'analysis'
MIN_SOCK, NRACK, KMAX = 100, 49, 5
B_RESAMPLE = int(os.environ.get('COHORT_B', 2000))
ALPHAS = (20.0, 100.0, 500.0, 2000.0)
EPS = 1e-9

M = np.load(OUT / 'harvest_maps_full.npy').astype(np.int8)
K = np.load(OUT / 'harvest_keys_full.npy')
keep = K[:, 0] // 20 < NRACK            # nodes 980..983: no documented cabinet (Sec. IV-A)
X, node = M[keep], K[keep, 0]
rack = node // 20
N = len(X)
s0 = X[:, 0].astype(float)
pid = cm.pattern_ids(X)
sizes = np.bincount(rack, minlength=NRACK)
racks_idx = [np.flatnonzero(rack == r) for r in range(NRACK)]
F_CB, F_PW = cm.features('cb'), cm.features('pairwise')
print(f'sockets {N:,} in {NRACK} racks (2 sockets on node 980, outside the documented '
      f'cabinets, excluded)')


# ============================================================ 1. boundary uncertainty
def cands_for(rk, nr):
    return [c for c in range(1, nr) if (rk < c).sum() >= MIN_SOCK and (rk >= c).sum() >= MIN_SOCK]


CANDS = cands_for(rack, NRACK)


def welch_profile_counts(y, n, cands):
    """Welch t for P(slice 0) at each boundary, from per-rack counts (exact, 0/1 data)."""
    cy, cn = np.cumsum(y), np.cumsum(n)
    out = []
    for c in cands:
        ya, na = cy[c - 1], cn[c - 1]
        yb, nb = cy[-1] - ya, cn[-1] - na
        pa, pb = ya / na, yb / nb
        va = pa * (1 - pa) * na / (na - 1) / na
        vb = pb * (1 - pb) * nb / (nb - 1) / nb
        out.append((pa - pb) / np.sqrt(va + vb + EPS))
    return np.array(out)


y_r = np.array([s0[ix].sum() for ix in racks_idx])


def argmax_cut(y, n):
    return CANDS[int(np.argmax(np.abs(welch_profile_counts(y, n, CANDS))))]


def section1():
    """Boundary uncertainty; returns (json-ready dict, observed cut)."""
    prof = welch_profile_counts(y_r, sizes, CANDS)
    obs_cut = CANDS[int(np.argmax(np.abs(prof)))]
    print(f'\n=== 1. boundary uncertainty ===\n  argmax |t| = {np.abs(prof).max():.1f} at rack '
          f'{obs_cut}')
    rng = np.random.default_rng(22)
    hist = {}
    # a. sockets within rack
    bc = []
    for _ in range(B_RESAMPLE):
        y = np.array([s0[ix[rng.integers(0, len(ix), len(ix))]].sum() for ix in racks_idx])
        bc.append(argmax_cut(y, sizes))
    hist['within_rack_sockets'] = np.array(bc)
    # b. nodes within rack (both sockets of a node move together)
    node_rows = {}
    for i, nd in enumerate(node):
        node_rows.setdefault(nd, []).append(i)
    rack_nodes = [[np.array(node_rows[nd]) for nd in sorted(set(node[ix]))] for ix in racks_idx]
    bc = []
    for _ in range(B_RESAMPLE):
        y, n = np.zeros(NRACK), np.zeros(NRACK)
        for r, nl in enumerate(rack_nodes):
            pick = rng.integers(0, len(nl), len(nl))
            rows = np.concatenate([nl[j] for j in pick])
            y[r], n[r] = s0[rows].sum(), len(rows)
        bc.append(argmax_cut(y, n))
    hist['within_rack_nodes'] = np.array(bc)
    # c. whole racks resampled within each side, positions kept
    sideA, sideB = np.arange(obs_cut), np.arange(obs_cut, NRACK)
    bc = []
    for _ in range(B_RESAMPLE):
        pick = np.concatenate([rng.choice(sideA, len(sideA)), rng.choice(sideB, len(sideB))])
        bc.append(argmax_cut(y_r[pick], sizes[pick]))
    hist['rack_block_by_side'] = np.array(bc)
    # d. parametric beta-binomial at the rack level
    ll22, muA, muB, rho = cm.betabinom_two_segment(y_r, sizes, obs_cut)
    mu_r = np.where(np.arange(NRACK) < obs_cut, muA, muB)
    sfac = 1 / rho - 1
    bc = []
    for _ in range(B_RESAMPLE):
        y = betabinom.rvs(sizes, mu_r * sfac, (1 - mu_r) * sfac, random_state=rng)
        bc.append(argmax_cut(y, sizes))
    hist['parametric_betabinomial'] = np.array(bc)
    # e. profile likelihood over every admissible boundary
    pl = np.array([cm.betabinom_two_segment(y_r, sizes, c)[0] for c in CANDS])
    order = np.argsort(-pl)
    d_ll = float(pl[order[0]] - pl[order[1]])
    post = np.exp(pl - pl.max())
    post /= post.sum()
    res1 = {'observed_cut': int(obs_cut), 'max_abs_t': float(np.abs(prof).max()),
            'betabinomial_fit': {'muA': muA, 'muB': muB, 'rho': rho},
            'profile_ll_best': int(CANDS[order[0]]), 'profile_ll_second': int(CANDS[order[1]]),
            'profile_delta_ll_best_vs_second': d_ll,
            'profile_posterior_mass_at_best_flat_prior': float(post.max()),
            'resampling': {}}
    print(f'  rack-level beta-binomial fit at rack {obs_cut}: mu_A {muA:.3f}  mu_B {muB:.3f}  '
          f'intra-rack correlation rho {rho:.3f}')
    for name, h in hist.items():
        vals, cnt = np.unique(h, return_counts=True)
        lo, hi = np.percentile(h, [2.5, 97.5])
        res1['resampling'][name] = {'B': int(len(h)), 'share_at_observed': float((h == obs_cut).mean()),
                                    'pct_2p5': float(lo), 'pct_97p5': float(hi),
                                    'histogram': {int(v): int(c) for v, c in zip(vals, cnt)}}
        print(f'  {name:<26}: rack {obs_cut} in {100 * (h == obs_cut).mean():5.1f}% of {len(h)}   '
              f'2.5-97.5 pct [{lo:.0f}, {hi:.0f}]   {dict(zip(vals.tolist(), cnt.tolist()))}')
    print(f'  profile log-likelihood: best rack {CANDS[order[0]]}, next rack {CANDS[order[1]]}, '
          f'difference {d_ll:.1f} (likelihood ratio {np.exp(min(d_ll, 700)):.2e}); flat-prior '
          f'posterior mass at the best boundary {post.max():.6f}')
    np.save(OUT / 'changepoint_boot_hist.npy',
            np.vstack([hist[k] for k in ('within_rack_sockets', 'within_rack_nodes',
                                         'rack_block_by_side', 'parametric_betabinomial')]))
    return res1, obs_cut


# ============================================================ 2 + 3. segmentation, per fold
C12 = np.array([X[ix].sum(0) for ix in racks_idx], dtype=float)          # slice counts per rack
ADJR = np.array([cm.ADJ[pid[ix]].sum() for ix in racks_idx])
PATR = np.array([np.bincount(pid[ix], minlength=cm.NPAT) for ix in racks_idx], dtype=float)
MODELS = ['bernoulli', 'cb', 'pairwise'] + [f'dirichlet_{a:g}' for a in ALPHAS]


def segment_tables(rlist):
    """In-sample LL and fitted parameters of every contiguous segment of the rack list."""
    R = len(rlist)
    c12, adj, pat = np.cumsum(C12[rlist], 0), np.cumsum(ADJR[rlist]), np.cumsum(PATR[rlist], 0)
    nn = np.cumsum(sizes[rlist])
    z = lambda a: np.vstack([np.zeros((1,) + a.shape[1:]), a]) if a.ndim > 1 else np.r_[0, a]
    c12, adj, pat, nn = z(c12), z(adj), z(pat), z(nn)
    cost = {m: np.full((R + 1, R + 1), -np.inf) for m in MODELS}
    par = {}
    th_cb = th_pw = None
    for i in range(R):
        for j in range(i + 1, R + 1):
            n = nn[j] - nn[i]
            c = c12[j] - c12[i]
            p = np.clip(c / n, EPS, 1 - EPS)
            cost['bernoulli'][i, j] = float((c * np.log(p) + (n - c) * np.log(1 - p)).sum())
            th_cb, ll_cb = cm.fit_expfam(F_CB, c, n, theta0=th_cb)
            T = np.r_[c, adj[j] - adj[i]]
            th_pw, ll_pw = cm.fit_expfam(F_PW, T, n, theta0=th_pw)
            cost['cb'][i, j], cost['pairwise'][i, j] = ll_cb, ll_pw
            q = np.exp(cm.logprob_expfam(F_CB, th_cb))
            cnts = pat[j] - pat[i]
            for a in ALPHAS:
                pr = cm.dirichlet_predictive(cnts, q, a)
                cost[f'dirichlet_{a:g}'][i, j] = float((cnts * np.log(pr)).sum())
            par[(i, j)] = (p, th_cb, th_pw, cnts, q)
    return cost, par


def score_heldout(model, prm, rows):
    p, th_cb, th_pw, cnts, q = prm
    if model == 'bernoulli':
        return float(cm.bernoulli_logprob(X[rows], p).sum())
    if model == 'cb':
        return float(cm.logprob_expfam(F_CB, th_cb)[pid[rows]].sum())
    if model == 'pairwise':
        return float(cm.logprob_expfam(F_PW, th_pw)[pid[rows]].sum())
    a = float(model.split('_')[1])
    return float(np.log(cm.dirichlet_predictive(cnts, q, a))[pid[rows]].sum())


def welch_cut_reduced(rlist):
    y, n = y_r[rlist], sizes[rlist]
    rk = np.repeat(np.arange(len(rlist)), n)
    cd = cands_for(rk, len(rlist))
    return cd[int(np.argmax(np.abs(welch_profile_counts(y, n, cd))))]


def fold(r):
    """Everything computed without rack r (r = -1: the full data, no rack held out)."""
    rlist = np.array([x for x in range(NRACK) if x != r])
    cost, par = segment_tables(rlist)
    out = {'r': r, 'seg': {}}
    for m in MODELS:
        dp = cm.segment_dp(cost[m], sizes[rlist], KMAX, MIN_SOCK)
        out['seg'][m] = {k: (ll, [int(c) for c in cuts]) for k, (ll, cuts) in dp.items()}
        if r < 0:
            continue
        pos = r - 1 if r >= 1 else 0                  # held-out rack joins its left neighbour
        ho = {}
        for k, (_, cuts) in dp.items():
            b = [0] + list(cuts) + [len(rlist)]
            s = next(t for t in range(len(b) - 1) if b[t] <= pos < b[t + 1])
            ho[k] = score_heldout(m, par[(b[s], b[s + 1])], racks_idx[r])
        out.setdefault('heldout', {})[m] = ho
    if r >= 0:
        # nested classification: boundary from the 48 training racks only
        rows_ho = racks_idx[r]
        for bname in ('welch', 'cb_k2'):
            if bname == 'welch':
                cr = welch_cut_reduced(rlist)
            else:
                cr = out['seg']['cb'][2][1][0]
            cut_orig = cm.reduced_to_original(cr, r)
            tr = np.concatenate([racks_idx[x] for x in rlist])
            labA = rack[tr] < cut_orig
            label_pos = int(r >= cut_orig)                   # held-out rack's side, by position
            gap = cr == r                                    # boundary fell in the vacated slot
            preds = {}
            for cname in ('bernoulli', 'cb', 'pairwise'):
                Xa, Xb = X[tr[labA]], X[tr[~labA]]
                if cname == 'bernoulli':
                    la = cm.bernoulli_logprob(X[rows_ho], Xa.mean(0))
                    lb = cm.bernoulli_logprob(X[rows_ho], Xb.mean(0))
                else:
                    F = F_CB if cname == 'cb' else F_PW
                    ta, _ = cm.fit_expfam(F, F[pid[tr[labA]]].sum(0), len(Xa))
                    tb, _ = cm.fit_expfam(F, F[pid[tr[~labA]]].sum(0), len(Xb))
                    la = cm.logprob_expfam(F, ta)[pid[rows_ho]]
                    lb = cm.logprob_expfam(F, tb)[pid[rows_ho]]
                preds[cname] = (lb > la).astype(int).tolist()
            out.setdefault('nested', {})[bname] = {'cut': int(cut_orig), 'label_pos': label_pos,
                                                   'gap': bool(gap), 'pred': preds}
    return out


if __name__ == '__main__':
    res1, obs_cut = section1()
    with Pool(min(24, os.cpu_count() or 1)) as pool:
        folds = pool.map(fold, list(range(-1, NRACK)))
    full, lo = folds[0], folds[1:]

    print('\n=== 2. how many populations? k = 1..5, rack-held-out log-likelihood per socket ===')
    print('   model              ' + ''.join(f'   k={k}   ' for k in range(1, KMAX + 1))
          + '  best   k=2 cut')
    res2 = {}
    for m in MODELS:
        ho = np.array([[f['heldout'][m][k] for k in range(1, KMAX + 1)] for f in lo]).sum(0) / N
        kbest = int(np.argmax(ho)) + 1
        cut2 = full['seg'][m][2][1]
        # rack-level paired comparison k=2 vs every other k (racks as the unit)
        per_rack = np.array([[f['heldout'][m][k] for k in range(1, KMAX + 1)] for f in lo])
        d = {f'k2_minus_k{k}': {'total': float((per_rack[:, 1] - per_rack[:, k - 1]).sum()),
                                'racks_favouring_k2': int((per_rack[:, 1] > per_rack[:, k - 1]).sum())}
             for k in range(1, KMAX + 1) if k != 2}
        cuts_all = {k: full['seg'][m][k][1] for k in range(1, KMAX + 1)}
        res2[m] = {'heldout_ll_per_socket': {k: float(ho[k - 1]) for k in range(1, KMAX + 1)},
                   'best_k': kbest, 'insample_cuts': cuts_all,
                   'insample_ll': {k: full['seg'][m][k][0] for k in range(1, KMAX + 1)},
                   'k2_vs_other': d}
        print(f'   {m:<18}' + ''.join(f'{v:9.4f} ' for v in ho) + f'  k={kbest}   {cut2}')
    print('   (Bernoulli scores live on 2^12 outcomes, the others on the 495 valid maps, so '
          'compare\n    across k within a row, not across rows.)')
    for m in MODELS:
        dd = res2[m]['k2_vs_other']
        print(f'   {m:<18} k=2 minus k=1,3,4,5 (nats, summed over held-out racks): '
              + ', '.join(f'{dd[f"k2_minus_k{k}"]["total"]:+.1f}' for k in (1, 3, 4, 5)))

    print('\n=== 3. nested held-out classification (boundary re-found without the held-out rack) ===')
    res3 = {}
    truth_full = (np.arange(NRACK) >= obs_cut).astype(int)
    for bname in ('welch', 'cb_k2'):
        cuts = [f['nested'][bname]['cut'] for f in lo]
        gaps = [f['r'] for f in lo if f['nested'][bname]['gap']]
        res3[bname] = {'boundary_per_fold': {int(f['r']): int(c) for f, c in zip(lo, cuts)},
                       'folds_boundary_in_gap': gaps}
        v_, c_ = np.unique(cuts, return_counts=True)
        print(f'  boundary rule "{bname}": training-fold boundary values '
              f'{dict(zip(v_.tolist(), c_.tolist()))}; lands in the held-out gap for racks {gaps} '
              f'(position alone cannot place a held-out rack that sits in the gap)')
        for cname in ('bernoulli', 'cb', 'pairwise'):
            s_pos = s_full = r_pos = r_full = 0
            wrong = []
            for f in lo:
                r = f['r']
                pr = np.array(f['nested'][bname]['pred'][cname])
                lp = f['nested'][bname]['label_pos']
                s_pos += int((pr == lp).sum())
                s_full += int((pr == truth_full[r]).sum())
                vote = int(pr.mean() > 0.5)
                r_pos += int(vote == lp)
                r_full += int(vote == truth_full[r])
                if vote != truth_full[r]:
                    wrong.append((r, round(float(pr.mean()), 3)))
            res3[bname][cname] = {'socket_acc_vs_fold_label': s_pos / N,
                                  'socket_acc_vs_full_label': s_full / N,
                                  'rack_acc_vs_fold_label': r_pos / NRACK,
                                  'rack_acc_vs_full_label': r_full / NRACK,
                                  'racks_correct_vs_full_label': r_full,
                                  'misclassified_racks_share_predicted_B': wrong}
            print(f'    classifier {cname:<9}: sockets {100 * s_pos / N:5.1f}% (fold label) '
                  f'{100 * s_full / N:5.1f}% (full-data label);  racks {r_pos}/{NRACK} '
                  f'(fold) {r_full}/{NRACK} (full);  misclassified {wrong}')
    base = max(truth_full[rack].mean(), 1 - truth_full[rack].mean())
    res3['socket_majority_baseline'] = float(base)
    print(f'  majority-class baseline at socket level: {100 * base:.1f}%')

    # ======================================================== 4. step or gradient (slice 0)
    from scipy.optimize import minimize

    def fit_logit(Z, y):
        def nll(b):
            eta = Z @ b
            return float(np.sum(np.logaddexp(0, eta) - y * eta)), Z.T @ (1 / (1 + np.exp(-eta)) - y)
        return minimize(nll, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B').x

    def design(rk, cut, kind):
        z = (rk - 24.0) / 14.0
        one = np.ones_like(z)
        if kind == 'constant':
            return one[:, None]
        if kind == 'trend':
            return np.column_stack([one, z])
        stp = (rk >= cut).astype(float)
        if kind == 'step':
            return np.column_stack([one, stp])
        return np.column_stack([one, stp, z * (1 - stp), z * stp])      # step + trends

    res4 = {}
    for kind in ('constant', 'trend', 'step', 'step_trends'):
        tot = 0.0
        for r in range(NRACK):
            tr, te = rack != r, rack == r
            rl = np.array([x for x in range(NRACK) if x != r])
            cut = cm.reduced_to_original(welch_cut_reduced(rl), r)
            b = fit_logit(design(rack[tr], cut, kind), s0[tr])
            eta = design(rack[te], cut, kind) @ b
            tot += float(np.sum(s0[te] * eta - np.logaddexp(0, eta)))
        res4[kind] = tot / N
    print('\n=== 4. step or gradient? slice-0 rack-held-out log-likelihood per socket ===')
    for kind, v in res4.items():
        print(f'  {kind:<12}: {v:.4f}')

    # ======================================================== 5. mixture, no rack information
    def mixture_cb(Xm, k=2, restarts=20, seed=5):
        rr = np.random.default_rng(seed)
        pids = cm.pattern_ids(Xm)
        best = None
        for _ in range(restarts):
            w = rr.dirichlet(np.ones(k), size=len(Xm))
            ll_old = -np.inf
            for _it in range(500):
                ths = [cm.fit_expfam(F_CB, F_CB[pids].T @ w[:, j], w[:, j].sum())[0]
                       for j in range(k)]
                pis = w.mean(0)
                lp = np.column_stack([np.log(pis[j]) + cm.logprob_expfam(F_CB, ths[j])[pids]
                                      for j in range(k)])
                lse = np.logaddexp.reduce(lp, axis=1)
                w = np.exp(lp - lse[:, None])
                ll = lse.sum()
                if ll - ll_old < 1e-7:
                    break
                ll_old = ll
            if best is None or ll > best[0]:
                best = (ll, w, ths, pis)
        return best

    llm, W, ths, pis = mixture_cb(X)
    s0_comp = [float(np.exp(cm.logprob_expfam(F_CB, th)) @ cm.A_SLICE[:, 0]) for th in ths]
    jA = int(np.argmax(s0_comp))                    # the slice-0-heavy component
    wA = W[:, jA]
    rack_share = np.array([wA[ix].mean() for ix in racks_idx])
    # scan the rack profile of the unsupervised membership for its own breakpoint, and
    # calibrate its max |t| against whole-rack relabelling, as for slice 0 in changepoint.py
    def wprof(w, rk):
        sw, sq, nn = (np.bincount(rk, weights=w, minlength=NRACK),
                      np.bincount(rk, weights=w * w, minlength=NRACK), np.bincount(rk, minlength=NRACK))
        cs, cq, cn = np.cumsum(sw), np.cumsum(sq), np.cumsum(nn)
        t = []
        for c in CANDS:
            na, nb = cn[c - 1], cn[-1] - cn[c - 1]
            ma, mb = cs[c - 1] / na, (cs[-1] - cs[c - 1]) / nb
            va = (cq[c - 1] - na * ma * ma) / (na - 1)
            vb = (cq[-1] - cq[c - 1] - nb * mb * mb) / (nb - 1)
            t.append((ma - mb) / np.sqrt(va / na + vb / nb))
        return np.array(t)

    pw_obs = wprof(wA, rack)
    cut_w = CANDS[int(np.argmax(np.abs(pw_obs)))]
    rr = np.random.default_rng(55)
    exceed = sum(np.abs(wprof(wA, rr.permutation(NRACK)[rack])).max() >= np.abs(pw_obs).max()
                 for _ in range(B_RESAMPLE))
    p_w = (exceed + 1) / (B_RESAMPLE + 1)
    res5 = {'mixture_weights': pis.tolist(),
            'component_slice0_share': s0_comp,
            'heavy_component_mean_membership_racks_lt_cut': float(wA[rack < obs_cut].mean()),
            'heavy_component_mean_membership_racks_ge_cut': float(wA[rack >= obs_cut].mean()),
            'membership_profile_breakpoint': int(cut_w),
            'membership_profile_max_abs_t': float(np.abs(pw_obs).max()),
            'membership_profile_perm_p': float(p_w),
            'rack_share_heavy_component': rack_share.round(3).tolist()}
    print('\n=== 5. two-component mixture over maps, fitted with NO rack information ===')
    print(f'  component weights {np.round(pis, 3)}; P(slice 0) inside each component '
          f'{np.round(s0_comp, 3)}')
    print(f'  mean membership of the slice-0-heavy component: racks 0..{obs_cut - 1} '
          f'{W[rack < obs_cut, jA].mean():.3f}, racks {obs_cut}..48 {W[rack >= obs_cut, jA].mean():.3f}')
    print(f'  its rack profile breaks at rack {cut_w} (max |t| {np.abs(pw_obs).max():.1f}, '
          f'whole-rack permutation p = {p_w:.2e}, (b+1)/(B+1), B = {B_RESAMPLE})')
    print('  The components are not the lots: Lot A is a mix of both, Lot B mostly one. What the')
    print('  mixture shows is that maps alone, with the rack order withheld, carry a two-population')
    print('  structure whose rack profile changes at the same boundary.')

    json.dump({'section1_boundary_uncertainty': res1, 'section2_k_selection': res2,
               'section3_nested_classification': res3, 'section4_step_vs_trend': res4,
               'section5_unsupervised_mixture': res5, 'n_sockets': int(N), 'n_racks': NRACK,
               'B_resample': B_RESAMPLE, 'dirichlet_alphas': list(ALPHAS)},
              open(OUT / 'cohort_boundary.json', 'w'), indent=1, default=lambda o: o.tolist()
              if hasattr(o, 'tolist') else str(o))
    print('\nwrote analysis/cohort_boundary.json, analysis/changepoint_boot_hist.npy')
