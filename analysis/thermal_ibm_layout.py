"""Thermal coupling against IBM's documented POWER9 core layout.

Written for the external review of 2026-09-11 (THERMAL items TH-11/12/13). The paper read the
1x12 sensor-index line as the physical order of the slices. IBM's POWER9 Processor Registers
Specification (Vol. 1, v1.2, 29 Mar 2021, Fig. 2, p. 33) documents the layout instead: the core
chiplets EC00..EC23 (the sensor index) form six quads in two bands of three, each quad's four
cores stacked in one column, so index neighbours in different quads (cores 4q+3, 4q+4; slices
2q+1, 2q+2) are not neighbours on the die (thermal_lib.ibm_core_xy). This script asks what the
thermal census says about that layout, from the committed per-socket summaries:

  1. scores (Pearson over the 66 slice pairs, as Sec. IV-F) of IBM's layout at the diagram's own
     proportions, the 1x12 line and the exact best 1-D ordering; a relabelling null for IBM's
     layout; the same without the six same-quad slice pairs; sensitivity to the proportions,
     which the diagram does not guarantee;
  2. a 49-rack bootstrap of those scores and their differences;
  3. within-socket regressions of cross-quad slice coupling on IBM's topology (column distance,
     same band), on the logical order (rank distance among active slices) and on index distance;
  4. the harvested-between contrast ('natural experiment') and the rank slope within a slice pair,
     observed and predicted by physical models on IBM's layout: a stationary covariance field
     sampled at the active sensors, and a source model in which harvested cores are dark
     (thermal_lib.heat_source_corr); plus a source model whose power is correlated in logical order.

Inputs : analysis/thermal_socket_{meta,c5}_<YM>.{csv,npy} (thermal_socket_build.py)
Output : analysis/thermal_ibm_layout.json
"""
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

A = Path(__file__).resolve().parent
MONTHS = [m for m in os.environ.get('THERM_MONTHS', '22-08,22-02,21-05').split(',')
          if (A / f'thermal_socket_c5_{m}.npy').exists()]
B_MAIN = int(os.environ.get('THERM_BOOT', '1000'))
NP = tl.NP
PAIRS = tl.SLICE_PAIRS
SQ_CELL = np.array([a // 2 == b // 2 for a, b in PAIRS])        # the six same-quad slice pairs
TOPO = tl.ibm_topology()                                         # column, band, EX per slice
XY = tl.ibm_slice_xy()
CORE_XY = tl.ibm_core_xy()
LIN = tl.CANDIDATES['1 x 12 linear']
OUT = {'source': 'IBM POWER9 Processor Registers Specification Vol. 1 v1.2 (29 Mar 2021), Fig. 2, '
                 'p. 33; core-box centroids segmented from the embedded figure image',
       'slice_xy_column_units': np.round(XY, 4).tolist(), 'months': {}}


def r4(x):
    return None if x is None else round(float(x), 4)


def ci4(v):
    return [r4(x) for x in tl.ci(v)]


def load(ym):
    meta = pd.read_csv(A / f'thermal_socket_meta_{ym}.csv')
    act = np.array([list(map(int, s.split('-'))) for s in meta.active])
    c5 = np.load(A / f'thermal_socket_c5_{ym}.npy').astype('float64')
    M5 = np.stack([tl.unpack(c5[i], act[i]) for i in range(len(meta))])
    return meta, act, c5, M5


def scores(S):
    cv = tl.cells(S)
    cvx = cv.copy(); cvx[SQ_CELL] = np.nan
    opt, order = tl.best_order(S)
    return dict(ibm=tl.coords_score(cv, XY), linear=tl.coords_score(cv, LIN), opt=opt,
                ibm_noquad=tl.coords_score(cvx, XY), linear_noquad=tl.coords_score(cvx, LIN),
                order=tl.canonical(order).tolist())


def relabel_null(cv, B, rng):
    """Random assignment of the 12 slices to IBM's 12 slice positions."""
    a, b = np.array(PAIRS).T
    v = (cv - cv.mean()) / cv.std()
    out = np.empty(B)
    for k in range(B):
        p = rng.permutation(NP)
        d = np.linalg.norm(XY[p[a]] - XY[p[b]], axis=1)
        out[k] = np.mean((d - d.mean()) / d.std() * v)
    return out


def pair_rows(act, SSS, rack):
    rows = []
    for i in range(len(act)):
        sl = np.unique(act[i] // 2)
        rk = {s: r for r, s in enumerate(sl)}
        for a, b in combinations(sl, 2):
            rows.append((i, a, b, b - a, rk[b] - rk[a], int(a // 2 == b // 2),
                         int(rk[a] // 2 == rk[b] // 2), SSS[i, a, b]))
    P = pd.DataFrame(rows, columns=['s', 'a', 'b', 'didx', 'drank', 'samequad', 'samelgroup', 'y'])
    P['rack'] = rack[P.s.values]
    P['coldist'] = np.abs(TOPO[P.a.values, 0] - TOPO[P.b.values, 0]).astype(float)
    P['sameband'] = (TOPO[P.a.values, 1] == TOPO[P.b.values, 1]).astype(float)
    P['dphys'] = np.linalg.norm(XY[P.a.values] - XY[P.b.values], axis=1)
    P['cell'] = P.a * NP + P.b
    return P


MODELS = {'topology': ['coldist', 'sameband'], 'rank': ['drank'],
          'topology+rank': ['coldist', 'sameband', 'drank'],
          'topology+rank+index': ['coldist', 'sameband', 'drank', 'didx'],
          'diagram_distance+rank': ['dphys', 'drank']}


def regressions(P, B, seed=5):
    """Socket-fixed-effect OLS on cross-quad slice pairs, 95% CIs by resampling racks. Socket
    demeaning is done once: a rack bootstrap keeps or drops whole sockets, so it is unaffected."""
    cq = P[P.samequad == 0].reset_index(drop=True)
    cols = sorted({c for v in MODELS.values() for c in v}) + ['y']
    Z = cq[cols].values.astype('float64')
    _, inv = np.unique(cq.s.values, return_inverse=True)
    cnt = np.bincount(inv)
    for k in range(Z.shape[1]):
        Z[:, k] -= (np.bincount(inv, Z[:, k]) / cnt)[inv]
    col = {c: j for j, c in enumerate(cols)}
    labs = np.unique(cq.rack.values)
    idx = {r: np.flatnonzero(cq.rack.values == r) for r in labs}
    rr = np.random.default_rng(seed)
    draws = [np.concatenate([idx[r] for r in rr.choice(labs, len(labs))]) for _ in range(B)]
    out = {}
    y = Z[:, col['y']]
    for name, xs in MODELS.items():
        X = Z[:, [col[c] for c in xs]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - np.mean((y - X @ beta) ** 2) / np.mean(y ** 2)
        bb = np.array([np.linalg.lstsq(X[j], y[j], rcond=None)[0] for j in draws])
        out[name] = {c: dict(beta=r4(beta[k]), ci=ci4(bb[:, k])) for k, c in enumerate(xs)}
        out[name]['within_r2'] = r4(r2)
    out['pairs'] = int(len(cq))
    # the stringent specification: socket FE + additive slice effects (each slice's overall
    # coupling level) + categorical index distance + categorical rank distance. Column distance
    # and same band are the only layout terms left; a logical-order mechanism cannot know the band.
    Dsl = np.zeros((len(cq), NP))
    Dsl[np.arange(len(cq)), cq.a.values] += 1; Dsl[np.arange(len(cq)), cq.b.values] += 1
    X = np.column_stack([cq[['sameband', 'coldist']].values,
                         pd.get_dummies(cq.didx).values[:, 1:], pd.get_dummies(cq.drank).values[:, 1:],
                         Dsl[:, 1:]]).astype('float64')
    Zs = np.column_stack([X, cq.y.values.astype('float64')])
    for k in range(Zs.shape[1]):
        Zs[:, k] -= (np.bincount(inv, Zs[:, k]) / cnt)[inv]
    beta = np.linalg.lstsq(Zs[:, :-1], Zs[:, -1], rcond=None)[0]
    bb = np.array([np.linalg.lstsq(Zs[j, :-1], Zs[j, -1], rcond=None)[0][:2] for j in draws])
    out['stringent'] = dict(spec='socket FE + slice effects + categorical index and rank distance',
                            sameband=dict(beta=r4(beta[0]), ci=ci4(bb[:, 0])),
                            coldist=dict(beta=r4(beta[1]), ci=ci4(bb[:, 1])))
    # leave one column out (topology+rank model): is any layout term carried by one column?
    loo = {}
    for drop in range(3):
        keep = (TOPO[cq.a.values, 0] != drop) & (TOPO[cq.b.values, 0] != drop)
        q = cq[keep]
        b = tl.within_ols(q.y.values, q[['coldist', 'sameband', 'drank']].values, q.s.values)
        loo[f'without_column_{drop}'] = dict(coldist=r4(b[0]), sameband=r4(b[1]), drank=r4(b[2]))
    out['leave_one_column_out'] = loo
    return out


def diagnostics(P, SSS, SL):
    """Statistics that the phase-1 analysis read as physical-versus-logical tests."""
    ne = tl.natural_experiment(SSS, SL)
    slope_pair = tl.within_ols(P.y.values, P[['drank']].values, P.s.values, fe=(P.cell.values,))[0]
    cq = P[P.samequad == 0]
    b = tl.within_ols(cq.y.values, cq[['coldist', 'sameband', 'drank']].values, cq.s.values)
    ra = cq[cq.drank == 1].groupby('samelgroup').y.mean()
    return dict(natural_experiment=r4(ne), rank_slope_within_slice_pair=r4(slope_pair),
                crossquad_coldist=r4(b[0]), crossquad_sameband=r4(b[1]), crossquad_drank=r4(b[2]),
                rank_adjacent_crossquad_by_logical_group={str(int(k)): r4(v) for k, v in ra.items()},
                logical_group_gap=r4(ra.get(1, np.nan) - ra.get(0, np.nan)))


def model_vectors(kind, theta, act):
    iu = tl.IU16
    out = np.empty((len(act), 120))
    rk = np.arange(16.0)
    for i, a in enumerate(act):
        if kind == 'field':
            lam, noise = theta
            D = np.linalg.norm(CORE_XY[a][:, None] - CORE_XY[a][None], axis=-1)
            C = tl.projected_corr(np.exp(-D / lam) + noise * np.eye(16), np.arange(16))
        elif kind == 'dark_source':
            lam, noise = theta
            C = tl.heat_source_corr(a, CORE_XY, lam, noise)
        else:                                                    # dark_source + logical order
            lam, noise, wr, mu = theta
            Cp = np.eye(16) + wr * np.exp(-np.abs(rk[:, None] - rk[None]) / mu)
            C = tl.heat_source_corr(a, CORE_XY, lam, noise, power_cov=Cp)
        out[i] = C[iu]
    return out


def socket_slices(V, act):
    return np.stack([tl.slice_similarity(tl.unpack(V[i], act[i])) for i in range(len(act))])


# ============================================================================ per month
for ym in MONTHS:
    meta, act, c5, M5 = load(ym)
    ns = len(meta)
    rack = meta.rack.values
    res = {'sockets': ns}
    print(f'\n===== {ym}: {ns} sockets =====')
    pooled, _ = tl.pool(M5)
    S = tl.slice_similarity(pooled)
    sc = scores(S)
    nl = relabel_null(tl.cells(S), 20000, np.random.default_rng(12))
    res['pooled'] = {k: (r4(v) if not isinstance(v, list) else v) for k, v in sc.items()}
    res['pooled'].update(ibm_relabel_null_mean=r4(nl.mean()), ibm_relabel_null_sd=r4(nl.std()),
                         ibm_relabel_z=r4((sc['ibm'] - nl.mean()) / nl.std()),
                         ibm_relabel_p=tl.mc_pvalue(nl, sc['ibm']), ibm_relabel_B=len(nl))
    print(f'  pooled: IBM layout {sc["ibm"]:+.3f} (relabel null {nl.mean():+.3f}+/-{nl.std():.3f}, '
          f'p = {tl.mc_pvalue(nl, sc["ibm"]):.1e}), 1x12 line {sc["linear"]:+.3f}, best 1-D '
          f'{sc["opt"]:+.3f} {sc["order"]}')
    print(f'  without the six same-quad slice pairs: IBM {sc["ibm_noquad"]:+.3f}, '
          f'1x12 {sc["linear_noquad"]:+.3f}')
    cv = tl.cells(S)
    grid = []
    for fc in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for fg in (0.0, 0.5, 1.0, 2.0, 3.0):
            for fp in (0.5, 1.0, 2.0):
                grid.append(tl.coords_score(cv, tl.ibm_slice_xy(col_scale=fc, gap_scale=fg, pitch_scale=fp)))
    grid = np.array(grid)
    eq = XY.copy()                                   # the three columns equally spaced instead
    eq[:, 0] = TOPO[:, 0] * XY[:, 0].max() / 2
    res['proportions_sensitivity'] = dict(
        geometries=len(grid), col_scale=[0.5, 0.75, 1, 1.5, 2, 3], gap_scale=[0, 0.5, 1, 2, 3],
        pitch_scale=[0.5, 1, 2], ibm_min=r4(grid.min()), ibm_max=r4(grid.max()),
        n_better_than_best_1d=int((grid < sc['opt']).sum()),
        equal_column_spacing=r4(tl.coords_score(cv, eq)))
    print(f'  IBM score over {len(grid)} rescalings of the diagram: {grid.min():+.3f} .. {grid.max():+.3f}; '
          f'better than the best 1-D ordering in {int((grid < sc["opt"]).sum())}')

    # ------------------------------------------------ rack bootstrap
    B = B_MAIN if ym == '22-08' else max(200, B_MAIN // 3)
    rb = np.random.default_rng(31)
    rows = []
    for _ in range(B):
        ix = tl.cluster_resample(rack, rb)
        Sb = tl.slice_similarity(tl.pool(M5[ix])[0])
        if np.isnan(tl.cells(Sb)).any():
            continue
        s = scores(Sb)
        rows.append([s['ibm'], s['linear'], s['opt'], s['ibm_noquad'], s['linear_noquad']])
    rows = np.array(rows)
    res['rack_bootstrap'] = dict(
        replicates=int(len(rows)), ibm_ci=ci4(rows[:, 0]), linear_ci=ci4(rows[:, 1]),
        best_1d_ci=ci4(rows[:, 2]), ibm_minus_linear_ci=ci4(rows[:, 0] - rows[:, 1]),
        ibm_minus_best_1d_ci=ci4(rows[:, 0] - rows[:, 2]),
        frac_ibm_better_than_best_1d=r4(np.mean(rows[:, 0] < rows[:, 2])),
        noquad_ibm_minus_linear_ci=ci4(rows[:, 3] - rows[:, 4]),
        noquad_frac_ibm_better=r4(np.mean(rows[:, 3] < rows[:, 4])))
    rbx = res['rack_bootstrap']
    print(f'  rack bootstrap ({len(rows)}): IBM {rbx["ibm_ci"]}, 1x12 {rbx["linear_ci"]}, best 1-D '
          f'{rbx["best_1d_ci"]}; IBM - best 1-D {rbx["ibm_minus_best_1d_ci"]} (IBM better in '
          f'{100 * rbx["frac_ibm_better_than_best_1d"]:.1f}%); without same-quad pairs IBM - 1x12 '
          f'{rbx["noquad_ibm_minus_linear_ci"]}')

    # ------------------------------------------------ within-socket regressions (cross-quad pairs)
    SSS = np.stack([tl.slice_similarity(M5[i]) for i in range(ns)])
    SL = np.zeros((ns, NP), dtype=bool)
    for i in range(ns):
        SL[i, np.unique(act[i] // 2)] = True
    P = pair_rows(act, SSS, rack)
    reg = regressions(P, 400 if ym == '22-08' else 200)
    res['crossquad_regressions'] = reg
    for name in MODELS:
        print(f'  cross-quad, socket FE, [{name}]: ' + '  '.join(
            f'{c} {v["beta"]:+.4f} {v["ci"]}' for c, v in reg[name].items() if c != 'within_r2')
            + f'  R2 {reg[name]["within_r2"]}')
    s_ = reg['stringent']
    print(f'  cross-quad, {s_["spec"]}: same band {s_["sameband"]["beta"]:+.4f} {s_["sameband"]["ci"]}, '
          f'column distance {s_["coldist"]["beta"]:+.4f} {s_["coldist"]["ci"]}')
    print('  leave one column out (topology+rank): ' + '; '.join(
        f'{k}: coldist {v["coldist"]:+.4f} sameband {v["sameband"]:+.4f}' for k, v in reg['leave_one_column_out'].items()))
    res['observed_diagnostics'] = diagnostics(P, SSS, SL)
    print(f'  observed: {res["observed_diagnostics"]}')

    # ------------------------------------------------ physical models (paper month only)
    if ym == '22-08':
        sub = np.random.default_rng(0).choice(ns, 400, replace=False)
        fits = {}
        for kind, p0 in (('field', [np.log(0.3), np.log(1.0)]),
                         ('dark_source', [np.log(0.3), np.log(1.0)]),
                         ('dark_source+logical_order', [np.log(0.3), np.log(1.0), np.log(0.5), np.log(2.0)])):
            r = minimize(lambda p: ((c5[sub] - model_vectors(kind, np.exp(p), act[sub])) ** 2).sum(), p0,
                         method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
            th = np.exp(r.x)
            V = model_vectors(kind, th, act)
            Ss = socket_slices(V, act)
            Pm = P.copy(); Pm['y'] = Ss[Pm.s.values, Pm.a.values, Pm.b.values]
            fits[kind] = dict(theta=[r4(t) for t in th],
                              r2=r4(1 - ((c5 - V) ** 2).sum() / ((c5 - c5.mean()) ** 2).sum()),
                              **diagnostics(Pm, Ss, SL))
            print(f'  model [{kind}] theta {np.round(th, 3).tolist()}: {fits[kind]}')
        sweep = []
        for lam in (0.1, 0.2, 0.4):
            for noise in (0.1, 1.0):
                V = model_vectors('dark_source', (lam, noise), act)
                Ss = socket_slices(V, act)
                Pm = P.copy(); Pm['y'] = Ss[Pm.s.values, Pm.a.values, Pm.b.values]
                d = diagnostics(Pm, Ss, SL)
                sweep.append(dict(lam=lam, noise=noise, natural_experiment=d['natural_experiment'],
                                  rank_slope_within_slice_pair=d['rank_slope_within_slice_pair'],
                                  logical_group_gap=d['logical_group_gap']))
        fits['dark_source_sweep'] = sweep
        ne = [s['natural_experiment'] for s in sweep]
        rs = [s['rank_slope_within_slice_pair'] for s in sweep]
        print(f'  dark-source sweep: natural experiment {min(ne):+.3f} .. {max(ne):+.3f}; rank slope '
              f'within slice pair {min(rs):+.3f} .. {max(rs):+.3f}')
        res['physical_models'] = fits
    OUT['months'][ym] = res

json.dump(OUT, open(A / 'thermal_ibm_layout.json', 'w'), indent=1)
print('\nwrote analysis/thermal_ibm_layout.json')
