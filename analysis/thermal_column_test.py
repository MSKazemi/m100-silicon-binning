"""Does thermal coupling beyond the quad follow IBM's layout or the core numbering?

External review 2026-09-11, THERMAL items TH-11/TH-12. IBM's POWER9 Processor Registers
Specification (Vol. 1, v1.2, 29 Mar 2021, Fig. 2, p. 33) stacks the four cores of each quad in one
column, the quads of a column in two bands around the central nest (thermal_lib.ibm_core_xy). In
every column c, slice 4c+1 (top band, next to the nest) is therefore nearer to slice 4c+3 (bottom
band, next to the nest) than to slice 4c+2 (bottom band, at the far die edge), while the index and
the operating system's numbering of surviving cores put 4c+2 next to 4c+1. On sockets where all
three slices are active this is a position-matched test with no harvested slice in between:

    d = S[4c+1, 4c+3] - S[4c+1, 4c+2]      (> 0: coupling follows the diagram; < 0: the order)

Also the slope of coupling of the same two slices on the number of active slices between them
(socket and slice-pair fixed effects). Both are reported per month with 95 % intervals from
resampling the 49 racks; d is also compared with what purely physical source models on IBM's
layout predict for the same sockets (thermal_lib.heat_source_corr, harvested cores dark;
parameters: the fit and the sweep of thermal_ibm_layout.py).

Inputs : analysis/thermal_socket_{meta,c5}_<YM>.{csv,npy}; analysis/thermal_ibm_layout.json
Output : analysis/thermal_column_test.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_lib as tl  # noqa: E402

A = Path(__file__).resolve().parent
MONTHS = [m for m in os.environ.get('THERM_MONTHS', '22-08,22-02,21-05').split(',')
          if (A / f'thermal_socket_c5_{m}.npy').exists()]
B = int(os.environ.get('THERM_BOOT', '2000'))
CORE_XY = tl.ibm_core_xy()


def r4(x):
    return round(float(x), 4)


def load(ym):
    meta = pd.read_csv(A / f'thermal_socket_meta_{ym}.csv')
    act = np.array([list(map(int, s.split('-'))) for s in meta.active])
    c5 = np.load(A / f'thermal_socket_c5_{ym}.npy').astype('float64')
    SSS = np.stack([tl.slice_similarity(tl.unpack(c5[i], act[i])) for i in range(len(meta))])
    SL = np.zeros((len(meta), tl.NP), dtype=bool)
    for i in range(len(meta)):
        SL[i, np.unique(act[i] // 2)] = True
    return meta, act, SSS, SL


def summarise(d, per_col, rack, rng):
    ok = np.isfinite(d)
    ix = np.flatnonzero(ok)
    rk = rack[ix]
    boots = [np.mean(d[ix][tl.cluster_resample(rk, rng)]) for _ in range(B)]
    return dict(mean=r4(d[ok].mean()), rack_ci=[r4(x) for x in tl.ci(boots)], sockets=int(ok.sum()),
                frac_positive=r4(np.mean(d[ok] > 0)),
                by_column=[r4(np.nanmean(per_col[:, c])) if np.isfinite(per_col[:, c]).any() else None
                           for c in range(3)])


def rank_slope(act, SSS, rack, rng, B_rank):
    """Coupling of the SAME two slices against the number of active slices between them: socket
    and slice-pair fixed effects, slope on the rank distance among active slices; 95 % interval
    from resampling racks. thermal_ibm_layout.py reports what physical models give for it."""
    rows = []
    for i, a in enumerate(act):
        sl = np.unique(a // 2)
        for x in range(len(sl)):
            for y in range(x + 1, len(sl)):
                rows.append((i, sl[x] * tl.NP + sl[y], y - x, SSS[i, sl[x], sl[y]]))
    P = np.array(rows)
    s, cell, dr, yv = P[:, 0].astype(int), P[:, 1].astype(int), P[:, 2:3], P[:, 3]
    b = tl.within_ols(yv, dr, s, fe=(cell,))[0]
    labs = np.unique(rack)
    members = {r: np.flatnonzero(rack[s] == r) for r in labs}
    bb = []
    for _ in range(B_rank):
        ix = np.concatenate([members[r] for r in rng.choice(labs, len(labs))])
        bb.append(tl.within_ols(yv[ix], dr[ix], s[ix], fe=(cell[ix],))[0])
    return dict(slope=r4(b), rack_ci=[r4(x) for x in tl.ci(bb)], pairs=int(len(P)), B=B_rank)


def model_d(act, SL, kind, theta):
    V = []
    for a in act:
        lam, noise = theta[0], theta[1]
        if kind == 'dark_source':
            C = tl.heat_source_corr(a, CORE_XY, lam, noise)
        else:                                            # stationary field sampled at the sensors
            D = np.linalg.norm(CORE_XY[a][:, None] - CORE_XY[a][None], axis=-1)
            C = tl.projected_corr(np.exp(-D / lam) + noise * np.eye(16), np.arange(16))
        V.append(tl.slice_similarity(tl.embed24(C, a)))
    d, _ = tl.column_contrast(np.stack(V), SL)
    return r4(np.nanmean(d))


OUT = {'test': 'S[4c+1,4c+3] - S[4c+1,4c+2] on sockets with all three slices active; >0 follows '
               "IBM's diagram, <0 follows the index / OS order", 'rack_bootstrap_B': B, 'months': {}}
ibm = json.load(open(A / 'thermal_ibm_layout.json'))['months'] if (A / 'thermal_ibm_layout.json').exists() else {}
for ym in MONTHS:
    meta, act, SSS, SL = load(ym)
    d, per_col = tl.column_contrast(SSS, SL)
    res = {'observed': summarise(d, per_col, meta.rack.values, np.random.default_rng(41))}
    o = res['observed']
    print(f'{ym}: observed d = {o["mean"]:+.4f} {o["rack_ci"]} on {o["sockets"]} sockets '
          f'({100 * o["frac_positive"]:.0f}% positive); by column {o["by_column"]}')
    res['rank_slope_within_slice_pair'] = rank_slope(act, SSS, meta.rack.values, np.random.default_rng(43),
                                                     int(os.environ.get('THERM_BOOT_RANK', '400')))
    rs = res['rank_slope_within_slice_pair']
    print(f'  same two slices, per active slice between them: {rs["slope"]:+.4f} {rs["rack_ci"]} '
          f'({rs["pairs"]} slice pairs, {rs["B"]} rack replicates)')
    pm = ibm.get(ym, {}).get('physical_models')
    if pm:
        pred = {}
        for kind in ('field', 'dark_source'):
            pred[kind] = dict(theta=pm[kind]['theta'][:2], d=model_d(act, SL, kind, pm[kind]['theta']))
        pred['dark_source_sweep'] = [dict(lam=s['lam'], noise=s['noise'],
                                          d=model_d(act, SL, 'dark_source', (s['lam'], s['noise'])))
                                     for s in pm['dark_source_sweep']]
        res['physical_model_predictions'] = pred
        sw = [s['d'] for s in pred['dark_source_sweep']]
        print(f'  physical models on IBM layout predict d = {pred["field"]["d"]:+.4f} (field fit), '
              f'{pred["dark_source"]["d"]:+.4f} (dark-source fit), sweep {min(sw):+.4f} .. {max(sw):+.4f}')
    OUT['months'][ym] = res

json.dump(OUT, open(A / 'thermal_column_test.json', 'w'), indent=1)
print('\nwrote analysis/thermal_column_test.json')
