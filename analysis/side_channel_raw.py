"""Recovery MEASURED on the raw 20 s timestamps of one month, not modelled from daily counts.

side_channel.py works from the committed daily counts, so everything it says about a window
shorter than a day, about cadence, or about deletion that is correlated in time or across
cores rests on a placement model. This script removes the model: it rebuilds, for every
(node, socket, core), the exact set of 20 s slots that carried a sample in month YM, and
measures directly

  A. exact recovery from a window of w consecutive slots, for w = 1 slot (20 s) to one day,
     under the gaps the collector actually produced, next to the uniform-independent
     placement model evaluated on the same socket-days' counts;
  B. exact recovery after deterministic decimation to a coarser cadence (every k-th slot,
     averaged over ALL k phases; with k = 4,320 this is exactly the single-slot rate);
  C. exact recovery after deleting whole timestamps at random (deletion correlated across
     all sensors of a node), simulated on the measured presence and bracketed analytically by
     side_channel_model.timestamp_deletion_bounds, against the independent-deletion product;
  D. exact recovery after per-core bursty deletion (a two-state Markov chain in time),
     simulated on the measured presence, against the closed form p_silent_bursty;
  E. the structure of the real missingness: how much of it hits a whole socket at once, and
     how long the whole-socket gaps last (slot-weighted run lengths).

A, B and E are measurements; C and D apply a simulated deletion process to measured presence.
Ground truth is each socket's modal 16-core map over the month (as in side_channel.py). Only
socket-days whose full-day reporting set equals that map are scored (60,707 of the 60,760
socket-days of 16-core sockets in 22-08), so the measurements isolate the cost of observation
from genuine map changes; for the same reason whole-day outages are outside E, and runs are
cut at midnight, so E understates the longest gaps.

Input: the per-core temperature partitions of the M100 ExaData archive (Zenodo
10.5281/zenodo.7588815). Pass an extracted directory (argv[1], containing
year_month=YM/plugin=ipmi_pub/metric=pX_coreY_temp/a_0.parquet), or set M100_RAW to the folder
of monthly tars; the 48 core partitions (~2.2 GB for 22-08) are then extracted to a temporary
directory and removed afterwards. Peak memory ~6.5 GB. Output: analysis/side_channel_raw.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from side_channel_model import (SLOTS, decimation_all_phases, p_exact, p_silent_bursty,  # noqa: E402
                                p_silent_decimation, p_silent_iid, run_lengths,
                                timestamp_deletion_bounds)

root = Path(__file__).resolve().parent.parent
YM = os.environ.get('SC_MONTH', '22-08')
RAW = Path(os.environ.get('M100_RAW', '/home/mohsen/exadata/raw'))
rng = np.random.default_rng(2026)
t_start = time.time()

# ------------------------------------------------------------------ locate / extract
tmp = None
if len(sys.argv) > 1:
    base = Path(sys.argv[1])
else:
    tmp = Path(tempfile.mkdtemp(prefix='sc_raw_', dir=os.environ.get('SC_TMP')))
    tar = RAW / f'{YM}.tar'
    if not tar.exists():
        sys.exit(f'no extracted directory given and {tar} not found; see the docstring')
    r = subprocess.run(['tar', '-xf', str(tar), '-C', str(tmp), '--wildcards', '--touch',
                        '--warning=no-timestamp',
                        f'year_month={YM}/plugin=ipmi_pub/metric=p0_core*_temp',
                        f'year_month={YM}/plugin=ipmi_pub/metric=p1_core*_temp'],
                       capture_output=True, text=True)
    base = tmp
part = base / f'year_month={YM}' / 'plugin=ipmi_pub'
files = {(s, c): part / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
         for s in (0, 1) for c in range(24)}
missing = [k for k, f in files.items() if not f.exists()]
if missing:
    sys.exit(f'{len(missing)} of 48 core partitions missing under {part}; refusing to run '
             '(a partial extraction looks exactly like harvested cores)'
             + (f'; tar said: {r.stderr[:200]}' if tmp is not None else ''))

# ------------------------------------------------------------------ presence bitsets
d = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
d['day'] = pd.to_datetime(d['day'], utc=True)
days = sorted(d.day.unique())
ND = len(days)
t0 = int(pd.Timestamp(days[0]).value // 10**6)          # ms
NN = int(d.node.max()) + 1
MS = ND * SLOTS                                          # slots in the month
bits = {}                                                # (s, c) -> packed (NN, MS/8)
n_rows = 0
for (s, c), f in files.items():
    t = pq.read_table(f, columns=['timestamp', 'node'])
    ts = t.column('timestamp').cast('int64').to_numpy()
    nd = pc.cast(t.column('node'), 'int32').to_numpy()
    del t
    if (ts % 20000 != 0).any():
        sys.exit('timestamps are not on the 20 s grid; the slot model below would be wrong')
    slot = (ts - t0) // 20000
    ok = (slot >= 0) & (slot < MS) & (nd >= 0) & (nd < NN)
    pres = np.zeros((NN, MS), dtype=bool)
    pres[nd[ok], slot[ok]] = True
    n_rows += int(ok.sum())
    bits[(s, c)] = np.packbits(pres, axis=1)
    del ts, nd, slot, ok, pres
print(f'month {YM}: {n_rows:,} core-temperature samples on the 20 s grid, '
      f'{NN} node ids, {ND} days  ({time.time() - t_start:.0f} s)')

# ------------------------------------------------------------------ ground truth
act = d.groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index()
truth = act.groupby(['node', 'socket'])['c'].agg(lambda s: s.value_counts().index[0])
truth = truth[truth.apply(len) == 16]
T_idx = [(int(n), int(s)) for n, s in truth.index]
T_core = np.array([sorted(m) for m in truth], dtype=np.int64)          # (S, 16)
S_all = len(T_idx)
print(f'sockets with a 16-core modal map: {S_all:,}')

cnt = d.pivot_table(index=['node', 'socket', 'day'], columns='core', values='n',
                    aggfunc='sum', fill_value=0)


def day_presence(di):
    """(NN, 2, 24, SLOTS) bool presence for day index di."""
    P = np.zeros((NN, 2, 24, SLOTS), dtype=bool)
    lo, hi = di * SLOTS // 8, (di + 1) * SLOTS // 8
    for (s, c), b in bits.items():
        P[:, s, c, :] = np.unpackbits(b[:, lo:hi], axis=1)[:, :SLOTS].astype(bool)
    return P


W = [1, 2, 3, 4, 5, 6, 7, 9, 15, 30, 45, 90, 180, 360, 1080, 2160, 4320]
DEC = [3, 15, 45, 180, 1080, 4320]                       # 1 min ... 24 h
F_TS = [0.9, 0.99, 0.999, 0.9999]
BURSTY = [(0.99, 15), (0.999, 15), (0.99, 180)]
acc = {w: [0, 0, 0] for w in W}                          # exact, socket-reporting, total
win_model = {w: 0.0 for w in W}                          # uniform-independent model, summed
dec_meas = {k: [0.0, 0] for k in DEC}
dec_model = {k: [0.0, 0] for k in DEC}
ts_meas = {f: [0, 0] for f in F_TS}
ts_iid = {f: [0.0, 0] for f in F_TS}
ts_lo = {f: 0.0 for f in F_TS}                           # analytic bounds, summed over sockets
ts_hi = {f: 0.0 for f in F_TS}
gap_hist = np.zeros(SLOTS + 1)                           # whole-socket gap run lengths (count)
bur_meas = {fl: [0, 0] for fl in BURSTY}
bur_model = {fl: [0.0, 0] for fl in BURSTY}
single_model, single_meas = [], []
miss = dict(total=0, socket_silent=0, node_silent=0)
runs_len, runs_n = 0, 0
count_mismatch = 0
scored_days = 0
MC_DAYS = set(range(0, ND, 4))                           # every 4th day for the MC parts
REPS = 4

for di, day in enumerate(days):
    P = day_presence(di)
    # consistency: the bitsets must reproduce the committed daily counts exactly
    dayc = cnt.xs(day, level='day')
    for (n, s) in T_idx[:: 97]:
        if (n, s) in dayc.index:
            if not np.array_equal(P[n, s].sum(1), dayc.loc[(n, s)].reindex(range(24), fill_value=0).values):
                count_mismatch += 1
    node_any = P.any(axis=(1, 2))                                           # (NN, SLOTS)
    keep, node_of = [], []
    for j, (n, s) in enumerate(T_idx):
        daymask = P[n, s].any(1)
        if daymask.sum() == 16 and daymask[T_core[j]].all():
            keep.append(j); node_of.append(n)
    if not keep:
        continue
    keep = np.array(keep); node_of = np.array(node_of)
    A = np.stack([P[T_idx[j][0], T_idx[j][1]][T_core[j]] for j in keep])  # (S, 16, SLOTS)
    del P
    S = len(keep)
    scored_days += 1
    n_i = A.sum(2)                                                          # (S, 16)

    # A. sliding windows of w consecutive slots within the day
    cs = np.zeros((S, 16, SLOTS + 1), dtype=np.int16)
    np.cumsum(A, axis=2, out=cs[:, :, 1:])
    for w in W:
        pw = (cs[:, :, w:] - cs[:, :, :-w]) > 0
        acc[w][0] += int(pw.all(1).sum()); acc[w][1] += int(pw.any(1).sum())
        acc[w][2] += pw.shape[0] * pw.shape[2]
        # uniform placement of each core's n_i samples, independent across cores: a fixed
        # window of w slots is statistically the same as w kept slots
        win_model[w] += float(p_exact(p_silent_decimation(n_i, w)).sum())
    del cs, pw
    single_model.append(p_exact(p_silent_decimation(n_i, 1)))
    single_meas.append(A.all(1).mean(1))

    # B. deterministic decimation: keep every k-th slot, averaged over all k phases
    for k in DEC:
        dec_meas[k][0] += float(decimation_all_phases(A, k).sum()); dec_meas[k][1] += S
        dec_model[k][0] += float(p_exact(p_silent_decimation(n_i, SLOTS // k)).sum())
        dec_model[k][1] += S

    # E. structure of the real missingness among active cores
    m = ~A
    miss['total'] += int(m.sum())
    sock_silent = ~A.any(1)                                                 # (S, SLOTS)
    miss['socket_silent'] += int((m & sock_silent[:, None, :]).sum())
    miss['node_silent'] += int((m & (~node_any[node_of])[:, None, :]).sum())
    dm = np.diff(m.astype(np.int8), axis=2)
    runs_n += int((dm == 1).sum() + m[:, :, 0].sum())
    runs_len += int(m.sum())
    gap_hist += np.bincount(run_lengths(sock_silent), minlength=SLOTS + 1)

    if di in MC_DAYS:
        # C. whole-timestamp deletion, shared by all sensors of a node
        un, inv = np.unique(node_of, return_inverse=True)
        for f in F_TS:
            ts_iid[f][0] += float(p_exact(p_silent_iid(n_i, f)).sum()); ts_iid[f][1] += S
            lo, hi = timestamp_deletion_bounds(A, f)
            ts_lo[f] += float(lo.sum()); ts_hi[f] += float(hi.sum())
            for _ in range(REPS):
                kept = rng.random((len(un), SLOTS)) >= f
                ts_meas[f][0] += int((A & kept[inv][:, None, :]).any(2).all(1).sum())
                ts_meas[f][1] += S
        # D. per-core bursty deletion in time (two-state Markov chain over slots)
        for (f, L) in BURSTY:
            a, b = (1 - f) / (f * L), 1.0 / L
            bur_model[(f, L)][0] += float(p_exact(p_silent_bursty(n_i, f, L)).sum())
            bur_model[(f, L)][1] += S
            for _ in range(REPS):
                st = rng.random((S, 16)) >= f
                seen = A[:, :, 0] & st
                for t in range(1, SLOTS):
                    u = rng.random((S, 16))
                    st = np.where(st, u >= b, u < a)
                    seen |= A[:, :, t] & st
                bur_meas[(f, L)][0] += int(seen.all(1).sum()); bur_meas[(f, L)][1] += S
    del A
    print(f'  day {di + 1:2d}/{ND}: {S:,} sockets scored  ({time.time() - t_start:.0f} s)',
          flush=True)

if tmp is not None:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------ report
res = {'month': YM, 'samples': n_rows, 'sockets_modal16': S_all, 'days_scored': scored_days,
       'count_check_mismatches': count_mismatch}
print(f'\nconsistency: bitset slot counts vs committed daily counts, sampled sockets: '
      f'{count_mismatch} mismatches')
sm, sd = np.concatenate(single_model), np.concatenate(single_meas)
res['socket_days_scored'] = int(len(sm))
print(f'socket-days scored (full-day set == modal map): {len(sm):,}')

print('\n=== A. exact recovery from w consecutive 20 s slots, measured on real gaps ===')
print('   window   P(exact)   P(exact | socket reports in window)   uniform-independent model')
res['window'] = []
for w in W:
    e, rep, tot = acc[w]
    mo = win_model[w] / len(sm)
    res['window'].append({'slots': w, 'seconds': 20 * w, 'p_exact': e / tot,
                          'p_exact_given_reporting': e / max(rep, 1), 'windows': tot,
                          'model_uniform_independent': mo})
    lab = f'{20 * w} s' if w < 180 else f'{w / 180:g} h'
    print(f'  {lab:>7}   {e / tot:8.5f}   {e / max(rep, 1):8.5f}   {mo:10.6f}')
opt = [r['seconds'] for r in res['window'] if r['p_exact'] < r['model_uniform_independent']]
pes = [r['seconds'] for r in res['window'] if r['p_exact'] > r['model_uniform_independent']]
res['window_model_optimistic_from_s'] = min(opt) if opt else None
print(f'  measured beats the model for windows of {pes} s (cross-core correlation of gaps);')
print(f'  the model is optimistic for windows of {min(opt) if opt else "-"} s and longer: real gaps '
      'silence whole sockets for long stretches.')
res['single_interval'] = {'measured': float(sd.mean()),
                          'model_uniform_independent': float(sm.mean())}
print(f'  single slot: measured {sd.mean():.5f} vs uniform-independent model {sm.mean():.5f}')

print('\n=== B. deterministic decimation (every k-th slot, averaged over all phases), one day ===')
print('   cadence   kept slots/day   measured   uniform-placement model')
res['decimation'] = []
for k in DEC:
    me, mo = dec_meas[k][0] / dec_meas[k][1], dec_model[k][0] / dec_model[k][1]
    res['decimation'].append({'every_k_slots': k, 'seconds': 20 * k, 'kept_per_day': SLOTS // k,
                              'measured': me, 'model': mo})
    lab = f'{20 * k // 60} min' if k < 180 else f'{k // 180} h'
    print(f'  {lab:>8}   {SLOTS // k:14,}   {me:8.5f}   {mo:8.5f}')

print('\n=== C. delete whole timestamps (all sensors of a node together), one day ===')
print('   deleted   simulated   analytic bounds       independent-per-core product')
res['timestamp_deletion'] = []
for f in F_TS:
    me, mo = ts_meas[f][0] / ts_meas[f][1], ts_iid[f][0] / ts_iid[f][1]
    n_sd = ts_iid[f][1]
    lo, hi = ts_lo[f] / n_sd, ts_hi[f] / n_sd
    res['timestamp_deletion'].append({'f': f, 'measured': me, 'bound_lo': lo, 'bound_hi': hi,
                                      'iid_model': mo, 'draws': ts_meas[f][1]})
    print(f'  {100 * f:7.2f}%   {me:9.5f}   [{lo:.5f}, {hi:.5f}]   {mo:10.3g}')
print('  ("measured" in the JSON is the simulation on measured presence; the bounds are exact)')

print('\n=== D. per-core bursty deletion in time (Markov chain), one day ===')
print('   deleted   mean kept run   measured   closed form on counts')
res['bursty'] = []
for (f, L) in BURSTY:
    me, mo = bur_meas[(f, L)][0] / bur_meas[(f, L)][1], bur_model[(f, L)][0] / bur_model[(f, L)][1]
    res['bursty'].append({'f': f, 'L_slots': L, 'measured': me, 'model': mo,
                          'draws': bur_meas[(f, L)][1]})
    print(f'  {100 * f:6.1f}%   {L:6d} slots    {me:8.5f}   {mo:8.5f}')

print('\n=== E. how the real missingness is structured (active cores, scored socket-days) ===')
tot_slots = res['socket_days_scored'] * 16 * SLOTS
L = np.arange(SLOTS + 1)
gap_slots = gap_hist * L                                   # slot-weighted
res['missingness'] = {'missing_fraction': miss['total'] / tot_slots,
                      'share_whole_socket_silent': miss['socket_silent'] / max(miss['total'], 1),
                      'share_whole_node_silent': miss['node_silent'] / max(miss['total'], 1),
                      'mean_missing_run_slots': runs_len / max(runs_n, 1),
                      'socket_gap_runs': int(gap_hist.sum()),
                      'socket_gap_mean_run_slots': float(gap_slots.sum() / max(gap_hist.sum(), 1)),
                      'socket_gap_slots_in_runs_ge': {
                          str(t): float(gap_slots[t:].sum() / max(gap_slots.sum(), 1))
                          for t in (2, 3, 15, 45, 180, 1080)}}
for k, v in res['missingness'].items():
    if isinstance(v, dict):
        for t, s in v.items():
            print(f'  share of socket-silent slots in gaps >= {int(t) * 20:>6} s: {s:.4f}')
    else:
        print(f'  {k:28s} {v:.4g}')
print('  (runs are cut at midnight and whole-day outages are excluded, so long gaps are understated)')
# The wall-clock runtime is printed, never written into the JSON: this file is pinned in
# DATA.sha256, and a timing field would make the checksum change on every re-run even when every
# number is bit-identical.
(root / 'analysis' / 'side_channel_raw.json').write_text(json.dumps(res, indent=1))
print(f'\nwrote analysis/side_channel_raw.json  ({round(time.time() - t_start, 1)} s)')
