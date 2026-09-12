"""How much telemetry does the disclosure actually take, and what stops it?

The paper reports that harvest maps are recoverable; this quantifies how cheaply, which sets the
bar for anyone reproducing the result and tells an operator publishing a dataset how much
redaction is enough.

RECOVERY RULE. A core index is "present" for a socket if it emits at least one sample in the
observation window; the presence set is the recovered map. Recovery is *exact* when that set
equals the socket's modal map over the month. A harvested core can never appear, so there are
no false positives: the only way to fail is for an active core to stay silent for the whole
window.

MODELS (analysis/side_channel_model.py, tested in tests/test_side_channel.py). With cores
deleted independently of one another, P(exact) = prod_i (1 - P(core i silent)) over the 16
active cores, evaluated on the EMPIRICAL per-core sample counts n_i of each socket-day:

  * iid deletion of each sample with probability f:      P(silent) = f ** n_i
  * Poisson surviving count with mean lam per core:     P(silent) = exp(-lam)
  * deterministic decimation, K of 4,320 slots kept:    P(silent) = C(4320-K, n_i) / C(4320, n_i)
  * per-core bursty (two-state Markov) deletion:        P(silent) = f (1 - (1-f)/(f L)) ** (n_i-1)

A previous version fed the fractional count n (1 - f) into the decimation formula with
K = 4319, which is not a deletion model and reported recovery 1.000 at 99.9 % deletion; the
correct one-day value is ~0.81 (external review, 2026-09). Deletion correlated ACROSS cores,
windows shorter than a day, and real collector gaps are measured on raw timestamps by
side_channel_raw.py; if its output analysis/side_channel_raw.json is present it is reported
alongside. Day-granularity recovery (section 2) is measured directly and needs no model.

TWO ATTACKERS. "Exact recovery" above is the per-core rule: every one of the 16 active cores
must keep a sample. Every map is a union of whole slices (asserted below), so an attacker who
knows the slice rule needs one surviving sample per active SLICE (8 events, not 16); the
per-core numbers are the conservative ones, and the slice-aware ones are what a publisher
should plan against. When exact recovery fails, the surviving samples still reveal active
slices with no false positives; section 4 reports how many, and how many of the 495 possible
maps remain consistent with them.

Output: analysis/side_channel_results.json (all numbers below, and the data for Fig. 11 right).
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from side_channel_model import (SLOTS, deletion_for_target, expected_unique_share,  # noqa: E402
                                mc_iid_thinning, p_exact, p_silent_bursty, p_silent_decimation,
                                p_silent_iid, poisson_curve, slice_counts, slices_revealed_iid)

root = Path(__file__).resolve().parent.parent
YM = '22-08'
rng = np.random.default_rng(17)
R = {'month': YM}


def lam_for(p, cores=16):
    """Poisson mean per core at which (1 - exp(-lam)) ** cores = p."""
    return float(-np.log(1.0 - p ** (1.0 / cores)))


d = pd.read_parquet(root / 'daily' / f'daily_{YM}.parquet')
d['day'] = pd.to_datetime(d['day'], utc=True)
print(f'month {YM}: {d.node.nunique()} nodes, {len(d):,} (socket,core,day) rows')

# ground truth = modal 16-core map per socket over the month
act = (d.groupby(['node', 'socket', 'day'])['core'].apply(frozenset)
        .rename('c').reset_index())
truth = (act.groupby(['node', 'socket'])['c']
            .agg(lambda s: s.value_counts().index[0]).rename('map'))
truth = truth[truth.apply(len) == 16]
print(f'sockets with a 16-core modal map: {len(truth):,}')
R['sockets_modal16'] = int(len(truth))
# the slice-aware attacker and slice_counts() need every map to be a union of whole slices
n_aligned = int(sum(all((c ^ 1) in m for c in m) for m in truth))
print(f'maps that are unions of whole slices (cores 2j, 2j+1): {n_aligned:,} of {len(truth):,}')
assert n_aligned == len(truth), 'slice-aware analysis assumes slice-aligned maps'
R['maps_slice_aligned'] = n_aligned

# per-core counts of every socket-day, restricted to the socket's modal cores
days = sorted(d.day.unique())
wide = d.pivot_table(index=['node', 'socket', 'day'], columns='core', values='n',
                     aggfunc='sum', fill_value=0).reindex(columns=range(24), fill_value=0)
wide = wide[wide.index.droplevel('day').isin(truth.index)]
tmask = np.zeros((len(truth), 24), dtype=bool)
for j, m in enumerate(truth):
    tmask[j, sorted(m)] = True
tpos = pd.Series(np.arange(len(truth)), index=truth.index)
row_t = tpos.reindex(wide.index.droplevel('day')).values
W = wide.values
row_day = pd.Index(days).get_indexer(wide.index.get_level_values('day'))


def window_counts(D):
    """Pooled per-core counts (pairs x 16) over non-overlapping windows of D days, for
    (socket, window) pairs whose reporting set over the window equals the modal map."""
    out = []
    for start in range(0, len(days) - D + 1, D):
        sel = (row_day >= start) & (row_day < start + D)
        pooled = pd.DataFrame(W[sel]).groupby(row_t[sel]).sum()
        t = tmask[pooled.index]
        present = pooled.values > 0
        ok = (present == t).all(1)
        out.append(pooled.values[ok][t[ok]].reshape(-1, 16))
    return np.concatenate(out)


C1 = window_counts(1)                       # every clean socket-day of the month
R['socket_days_clean'] = int(len(C1))

# ------------------------------------------------------------------ 1. what actually binds?
day = d[d.day == d.day.value_counts().idxmax()]
g = day.groupby(['node', 'socket'])['n'].apply(list)
g = g[g.apply(len) == 16]
G = np.array(g.tolist())
med = float(np.median(np.median(G, axis=1)))
print('\n=== 1. what does recovery actually cost? ===')
print(f'  {len(g):,} fully-configured socket-days on the modal day; median per-core samples/day '
      f'= {med:,.0f} of {SLOTS:,}')
single_modal = float(p_exact(p_silent_decimation(G, 1)).mean())
single_all = float(p_exact(p_silent_decimation(C1, 1)).mean())
print(f'  P(exact from a SINGLE 20 s interval), uniform-placement model on daily counts:')
print(f'    modal day {single_modal:.4f};  all {len(C1):,} clean socket-days {single_all:.4f}')
R['single_interval_model'] = {'modal_day': single_modal, 'all_days': single_all,
                              'modal_day_sockets': int(len(g)), 'median_samples_per_core': med}
raw_path = root / 'analysis' / 'side_channel_raw.json'
RAW = json.loads(raw_path.read_text()) if raw_path.exists() else None
if RAW:
    wmap = {r['slots']: r for r in RAW['window']}
    print(f'  MEASURED on raw timestamps ({RAW["socket_days_scored"]:,} socket-days, real gaps):')
    for w in (1, 3, 6, 7, 15, 180):
        r = wmap[w]
        print(f'    {20 * w:5d} s window: P(exact) = {r["p_exact"]:.5f}  '
              f'(given the socket reports in the window: {r["p_exact_given_reporting"]:.7f}; '
              f'uniform-independent model {r["model_uniform_independent"]:.5f})')
    print(f'  the model is optimistic from {RAW["window_model_optimistic_from_s"]} s up: the misses'
          ' are whole-socket outages lasting hours, not single cores')
print('  Healthy collection is not the binding case. What binds is how many samples per core')
print('  survive into the window, so we vary that directly.')

print('\n  surviving samples per core ~ Poisson(mean lam), independent across the 16 cores')
print('   mean lam   P(exact) = (1-exp(-lam))^16   wall-clock if one sample per 20 s slot')
lams = [0.1, 0.25, 0.5, 1, 2, 3, 4.32, 5, 7, 10, 20]
for lam in lams:
    print(f'  {lam:9.2f}   {poisson_curve(lam):27.4f}   {20 * lam:22.0f} s')
knee = {str(p): lam_for(p) for p in (0.5, 0.9, 0.99)}
print(f'  lam needed: 0.5 -> {knee["0.5"]:.2f}, 0.9 -> {knee["0.9"]:.2f}, '
      f'0.99 -> {knee["0.99"]:.2f} samples per core')
R['poisson'] = {'lam': lams, 'p_exact': [float(poisson_curve(x)) for x in lams],
                'lam_for_target': knee}

# ------------------------------------------------------------------ 2. multi-day, measured
print('\n=== 2. exact-map recovery vs number of days (measured, no model) ===')
print('   days   sockets scored   exact recovery')
R['measured_days'] = []
for D in [1, 2, 3, 7, 14, 21, len(days)]:
    D = min(D, len(days))
    hits = tot = 0
    for start in range(0, len(days) - D + 1, max(1, (len(days) - D + 1) // 8)):
        win = d[d.day.isin(days[start:start + D])]
        rec = win.groupby(['node', 'socket'])['core'].apply(frozenset)
        common = rec.index.intersection(truth.index)
        hits += int((rec.loc[common] == truth.loc[common]).sum())
        tot += len(common)
    R['measured_days'].append({'days': D, 'scored': tot, 'exact': hits / max(tot, 1)})
    print(f'  {D:5,}   {tot:14,}   {hits / max(tot, 1):14.4f}')

# ------------------------------------------------------------------ 3. decimated cadence
print('\n=== 3. defence: publish at a coarser cadence (keep every k-th slot, one day) ===')
print('   cadence      kept slots/day   P(exact), model   measured (raw)')
R['decimation'] = []
dmeas = {r['every_k_slots']: r['measured'] for r in RAW['decimation']} if RAW else {}
for label, k in [('20 s (as published)', 1), ('1 min', 3), ('5 min', 15), ('15 min', 45),
                 ('1 h', 180), ('6 h', 1080), ('24 h', 4320)]:
    pm = float(p_exact(p_silent_decimation(C1, SLOTS // k)).mean())
    me = dmeas.get(k)
    R['decimation'].append({'cadence': label, 'every_k_slots': k, 'kept_per_day': SLOTS // k,
                            'model': pm, 'measured': me})
    print(f'  {label:>19}   {SLOTS // k:14,}   {pm:15.4f}   '
          f'{"" if me is None else f"{me:.4f}":>14}')
print('  Decimation alone is not a defence: one kept sample per core per day still names the')
print('  core, and a coarser cadence only lengthens the window an attacker needs.')

# ------------------------------------------------------------------ 4. iid deletion
print('\n=== 4. defence: delete each sample independently with probability f ===')
F = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
WIN = {1: C1, 2: window_counts(2), 7: window_counts(7), 31: window_counts(31)}
print('   deleted   ' + '   '.join(f'{D:>2}-day window' for D in WIN))
R['iid'] = {'f': F, 'windows': {}}
for D, C in WIN.items():
    R['iid']['windows'][str(D)] = {
        'pairs': int(len(C)), 'mean_samples_per_core': float(C.mean()),
        'p_exact': [float(p_exact(p_silent_iid(C, f)).mean()) for f in F],
        'mean_surviving_per_core': [float(C.mean() * (1 - f)) for f in F]}
for j, f in enumerate(F):
    print(f'  {100 * f:8.3f}%  ' + '  '.join(f'{R["iid"]["windows"][str(D)]["p_exact"][j]:14.4g}'
                                           for D in WIN))
print('  (pairs scored: ' + ', '.join(f'{D} d: {len(C):,}' for D, C in WIN.items()) + ')')

print('\n  slice-aware attacker (one surviving sample per active slice suffices)')
print('   deleted   ' + '   '.join(f'{D:>2}-day window' for D in WIN))
for D, C in WIN.items():
    SC = slice_counts(C)
    R['iid']['windows'][str(D)]['p_exact_slice_aware'] = [
        float(p_exact(p_silent_iid(SC, f)).mean()) for f in F]
for j, f in enumerate(F):
    print(f'  {100 * f:8.3f}%  ' + '  '.join(
        f'{R["iid"]["windows"][str(D)]["p_exact_slice_aware"][j]:14.4g}' for D in WIN))

print('\n  what survives when exact recovery fails: active slices revealed (of 8, no false')
print('  positives) and maps of the 495 still consistent with them under the slice rule;')
print('  P(exact) is the per-core rule of the table above')
print('   window  deleted   P(exact)   slices revealed   maps still consistent')
R['iid']['partial'] = []
for D, f in [(1, 0.999), (1, 0.9999), (1, 0.99999), (7, 0.99999), (31, 0.99999)]:
    k, cand = slices_revealed_iid(WIN[D], f)
    pe = R['iid']['windows'][str(D)]['p_exact'][F.index(f)]
    R['iid']['partial'].append({'days': D, 'f': f, 'p_exact': pe,
                                'mean_slices_revealed': float(k.mean()),
                                'mean_maps_consistent': float(cand.mean())})
    print(f'  {D:4d} d  {100 * f:8.3f}%   {pe:8.3g}   {k.mean():15.2f}   {cand.mean():20.1f}')

print('\n  Monte Carlo check: binomially thin every core\'s actual count, recover the map')
print('   window  deleted   analytic   Monte Carlo (+- 1 SE)   draws')
R['iid']['monte_carlo'] = []
for D, f, reps in [(1, 0.99, 20), (1, 0.999, 20), (1, 0.9999, 20), (2, 0.999, 20),
                   (7, 0.9999, 50), (31, 0.99999, 200)]:
    C = WIN[D]
    an = float(p_exact(p_silent_iid(C, f)).mean())
    mc, se = mc_iid_thinning(C, f, reps, rng)
    R['iid']['monte_carlo'].append({'days': D, 'f': f, 'analytic': an, 'mc': mc, 'se': se,
                                    'draws': reps * len(C)})
    print(f'  {D:4d} d  {100 * f:8.3f}%   {an:8.4g}   {mc:10.4g} +- {se:8.2g}   {reps * len(C):,}')

print('\n  deletion needed before exact recovery falls to a target (iid, empirical counts)')
print('   window   P(exact) <= 0.5   P(exact) <= 0.01   slice-aware: <= 0.5   <= 0.01')
R['iid']['needed'] = []
for D, C in WIN.items():
    f5, f1 = deletion_for_target(C, 0.5), deletion_for_target(C, 0.01)
    s5, s1 = deletion_for_target(slice_counts(C), 0.5), deletion_for_target(slice_counts(C), 0.01)
    R['iid']['needed'].append({'days': D, 'f_for_0.5': f5, 'f_for_0.01': f1,
                               'slice_aware_f_for_0.5': s5, 'slice_aware_f_for_0.01': s1})
    print(f'  {D:4d} d   {100 * f5:14.5f}%   {100 * f1:15.5f}%   {100 * s5:18.5f}%   '
          f'{100 * s1:9.5f}%')
for D in (365, 858):   # Poisson extrapolation at the healthy 4,320 samples/core/day
    f5 = 1 - lam_for(0.5) / (SLOTS * D)
    f1 = 1 - lam_for(0.01) / (SLOTS * D)
    s5 = 1 - lam_for(0.5, cores=8) / (2 * SLOTS * D)
    s1 = 1 - lam_for(0.01, cores=8) / (2 * SLOTS * D)
    R['iid']['needed'].append({'days': D, 'f_for_0.5': f5, 'f_for_0.01': f1,
                               'slice_aware_f_for_0.5': s5, 'slice_aware_f_for_0.01': s1,
                               'extrapolated': True})
    print(f'  {D:4d} d   {100 * f5:14.5f}%   {100 * f1:15.5f}%   {100 * s5:18.5f}%   '
          f'{100 * s1:9.5f}%   (Poisson, 4,320/core/day)')
print('  Every surviving sample is a full disclosure of one core index, and the map is static')
print('  for months, so an attacker pools days: deletion that hides the map for one day leaves')
print('  it exposed over a week.')

# ------------------------------------------------------------------ 5. correlated deletion
print('\n=== 5. deletion correlated in time or across cores (one day) ===')
print('  per-core bursty deletion: survivors arrive in runs of mean length L slots')
print('   deleted   iid (L=1/f)    L=3      L=15     L=180')
R['bursty'] = []
for f in (0.9, 0.99, 0.999):
    row = {'f': f, 'iid': float(p_exact(p_silent_iid(C1, f)).mean())}
    for L in (3, 15, 180):
        row[f'L{L}'] = float(p_exact(p_silent_bursty(C1, f, L)).mean())
    R['bursty'].append(row)
    print(f'  {100 * f:6.1f}%   {row["iid"]:9.4f}   {row["L3"]:7.4f}  {row["L15"]:7.4f}  '
          f'{row["L180"]:7.4f}')
print('  Clumping the survivors into runs of L divides the number of independent chances to be')
print('  seen by ~L, i.e. it acts like L-fold heavier iid deletion. Pooled over a month:')
print('   deleted   L=15, 31 days   L=180, 31 days')
C31 = WIN[31]
R['bursty_month'] = []
for f in (0.99, 0.999, 0.9999):
    a15 = float(p_exact(p_silent_bursty(C31, f, 15)).mean())
    a180 = float(p_exact(p_silent_bursty(C31, f, 180)).mean())
    R['bursty_month'].append({'f': f, 'L15': a15, 'L180': a180})
    print(f'  {100 * f:7.2f}%   {a15:13.4f}   {a180:14.4f}')
if RAW:
    print('  simulated on the measured raw-timestamp presence (side_channel_raw.json):')
    for r in RAW['timestamp_deletion']:
        print(f'    delete whole timestamps {100 * r["f"]:7.2f}%: P(exact) = {r["measured"]:.4f}'
              f'  in analytic [{r["bound_lo"]:.4f}, {r["bound_hi"]:.4f}]'
              f'   (independent-per-core product: {r["iid_model"]:.3g})')
    for r in RAW['bursty']:
        print(f'    per-core bursty in time, {100 * r["f"]:.1f}%, L={r["L_slots"]}: simulated '
              f'{r["measured"]:.4f} vs closed form {r["model"]:.4f}')
    m = RAW['missingness']
    long_share = m['socket_gap_slots_in_runs_ge']['180']
    print(f'  measured collector gaps: {100 * m["missing_fraction"]:.3f}% of active-core slots '
          f'missing; {100 * m["share_whole_socket_silent"]:.2f}% of them with the whole socket '
          f'silent. Most whole-socket gaps are short (mean {m["socket_gap_mean_run_slots"]:.1f} '
          f'slots), but {100 * long_share:.0f}% of the silent slots lie in gaps of an hour or more.')
    R['raw'] = RAW

# ------------------------------------------------------------------ 6. structural defences
print('\n=== 6. defences that remove the position ===')
T = np.array([sorted(m) for m in truth])
n_uniq = len({tuple(r) for r in T})
print(f'  as published                : {len(truth):,} sockets, {n_uniq} distinct maps recovered')
print('  per-socket aggregate only   : 0 maps -- no per-core sensor, nothing to recover')
print('  compact renumbering (0..15) : 0 maps -- every socket reports the same set by')
print('                                construction; this is the one the BMC nearly did')
perm_keep = []
for r in T:
    p = rng.permutation(24)
    perm_keep.append(tuple(sorted(p[r])))
n_ps = len(set(perm_keep))
# a uniformly random permutation sends a fixed 16-set to a uniformly random 16-of-24 set, so
# the expected share of sockets whose pseudonymous set is unique in the fleet is exact:
exp_unique = expected_unique_share(len(truth), comb(24, 16))
map_counts = pd.Series([tuple(r) for r in T]).value_counts()
true_unique = float((map_counts == 1).sum() / len(truth))
print(f'  per-node random renumbering : the map is destroyed (the pseudonymous set is unlinked to')
print(f'                                position) but the count (16 of 24) still leaks, and a')
print(f'                                permutation that is FIXED over time makes the set itself a')
print(f'                                fingerprint: expected {100 * exp_unique:.1f}% of sockets unique '
      f'(one draw: {n_ps} distinct')
print(f'                                sets of {len(truth):,}), against {100 * true_unique:.1f}% for the '
      f'true maps. It re-identifies a')
print(f'                                socket across releases even if node names change, and still')
print(f'                                exposes every map change. Redraw it per release (or per sample).')
R['structural'] = {'distinct_maps': n_uniq, 'pseudonymous_sets_distinct_one_draw': n_ps,
                   'pseudonymous_expected_unique_share': exp_unique,
                   'true_map_unique_share': true_unique, 'sockets': int(len(truth))}
print('\n  Only defences that remove the per-core POSITION close the channel. Reducing resolution')
print('  does not, because the channel is carried by which sensors exist, not by what they report.')

print('\n=== summary (stable lines for CI) ===')
one = dict(zip(F, R['iid']['windows']['1']['p_exact']))
mon = dict(zip(F, R['iid']['windows']['31']['p_exact']))
f31 = next(x['f_for_0.5'] for x in R['iid']['needed'] if x['days'] == 31)
print(f'iid 99.9% deletion, one day: exact recovery {one[0.999]:.2f}')
print(f'iid 99.99% deletion, one day: exact recovery {one[0.9999]:.1e}')
print(f'iid 99.99% deletion, one month: exact recovery {mon[0.9999]:.4f}')
print(f'iid deletion needed for one-month recovery 0.5: {100 * f31:.3f}%')
s1 = dict(zip(F, R['iid']['windows']['1']['p_exact_slice_aware']))
s31 = next(x['slice_aware_f_for_0.5'] for x in R['iid']['needed'] if x['days'] == 31)
print(f'slice-aware attacker, iid 99.9% deletion, one day: exact recovery {s1[0.999]:.4f}')
print(f'slice-aware attacker, deletion needed for one-month recovery 0.5: {100 * s31:.4f}%')
if RAW:
    print(f'single 20 s interval, measured on raw timestamps: {RAW["window"][0]["p_exact"]:.4f}')
    print('whole-timestamp 99.9% deletion, one day, simulated on raw timestamps: '
          f'{RAW["timestamp_deletion"][2]["measured"]:.2f}')

(root / 'analysis' / 'side_channel_results.json').write_text(json.dumps(R, indent=1))
print('\nwrote analysis/side_channel_results.json')
