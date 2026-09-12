"""How diverse, and how identifying, are the harvest maps? With bias-corrected estimators and
uncertainty that respects the rack clustering.

Responds to an external review of the Discussion ("How identifying is a harvest map?") and of
Fig. 8 / Tables IV and IX:

  D1  ENTROPY. The paper quoted the plug-in (maximum-likelihood) entropy, 8.08 bits (log base
      2). With 1,962 sockets spread over up to 495 maps the plug-in is biased low. Reported
      here beside the Miller-Madow and Chao-Shen corrections, with a delete-one-rack jackknife
      standard error (a socket bootstrap SE for comparison). Bootstrap INTERVALS are not used:
      resampling with replacement duplicates sockets/racks, which by itself lowers entropy and
      richness, so the bootstrap distribution is biased for exactly these statistics.
  D2  DIVERSITY BY LOT (Fig. 8). Rarefied richness at equal size, Chao1, Chao-Shen entropy,
      Hill numbers of order 1 and 2, each with a delete-one-rack jackknife SE, and the Lot B
      minus Lot A difference with a whole-rack relabelling permutation p, (b+1)/(B+1).
  D3  IDENTIFIABILITY UNDER AUXILIARY INFORMATION. Fleet-wide uniqueness of a single map is
      a property of THIS fleet and its size. Uniqueness is recomputed for an adversary who also
      knows the lot, the rack, or who sees a node's two maps together, and as a function of
      the size of the population searched.
  D4  TABLE IX, 442 vs 443 PATTERNS. Which socket and which map make the difference.
  D5  TABLE IV. Which sockets the rack- and node-level lot counts include.

Writes analysis/cohort_diversity.json.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort_models as cm  # noqa: E402

root = Path(__file__).resolve().parent.parent
OUT = root / 'analysis'
B = int(os.environ.get('COHORT_B', 2000))
rng = np.random.default_rng(2026)
LOG2_495 = np.log2(495)

M = np.load(OUT / 'harvest_maps_full.npy').astype(np.int8)
K = np.load(OUT / 'harvest_keys_full.npy')
node, sock = K[:, 0], K[:, 1]
rack = node // 20
pid = cm.pattern_ids(M)
N = len(M)
res = {}


def counts_of(ids):
    return np.bincount(ids, minlength=cm.NPAT)


def estimators(ids):
    c = counts_of(ids)
    return {'plugin': cm.entropy_plugin(c), 'miller_madow': cm.entropy_miller_madow(c),
            'chao_shen': cm.entropy_chao_shen(c)}


jack_se = cm.jack_se


# ================================================================== D1 entropy
# Resampling WITH replacement duplicates sockets or racks, which by itself lowers any entropy or
# richness statistic, so bootstrap intervals for these are biased (a basic interval can even
# exclude the point estimate). Standard errors therefore come from a delete-one-rack jackknife,
# which never duplicates anything; the socket bootstrap SE is shown only for comparison.
print('=== D1  entropy of the fleet map distribution (bits = log base 2) ===')
est = estimators(pid)
kobs = int((counts_of(pid) > 0).sum())
sock_b = {k: [] for k in est}
for _ in range(B):
    e = estimators(pid[rng.integers(0, N, N)])
    for k in est:
        sock_b[k].append(e[k])
jk = {k: [] for k in est}
for r in np.unique(rack):
    e = estimators(pid[rack != r])
    for k in est:
        jk[k].append(e[k])
d1 = {'n_sockets': int(N), 'distinct_maps': kobs, 'log2_495': float(LOG2_495),
      'singletons': int((counts_of(pid) == 1).sum()), 'doubletons': int((counts_of(pid) == 2).sum())}
print(f'  sockets {N}, distinct maps {kobs}, singletons {d1["singletons"]}, doubletons '
      f'{d1["doubletons"]}, log2(495) = {LOG2_495:.3f}')
print(f'  {"estimator":<13}{"H bits":>8}{"2^H":>7}{"% of max":>9}{"socket-boot SE":>16}'
      f'{"rack-jackknife SE":>19}{"95% CI (H +/- 1.96 SE)":>26}{"2^H CI":>14}')
for k in est:
    jse = jack_se(jk[k])
    ci = [est[k] - 1.96 * jse, est[k] + 1.96 * jse]
    d1[k] = {'H_bits': est[k], 'effective_maps_2^H': float(2 ** est[k]),
             'pct_of_log2_495': float(100 * est[k] / LOG2_495),
             'socket_boot_se': float(np.std(sock_b[k])), 'rack_jackknife_se': jse,
             'ci95_rack_jackknife': ci, 'effective_maps_ci': [float(2 ** ci[0]), float(2 ** ci[1])]}
    print(f'  {k:<13}{est[k]:8.3f}{2 ** est[k]:7.0f}{100 * est[k] / LOG2_495:9.1f}'
          f'{np.std(sock_b[k]):16.3f}{jse:19.3f}      [{ci[0]:.3f}, {ci[1]:.3f}]'
          f'   [{2 ** ci[0]:.0f}, {2 ** ci[1]:.0f}]')
print('  The plug-in is biased low; the corrections move it by '
      f'{est["miller_madow"] - est["plugin"]:+.2f} (Miller-Madow) and '
      f'{est["chao_shen"] - est["plugin"]:+.2f} bits (Chao-Shen).')
res['D1_entropy'] = d1

# ================================================================== D2 diversity by lot
print('\n=== D2  diversity by lot (rack-indexed sockets; node 980 excluded) ===')
idx = rack < 49
lotA, lotB = idx & (rack < 22), idx & (rack >= 22)
nA, nB = int(lotA.sum()), int(lotB.sum())
M_RARE = 800            # below the smallest delete-one-rack size of either lot (840)


def div(ids):
    c = counts_of(ids)
    q0, q1, q2 = cm.hill_numbers(c)
    return {'n': int(len(ids)), 'richness': q0, 'richness_rarefied_800': cm.rarefied_richness(c, M_RARE),
            'chao1': cm.chao1(c), 'H_plugin': cm.entropy_plugin(c),
            'H_mm': cm.entropy_miller_madow(c), 'H_cs': cm.entropy_chao_shen(c),
            'hill1_plugin': q1, 'hill1_cs': float(2 ** cm.entropy_chao_shen(c)),
            'hill2_unbiased': cm.inverse_simpson_unbiased(c)}


keys = ['richness_rarefied_800', 'chao1', 'H_mm', 'H_cs', 'hill1_cs', 'hill2_unbiased']
d2 = {}
for lab, msk in (('A', lotA), ('B', lotB)):
    e = div(pid[msk])
    reps = [div(pid[msk & (rack != r)]) for r in np.unique(rack[msk])]
    e['rack_jackknife_se'] = {k: jack_se([x[k] for x in reps]) for k in keys}
    d2[lab] = e
# B - A: SE from the two independent jackknives; p from relabelling whole racks between lots
ur = np.unique(rack[idx])
nrA = len(np.unique(rack[lotA]))
obs_d = {k: d2['B'][k] - d2['A'][k] for k in keys}
exceed = {k: 0 for k in keys}
for _ in range(B):
    perm = rng.permutation(ur)
    ga = np.isin(rack, perm[:nrA]) & idx
    gb = np.isin(rack, perm[nrA:]) & idx
    ea, eb = div(pid[ga]), div(pid[gb])
    for k in keys:
        exceed[k] += abs(eb[k] - ea[k]) >= abs(obs_d[k])
d2['B_minus_A'] = {k: {'estimate': float(obs_d[k]),
                       'se': float(np.hypot(d2['A']['rack_jackknife_se'][k],
                                            d2['B']['rack_jackknife_se'][k])),
                       'rack_permutation_p': float((exceed[k] + 1) / (B + 1))} for k in keys}
print(f'  {"":<24}{"Lot A":>18}{"Lot B":>18}{"B - A":>18}{"perm p":>11}')
for k, lab in [('richness_rarefied_800', 'distinct maps @ n=800'), ('chao1', 'Chao1 richness'),
               ('H_cs', 'H Chao-Shen [bits]'), ('hill1_cs', 'effective maps 2^H_CS'),
               ('hill2_unbiased', 'inverse Simpson')]:
    a_, b_, dd = d2['A'], d2['B'], d2['B_minus_A'][k]
    fa = f'{a_[k]:.2f} +/- {a_["rack_jackknife_se"][k]:.2f}'
    fb = f'{b_[k]:.2f} +/- {b_["rack_jackknife_se"][k]:.2f}'
    fd = f'{dd["estimate"]:+.2f} +/- {dd["se"]:.2f}'
    print(f'  {lab:<24}{fa:>18}{fb:>18}{fd:>18}{dd["rack_permutation_p"]:>11.1e}')
print(f'  raw distinct maps: Lot A {d2["A"]["richness"]:.0f} (n={nA}), Lot B '
      f'{d2["B"]["richness"]:.0f} (n={nB}). +/- is a delete-one-rack jackknife SE; p relabels whole')
print(f'  racks between the lots, (b+1)/(B+1), B = {B}.')
res['D2_diversity_by_lot'] = d2

# ================================================================== D3 identifiability
print('\n=== D3  uniqueness of a map, by what else the adversary knows ===')
pat = [tuple(np.flatnonzero(r)) for r in M]
lot = np.where(rack < 22, 'A', 'B')
d3 = {}
for lab, key in [('map alone (fleet)', pat), ('map + lot', list(zip(pat, lot))),
                 ('map + rack', list(zip(pat, rack)))]:
    u = cm.unique_fraction(key)
    d3[lab] = {'unique_fraction': u, 'unique_count': int(round(u * N)), 'n': int(N)}
    print(f'  {lab:<34}: {100 * u:5.1f}%  ({int(round(u * N))} of {N} sockets)')
# a node's two maps, as an (ordered) pair and as an unordered pair
bynode = {}
for (n_, s_), p_ in zip(K, pat):
    bynode.setdefault(int(n_), {})[int(s_)] = p_
full = {n_: v for n_, v in bynode.items() if len(v) == 2}
ordered = [(v[0], v[1]) for v in full.values()]
unordered = [tuple(sorted((v[0], v[1]))) for v in full.values()]
for lab, key in [('node: ordered (p0, p1) pair', ordered),
                 ('node: unordered pair of maps', unordered)]:
    u = cm.unique_fraction(key)
    d3[lab] = {'unique_fraction': u, 'unique_count': int(round(u * len(key))), 'n': len(key)}
    print(f'  {lab:<34}: {100 * u:5.1f}%  ({int(round(u * len(key)))} of {len(key)} nodes)')
# uniqueness against the size of the population searched: m racks drawn without replacement
ur = np.unique(rack[idx])
curve = {}
for m in (1, 2, 5, 10, 25, 49):
    vals = []
    for _ in range(400 if m < 49 else 1):
        pick = set(rng.choice(ur, m, replace=False).tolist())
        sub = [p_ for p_, r in zip(pat, rack) if r in pick]
        vals.append(cm.unique_fraction(sub))
    curve[m] = {'sockets': int(sum(1 for r in rack if r in pick)),
                'mean_unique_fraction': float(np.mean(vals)),
                'pct_2p5': float(np.percentile(vals, 2.5)), 'pct_97p5': float(np.percentile(vals, 97.5))}
print('  map alone, searched among the sockets of m randomly chosen racks:')
for m, v in curve.items():
    print(f'    m = {m:2d} racks (~{v["sockets"]:4d} sockets): unique {100 * v["mean_unique_fraction"]:5.1f}%'
          f'  (2.5-97.5 pct over draws [{100 * v["pct_2p5"]:.1f}, {100 * v["pct_97p5"]:.1f}])')
d3['uniqueness_vs_racks_searched'] = curve
cnt = Counter(pat)
d3['mean_sockets_sharing_a_map'] = float(N / len(cnt))
d3['note'] = ('Uniqueness is a census count for this fleet, not a sample proportion: a binomial '
              '(Clopper-Pearson) interval on 97/1962 treats sockets as independent trials, which '
              'they are not. The curve over racks searched shows how strongly it depends on the '
              'size of the population an adversary must search.')
res['D3_identifiability'] = d3

# ================================================================== D4 Table IX 442 vs 443
print('\n=== D4  Table IX: why 442 patterns at thresholds 0.50/0.80 and under "first" ===')
EXPECT = 4320
df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
act = df.groupby(['node', 'socket', 'day']).size().rename('active')


def maps_for(thresh, rule='mode'):
    """Identical to sensitivity.py's maps_for, returning {(node, socket): pattern}."""
    fu = df[df.n > thresh * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
    t = pd.concat([act, fu], axis=1).fillna(0).reset_index()
    clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
    d = df.merge(clean, on=['node', 'socket', 'day'])
    sets = d.groupby(['node', 'socket', 'day'])['core'].apply(
        lambda c: frozenset(x // 2 for x in c)).rename('p').reset_index()
    out = {}
    for (n_, s_), g in sets.groupby(['node', 'socket']):
        g = g.sort_values('day')
        vc = g['p'].value_counts()
        pick = {'mode': vc.index[0], 'first': g['p'].iloc[0], 'last': g['p'].iloc[-1]}[rule]
        if len(pick) == 8:
            out[(int(n_), int(s_))] = (tuple(sorted(set(range(12)) - set(pick))),
                                       int(vc.iloc[0]), int(vc.get(pick, 0)), len(g))
    return out


ref = maps_for(0.90, 'mode')
ref_pats = Counter(v[0] for v in ref.values())
d4 = {}
for lab, (th, rule) in {'0.50': (0.50, 'mode'), '0.80': (0.80, 'mode'),
                        'first': (0.90, 'first')}.items():
    alt = maps_for(th, rule)
    alt_pats = Counter(v[0] for v in alt.values())
    lost = sorted(set(ref_pats) - set(alt_pats))
    gained = sorted(set(alt_pats) - set(ref_pats))
    changed = {k: (ref[k][0], alt[k][0]) for k in ref if k in alt and ref[k][0] != alt[k][0]}
    culprits = []
    for pp in lost:
        who = [k for k, v in ref.items() if v[0] == pp]
        for k in who:
            culprits.append({'socket': f'node {k[0]} p{k[1]}', 'map_at_0.90_mode': list(pp),
                             'map_under_alternative': list(alt[k][0]) if k in alt else None,
                             'clean_days_0.90': ref[k][3],
                             'days_supporting_mode_0.90': ref[k][1],
                             'clean_days_alternative': alt[k][3] if k in alt else 0,
                             'days_supporting_choice_alternative': alt[k][2] if k in alt else 0})
    d4[lab] = {'patterns': len(alt_pats), 'sockets': len(alt), 'patterns_lost': [list(p) for p in lost],
               'patterns_gained': [list(p) for p in gained], 'maps_changed': len(changed),
               'culprits': culprits}
    print(f'  {lab:>5}: {len(alt_pats)} patterns on {len(alt)} sockets; vs 0.90/mode lost '
          f'{[list(p) for p in lost]}, gained {[list(p) for p in gained]}; {len(changed)} maps differ')
    for c in culprits:
        print(f'         {c["socket"]}: 0.90/mode map {c["map_at_0.90_mode"]} '
              f'({c["days_supporting_mode_0.90"]} of {c["clean_days_0.90"]} clean days) -> '
              f'{c["map_under_alternative"]} under the alternative '
              f'({c["days_supporting_choice_alternative"]} of {c["clean_days_alternative"]})')
res['D4_table_IX_442_vs_443'] = d4

# ================================================================== D5 Table IV
print('\n=== D5  Table IV: what the lot counts contain ===')
s0 = M[:, 0]
rows = []
for lab, msk in [('Lot A, racks 0-21', rack < 22),
                 ('Lot B, rack >= 22 (as tabulated)', rack >= 22),
                 ('Lot B, racks 22-48 only', (rack >= 22) & (rack < 49)),
                 ('Lot A, nodes 0-438', node <= 438),
                 ('Lot B, node >= 439 (as tabulated)', node >= 439),
                 ('Lot B, nodes 439-979 only', (node >= 439) & (node <= 979))]:
    rows.append({'population': lab, 'sockets': int(msk.sum()),
                 'P_slice0_pct': float(100 * s0[msk].mean()),
                 'patterns': int(len(set(pid[msk])))})
    print(f'  {lab:<36} {msk.sum():5d} sockets  P(slice 0) {100 * s0[msk].mean():5.1f}%  '
          f'{len(set(pid[msk])):4d} patterns')
print(f'  sockets outside racks 0-48 (nodes 980-983): {int((rack >= 49).sum())} '
      f'on node(s) {sorted(set(node[rack >= 49].tolist()))}')
res['D5_table_IV'] = rows

json.dump(res, open(OUT / 'cohort_diversity.json', 'w'), indent=1,
          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('\nwrote analysis/cohort_diversity.json')
