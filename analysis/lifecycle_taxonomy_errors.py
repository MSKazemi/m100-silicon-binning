"""How wrong can the change taxonomy be, and how uncertain is the 2.1 % replacement rate?

A harvest map is a low-cardinality signature, not a serial number, so the RELABEL / CPU-SWAP /
NODE-SWAP classes of change_taxonomy.py are map-change classes whose most plausible physical
reading is the named event. This script quantifies the ways that reading can fail, from the
committed data alone, and propagates them into the rate.

False negatives (a replacement that leaves no trace)
  FN1  the replacement die carries the SAME map. Probability = collision probability of the pool
       it is drawn from, sum_p n_p(n_p-1)/(N(N-1)), computed per lot.
  FN2  a NODE-SWAP whose two new dies happen to mirror the old ones is read as RELABEL.
False positives (a map change with no new die)
  FP1  a firmware / BMC sensor-ID remap. It applies ONE permutation to every socket it touches, so
       a date cluster of changes produced by it must admit a common slice permutation; tested
       exactly (bipartite matching) and calibrated against independent same-lot dies.
  FP2  an in-place reconfiguration (a different set of cores enabled on the same die). It predicts
       new maps CLOSE to the old one; independent new dies predict the overlap distribution of two
       random maps of the same lot. Compared by a goodness-of-fit test.
  FP3  a relabel coinciding with one new die ("half mirror"): one socket takes its partner's old
       map, the other gets a new one. Classified NODE-SWAP (two dies), plausibly one die.
  FP4  a die MOVED from another socket of the fleet (a cross-node exchange). The socket changes
       die, but no new processor enters service. Traced exactly: the new map is the old map of
       another socket that changed on the same day, against the chance of such a match between
       independent dies of the same lot. FP3 is the within-node special case.
Positive evidence (not a validation against service records, which we do not have)
  PE1  CPU-SWAP events preceded by a core or slice deconfiguration that the OS's own CPU count
       confirms (lifecycle_os_crosscheck.py): a failing processor whose map then changes.
Rate
  exact Poisson (independent events), node- and rack-cluster bootstrap (NODE-SWAP moves two dies
  at once; replacements cluster by rack), and sensitivity bounds combining FN1 and FP3.

Machine-readable results: analysis/lifecycle_taxonomy_errors.json.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

import lifecycle_common as L

root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(2026)
OUT = {}

df = L.load_daily(root)
sets = L.clean_sets(df)
del df
E = L.classify_events(sets)
ss = E[E.era == 'steady'].reset_index(drop=True)

# ------------------------------------------------------------------ denominators
print('=== denominators (steady state, 2020-06-01 .. 2022-09-28) ===')
n_ev = len(ss)
n_tr = int(ss.nsock.sum())
kinds = ss.kind.value_counts().to_dict()
dies = int(ss.dies.sum())
print(f'  node-level events: {n_ev}  (= {n_tr} socket-level transitions)')
for k in ['RELABEL', 'CPU-SWAP', 'CPU-SWAP?', 'NODE-SWAP']:
    print(f'    {k:<10} {kinds.get(k, 0):4d}  {100 * kinds.get(k, 0) / n_ev:5.1f}% of the {n_ev} node-level events')
cs = kinds.get('CPU-SWAP', 0) + kinds.get('CPU-SWAP?', 0)
print(f'    CPU-SWAP incl. CPU-SWAP?  {cs} = {100 * cs / n_ev:.1f}%')
print(f'  dies moved (1 per CPU-SWAP[?], 2 per NODE-SWAP): {dies} = the {n_tr} socket transitions '
      f'minus the {2 * kinds.get("RELABEL", 0)} of RELABEL events')
OUT['denominators'] = dict(node_events=n_ev, socket_transitions=n_tr, kinds=kinds,
                           cpu_swap_incl_q=cs, dies=dies)

# ------------------------------------------------------------------ lot map distributions
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
maps = [frozenset(np.flatnonzero(r).tolist()) for r in M]
rack = K[:, 0] // 20
lotA = [m for m, r in zip(maps, rack) if r < 22]
lotB = [m for m, r in zip(maps, rack) if 22 <= r <= 48]
POOL = {}
for lab, mm in [('A', lotA), ('B', lotB)]:
    c = Counter(tuple(sorted(m)) for m in mm)
    POOL[lab] = (list(c.keys()), np.array(list(c.values()), float) / len(mm))


def lot(node):
    return 'A' if node // 20 < 22 else 'B'


# ------------------------------------------------------------------ FN1, FN2
print('\n=== FN1: a replacement die with the same map is invisible ===')
qA, qB = L.collision_probability(lotA), L.collision_probability(lotB)
qAB = L.cross_collision_probability(lotA, lotB)
qBA = L.cross_collision_probability(lotB, lotA)
print(f'  collision probability, same-lot pool: Lot A {qA:.4f}  Lot B {qB:.4f}  '
      f'(uniform over 495 patterns: {1 / 495:.4f})')
print(f'  cross-lot pool: A socket / B die {qAB:.4f}, B socket / A die {qBA:.4f}')
dies_ev = []
for r in ss.itertuples():
    if r.dies:
        dies_ev += [lot(r.node)] * r.dies
nA, nB = dies_ev.count('A'), dies_ev.count('B')
miss_same = nA * qA / (1 - qA) + nB * qB / (1 - qB)
miss_max = nA * max(qA, qAB) / (1 - max(qA, qAB)) + nB * max(qB, qBA) / (1 - max(qB, qBA))
print(f'  observed dies by lot: A {nA}, B {nB}; expected missed: {miss_same:.2f} (same-lot pool), '
      f'{miss_max:.2f} (worst of same-/cross-lot pool)')
print('\n=== FN2: a NODE-SWAP whose new dies mirror the old ones reads as RELABEL ===')
pairs = {}
for (n, s), m in zip(map(tuple, K), maps):
    pairs.setdefault(int(n), {})[int(s)] = m
pm = []
for n, pr in pairs.items():
    if len(pr) == 2 and n < 980:
        keys, p = POOL[lot(n)]
        f = dict(zip(keys, p))
        pm.append(f.get(tuple(sorted(pr[1])), 0) * f.get(tuple(sorted(pr[0])), 0))
p_mirror = float(np.mean(pm))
print(f'  P(new p0 = old p1 and new p1 = old p0 | independent same-lot dies) = {p_mirror:.2e} per '
      f'node swap; expected misreads among {kinds.get("NODE-SWAP", 0)} NODE-SWAPs: '
      f'{kinds.get("NODE-SWAP", 0) * p_mirror:.1e}')
OUT['FN'] = dict(collision_lotA=qA, collision_lotB=qB, collision_A_socket_B_die=qAB,
                 collision_B_socket_A_die=qBA, dies_lotA=nA, dies_lotB=nB,
                 expected_missed_same_lot=miss_same, expected_missed_worst=miss_max,
                 mirror_collision_per_node_swap=p_mirror)

# ------------------------------------------------------------------ FP2: overlap GOF
print('\n=== FP2: are new maps close to the old ones (in-place reconfiguration)? ===')
tr = []
for r in ss.itertuples():
    if r.kind == 'RELABEL':
        continue
    for s, (o, w, _) in r.per.items():
        tr.append((r.node, s, L.harvested_slices(o), L.harvested_slices(w), r.kind))
obs = np.zeros(4)
expv = np.zeros(4)
null_probs = []
for n, s, ho, hn, _ in tr:
    obs[len(ho & hn)] += 1
    keys, p = POOL[lot(n)]
    old = tuple(sorted(ho))
    keep = np.array([k != old for k in keys])
    pp = p * keep
    pp /= pp.sum()
    ov = np.array([len(set(k) & ho) for k in keys])
    pj = np.array([pp[ov == j].sum() for j in range(4)])
    expv += pj
    null_probs.append(pj)
x2 = float(((obs - expv) ** 2 / expv).sum())
B = 20_000
sim = np.zeros((B, 4))
for pj in null_probs:
    sim += rng.multinomial(1, pj, size=B)
x2s = ((sim - expv) ** 2 / expv).sum(1)
b = int((x2s >= x2).sum())
print(f'  shared harvested slices between old and new map (0..3) over {len(tr)} non-relabel socket changes')
print(f'    observed                         {obs.astype(int).tolist()}')
print(f'    independent same-lot new die     {np.round(expv, 1).tolist()}')
print(f'    chi2 = {x2:.2f} (df 3, asymptotic p = {chi2_dist.sf(x2, 3):.2f}); Monte-Carlo p = '
      f'{(b + 1) / (B + 1):.2f} (b={b}, B={B})')
print(f'    one-slice exchanges (3 shared): {int(obs[3])} observed vs {expv[3]:.1f} expected -- '
      f'no excess of minimal changes')
OUT['FP2_overlap'] = dict(n=len(tr), observed=obs.tolist(), expected=expv.tolist(), chi2=x2,
                          p_asymptotic=float(chi2_dist.sf(x2, 3)), B=B, b=b, p_mc=(b + 1) / (B + 1))

# ------------------------------------------------------------------ FP1: remap test
print('\n=== FP1: could a date cluster be one firmware sensor-ID remap? ===')
allE = E.copy()
rows = []
for r in allE.itertuples():
    if r.kind == 'RELABEL':
        continue
    for s, (o, w, _) in r.per.items():
        rows.append(dict(node=r.node, day=r.day.date(), era=r.era,
                         ho=L.harvested_slices(o), hn=L.harvested_slices(w)))
TR = pd.DataFrame(rows)


def p_common_chance(k, lab_nodes, B=2000):
    """P(a common permutation exists) for k independent (old, new) same-lot map pairs."""
    hit = 0
    for _ in range(B):
        pr = []
        for n in lab_nodes[:k]:
            keys, p = POOL[lot(n)]
            i, j = rng.choice(len(keys), 2, p=p)
            while i == j:
                i, j = rng.choice(len(keys), 2, p=p)
            pr.append((set(keys[i]), set(keys[j])))
        hit += L.common_slice_permutation(pr) is not None
    return hit / B


OUT['FP1_remap'] = []
TR['socket'] = [s for r in allE.itertuples() if r.kind != 'RELABEL' for s in r.per]
for (era, day), g in TR.groupby(['era', 'day']):
    if len(g) < 3:
        continue
    sig = L.common_slice_permutation(list(zip(g.ho, g.hn)))
    # a remap could differ between the p0 and p1 sensor namespaces, or between the two lots (e.g. a
    # firmware table keyed on the part's revision): test each group on its own
    per_sock = {int(s): (L.common_slice_permutation(list(zip(h.ho, h.hn))) is not None)
                for s, h in g.groupby('socket') if len(h) >= 3}
    per_lot = {lab: (L.common_slice_permutation(list(zip(h.ho, h.hn))) is not None)
               for lab, h in g.groupby(g.node.map(lot)) if len(h) >= 3}
    pc = p_common_chance(len(g), list(g.node)) if sig is not None else None
    OUT['FP1_remap'].append(dict(era=era, day=str(day), changes=len(g), nodes=int(g.node.nunique()),
                                 common_permutation=sig, per_socket_namespace=per_sock,
                                 per_lot=per_lot, p_by_chance=pc))
    ps = '' if not per_sock else ('; per socket: ' + ', '.join(
        f'p{s} {"exists" if v else "none"} ({int((g.socket == s).sum())})' for s, v in per_sock.items()))
    pl = '' if not per_lot else ('; per lot: ' + ', '.join(
        f'{k} {"exists" if v else "none"} ({int((g.node.map(lot) == k).sum())})' for k, v in per_lot.items()))
    print(f'  {era:<13} {day}  {len(g):3d} non-relabel socket changes on {g.node.nunique():2d} nodes: '
          f'{"a common permutation EXISTS" if sig else "no common slice permutation"}{ps}{pl}'
          f'{f" (P by chance for {len(g)} independent dies = {pc:.2f})" if pc is not None else ""}')
big = [x for x in OUT['FP1_remap'] if x['era'] == 'steady' and x['changes'] >= 4]
print(f'  steady-state dates with >= 4 changes: {len(big)}, of which admit one remap: '
      f'{sum(x["common_permutation"] is not None for x in big)}')

# ------------------------------------------------------------------ FP3: half mirrors
print('\n=== FP3: NODE-SWAPs that are a relabel plus ONE new die ===')
hm = ss[(ss.kind == 'NODE-SWAP') & ss.per.map(L.half_mirror)]
for r in hm.itertuples():
    print(f'  node {r.node} {r.day.date()}: one socket takes its partner\'s old map, the other a new one')
print(f'  {len(hm)} of {kinds.get("NODE-SWAP", 0)} steady-state NODE-SWAPs; die count would fall by {len(hm)}')
OUT['FP3_half_mirror'] = [dict(node=int(r.node), day=str(r.day.date())) for r in hm.itertuples()]

# ------------------------------------------------------------------ FP4: dies moved in the fleet
print('\n=== FP4: dies moved between sockets, traced by their map ===')
CH = []                                   # every steady-state socket change, relabels included
for r in ss.itertuples():
    for s, (o, w, _) in r.per.items():
        CH.append(dict(node=r.node, socket=s, day=r.day, kind=r.kind,
                       old=tuple(sorted(L.harvested_slices(o))), new=tuple(sorted(L.harvested_slices(w)))))
CH = pd.DataFrame(CH)
NR = CH[CH.kind != 'RELABEL'].reset_index(drop=True)
PA = {lab: dict(zip(POOL[lab][0], POOL[lab][1])) for lab in POOL}


def trace(W):
    """For each non-relabel change: is its new map the old map of another socket that changed
    within +/- W days? Also the expected number of such matches between independent dies."""
    hit, exp = [], 0.0
    for r in NR.itertuples():
        cand = CH[((CH.day - r.day).dt.days.abs() <= W) & ~((CH.node == r.node) & (CH.socket == r.socket))]
        m = cand[cand.old == r.new]
        hit.append(len(m) > 0)
        exp += sum(PA[lot(r.node)].get(o, 0.0) for o in cand.old)
    return np.array(hit), exp


tr0, ex0 = trace(0)
tr7, ex7 = trace(7)
NR['traced'] = tr0
NR['traced7'] = tr7


def donors(W):
    """Per change: donor on ANOTHER node within +/- W days; and the reciprocal exchanges (two
    sockets that swapped maps within +/- W days of each other)."""
    xn, rc = [], set()
    for r in NR.itertuples():
        near = ((CH.day - r.day).dt.days.abs() <= W) & ~((CH.node == r.node) & (CH.socket == r.socket))
        xn.append(bool((near & (CH.node != r.node) & (CH.old == r.new)).any()))
        for y in CH[near & (CH.old == r.new) & (CH.new == r.old)].itertuples():
            rc.add(tuple(sorted([(int(r.node), int(r.socket)), (int(y.node), int(y.socket))])))
    return np.array(xn), rc


xnode, rec = donors(0)
xnode7, rec7 = donors(7)
print(f'  of {len(NR)} non-relabel socket changes, the new map is the old map of another socket that')
print(f'    changed the SAME day : {int(tr0.sum())}   (expected between independent dies: {ex0:.2f})')
print(f'    changed within 7 days: {int(tr7.sum())}   (expected: {ex7:.2f})')
for lab, tr_, xn_, rc_ in [('same-day', tr0, xnode, rec), ('7-day', tr7, xnode7, rec7)]:
    print(f'    {lab:>8}: donor on another node {int((tr_ & xn_).sum())}, only the partner socket of the same '
          f'node (FP3 half mirrors) {int((tr_ & ~xn_).sum())}; reciprocal exchanges {len(rc_)} pairs '
          f'({2 * len(rc_)} socket changes; between different nodes {sum(a[0] != b[0] for a, b in rc_)}; '
          f'node pairs {len({tuple(sorted((a[0], b[0]))) for a, b in rc_})})')
print(f'  e.g. {", ".join(f"{a[0]}p{a[1]}<->{b[0]}p{b[1]}" for a, b in sorted(rec)[:6])}')
bk = NR.groupby('kind').traced.agg(['sum', 'count'])
for k, v in bk.iterrows():
    print(f'    {k:<10} {int(v["sum"]):3d} of {int(v["count"]):3d} socket changes traced to another socket')
OUT['FP4_die_moves'] = dict(changes=len(NR), traced_same_day=int(tr0.sum()), expected_same_day=ex0,
                            traced_same_day_cross_node=int((tr0 & xnode).sum()),
                            traced_same_day_within_node_only=int((tr0 & ~xnode).sum()),
                            traced_7d=int(tr7.sum()), expected_7d=ex7,
                            traced_7d_cross_node=int((tr7 & xnode7).sum()),
                            traced_7d_within_node_only=int((tr7 & ~xnode7).sum()),
                            reciprocal_pairs=sorted(rec),
                            reciprocal_pairs_cross_node=int(sum(a[0] != b[0] for a, b in rec)),
                            reciprocal_pairs_7d=sorted(rec7),
                            by_kind={k: [int(v['sum']), int(v['count'])] for k, v in bk.iterrows()})
for x in OUT['FP1_remap']:
    if x['common_permutation'] is not None:
        d = pd.Timestamp(x['day'], tz='UTC')
        g = NR[NR.day == d]
        x['traced_die_moves'] = int(g.traced.sum())
        x['changes'] = int(len(g))
        print(f'  FP1 date {x["day"]} admitting a common permutation: {x["traced_die_moves"]} of its '
              f'{len(g)} changes are traced die moves; an exchange of two maps is itself a permutation, '
              f'so this is a die exchange, not a remap')

# ------------------------------------------------------------------ PE1: OS-corroborated
print('\n=== PE1: CPU-SWAPs preceded by a deconfiguration the OS confirms ===')
xc_path = root / 'analysis' / 'lifecycle_os_crosscheck.json'
if not xc_path.exists():
    raise SystemExit('run analysis/lifecycle_os_crosscheck.py first')
xc = json.loads(xc_path.read_text())['os_to_bmc']['episodes']
pe = []
for e in xc:
    if not e.get('map_changed'):
        continue
    s = int(next(iter(e['bmc_lost'])))
    d = pd.Timestamp(e['next_clean_day'], tz='UTC')
    hit = ss[(ss.node == e['node']) & (ss.day == d)]
    kind = hit.kind.iloc[0] if len(hit) else None
    lag = (d - pd.Timestamp(e['end'], tz='UTC')).days
    sel = NR[(NR.node == e['node']) & (NR.socket == s) & (NR.day == d)]
    traced = bool(sel.traced.iloc[0]) if len(sel) else None
    traced7 = bool(sel.traced7.iloc[0]) if len(sel) else None
    pe.append(dict(node=e['node'], socket=s, os=e['os_value'], deconfig=f"{e['start']}..{e['end']}",
                   change_day=e['next_clean_day'], days_after=lag, kind=kind, new_map_traced=traced,
                   new_map_traced_7d=traced7))
    print(f"  node {e['node']:>3} p{s}: OS {e['os_value']} {e['start']}..{e['end']} -> map change "
          f"{e['next_clean_day']} ({lag} d later), classified {kind}; "
          f"{'die moved from another socket' if traced else 'new map seen nowhere else that day'}"
          f"{'' if traced7 else ' or within 7 days'}")
n_pe = sum(1 for x in pe if x['kind'] == 'CPU-SWAP')
print(f'  {sum(1 for x in pe if x["new_map_traced"] is False)} of {len(pe)} bring in a map that no other '
      f'socket gave up that day ({sum(1 for x in pe if x["new_map_traced_7d"] is False)} within 7 days): '
      f'the sequence a failing processor, deconfigured and then replaced by a new one, would produce')
# chance that a CPU-SWAP of the same socket starts within 7 days of an arbitrary steady-state day
n_cpu = kinds.get('CPU-SWAP', 0)
p_week = n_cpu / (1960 * 850) * 7
print(f'  {n_pe} of the {n_cpu} steady-state CPU-SWAPs follow an OS-confirmed deconfiguration of the same '
      f'socket; chance of a CPU-SWAP within 7 days of an arbitrary socket-day: {p_week:.1e}')
OUT['PE1_os_corroborated'] = dict(events=pe, cpu_swaps_corroborated=n_pe, cpu_swaps=n_cpu,
                                  p_cpu_swap_within_7d_by_chance=p_week)

# ------------------------------------------------------------------ rate
print('\n=== the rate: 96 dies over the steady-state calendar exposure ===')
ssdays = sets[sets.day >= L.CUT]
d0, d1 = ssdays.day.min(), ssdays.day.max()
span = (d1 - d0).days + 1
socks = ssdays.groupby(['node', 'socket']).size().reset_index()[['node', 'socket']]
socks['expo'] = span / L.DAYS_PER_YEAR
expo = float(socks.expo.sum())
per_node_d = ss.groupby('node').dies.sum()
node_tab = socks.groupby('node').expo.sum().to_frame()
node_tab['dies'] = per_node_d.reindex(node_tab.index, fill_value=0)
node_tab['rack'] = node_tab.index // 20
lo, hi = L.poisson_ci(dies, expo)
print(f'  {dies} dies / {expo:,.0f} socket-years = {100 * dies / expo:.2f}% per socket-year')
print(f'    exact Poisson 95% CI      [{100 * lo:.2f}, {100 * hi:.2f}]   (assumes independent single-die events)')
bn = L.cluster_bootstrap_rate(node_tab.dies, node_tab.expo, B=20_000, rng=rng)
rk = node_tab.groupby('rack')[['dies', 'expo']].sum()
br = L.cluster_bootstrap_rate(rk.dies, rk.expo, B=20_000, rng=rng)
print(f'    node-cluster bootstrap    [{100 * bn["lo"]:.2f}, {100 * bn["hi"]:.2f}]   ({bn["clusters"]} nodes)')
print(f'    rack-cluster bootstrap    [{100 * br["lo"]:.2f}, {100 * br["hi"]:.2f}]   ({br["clusters"]} racks)')
exp_r = rk.expo * dies / expo
disp = float(((rk.dies - exp_r) ** 2 / exp_r).sum() / (len(rk) - 1))
print(f'    Pearson dispersion across racks: {disp:.2f} (1 = Poisson)')
OUT['rate'] = dict(label='die-change rate: sockets receiving a different die, per socket-year',
                   dies=dies, exposure_socket_years=expo, rate=dies / expo, poisson=[lo, hi],
                   node_bootstrap=bn, rack_bootstrap=br, rack_dispersion=disp)

print('\n=== ... and the part of it that is new silicon ===')
new0 = int((~NR.traced).sum())
new7 = int((~tr7).sum())
nd = NR[~NR.traced].groupby('node').size()
node_tab['new'] = nd.reindex(node_tab.index, fill_value=0)
rk2 = node_tab.groupby('rack')[['new', 'expo']].sum()
lo2, hi2 = L.poisson_ci(new0, expo)
bn2 = L.cluster_bootstrap_rate(node_tab.new, node_tab.expo, B=20_000, rng=rng)
br2 = L.cluster_bootstrap_rate(rk2.new, rk2.expo, B=20_000, rng=rng)
qn = (sum(qA / (1 - qA) for n in NR[~NR.traced].node if lot(n) == 'A') +
      sum(qB / (1 - qB) for n in NR[~NR.traced].node if lot(n) == 'B'))
print(f'  changes to a map no other socket gave up that day: {new0} = {100 * new0 / expo:.2f}% per socket-year')
print(f'    exact Poisson [{100 * lo2:.2f}, {100 * hi2:.2f}]; node-cluster bootstrap '
      f'[{100 * bn2["lo"]:.2f}, {100 * bn2["hi"]:.2f}]; rack-cluster bootstrap [{100 * br2["lo"]:.2f}, {100 * br2["hi"]:.2f}]')
nd7 = NR[~NR.traced7].groupby('node').size()
node_tab['new7'] = nd7.reindex(node_tab.index, fill_value=0)
rk3 = node_tab.groupby('rack')[['new7', 'expo']].sum()
lo3, hi3 = L.poisson_ci(new7, expo)
br3 = L.cluster_bootstrap_rate(rk3.new7, rk3.expo, B=20_000, rng=rng)
print(f'    tracing within 7 days instead: {new7} = {100 * new7 / expo:.2f}%; exact Poisson '
      f'[{100 * lo3:.2f}, {100 * hi3:.2f}]; rack-cluster bootstrap [{100 * br3["lo"]:.2f}, {100 * br3["hi"]:.2f}]')
print(f'    same-map new dies missed (FN1): +{qn:.2f}')
print(f'  -> new processors enter service at {100 * new7 / expo:.1f}-{100 * (new0 + qn) / expo:.1f}% per '
      f'socket-year; the {dies} die changes ({100 * dies / expo:.1f}%) include {int(tr0.sum())} dies that moved '
      f'between sockets the same day ({int(tr7.sum())} within 7 days)')
OUT['rate_new_dies'] = dict(label='changes to a map no other socket gave up the same day (new silicon)',
                            new_same_day=new0, new_7d=new7, rate=new0 / expo, poisson=[lo2, hi2],
                            node_bootstrap=bn2, rack_bootstrap=br2, fn_collision_add=qn,
                            rate_7d=new7 / expo, poisson_7d=[lo3, hi3], rack_bootstrap_7d=br3,
                            range_rate=[new7 / expo, (new0 + qn) / expo])

(root / 'analysis' / 'lifecycle_taxonomy_errors.json').write_text(json.dumps(OUT, indent=1, default=str))
print('\nwrote analysis/lifecycle_taxonomy_errors.json')
