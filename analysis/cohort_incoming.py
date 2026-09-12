"""Does the lot signature belong to the rack, or to the dies that were put in it?

Sec. IV-G argues that rack position cannot cause a harvest map because the map exists before
installation. The review points out that rack index can still proxy other things that differ
between racks: installation chronology, per-rack configuration or firmware, the room. One
consequence of each reading can be tested inside the racks themselves.

Dies arrive in racks after commissioning through map changes of the processor- and node-
replacement classes (Sec. IV-I). If Lot A's signature (slice 0 harvested on 88% of sockets) were
imposed by anything attached to the rack position -- the room, a per-rack configuration, a
per-rack firmware lineage -- dies arriving later in Lot A racks would carry it too. If instead
it is a property of the population of dies originally installed there, arriving dies should
look like wherever spares come from.

Events are the transitions of change_taxonomy.py (consecutive clean days whose 16-core set
differs), with RELABEL excluded because no silicon moves. An incoming map is counted once per
socket and only if it had not been seen on that socket before, so a die returning to its old
socket is not counted as a new arrival.

Writes analysis/cohort_incoming.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

root = Path(__file__).resolve().parent.parent
EXPECT = 4320
CUT = pd.Timestamp('2020-06-01', tz='UTC')          # post-commissioning, as change_taxonomy.py
rng = np.random.default_rng(439)

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
a = df.groupby(['node', 'socket', 'day']).size().rename('active')
fu = df[df.n > 0.9 * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
t = pd.concat([a, fu], axis=1).fillna(0).reset_index()
clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
sets = (df.merge(clean, on=['node', 'socket', 'day'])
          .groupby(['node', 'socket', 'day'])['core'].apply(frozenset).rename('c').reset_index())
S = {(n, s, d): c for n, s, d, c in zip(sets.node, sets.socket, sets.day, sets.c)}
days_of = {k: sorted(g.day) for k, g in sets.groupby(['node', 'socket'])}


def harvested(cores):
    return frozenset(range(12)) - frozenset(c // 2 for c in cores)


events = {}
for (n, s), ds in days_of.items():
    for i in range(1, len(ds)):
        o, w = S[(n, s, ds[i - 1])], S[(n, s, ds[i])]
        if o != w:
            events.setdefault((n, ds[i]), {})[s] = (o, w)

rows, seen = [], {}
for (n, d), per in sorted(events.items(), key=lambda kv: kv[0][1]):
    if len(per) == 2 and per[0][1] == per[1][0] and per[1][1] == per[0][0]:
        continue                                                   # RELABEL: no silicon moved
    kind = 'NODE-SWAP' if len(per) == 2 else 'CPU-SWAP'
    for s, (o, w) in per.items():
        ho, hw = harvested(o), harvested(w)
        if len(ho) != 4 or len(hw) != 4:
            continue
        key = (n, s)
        seen.setdefault(key, {ho})
        if hw in seen[key]:
            continue                                               # a die returning, not arriving
        seen[key].add(hw)
        rows.append((int(n), int(s), d, kind, int(0 in ho), int(0 in hw), d >= CUT))
E = pd.DataFrame(rows, columns=['node', 'socket', 'day', 'kind', 's0_out', 's0_in', 'steady'])
E['lot'] = np.where(E.node // 20 < 22, 'A', 'B')
E = E[E.node // 20 < 49]

M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
K = np.load(root / 'analysis' / 'harvest_keys_full.npy')
rk = K[:, 0] // 20
base = {'A': float(M[rk < 22, 0].mean()), 'B': float(M[(rk >= 22) & (rk < 49), 0].mean()),
        'fleet': float(M[:, 0].mean())}


def node_boot_ci(sub, B=4000):
    """95% percentile CI for P(slice 0 harvested) among arriving dies, nodes resampled."""
    g = sub.groupby('node')['s0_in'].agg(['sum', 'count'])
    v = g.values
    bs = [(lambda x: x[:, 0].sum() / x[:, 1].sum())(v[rng.integers(0, len(v), len(v))])
          for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {'baseline_P_slice0': base}
print('P(slice 0 harvested): Lot A {A:.3f}, Lot B {B:.3f}, fleet {fleet:.3f}'.format(**base))
for era, msk in (('steady', E.steady), ('all', np.ones(len(E), bool))):
    sub = E[msk]
    res = {}
    print(f'\n=== {era} era: dies arriving in a socket after a non-relabel map change ===')
    for lot in ('A', 'B'):
        g = sub[sub.lot == lot]
        k_in, k_out, n = int(g.s0_in.sum()), int(g.s0_out.sum()), len(g)
        ci = node_boot_ci(g)
        p_own = binomtest(k_in, n, base[lot]).pvalue
        p_fleet = binomtest(k_in, n, base['fleet']).pvalue
        res[lot] = {'n': n, 'nodes': int(g.node.nunique()), 'out_s0': k_out, 'in_s0': k_in,
                    'in_rate': k_in / n, 'in_rate_node_boot_ci': ci,
                    'binom_p_vs_own_lot_rate': float(p_own),
                    'binom_p_vs_fleet_rate': float(p_fleet),
                    'by_kind': g.groupby('kind').s0_in.agg(['sum', 'count']).to_dict('index')}
        print(f'  Lot {lot} racks: {n} arriving dies on {g.node.nunique()} nodes; outgoing had slice 0 '
              f'harvested {k_out}/{n} ({100 * k_out / n:.0f}%), arriving {k_in}/{n} '
              f'({100 * k_in / n:.1f}%, node-bootstrap 95% CI [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}])')
        print(f'      exact binomial vs own-lot rate {100 * base[lot]:.1f}%: p = {p_own:.2g};  '
              f'vs fleet rate {100 * base["fleet"]:.1f}%: p = {p_fleet:.2g}')
    ga, gb = sub[sub.lot == 'A'], sub[sub.lot == 'B']
    tab = [[int(ga.s0_in.sum()), len(ga) - int(ga.s0_in.sum())],
           [int(gb.s0_in.sum()), len(gb) - int(gb.s0_in.sum())]]
    p_f = fisher_exact(tab).pvalue
    # node-level permutation of the lot label of arriving dies
    nd = sub.groupby('node').agg(k=('s0_in', 'sum'), n=('s0_in', 'size'), lot=('lot', 'first'))
    obs = ga.s0_in.mean() - gb.s0_in.mean()
    isA = (nd.lot == 'A').values
    kk, nn = nd.k.values, nd.n.values
    Bp, exceed = 4000, 0
    for _ in range(Bp):
        pa = rng.permutation(isA)
        d = kk[pa].sum() / nn[pa].sum() - kk[~pa].sum() / nn[~pa].sum()
        exceed += abs(d) >= abs(obs)
    res['A_vs_B_arriving'] = {'difference': float(obs), 'fisher_p': float(p_f),
                              'node_permutation_p': (exceed + 1) / (Bp + 1)}
    print(f'  arriving dies, Lot A racks minus Lot B racks: {100 * obs:+.1f} pp; Fisher p = {p_f:.2g}; '
          f'node-permutation p = {(exceed + 1) / (Bp + 1):.2g}')
    out[era] = res

print('\nReading: dies that arrive in Lot A racks do not take on Lot A\'s slice-0 rate. The')
print('signature is carried by the parts installed at commissioning (or by how they were set up')
print('then), not imposed by the rack position, the room, or any setting that stays with the rack.')
print('It cannot, by itself, say whether the two original populations differ because of the')
print('delivery, the vendor-side provisioning or the installation-time setup. Lot B arrivals alone')
print('are inconclusive in the steady era; the Lot A contrast carries the test.')
json.dump(out, open(root / 'analysis' / 'cohort_incoming.json', 'w'), indent=1, default=str)
print('\nwrote analysis/cohort_incoming.json')
