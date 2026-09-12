"""Every departure of the OS logical-CPU count from 128, checked against the BMC, and back.

The paper validates sensor presence against the operating system in two ways: the OS reports
128 logical CPUs (2 sockets x 16 cores x SMT4) on 99.985 % of samples, and the only three nodes
that ever report 120 (= 128 - one two-core slice) are the three slice-loss episodes. It also
stated that six further nodes reporting 124 (= 128 - one core) are "operating-system core
offlining, invisible to the BMC". This script tests that last statement instead of asserting it,
and turns the comparison into a complete two-way census.

  A. OS -> BMC. For every run of node-days on which `cpu_num` reports 120 or 124, find the cores
     of the configuration in force (the last clean day before) that the BMC loses over the same
     days, on either socket, and compare the count with (128 - value) / 4.
  B. What follows. For each such episode, the affected socket's next clean map, and whether its
     partner stayed unchanged (the CPU-SWAP signature of change_taxonomy.py).
  C. BMC -> OS. Every run of socket-days on which the BMC loses one or two baseline cores while
     the socket otherwise reports (no new cores appearing), with the OS count on those days.
  D. Census of every socket-day that does not report exactly 16 cores: the clean-day filter
     conditions on 16, so this is what shows that no stable configuration of another size exists
     (counts above 16 are partially sampled days across a reconfiguration; odd counts split into
     OS-confirmed single-core deconfigurations and single-sensor dropouts).

Scope: both collectors report COUNTS. The OS cannot say WHICH cores are missing, so this
validates the number and the timing of deconfigurations, not the positions in a harvest map.
Machine-readable results: analysis/lifecycle_os_crosscheck.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lifecycle_common as L

root = Path(__file__).resolve().parent.parent
df = L.load_daily(root)
t = L.socket_days(df)
sets = L.clean_sets(df, t)
SB = L.by_socket(sets)
os_ = L.load_os(root)
boot = L.load_boot(root)
BOOTDAY = {(int(n), d) for n, d in zip(boot.node[boot.rebooted], boot.day[boot.rebooted])}
dev = os_[os_.value.isin([120, 124])]
# per-core sample counts are needed only on OS-deviation node-days and on sub-16 socket-days
need = pd.concat([dev[['node', 'day']].drop_duplicates(),
                  t[t.active < 16][['node', 'day']].drop_duplicates()], ignore_index=True).drop_duplicates()
sub = df.merge(need, on=['node', 'day'])
del df
CNT = {k: dict(zip(g.core, g.n)) for k, g in sub.groupby(['node', 'socket', 'day'])}
OSV = os_.groupby(['node', 'day']).apply(lambda g: dict(zip(g.value.astype(int), g.n.astype(int))),
                                         include_groups=False).to_dict()
OUT = {}

print(f'OS cpu_num coverage: {os_.node.nunique()} nodes, {os_.day.min().date()}..{os_.day.max().date()}; '
      f'values seen: {os_.value.value_counts().to_dict()}')
odd = os_[~os_.value.isin([128, 120, 124])]
print(f'  values other than 128/124/120 ({len(odd)} node-days) occur on nodes '
      f'{sorted(odd.node.unique().tolist())}: nodes >= 984 are not compute nodes; the rest are '
      f'single-day readings, not analysed here')

# ------------------------------------------------------------------ A + B
print('\n=== A. every OS departure to 120 or 124, against the BMC ===')
epis = []
for n, g in dev.groupby('node'):
    for run in L.runs(sorted(g.day.unique())):
        vals = sorted(set(g[g.day.isin(run)].value.astype(int)))
        start, end = run[0], run[-1]
        lost = {}
        for s in (0, 1):
            base, bday = L.last_clean_before(SB, n, s, start)
            ls = set()
            d = start
            while d <= end:
                ls |= L.core_loss(CNT.get((n, s, d), {}), base, onset=(d == start))
                d += pd.Timedelta(days=1)
            if ls:
                lost[s] = sorted(ls)
        expect = (128 - vals[0]) // 4 if len(vals) == 1 else None
        nlost = sum(len(v) for v in lost.values())
        e = dict(node=int(n), start=str(start.date()), end=str(end.date()), os_value=vals,
                 cores_expected=expect, bmc_lost={int(k): v for k, v in lost.items()},
                 match=(expect == nlost and len(lost) == 1),
                 onset_is_boot_day=(int(n), start) in BOOTDAY)
        # B: what follows on the affected socket
        if len(lost) == 1:
            s = next(iter(lost))
            base, _ = L.last_clean_before(SB, n, s, start)
            nxt, nday = L.first_clean_after(SB, n, s, end)
            pb, _ = L.last_clean_before(SB, n, 1 - s, start)
            pa, _ = L.first_clean_after(SB, n, 1 - s, nday) if nday is not None else (None, None)
            if nxt is not None:
                ov = len(L.slices_of(base) & L.slices_of(nxt))
                e.update(next_clean_day=str(nday.date()), map_changed=bool(nxt != base),
                         partner_unchanged=bool(pb == pa) if pb is not None and pa is not None else None,
                         active_slices_kept=ov,
                         lost_slice_active_after=bool({c // 2 for c in lost[s]} <= L.slices_of(nxt)))
        epis.append(e)
        tag = 'MATCH' if e['match'] else 'no match'
        fol = (f"next clean map {e.get('next_clean_day')}: "
               f"{'CHANGED' if e.get('map_changed') else 'unchanged'}"
               f"{', partner unchanged' if e.get('partner_unchanged') else ''}") if 'next_clean_day' in e else ''
        print(f'  node {n:>3}  {start.date()}..{end.date()}  OS {vals}  expects {expect} core(s) lost; '
              f'BMC loses {e["bmc_lost"]} -> {tag}; onset on a boot day: {e["onset_is_boot_day"]}; {fol}')
E = pd.DataFrame(epis)
nm = int(E.match.sum())
print(f'\n  OS departures matched by a BMC loss of exactly the implied number of cores on one socket, '
      f'starting the same day: {nm} of {len(E)} episodes on {E.node.nunique()} nodes')
chg = E[E.get('map_changed', pd.Series(False, index=E.index)).fillna(False).astype(bool)]
cpu = chg[chg.partner_unchanged.fillna(False).astype(bool)]
print(f'  followed by a change of the same socket\'s map at its next clean day: {len(chg)} of {len(E)}; '
      f'with the partner socket unchanged (CPU-SWAP signature): {len(cpu)}')
OUT['os_to_bmc'] = dict(episodes=epis, n=len(E), matched=nm, followed_by_change=len(chg),
                        followed_by_cpu_swap_signature=len(cpu))

# ------------------------------------------------------------------ C
print('\n=== C. every BMC loss of one or two configured cores, against the OS ===')
loss_days = []
for (n, s, d), cnt in CNT.items():
    if len(cnt) >= 16:
        continue
    base, _ = L.last_clean_before(SB, n, s, d)
    ls = L.core_loss(cnt, base)
    if 1 <= len(ls) <= 2:
        loss_days.append((n, s, d, tuple(sorted(ls)), len(cnt),
                          sum(1 for v in cnt.values() if v > 0.9 * L.EXPECT) == len(cnt)))
LD = pd.DataFrame(loss_days, columns=['node', 'socket', 'day', 'lost', 'active', 'fully_sampled'])
bmc = []
for (n, s, lost), g in LD.groupby(['node', 'socket', 'lost']):
    for run in L.runs(sorted(g.day)):
        osv = [OSV.get((n, d)) for d in run]
        cov = [v for v in osv if v is not None]
        dv = sorted({k for v in cov for k in v if k != 128})
        status = ('no OS coverage' if not cov else
                  ('OS deviates ' + str(dv) if dv else 'OS reports 128 throughout'))
        full = int(g[g.day.isin(run)].fully_sampled.sum())
        bmc.append(dict(node=int(n), socket=int(s), lost=list(lost), start=str(run[0].date()),
                        end=str(run[-1].date()), days=len(run), fully_sampled_days=full,
                        era='steady' if run[0] >= L.CUT else 'commissioning', os=status))
        print(f'  node {n:>3} p{s}  lost {list(lost)}  {run[0].date()}..{run[-1].date()} '
              f'({len(run)} d, {full} fully sampled)  {status}')
OUT['bmc_to_os'] = bmc

# ------------------------------------------------------------------ D
# The clean-day filter keeps only exactly-16-core, fully-sampled socket-days, so on its own it can
# never reveal a STABLE configuration of another size. This census of every other socket-day is
# the test that there is none.
print('\n=== D. census of every socket-day that does not report exactly 16 cores ===')
non16 = t[t.active != 16]
full = non16.nfull == non16.active
print(f'  {len(non16)} socket-days: {non16.active.value_counts().sort_index().to_dict()}')
big = non16[non16.active > 16]
rel = []
for r in big.itertuples():
    a, _ = L.last_clean_before(SB, r.node, r.socket, r.day)
    b, _ = L.first_clean_after(SB, r.node, r.socket, r.day)
    rel.append(L.union_relation(L.mask_to_set(r.mask), a, b))
rel = pd.Series(rel, dtype=object)
print(f'  above 16 cores: {len(big)} socket-days on {big[["node", "socket"]].drop_duplicates().shape[0]} sockets, '
      f'fully sampled: {int(full[non16.active > 16].sum())}; relation to the clean maps on either side: '
      f'exact union {int((rel == "exact").sum())}, within the union {int((rel == "within").sum())}, '
      f'outside it {int((rel == "outside").sum())} (an intermediate map inside the gap)')
b24 = non16[non16.active == 24]
print(f'  24 cores: {len(b24)} socket-days on {b24[["node", "socket"]].drop_duplicates().shape[0]} sockets '
      f'({b24.node.nunique()} nodes), fully sampled: {int((b24.nfull == 24).sum())}')
odd = non16[non16.active % 2 == 1]
cat = {}
for r in odd.itertuples():
    a, _ = L.last_clean_before(SB, r.node, r.socket, r.day)
    s = L.mask_to_set(r.mask)
    if a is None:
        k = 'no earlier clean map'
    elif not s <= a:
        k = 'union across a reconfiguration'
    else:
        v = OSV.get((int(r.node), r.day))
        k = ('core lost, OS count not collected' if v is None else
             ('core lost, OS reports 128 throughout (sensor dropout)' if set(v) == {128} else
              f'core lost, OS reports {sorted(k2 for k2 in v if k2 != 128)} (deconfiguration)'))
    cat.setdefault(k, []).append(r)
print(f'  odd counts: {len(odd)} socket-days')
for k, v in sorted(cat.items(), key=lambda kv: -len(kv[1])):
    socks = sorted({(int(x.node), int(x.socket)) for x in v})
    print(f'    {len(v):3d}  {k}: {", ".join(f"{n}p{s}" for n, s in socks)}')
sub = non16[(non16.active < 16) & full]
print(f'  fully-sampled socket-days below 16 cores: {len(sub)} '
      f'({sub.active.value_counts().sort_index().to_dict()}): the guard episodes and the sensor '
      f'dropouts above; none is a stable configuration')
OUT['census_non16'] = dict(
    by_active={int(k): int(v) for k, v in non16.active.value_counts().sort_index().items()},
    above16=int(len(big)), above16_fully_sampled=int(full[non16.active > 16].sum()),
    above16_exact_union=int((rel == 'exact').sum()), above16_within_union=int((rel == 'within').sum()),
    above16_outside_union=int((rel == 'outside').sum()),
    cores24_days=int(len(b24)), cores24_sockets=int(b24[['node', 'socket']].drop_duplicates().shape[0]),
    odd={k: len(v) for k, v in cat.items()},
    below16_fully_sampled={int(k): int(v) for k, v in sub.active.value_counts().sort_index().items()})
(root / 'analysis' / 'lifecycle_os_crosscheck.json').write_text(json.dumps(OUT, indent=1, default=str))
print('\nwrote analysis/lifecycle_os_crosscheck.json')
