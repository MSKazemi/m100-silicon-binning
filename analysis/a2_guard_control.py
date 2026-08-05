"""A2 -- the decisive control for the three core-guard episodes.

If a guard event really deconfigured one slice, the BMC stops exposing that slice's two core
sensors because the HARDWARE is gone -- while every other sensor on the same node keeps
streaming normally. If instead the collector faulted, we expect collateral loss: other metrics
for the same node/socket should also gap out over the same window.

Usage: a2_guard_control.py <extract_root>
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

root = Path(sys.argv[1])
EXPECT = 4320

# episode -> (month, node, socket, lost cores, window)
EPISODES = [
    ('20-03', 347, 1, [8, 9],   '2020-03-13', '2020-03-16'),
    ('21-11', 411, 1, [12, 13], '2021-11-13', '2021-11-15'),
    ('20-08', 946, 0, [16, 17], '2020-08-31', '2020-08-31'),
]
# control metrics on the SAME node (and same socket where socket-specific)
CTRL_SOCK = ['p{s}_power', 'p{s}_vdd_temp', 'p{s}_io_power', 'p{s}_mem_power']
CTRL_NODE = ['total_power', 'ambient', 'pcie', 'fan0_0', 'dimm0_temp']

def load(month, metric, node):
    f = root / f'year_month={month}' / 'plugin=ipmi_pub' / f'metric={metric}' / 'a_0.parquet'
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=['timestamp', 'node'])
    d = d[d.node == str(node)]
    if not len(d):
        return None
    return d.groupby(d.timestamp.dt.floor('D')).size()

for month, node, sock, lost, d0, d1 in EPISODES:
    print(f'\n{"="*78}\nEPISODE  node {node} p{sock}  {d0}..{d1}  lost cores {lost}\n{"="*78}')
    rows = {}
    # the lost cores themselves
    for c in lost:
        s = load(month, f'p{sock}_core{c}_temp', node)
        rows[f'p{sock}_core{c}_temp  (LOST)'] = s
    # a surviving core on the same socket, as an internal reference. It must be part of the
    # configuration in force BEFORE the episode -- picking any core with data somewhere in the
    # month can select a slice that only appears at a later reconfiguration.
    surv = None
    for c in range(24):
        if c in lost: continue
        s = load(month, f'p{sock}_core{c}_temp', node)
        if s is not None and any(v > 0.9 * EXPECT for d, v in s.items() if str(d.date()) < d0):
            surv = c; break
    if surv is not None:
        rows[f'p{sock}_core{surv}_temp  (surviving)'] = load(month, f'p{sock}_core{surv}_temp', node)
    for m in CTRL_SOCK:
        mm = m.format(s=sock)
        rows[mm] = load(month, mm, node)
    for m in CTRL_NODE:
        rows[m] = load(month, m, node)

    days = sorted({d for s in rows.values() if s is not None for d in s.index})
    win = [d for d in days if str(d.date()) >= d0 and str(d.date()) <= d1]
    pre = [d for d in days if str(d.date()) < d0][-3:]
    show = pre + win + [d for d in days if str(d.date()) > d1][:2]

    hdr = 'metric'.ljust(30) + ''.join(f'{str(d.date())[5:]:>7}' for d in show)
    print(hdr); print('-' * len(hdr))
    for name, s in rows.items():
        if s is None:
            print(name.ljust(30) + '  (metric absent for this node)'); continue
        cells = ''.join(f'{int(s.get(d,0)):>7}' for d in show)
        print(name.ljust(30) + cells)
    print(f'\n(window = {d0}..{d1}; values are samples/day, {EXPECT} = full 20 s coverage)')

    # verdict
    ctrl = [n for n in rows if 'LOST' not in n and rows[n] is not None]
    ok = sum(1 for n in ctrl if all(rows[n].get(d, 0) > 0.9 * EXPECT for d in win))
    lostzero = all(rows[n].get(d, 0) == 0 for n in rows if 'LOST' in n for d in win)
    print(f'VERDICT: lost-core sensors silent all window: {lostzero}; '
          f'control metrics fully sampled all window: {ok}/{len(ctrl)}')
