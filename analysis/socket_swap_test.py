"""Are apparent 'hardware changes' actually socket relabelling?

If node N's socket p0 loses exactly the slices p1 gains (and vice versa), the two dies did
not change -- the collector/BMC swapped which physical socket is called p0.
Usage: socket_swap_test.py <extracted_dir> <year_month>
"""
import sys, pandas as pd
from pathlib import Path

base = Path(sys.argv[1]) / f'year_month={sys.argv[2]}' / 'plugin=ipmi_pub'
parts = []
for s in (0, 1):
    for c in range(24):
        f = base / f'metric=p{s}_core{c}_temp' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'node'])
        if not len(d): continue
        parts.append(d.assign(day=d.timestamp.dt.floor('D'), socket=s, core=c)
                      [['node', 'day', 'socket', 'core']])
df = pd.concat(parts, ignore_index=True)
df['node'] = df['node'].astype(int)

S = {k: frozenset(g['core']) for k, g in df.groupby(['node', 'socket', 'day'])}
days = sorted({d for (_, _, d) in S})
print('reporting days:', [str(d.date()) for d in days])

swap = same = other = 0
examples = []
for n in sorted({k[0] for k in S}):
    ds = [d for d in days if (n, 0, d) in S and (n, 1, d) in S]
    if len(ds) < 2: continue
    a0, a1 = S[(n, 0, ds[0])], S[(n, 0, ds[-1])]
    b0, b1 = S[(n, 1, ds[0])], S[(n, 1, ds[-1])]
    if a0 == a1 and b0 == b1:
        same += 1
    elif a1 == b0 and b1 == a0:
        swap += 1; examples.append((n, 'SWAP', sorted(a0), sorted(a1)))
    else:
        other += 1; examples.append((n, 'OTHER', sorted(a0 ^ a1), sorted(b0 ^ b1)))

tot = swap + same + other
print(f'\nnodes with both sockets reporting on >=2 days: {tot}')
print(f'  unchanged            : {same} ({100*same/tot:.2f}%)')
print(f'  pure socket SWAP     : {swap} ({100*swap/tot:.2f}%)   <- dies unchanged, labels flipped')
print(f'  genuinely different  : {other} ({100*other/tot:.2f}%)')
print('\nexamples:')
for n, kind, x, y in examples[:10]:
    print(f'  node {n:>3} {kind}: p0 {x}  |  p1-symmetric-difference {y}')
