import sys, pandas as pd, numpy as np
from pathlib import Path

root = Path(sys.argv[1])
ym = sys.argv[2]
base = root / f'year_month={ym}' / 'plugin=ipmi_pub'

# sample count per (node, socket, core)
recs = []
for p in range(2):
    for c in range(24):
        f = base / f'metric=p{p}_core{c}_temp' / 'a_0.parquet'
        if not f.exists():
            print(f'MISSING METRIC FILE: p{p}_core{c}_temp'); continue
        s = pd.read_parquet(f, columns=['node'])['node'].value_counts()
        for node, n in s.items():
            recs.append((node, p, c, n))

df = pd.DataFrame(recs, columns=['node', 'socket', 'core', 'n'])
df.to_parquet(root / f'counts_{ym}.parquet')

# pivot: rows = node, cols = (socket, core), values = sample count
mat = df.pivot_table(index='node', columns=['socket', 'core'], values='n', fill_value=0)
mat = mat.reindex(columns=pd.MultiIndex.from_product([[0, 1], range(24)]), fill_value=0)

print(f'=== {ym} ===')
print('nodes seen in any core metric:', len(mat))

# per-node reference = median samples across the cores that DO report
ref = mat.replace(0, np.nan).median(axis=1)
print('median samples per (node,core), fleet median: %.0f' % ref.median())

present = mat.gt(0)
n_present = present.sum(axis=1)
print('\n--- how many of the 48 core sensors report, per node ---')
print(n_present.value_counts().sort_index().to_string())

# partial reporters: present but far below that node's own median
partial = (mat.gt(0) & mat.lt(ref.values[:, None] * 0.5)).sum(axis=1)
print('\nnodes with >=1 core reporting but <50%% of that node per-core median: %d' % (partial > 0).sum())

full = (n_present == 48)
print('\nnodes with all 48 present: %d / %d (%.1f%%)' % (full.sum(), len(mat), 100 * full.mean()))

# which core indices are most often absent
print('\n--- absence rate by core index (fraction of nodes missing it) ---')
miss = (~present).mean(axis=0)
for p in range(2):
    row = ' '.join('%2d:%4.0f%%' % (c, 100 * miss[(p, c)]) for c in range(24))
    print(f'p{p}: {row}')

# per-node count of present cores per socket
p0 = present.loc[:, 0].sum(axis=1)
p1 = present.loc[:, 1].sum(axis=1)
print('\n--- cores present per socket ---')
print(pd.DataFrame({'p0': p0.value_counts(), 'p1': p1.value_counts()}).fillna(0).astype(int).sort_index().to_string())
