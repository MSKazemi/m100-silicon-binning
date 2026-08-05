import pandas as pd, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent

def load(ym, d):
    df = pd.read_parquet(root / d / f'counts_{ym}.parquet') if (root / d / f'counts_{ym}.parquet').exists() \
         else pd.read_parquet(root / f'counts_{ym}.parquet')
    return df

a = pd.read_parquet(root / 'counts_20-04.parquet')
b = pd.read_parquet(root / 'counts_21-03.parquet')

def sets(df):
    """node -> {socket: frozenset(present core idx)}"""
    out = {}
    for (node, sock), g in df.groupby(['node', 'socket']):
        out.setdefault(node, {})[sock] = frozenset(g['core'])
    return out

A, B = sets(a), sets(b)

# 1. is the disabled set always whole PAIRS (2k, 2k+1)?
bad = 0
for node, socks in A.items():
    for s, present in socks.items():
        missing = set(range(24)) - present
        if any((m ^ 1) not in missing for m in missing):
            bad += 1
print('1) PAIR STRUCTURE (20-04)')
print('   socket-instances where missing cores are NOT whole (2k,2k+1) pairs:', bad,
      'of', sum(len(v) for v in A.values()))

# 2. how many distinct patterns exist?
pats = pd.Series([tuple(sorted(set(range(24)) - p)) for socks in A.values() for p in socks.values()])
print('\n2) DISTINCT DISABLED-CORE PATTERNS (20-04):', pats.nunique(), 'distinct, over', len(pats), 'sockets')
print('   top 8 patterns (missing core ids -> #sockets):')
for pat, n in pats.value_counts().head(8).items():
    print('     %-28s %4d' % (str(pat), n))

# 3. stability across ~11 months: same node+socket, same present set?
common = sorted(set(A) & set(B))
same = diff = 0
examples = []
for node in common:
    for s in (0, 1):
        if s in A[node] and s in B[node]:
            if A[node][s] == B[node][s]:
                same += 1
            else:
                diff += 1
                if len(examples) < 5:
                    examples.append((node, s, sorted(A[node][s]), sorted(B[node][s])))
print('\n3) STABILITY 20-04 vs 21-03 (%d common nodes)' % len(common))
print('   identical present-core set: %d sockets (%.2f%%)' % (same, 100*same/(same+diff)))
print('   changed:                    %d sockets' % diff)
for e in examples:
    print('     node %s p%d: %s -> %s' % e)

# 4. concrete random nodes
rng = np.random.default_rng(0)
pick = rng.choice(common, 6, replace=False)
print('\n4) SIX RANDOM NODES (20-04) — which of core 0..23 report per socket')
for node in sorted(pick, key=int):
    print(f'   node {node}:')
    for s in (0, 1):
        pr = sorted(A[node].get(s, ()))
        miss = sorted(set(range(24)) - set(pr))
        print('     p%d  active(%2d): %s' % (s, len(pr), ','.join(map(str, pr))))
        print('         disabled  : %s' % ','.join(map(str, miss)))
