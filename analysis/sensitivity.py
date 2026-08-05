"""Sensitivity of every headline result to the analysis choices behind it.

The paper states its thresholds; this varies them. Four knobs:
  S1  clean-day threshold      (fraction of the 4,320 expected daily samples; paper uses 0.90)
  S2  modal-map rule           (paper: mode over clean days; alternatives: first, last, strict)
  S3  curveball draw count     (paper: 200 draws x 20,000 swaps)
  S4  thermal alignment grid   -- deferred, needs raw-series re-extraction (see CHANGELOG)
"""
import pandas as pd, numpy as np
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(77)
EXPECT, NP = 4320, 12

df = pd.concat([pd.read_parquet(f) for f in sorted((root / 'daily').glob('daily_*.parquet'))],
               ignore_index=True)
df['day'] = pd.to_datetime(df['day'], utc=True)
act = df.groupby(['node', 'socket', 'day']).size().rename('active')

def maps_for(thresh, rule='mode'):
    """Recover per-socket harvest maps under a given clean-day threshold and map rule."""
    fu = df[df.n > thresh * EXPECT].groupby(['node', 'socket', 'day']).size().rename('nfull')
    t = pd.concat([act, fu], axis=1).fillna(0).reset_index()
    clean = t[(t.active == 16) & (t.nfull == 16)][['node', 'socket', 'day']]
    d = df.merge(clean, on=['node', 'socket', 'day'])
    sets = d.groupby(['node', 'socket', 'day'])['core'].apply(
        lambda c: frozenset(x // 2 for x in c)).rename('p').reset_index()
    rows, keys = [], []
    for (n, s), g in sets.groupby(['node', 'socket']):
        g = g.sort_values('day')
        if rule == 'mode':   pick = g['p'].value_counts().index[0]
        elif rule == 'first': pick = g['p'].iloc[0]
        elif rule == 'last':  pick = g['p'].iloc[-1]
        elif rule == 'strict':
            if g['p'].nunique() != 1: continue
            pick = g['p'].iloc[0]
        v = np.ones(NP, dtype=np.int8); v[sorted(pick)] = 0
        if v.sum() == 4: rows.append(v); keys.append((int(n), int(s)))
    return np.array(rows), keys, len(clean)

def welch(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))

def summarize(M, keys):
    rack = np.array([n for n, _ in keys]) // 20
    A, B = M[rack < 22, 0], M[rack >= 22, 0]
    return dict(n=len(M), pats=len(set(map(tuple, map(np.flatnonzero, M)))),
                s0=100*M[:, 0].mean(), lotA=100*A.mean(), lotB=100*B.mean(), t=welch(A, B))

print('=== S1  clean-day threshold ===')
print(f'{"thresh":>7} {"clean s-days":>13} {"sockets":>8} {"patterns":>9} {"P(s0)%":>8} '
      f'{"LotA%":>7} {"LotB%":>7} {"Welch t":>8}')
base = None
for th in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
    M, K, ncl = maps_for(th)
    r = summarize(M, K)
    if th == 0.90: base = r
    print(f'{th:>7.2f} {ncl:>13,} {r["n"]:>8} {r["pats"]:>9} {r["s0"]:>8.1f} '
          f'{r["lotA"]:>7.1f} {r["lotB"]:>7.1f} {r["t"]:>8.1f}')

print('\n=== S2  modal-map rule (threshold fixed at 0.90) ===')
print(f'{"rule":>8} {"sockets":>8} {"patterns":>9} {"P(s0)%":>8} {"LotA%":>7} {"LotB%":>7} '
      f'{"Welch t":>8} {"maps differing from mode":>26}')
Mm, Km, _ = maps_for(0.90, 'mode')
ref = {k: tuple(np.flatnonzero(r)) for k, r in zip(Km, Mm)}
for rule in ('mode', 'first', 'last', 'strict'):
    M, K, _ = maps_for(0.90, rule)
    r = summarize(M, K)
    diff = sum(1 for k, row in zip(K, M)
               if k in ref and ref[k] != tuple(np.flatnonzero(row)))
    print(f'{rule:>8} {r["n"]:>8} {r["pats"]:>9} {r["s0"]:>8.1f} {r["lotA"]:>7.1f} '
          f'{r["lotB"]:>7.1f} {r["t"]:>8.1f} {diff:>26}')

print('\n=== S3  curveball draw count ===')
M = np.load(root / 'analysis' / 'harvest_maps_full.npy')
def curveball(mat, n):
    rs = [set(np.flatnonzero(r)) for r in mat]; R = len(rs)
    for _ in range(n):
        i, j = rng.integers(0, R, 2)
        if i == j: continue
        A, B = rs[i], rs[j]
        if not (A - B) or not (B - A): continue
        pool = list((A - B) | (B - A)); rng.shuffle(pool)
        na = (A & B) | set(pool[:len(A) - len(A & B)]); nb = (A | B) - na
        if len(na) == len(A) and len(nb) == len(B): rs[i], rs[j] = na, nb
    out = np.zeros_like(mat)
    for r, s in enumerate(rs): out[r, list(s)] = 1
    return out
def gap(m):
    tot = k = 0
    for r in m:
        idx = np.flatnonzero(r)
        for x, y in combinations(idx, 2): tot += abs(x - y); k += 1
    return tot / k
obs = gap(M)
print(f'{"draws":>7} {"null mean":>10} {"null sd":>9} {"z":>8}')
cur = M.copy(); vals = []
for target in (25, 50, 100, 200, 400):
    while len(vals) < target:
        cur = curveball(cur, 20000); vals.append(gap(cur))
    v = np.array(vals)
    print(f'{target:>7} {v.mean():>10.4f} {v.std():>9.4f} {(obs-v.mean())/v.std():>8.2f}')
print(f'observed mean index gap = {obs:.4f}')
