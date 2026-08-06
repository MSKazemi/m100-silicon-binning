"""Exact confidence intervals for the paper's headline rates.

Several rates were reported as point estimates. Small-count rates especially (3 guard episodes,
20 relabelling events) carry uncertainty that a reader cannot reconstruct from the point value.
Clopper-Pearson for proportions, exact Poisson for event counts.
"""
from scipy.stats import beta, chi2

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k)
    return lo, hi

def pois(k, expo, a=0.05):
    lo = 0.0 if k == 0 else chi2.ppf(a/2, 2*k)/2/expo
    hi = chi2.ppf(1-a/2, 2*(k+1))/2/expo
    return lo, hi

print('=== proportions (Clopper-Pearson, exact) ===')
for lab, k, n, unit in [
    ('sockets that never change configuration', 1756, 1962, '%'),
    ('steady-state changes that are relabelling', 20, 105, '%'),
    ('steady-state changes that are CPU-SWAP', 74, 105, '%'),
    ('sockets with a fleet-unique harvest map', 97, 1962, '%'),
    ('held-out socket-days predicted correctly', 811956, 839067, '%'),
]:
    lo, hi = cp(k, n)
    print(f'  {lab:<44} {100*k/n:6.2f}{unit}  95% CI [{100*lo:.2f}, {100*hi:.2f}]  (k={k}/{n})')

print('\n=== event rates (exact Poisson) ===')
SOCK, YEARS = 1968, 2.55
lo, hi = pois(3, SOCK*YEARS)
print(f'  core-guard episodes: 3 in {SOCK} sockets x {YEARS} y')
print(f'    rate = {1e4*3/(SOCK*YEARS):.2f} per 10,000 socket-years  '
      f'95% CI [{1e4*lo:.2f}, {1e4*hi:.2f}]')
print(f'    -> as a share of sockets over the record: {100*3/SOCK:.3f}% '
      f'(95% CI [{100*pois(3,SOCK)[0]:.3f}, {100*pois(3,SOCK)[1]:.3f}])')
lo, hi = pois(96, 1960*YEARS)
print(f'\n  processor replacements: 96 dies in 1,960 sockets x {YEARS} y')
print(f'    rate = {100*96/(1960*YEARS):.2f}% per socket-year  '
      f'95% CI [{100*lo:.2f}, {100*hi:.2f}]')
print('\n  -> the guard rate in particular is an order-of-magnitude statement, not a')
print('     precise one: with 3 events the interval spans roughly a factor of five.')
