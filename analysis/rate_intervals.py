"""Exact confidence intervals for the paper's headline proportions.

Several proportions were reported as point estimates. Small counts especially (20 relabelling
events of 105) carry uncertainty that a reader cannot reconstruct from the point value.
Clopper-Pearson throughout.

Event RATES are not computed here any more. An earlier version divided by hard-coded exposures
(1,968 or 1,960 sockets x 2.55 years) that matched neither the steady-state window the events were
counted in nor the observed exposure, and printed 5.98 per 10,000 socket-years and 1.92 % per
socket-year. The rates now come from the exposure they were measured over:
  exposure_rates.py              matched-exposure point rates and exact Poisson intervals
  lifecycle_taxonomy_errors.py   node- and rack-cluster bootstrap intervals, and the share of die
                                 changes that are dies moved between sockets rather than new silicon
"""
from lifecycle_common import cp_ci as cp

print('=== proportions (Clopper-Pearson, exact) ===')
for lab, k, n in [
    ('sockets that never change configuration', 1756, 1962),
    ('steady-state node-level events that are RELABEL', 20, 105),
    ('steady-state node-level events that are CPU-SWAP (incl. ?)', 74, 105),
    ('sockets with a fleet-unique harvest map', 97, 1962),
    ('held-out socket-days predicted correctly', 811956, 839067),
]:
    lo, hi = cp(k, n)
    print(f'  {lab:<58} {100*k/n:6.2f}%  95% CI [{100*lo:.2f}, {100*hi:.2f}]  (k={k}/{n})')
print('\nevent rates: see exposure_rates.py and lifecycle_taxonomy_errors.py')
