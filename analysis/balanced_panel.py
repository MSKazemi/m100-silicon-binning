"""Monthly per-slice harvest rates: population marginals versus a fixed cohort of sockets.

Fig. 12 plots, for each month, the per-slice harvest rate over all clean socket-days of that month.
The sockets contributing change from month to month (coverage varies from 18% to 99.4%, hardware is
replaced), so a flat line could in principle reflect composition rather than stable maps. This
script separates the two:

  population   per_month_marginals.csv as published: every clean socket-day of the month, one vote
               per socket-day (recomputed here and checked against the file);
  balanced     a FIXED cohort: sockets with at least one clean day in every post-commissioning month
               (2020-06 .. 2022-09, 28 months), each contributing one vote per month, its modal map
               that month;
  matched      per-socket month-to-month change: the share of consecutive cohort months in which a
               socket's modal map differs from the month before, and how many cohort sockets ever
               change.
Writes analysis/model_balanced_panel.json (monthly tables included) for Fig. 12.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
NP, EXPECT = 12, 4320
STEADY_FROM = '20-06'


def load_clean():
    df = pd.concat([pd.read_parquet(f, columns=['node', 'day', 'n', 'socket', 'core'])
                    for f in sorted((root / 'daily').glob('daily_*.parquet'))], ignore_index=True)
    df['full'] = df['n'] > 0.9 * EXPECT
    df['bit'] = np.left_shift(np.int64(1), df['core'].astype(np.int64))
    g = df.groupby(['node', 'socket', 'day'], sort=False).agg(active=('core', 'size'),
                                                              nfull=('full', 'sum'),
                                                              mask=('bit', 'sum'))
    g = g[(g.active == 16) & (g.nfull == 16)].reset_index()
    core_mask = g['mask'].to_numpy(np.int64)
    slice_active = np.zeros((len(g), NP), dtype=bool)
    for k in range(NP):
        slice_active[:, k] = ((core_mask >> (2 * k)) & 3) != 0
    harv = ~slice_active
    g['code'] = harv.astype(np.int64) @ (1 << np.arange(NP))
    g['month'] = pd.to_datetime(g['day'], utc=True).dt.strftime('%y-%m')
    return g[['node', 'socket', 'day', 'month', 'code']], harv


def bits(codes):
    codes = np.asarray(codes, dtype=np.int64)
    return ((codes[:, None] >> np.arange(NP)) & 1).astype(float)


if __name__ == '__main__':
    g, harv = load_clean()
    print(f'clean socket-days: {len(g):,}   sockets: {g.groupby(["node", "socket"]).ngroups}')

    # --- population marginals (as published), checked against per_month_marginals.csv
    pop = pd.DataFrame(harv.astype(float), columns=[f's{k}' for k in range(NP)])
    pop['month'] = g['month'].to_numpy()
    PM = pop.groupby('month').mean()
    PM.insert(0, 'socketdays', pop.groupby('month').size())
    ref = pd.read_csv(root / 'analysis' / 'per_month_marginals.csv').set_index('month')
    common = PM.index.intersection(ref.index)
    dmax = float(np.abs(PM.loc[common, [f's{k}' for k in range(NP)]].to_numpy()
                        - ref.loc[common, [f's{k}' for k in range(NP)]].to_numpy()).max())
    print(f'population marginals reproduce per_month_marginals.csv to {dmax:.1e} '
          f'({len(common)} months)')

    # --- socket-month modal maps
    sm = (g.groupby(['node', 'socket', 'month'])['code']
          .agg(lambda c: c.value_counts().index[0]).rename('code').reset_index())
    months_all = sorted(g['month'].unique())
    steady = [m for m in months_all if m >= STEADY_FROM]
    sms = sm[sm.month.isin(steady)]
    cover = sms.groupby(['node', 'socket'])['month'].nunique()
    cohort = cover[cover == len(steady)].index
    n_sockets_steady = int(cover.size)
    panel = sms.set_index(['node', 'socket']).loc[cohort].reset_index()
    print(f'balanced cohort: {len(cohort)} of {n_sockets_steady} sockets seen in the '
          f'post-commissioning window have a clean day in every one of its {len(steady)} months')

    BP = pd.DataFrame(bits(panel['code']), columns=[f's{k}' for k in range(NP)])
    BP['month'] = panel['month'].to_numpy()
    BPM = BP.groupby('month').mean()
    BPM.insert(0, 'sockets', BP.groupby('month').size())
    assert (BPM['sockets'] == len(cohort)).all()

    cols = [f's{k}' for k in range(NP)]
    pop_sd = PM.loc[steady, cols].std(axis=0, ddof=1)
    pop_sd_all = PM[cols].std(axis=0, ddof=1)
    bal_sd = BPM[cols].std(axis=0, ddof=1)
    diff = (PM.loc[steady, cols] - BPM[cols]).abs()

    # --- matched sockets: month-to-month changes inside the cohort
    wide = panel.pivot(index=['node', 'socket'], columns='month', values='code')[steady]
    W = wide.to_numpy(np.int64)
    changed = W[:, 1:] != W[:, :-1]
    n_trans = int(changed.size)
    n_changed = int(changed.sum())
    socks_changing = int(changed.any(1).sum())
    ever_modal = (W != np.apply_along_axis(lambda r: np.bincount(
        np.unique(r, return_inverse=True)[1]).argmax(), 1, W)[:, None])
    print('\n--- post-commissioning months (2020-06 .. 2022-09) ---')
    print(f'max across-month SD of a slice rate: population {100 * pop_sd.max():.2f} pp '
          f'(slice {int(pop_sd.idxmax()[1:])}), balanced cohort {100 * bal_sd.max():.2f} pp '
          f'(slice {int(bal_sd.idxmax()[1:])})')
    print(f'  all {len(PM)} months with clean data (2020-04 has none), population (as published): '
          f'{100 * pop_sd_all.max():.2f} pp')
    print(f'slice 0 range: population {100 * PM.loc[steady, "s0"].min():.1f}..'
          f'{100 * PM.loc[steady, "s0"].max():.1f}%, balanced cohort '
          f'{100 * BPM["s0"].min():.1f}..{100 * BPM["s0"].max():.1f}%')
    print(f'largest |population - cohort| difference in any slice-month: {100 * diff.to_numpy().max():.2f} pp')
    print(f'matched sockets: {n_changed} of {n_trans:,} consecutive cohort socket-months change map '
          f'({100 * n_changed / n_trans:.3f}% per socket per month); {socks_changing} of '
          f'{len(cohort)} cohort sockets ({100 * socks_changing / len(cohort):.1f}%) change at least '
          f'once in 28 months')

    out = {'steady_months': steady, 'cohort_sockets': int(len(cohort)),
           'sockets_seen_steady': n_sockets_steady,
           'population_check_max_abs_diff': dmax,
           'pop_max_sd_pp_steady': float(100 * pop_sd.max()),
           'months_with_clean_data': int(len(PM)),
           'pop_max_sd_pp_all_months': float(100 * pop_sd_all.max()),
           'bal_max_sd_pp': float(100 * bal_sd.max()),
           'bal_sd_pp_by_slice': (100 * bal_sd).round(4).tolist(),
           'pop_sd_pp_by_slice_steady': (100 * pop_sd).round(4).tolist(),
           'pop_s0_range_steady': [float(PM.loc[steady, 's0'].min()),
                                   float(PM.loc[steady, 's0'].max())],
           'bal_s0_range': [float(BPM['s0'].min()), float(BPM['s0'].max())],
           'max_abs_pop_minus_bal_pp': float(100 * diff.to_numpy().max()),
           'matched_transitions': n_trans, 'matched_changes': n_changed,
           'matched_change_rate_per_socket_month': n_changed / n_trans,
           'cohort_sockets_ever_changing': socks_changing,
           'cohort_socket_months_off_mode': int(ever_modal.sum()),
           'population': PM.reset_index().to_dict(orient='list'),
           'balanced': BPM.reset_index().to_dict(orient='list')}
    json.dump(out, open(root / 'analysis' / 'model_balanced_panel.json', 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote analysis/model_balanced_panel.json')
