"""Which thermal results replicate across months? (external review 2026-09-11, TH-04)

Reads analysis/thermal_socket_stats[_<YM>].json written by thermal_socket_stats.py for each month
built (2022-08 is the paper's month; 2022-02 and 2021-05 were added as independent replications)
and tabulates the load-bearing quantities side by side. Writes thermal_month_summary.json.
"""
import json
from pathlib import Path

A = Path(__file__).resolve().parent
MONTHS = ['22-08', '22-02', '21-05']
rows = {}
for ym in MONTHS:
    f = A / f'thermal_socket_stats{"" if ym == "22-08" else "_" + ym}.json'
    if not f.exists():
        continue
    d = json.load(open(f))
    c, q, fp = d['contrast_clustered'], d['quad_decomposition'], d['floorplan']
    ir = fp['index_vs_rank_distance']
    rows[ym] = {
        'sockets': d['replication']['sockets'],
        'within': q['within']['pooled'], 'samequad': q['samequad']['pooled'],
        'crossquad': q['crossquad']['pooled'],
        'paper_contrast': c['pooled_contrast'], 'paper_contrast_rack_ci': c['rack']['contrast_ci'],
        'racks_positive': f"{c['racks_positive']}/{c['racks']}",
        'within_minus_samequad': q['balanced_within_minus_samequad']['mean'],
        'within_minus_samequad_ci': q['balanced_within_minus_samequad']['rack_ci'],
        'within_minus_crossquad': q['balanced_within_minus_crossquad']['mean'],
        'within_minus_crossquad_ci': q['balanced_within_minus_crossquad']['rack_ci'],
        'floorplan_identity': fp['census']['identity'], 'exact_optimum': fp['census']['exact_optimum'],
        'exact_order': fp['census']['exact_order'],
        'identity_rank': fp['census']['layouts_strictly_better_than_identity'] + 1,
        'quad_level_identity_is_optimum': fp['quad_level']['identity'] == fp['quad_level']['exact_optimum'],
        'per_socket_score': fp['per_socket']['mean'],
        'beta_index': ir['within_socket_ols']['beta_index'],
        'beta_index_ci': ir['within_socket_ols']['ci_index'],
        'beta_rank': ir['within_socket_ols']['beta_rank'],
        'beta_rank_ci': ir['within_socket_ols']['ci_rank'],
        'natural_experiment': ir['natural_experiment']['diff'],
        'natural_experiment_ci': ir['natural_experiment']['rack_ci'],
        # a stationary kernel in INDEX distance, not a physical model (see thermal_ibm_layout.py)
        'index_kernel_predicts': ir['kernel_fits']['index']['natural_experiment'],
        'rank_kernel_predicts': ir['kernel_fits']['rank']['natural_experiment'],
    }
# IBM's documented layout (thermal_ibm_layout.py), where available
fi = A / 'thermal_ibm_layout.json'
if fi.exists():
    ibm = json.load(open(fi))['months']
    for ym, r in rows.items():
        if ym not in ibm:
            continue
        m = ibm[ym]
        reg = m['crossquad_regressions']
        r.update({
            'ibm_layout_score': m['pooled']['ibm'],
            'ibm_minus_best_1d_ci': m['rack_bootstrap']['ibm_minus_best_1d_ci'],
            'ibm_better_than_best_1d_frac': m['rack_bootstrap']['frac_ibm_better_than_best_1d'],
            'sameband_stringent': reg['stringent']['sameband']['beta'],
            'sameband_stringent_ci': reg['stringent']['sameband']['ci'],
            'coldist_stringent_ci': reg['stringent']['coldist']['ci'],
            'rank_slope_within_slice_pair': m['observed_diagnostics']['rank_slope_within_slice_pair'],
        })
json.dump(rows, open(A / 'thermal_month_summary.json', 'w'), indent=1)
keys = list(next(iter(rows.values())))
print(f'{"quantity":<32}' + ''.join(f'{ym:>26}' for ym in rows))
for k in keys:
    print(f'{k:<32}' + ''.join(f'{str(rows[ym][k]):>26}' for ym in rows))
print('\nwrote analysis/thermal_month_summary.json')
