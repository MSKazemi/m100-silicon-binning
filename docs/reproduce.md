---
title: "Reproduce — run the core-harvest map analysis yourself"
description: "Every headline number is reproducible from data committed in the repository. Setup, the two derived populations, expected output, what CI checks, and where each result and figure comes from."
---

# Reproduce

Every number reported here is reproducible from data committed in the repository. **No 49.9 TB
download is required.**

## Setup

```bash
git clone https://github.com/MSKazemi/m100-silicon-binning
cd m100-silicon-binning
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # exact environment of the results (Python 3.12)
.venv/bin/pip install -r requirements-dev.txt    # pytest
.venv/bin/python -m pytest -q tests/             # about 20 s
```

`requirements-lock.txt` pins every package of the environment that produced the paper's numbers
(Python 3.12.3, x86-64 Linux, OpenBLAS). `requirements.txt` gives version floors for development;
CI also runs against the newest versions those floors allow on Python 3.10.
[`DATA.md`](https://github.com/MSKazemi/m100-silicon-binning/blob/main/DATA.md) documents every
derived table; `DATA.sha256` checksums them (`sha256sum -c DATA.sha256`).

## Two derived populations — it matters which you run

- `analysis/counts_*.parquet` (117 KB) — a **single-month snapshot**
- `daily/` (16 MB, 31 files) — the **full 31-month record** the paper reports

They differ slightly. The paper's own stability analysis is why: across the 30 months with clean
data the largest standard deviation of any slice's monthly rate is 0.25 percentage points, so the
snapshot is representative — but it is not identical, and we would rather say so than have you find
it.

## The mechanical finding

```bash
.venv/bin/python analysis/binning_stats.py
```

```
sockets with exactly 4 disabled pairs: 1960
possible patterns C(12,4) = 495
observed distinct patterns: 442
  pair  0 (cores  0, 1):  61.7%   ##############################
  chi2 uniformity = 836.5 (df=11, crit_0.001=31.3) -> NON-uniform
mean index-gap between disabled pairs: observed 4.243, null 4.589 +/- 0.017  -> z = -20.11
```

## The lot boundary

```bash
.venv/bin/python analysis/changepoint.py
```

```
sockets 1,960   racks 49   candidate boundaries 3..46 (44 of 48, trimmed at 100 sockets per side)
  observed argmax boundary : rack 22   |t| = 26.6
  rack-permutation null    : max|t| = 7.08 +/- 2.41, 99th pct 14.41, max seen 17.95
  calibrated p             : 2.50e-04  ((b+1)/(B+1); 0/4000 null draws reach the observed max)
  breakpoint rack = 22   sockets resampled within rack: rack 22 in 100.0% of 2000 (2.5-97.5 pct [22, 22])
  fits that split the remaining racks exactly as the full-data boundary does: 49/49
  rack-level accuracy   : 95.9%  (47/49 racks)
```

## Everything else

| Script | Reproduces | Time |
|---|---|---|
| `analysis/reboot_evidence.py` | 126/132 transitions coincide with a real `boottime` change; epoch-dated 133/136; node-stratified exact test | 7 s |
| `analysis/lifecycle_os_crosscheck.py` | every OS departure from 128 logical CPUs matched by a BMC core loss (9 of 9); census of non-16 socket-days | 1 min |
| `analysis/lifecycle_taxonomy_errors.py` | taxonomy error mechanisms, dies moved between sockets, clustered rate intervals (run after the cross-check) | 5 s |
| `analysis/survival.py` | the reliability null (risk-set concordance 0.55, p = 0.12) | 4 min |
| `analysis/survival_legacy_check.py` | re-derives the first release's 0.448 and shows it was a cross-validation artefact | 1 min |
| `analysis/equivalence.py` | the TOST power analysis on contemporaneous maps, with placement and idle-day checks, 4,000 rack-clustered bootstraps | 1 min |
| `analysis/side_channel.py` | recovery cost, and which mitigations work (deletion models on the committed counts) | 7 s |
| `analysis/side_channel_raw.py` | the same measured on the raw 20 s timestamps (needs one monthly tar) | 6 min |
| `analysis/side_channel_values.py` | whether compact renumbering still leaks the map through the temperatures (needs one monthly tar) | |
| `analysis/yield_model.py`, `gamma_clustered_se.py`, `model_quad.py`, `model_marginals.py` | the pairwise-interaction model, rack-clustered uncertainty, the quad split, constrained marginal tests | seconds |
| `analysis/ppc.py` | parametric model-replicate checks of the clustering model | 10 s |
| `analysis/curveball_null.py` | pair-level co-harvest tests against 100,000 exchangeable curveball draws (`MODEL_WORKERS` sets the process count; output does not depend on it) | 12–17 min on 10–12 cores |
| `analysis/cohort_boundary.py` | rack-level boundary uncertainty and the number of populations under three likelihoods | 35 s on 24 cores |
| `analysis/layout_hypotheses.py` | slice-layout scores incl. IBM's documented layout and the exact best of all 12!/2 orderings (set `THERM_MATRIX=thermal_corr_allnodes.npy`) | 20 s |
| `analysis/thermal_socket_stats.py` | socket-level thermal inference from the committed per-socket files: rack bootstrap, quad decomposition, strata | 2 min |
| `analysis/thermal_ibm_layout.py` | thermal coupling against IBM's documented layout; physical versus logical-order tests | 80 s |
| `analysis/thermal_column_test.py` | position-matched test inside each die column: IBM geometry versus core numbering | 2.5 min |
| `analysis/exposure_rates.py` | replacement and guard rates on matched exposure windows | |
| `analysis/within_node_overlap.py` | the two sockets of a node overlap in 1.641 of 4 harvested slices against a fleet-wide reassignment null of 1.470 ± 0.026 (z = +6.6); the committed producer for Sec. IV-C | <1 s |
| `analysis/oi8_and_os_full.py` | OI-8: the p0−p1 power difference flips sign at 12 of 15 relabelling events where it exceeds 2 W on both sides; also the OS ground-truth census over all available months | 4 min |

Times are wall-clock on a 24-core workstation.

## What CI checks, and what it does not

The [`reproduce`](https://github.com/MSKazemi/m100-silicon-binning/blob/main/.github/workflows/reproduce.yml)
workflow runs on every push, every pull request and monthly:

- **structural** (Python 3.10 with the newest versions the floors allow; Python 3.12 with the exact
  lock file): the checksums of the derived tables, the full test suite, the single-month statistics
  (`binning_stats.py`), the constrained marginal test (`model_marginals.py`), the side-channel
  deletion models (`side_channel.py`), the pairwise model and its checks (`yield_model.py`,
  `gamma_clustered_se.py`, `model_quad.py`, `ppc.py`, with `sum(beta) = 0` and the optimiser's
  gradient norm asserted), the thermal layout scores from the committed matrix
  (`layout_hypotheses.py`), the within-node map overlap of Sec. IV-C (`within_node_overlap.py`), the
  committed curveball null and per-socket thermal summaries read back from their JSON, and the hero
  image;
- **longitudinal** (Python 3.12, lock file): the lot boundary (`changepoint.py`), reboots and the OS
  cross-check (`reboot_evidence.py`, `lifecycle_os_crosscheck.py`, `lifecycle_taxonomy_errors.py`),
  the power equivalence analysis (`equivalence.py`), the survival analysis and the legacy check
  (`survival.py`, `survival_legacy_check.py`), OI-8's relabelling sign test
  (`oi8_and_os_full.py`), then `tests/test_consistency.py` against the freshly written result
  files.

Each step asserts the values the paper reports. CI does **not** re-derive what needs the raw
archive: the per-socket thermal matrices (`thermal_socket_build.py`), the raw-timestamp
side-channel measurement (`side_channel_raw.py`, `side_channel_values.py`) and the SLURM core
placement (`equivalence_alloc.py`); their committed outputs are checksummed instead. It checks the
committed output of the 100,000-draw curveball null (`model_curveball.json`) rather than rerunning
it, and it does not run the slower thermal and cohort scripts in the table above. A green badge
therefore means the committed tables still produce the asserted numbers, not that every number in
the paper was recomputed from raw data.

## Figures

Each paper figure has one script; none needs the raw archive.

| Paper figure | File | Script |
|---|---|---|
| Fig. 1 | `f0_hierarchy.pdf` | `analysis/make_fig_hierarchy.py` |
| Fig. 2 | `f1_slice_schematic.pdf` | `analysis/make_fig_schematic.py` |
| Figs. 3, 4 | `f2_marginals.pdf`, `f3_cooccurrence.pdf` | `analysis/make_figures4.py` |
| Fig. 5 | `f4_thermal_decay.pdf` | `analysis/make_fig_thermal_census.py` |
| Fig. 6 | `f5_layout_hypotheses.pdf` | `analysis/layout_hypotheses.py` |
| Figs. 7, 8 | `f6_rack_heatmap.pdf`, `f7_pattern_rank.pdf` | `analysis/make_fig_lots.py` |
| Figs. 9, 10, 11 | `f12_changepoint.pdf`, `f13_model_check.pdf`, `f14_null_results.pdf` | `analysis/make_figures3.py` |
| Fig. 12 | `f8_month_stability.pdf` | `analysis/make_figures2.py` |
| Fig. 13 | `f9_guard_episodes.pdf` | `analysis/make_fig_guard_v2.py` |
| Fig. 14 | `f10_transition_dates.pdf` | `analysis/make_figures2.py f10` |
| Fig. 15 | `f11_operational_impact.pdf` | `analysis/make_fig_equivalence.py` |

Do not run `analysis/make_figures.py` as a whole: it is kept for the history of the thermal
subsample, and without the raw data it cannot rebuild what it once drew.

## Rebuilding from the raw archives

Only the raw per-core thermal series are not committed — roughly 400 GB. Their per-socket residual
matrices are (`analysis/thermal_socket_*`, built by `thermal_socket_build.py`), so every thermal
number re-derives without them. To rebuild any derived table from scratch, download the M100
ExaData archives from twelve Zenodo records (v1.0.0, one tar per month):
[7588815](https://doi.org/10.5281/zenodo.7588815) for 2020, then 7589131, 7589320, 7589630,
7589942, 7590061, 7590308, 7590547, 7590555, 7590565, 7590574 and 7590583 for 2021-01 to 2022-09.
Then:

```bash
python3 analysis/core_matrix.py <extracted_dir> 20-04     # count tables
python3 analysis/sweep_all_months.py                      # daily aggregation
python3 analysis/sweep_covariates.py                      # boottime + workload covariates
THERM_MONTH=22-08 python3 analysis/thermal_socket_build.py   # per-socket thermal matrices
```

A full pass is a sequential scan of about 400 GB, roughly two hours on a single node, dominated by
archive traversal rather than computation. Re-running every result from the derived tables takes
minutes. That asymmetry is why the derived tables are committed.

## A warning worth repeating

Run the sweeps **alone**. A concurrent pandas job was OOM-killed while a sweep held ~18 GB. The
sweeps are resumable — they skip months whose output already exists.

---

*Next: [FAQ](faq.md) · [Limitations](limitations.md) · [source on GitHub](https://github.com/MSKazemi/m100-silicon-binning)*
