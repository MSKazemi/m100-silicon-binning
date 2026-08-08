---
title: "Reproduce — run the core-harvest map analysis yourself"
description: "Every headline number is reproducible from data committed in the repository. Setup, the two derived populations, expected output, and where each result comes from."
---

# Reproduce

Every number reported here is reproducible from data committed in the repository. **No 49.9 TB
download is required.**

## Setup

```bash
git clone https://github.com/MSKazemi/m100-silicon-binning
cd m100-silicon-binning
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Two derived populations — it matters which you run

- `analysis/counts_*.parquet` (117 KB) — a **single-month snapshot**
- `daily/` (16 MB, 31 files) — the **full 31-month record** the paper reports

They differ slightly. The paper's own stability analysis is why: across 31 months the largest
standard deviation of any slice's monthly rate is 0.25 percentage points, so the snapshot is
representative — but it is not identical, and we would rather say so than have you find it.

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
mean index-gap between disabled pairs: observed 4.243, null 4.585 +/- 0.016  -> z = -20.81
```

## The procurement-lot result

```bash
.venv/bin/python analysis/changepoint.py
```

```
sockets 1,960   racks 49   candidate boundaries 3..46
observed argmax boundary : rack 22   |t| = 26.6
rack-permutation null    : max|t| = 7.08 +/- 2.41, max seen 17.95
calibrated p             : < 2.50e-04
breakpoint rack = 22   95% CI [22, 22]   (100% of 2,000 resamples pick it exactly)
socket-level accuracy : 72.9%      rack-level accuracy : 95.9%  (47/49 racks)
```

## Everything else

| Script | Reproduces |
|---|---|
| `analysis/reboot_evidence.py` | 126/132 transitions coincide with a real `boottime` change |
| `analysis/survival.py` | the reliability null (C-index 0.448) |
| `analysis/equivalence.py` | the TOST power analysis — slow, 400 rack-clustered bootstraps |
| `analysis/side_channel.py` | recovery cost, and which mitigations work |
| `analysis/ppc.py` | posterior-predictive checks of the clustering model |
| `analysis/layout_hypotheses.py` | die-floorplan scoring (set `THERM_MATRIX=thermal_corr_allnodes.npy`) |
| `analysis/exposure_rates.py` | replacement and guard rates on matched exposure windows |

## Rebuilding from the raw archives

Only the raw per-core thermal series are not committed — roughly 400 GB. To rebuild any derived
table from scratch, download the M100 ExaData archives from
[Zenodo](https://doi.org/10.5281/zenodo.7588815) (one tar per month), then:

```bash
python3 analysis/core_matrix.py <extracted_dir> 20-04     # count tables
python3 analysis/sweep_all_months.py                      # daily aggregation
python3 analysis/sweep_covariates.py                      # boottime + workload covariates
```

A full pass is a sequential scan of about 400 GB, roughly two hours on a single node, dominated by
archive traversal rather than computation. Re-running every result from the derived tables takes
minutes. That asymmetry is why the derived tables are committed.

## A warning worth repeating

Run the sweeps **alone**. A concurrent pandas job was OOM-killed while a sweep held ~18 GB. The
sweeps are resumable — they skip months whose output already exists.

---

*Next: [FAQ](faq.md) · [Limitations](limitations.md) · [source on GitHub](https://github.com/MSKazemi/m100-silicon-binning)*
