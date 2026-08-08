# m100-silicon-binning — recovering per-die CPU core-harvest maps from BMC telemetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: M100 ExaData](https://img.shields.io/badge/data-M100%20ExaData-blue)](https://doi.org/10.1038/s41597-023-02174-3)
[![Reproducible](https://img.shields.io/badge/results-reproducible%20from%20public%20data-brightgreen)](#reproducing)
[![Docs](https://img.shields.io/badge/docs-mskazemi.com-informational)](https://mskazemi.com/m100-silicon-binning/)

**Processor vendors sell partially defective dies by permanently fusing off the broken units. Which
units they fused is proprietary — you are told the part has 16 cores, never *which* 16. This
repository shows that the out-of-band telemetry supercomputers already collect gives it away, and
recovers the map for all 1,962 POWER9 sockets of CINECA's Marconi100.**

The mechanism is simple: a baseboard management controller exposes one thermal sensor per
*physical* core position and does **not** renumber the survivors compactly. So the set of
`pX_coreY_temp` metrics that report **is** the die's harvest map — published inadvertently, at
fleet scale, in a public dataset.

- **What this is** — an open-source analysis pipeline and the reproducible results of applying it.
- **Who it is for** — HPC operators, systems and reliability researchers, semiconductor-yield
  researchers, and anyone who publishes or consumes fleet telemetry datasets.
- **Why you might care** — you can audit a procurement without vendor cooperation, detect processor
  replacements with no asset database, spot field core-guard events, and check whether your own
  published telemetry carries the same disclosure.
- **How to start** — [Reproducing](#reproducing) runs the headline results in minutes from the
  117 KB of derived tables committed here.

> **Not to be confused with** the architectural sense of "core harvesting" — reclaiming *idle CPU
> cycles* for co-located work. That is unrelated. This is about *silicon*: which physical cores
> were fused off at manufacture.

> **Status: unpublished preprint.** The manuscript is [`paper/paper.tex`](paper/paper.tex)
> (builds to `paper/paper.pdf`). Read [`LIMITATIONS.md`](LIMITATIONS.md) before relying on any
> result.

## What we found

Across **1,962 sockets** with a well-defined map (981 nodes), over the full 31-month record
(2020-03 to 2022-09, 1.67 M socket-days):

| Finding | Evidence |
|---|---|
| Every socket reports exactly 16 of 24 cores | 99.9847% of 1.67 M socket-days; 0 sockets ever report 24 |
| Disabling is always at slice `(2k, 2k+1)` granularity | 1,962 / 1,962, zero exceptions |
| The map is per-die, not per-SKU | 443 of 495 possible patterns observed |
| Harvested slices cluster spatially | mean index gap z ≈ −20 vs a curveball null; 8 positive-excess pairs, all at \|Δk\| ≤ 2 |
| Slice sibling cores are physically abutted | within-slice r = +0.372 vs cross-slice +0.312 at equal index gap, over all 1,948 fully-configured sockets |
| The sensor index **is** the physical die ordering | 1×12 linear fits at −0.791 vs a random-ordering null of +0.001 ± 0.121 (z = −6.55); an unconstrained search returns the same order reversed |
| Two procurement lots, boundary at rack 22 | permutation-calibrated p < 2.5e-4; bootstrap CI [22,22]; unseen racks assigned to the right lot 95.9% of the time |
| Maps are stable over time | 96.8% of 839 k held-out socket-days predicted from a map measured a year earlier |
| Configuration changes coincide with **reboots**, measured | 126/132 transitions carry a `boottime` change vs 1.74% of controls |
| Processor replacement rate, with no asset database | 2.1% per socket-year (95% CI [1.7, 2.6]) |
| The map does **not** predict replacement | out-of-fold C-index 0.448 (permutation null 0.499 ± 0.031) |
| Its power effect is bounded, not absent | +7.9 W over the feature range, 90% CI [+0.9, +16.6], against a ±5 W margin |
| The disclosure is nearly free to exploit | exact recovery from a single 20 s interval; coarser cadence is **not** a mitigation |

Two rows above reverse earlier versions of this table. The sensor-index ordering was once reported
as "not established" and the abutment contrast as +0.51 vs +0.29 — both came from a three-day
sample that inverted the first conclusion and overstated the second threefold. That, and three
other self-corrections, are documented rather than quietly fixed; see
[`LIMITATIONS.md`](LIMITATIONS.md).

## Known limitations

Read these before relying on any result. The full register, with what would settle each item, is
[`LIMITATIONS.md`](LIMITATIONS.md).

- **Single system** (OI-10). Every number comes from one machine, Marconi100. Whether the lot
  structure or the clustering generalises to other POWER9 installations, or to other vendors, is
  untested. This is the largest remaining weakness.
- **The lot boundary has no external confirmation** (OI-9). The changepoint at rack 22 is
  statistically sharp, but no procurement or delivery record was available to check it against.
  "Procurement lot" remains the most parsimonious explanation, not a documented fact.
- **Defect-driven vs policy-driven harvesting is not identified** (OI-3). A correlated defect
  process and a binning *policy* that fuses contiguous blocks both predict the clustering we
  measure, and the fitted model cannot attribute it. That model is also demonstrably incomplete.
- **The map's effect on power is bounded, not negligible** (OI-13). An earlier version called it
  "operationally invisible"; that was withdrawn.
- **The p0/p1 socket labels occasionally swap** (OI-8) between collection periods.
- **The pooled thermal estimator is noisy** (OI-11). Some reported bands are spreads across core
  pairs, not bootstraps over sockets; the figures say which.

## Does this apply to my fleet?

The method needs one thing: a management controller that exposes **one sensor per physical core
position** and does not renumber survivors. Check by looking at whether every node reports the same
compact set of core indices.

- If every socket reports cores `0..15`, your BMC renumbers — the channel is closed, nothing to
  recover.
- If sockets report *arbitrary* 16-element subsets of `0..23`, the channel is open and the method
  applies directly.

For a fleet with 24-core dies selling as 16-core parts, that is 495 possible patterns; we observed
443 of them. See [`docs/faq.md`](docs/faq.md).

## Reproducing

Every number in the tables above is reproducible from data committed here — no 49.9 TB download
required. Set up once:

```bash
python3 -m venv .venv
.venv/bin/pip install pandas pyarrow numpy scipy matplotlib
```

**Two derived populations, and it matters which you run.** `analysis/counts_*.parquet` (117 KB) is
a *single-month snapshot*; `daily/` (16 MB, 31 files) is the *full 31-month record* the paper
reports. They differ slightly, and the paper says why: the snapshot is representative, but not
identical.

Single-month snapshot — fast, and shows the mechanical finding:

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

Full record — the procurement-lot result, calibrated against a permutation null:

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

Other entry points, all running off committed tables:

| Script | Reproduces |
|---|---|
| `analysis/reboot_evidence.py` | 126/132 transitions coincide with a real `boottime` change |
| `analysis/survival.py` | the reliability null (C-index 0.448) |
| `analysis/equivalence.py` | the TOST power analysis (slow — 400 rack-clustered bootstraps) |
| `analysis/side_channel.py` | recovery cost, and which mitigations work |
| `analysis/ppc.py` | posterior-predictive checks of the clustering model |
| `analysis/layout_hypotheses.py` | die-floorplan scoring (set `THERM_MATRIX=thermal_corr_allnodes.npy`) |

Only the raw thermal series are not committed — they are ~400 GB. To rebuild any derived table
from scratch, download the M100 ExaData archives
([Zenodo](https://doi.org/10.5281/zenodo.7588815), one tar per month), then use
`analysis/core_matrix.py`, `analysis/sweep_all_months.py` or `analysis/sweep_covariates.py`.

## Layout

```
paper/paper.tex               the manuscript
LIMITATIONS.md                open issues, with what would settle each
analysis/
  core_matrix.py              build (node, socket, core) sample counts from raw tars
  binning_stats.py            marginals, chi-square, curveball null, co-disable z-scores
  changepoint.py              permutation-calibrated max-t, bootstrap CI, held-out lot prediction
  thermal_all_nodes.py        residual thermal correlation, fleet-wide
  layout_hypotheses.py        scoring candidate die floorplans against thermal coupling
  sweep_covariates.py         one pass over the raw tars for boottime + workload covariates
  reboot_evidence.py          transitions against an independent boot signal
  survival.py                 harvest map -> field replacement (a null result)
  equivalence.py              TOST on socket power, covariate-adjusted, rack-clustered
  side_channel.py             recovery cost, and which mitigations actually work
  ppc.py                      posterior-predictive checks of the pairwise-interaction model
  counts_*.parquet            derived count tables
```

## How this differs from adjacent work

- **PUF and hardware fingerprinting** answer *which die is this*, for traceability and counterfeit
  detection, using designed-in circuitry or physical access. We answer *what did the vendor disable
  on it*, from telemetry alone. A harvest map is a partial identifier: only 4.9% of sockets have
  one unique in the fleet.
- **Manufacturing-variability studies in HPC** characterise the die by its *behaviour* — frequency
  and power spread under a cap, turbo variation. They treat the die as a black box; we recover its
  internal structure.
- **Yield modelling** describes clustered killer defects directly, but is validated on fab data
  unavailable to system owners. This is the same statistics, measured from the field.
- **Node-level anomaly detection** establishes *that* nodes differ without recovering the hardware
  structure responsible.

## Citing

If you use this software or its findings, cite the manuscript. `CITATION.cff` is machine-readable
and GitHub renders a "Cite this repository" button from it. The manuscript is currently an
unpublished preprint — check back for the published reference.

## Data source

Borghesi, A., Di Santi, C., Molan, M., Ardebili, M.S., Mauri, A., Guarrasi, M., Galetti, D.,
Cestari, M., Barchi, F., Benini, L., Beneventi, F., Bartolini, A.
"M100 ExaData: a data collection campaign on the CINECA's Marconi100 Tier-0 supercomputer."
*Scientific Data* **10**, 288 (2023). https://doi.org/10.1038/s41597-023-02174-3

## License

Code is [MIT](LICENSE). The manuscript text and figures are CC BY 4.0, matching the licence of the
underlying M100 ExaData release.
