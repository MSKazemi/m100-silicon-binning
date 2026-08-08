# M100 Silicon Binning Study

Recovering per-die **core-harvesting maps** for CINECA's Marconi100 (M100) from out-of-band
BMC/IPMI telemetry in the public [M100 ExaData](https://doi.org/10.1038/s41597-023-02174-3)
release.

> **Status: unpublished working draft.** The paper is [`paper/paper.tex`](paper/paper.tex)
> (compiles to `paper/paper.pdf`). [`PAPER.md`](PAPER.md) is the earlier markdown draft, partly
> superseded.

### Known limitations

Read these before relying on any result here. The full register, with what would settle each item,
is [`LIMITATIONS.md`](LIMITATIONS.md).

- **Single system** (OI-10). Every number comes from one machine, Marconi100. Whether the lot
  structure or the clustering generalises to other POWER9 installations, or to other vendors, is
  untested. This is the largest remaining weakness.
- **The lot boundary has no external confirmation** (OI-9). The changepoint at rack 22 is
  statistically sharp (permutation-calibrated *p* < 2.5e-4, bootstrap CI [22,22]), but no
  procurement or delivery record was available to check it against. "Procurement lot" remains the
  most parsimonious explanation, not a documented fact.
- **Defect-driven vs policy-driven harvesting is not identified** (OI-3). A correlated defect
  process and a binning *policy* that fuses contiguous blocks both predict the clustering we
  measure, and the fitted model cannot attribute it. That model is also demonstrably incomplete: a
  single position-independent adjacency term misplaces adjacency across the die.
- **The map's effect on power is bounded, not negligible** (OI-13). An earlier version called it
  "operationally invisible"; that was withdrawn. Over the full record the strongest feature moves
  socket power +7.9 W across its range, 90% CI [+0.9, +16.6] against a ±5 W margin — comparable to
  which slot a processor occupies, and not bounded below relevance.
- **The p0/p1 socket labels occasionally swap** (OI-8) between collection periods. Handled by
  treating the node as the unit where it matters; 12 of 15 resolvable events look collector-side,
  but the mechanism is unconfirmed.
- **The pooled thermal estimator is noisy** (OI-11). Reported bands are spreads across core pairs,
  not bootstraps over sockets, wherever the per-pair samples were not retained; figures say which.

## Summary of findings

Marconi100 nodes are IBM AC922 8335-GTG with 2 × 16-core POWER9. A 16-core POWER9 is a **24-core
die with 4 of its 12 slices fused off**. The BMC exposes a sensor per *physical* core position, so
the set of reporting `pX_coreY_temp` metrics reveals which slices were harvested on each die.

Across **1,962 sockets** with a well-defined map (981 nodes), over the full 31-month record
(2020-03 to 2022-09, 1.67 M socket-days):

| Finding | Evidence |
|---|---|
| Every socket reports exactly 16 of 24 cores | 99.9847% of 1.67 M socket-days; 0 sockets ever report 24 |
| Disabling is always at slice `(2k, 2k+1)` granularity | 1,962 / 1,962, zero exceptions |
| Map is per-die, not per-SKU | 443 of 495 possible patterns observed |
| Harvested slices cluster spatially | mean index gap z ≈ −20 vs curveball null; 8 positive-excess pairs, all at \|Δk\| ≤ 2 |
| Slice sibling cores are physically abutted | within-slice r = +0.372 vs cross-slice +0.312 at equal index gap, over all 1,948 fully-configured sockets |
| Sensor index **is** the physical die ordering | 1×12 linear fits at −0.791 vs a random-ordering null of +0.001 ± 0.121 (z = −6.55); an unconstrained search returns the same order reversed |
| Two procurement lots, boundary at rack 22 | permutation-calibrated p < 2.5e-4; bootstrap CI [22,22]; unseen racks assigned to the right lot 95.9% of the time |
| Maps are stable over time | 96.8% of 839 k held-out socket-days predicted from a map measured a year earlier |
| Changes coincide with **reboots**, measured | 126/132 transitions carry a `boottime` change vs 1.74% of controls |
| Map does **not** predict replacement | out-of-fold C-index 0.448 (null 0.499 ± 0.031) |
| Map's power effect is bounded, not absent | +7.9 W over the feature range, 90% CI [+0.9, +16.6], vs a ±5 W margin |
| The disclosure is nearly free | exact recovery from one 20 s interval; coarser cadence is not a mitigation |

Two rows above reverse earlier versions of this table. The sensor-index ordering was once reported
as "not established" and the abutment contrast as +0.51 vs +0.29 — both came from a three-day
sample that inverted the first conclusion and overstated the second threefold. See
[`LIMITATIONS.md`](LIMITATIONS.md) (OI-1).

## Layout

```
PAPER.md                     working draft
analysis/
  core_matrix.py             build (node, socket, core) sample counts from raw tars
  core_pattern.py            pair structure, pattern diversity, cross-month stability
  binning_stats.py           marginals, chi-square, curveball null, co-disable z-scores
  thermal_adjacency.py       residual thermal correlation vs core index distance
  rack_lot.py                rack/row clustering, permutation tests, lot changepoint
  counts_20-04.parquet       derived count table (2020-04)
  counts_21-03.parquet       derived count table (2021-03)

  # covariate sweep and the analyses that depend on it
  sweep_covariates.py        one pass over the raw tars for boottime + workload covariates
  reboot_evidence.py         transitions vs an independent boot signal
  changepoint.py             permutation-calibrated max-t, bootstrap CI, k-selection, held-out lot
  survival.py                harvest map -> field replacement (a null)
  equivalence.py             TOST on socket power, covariate-adjusted, rack-clustered
  side_channel.py            recovery cost and which mitigations work
  ppc.py                     posterior-predictive checks of the pairwise-interaction model
  exposure_rates.py          rates on matched exposure windows
```

## Reproducing

The two `counts_*.parquet` tables (117 KB total) are sufficient for every result except the
thermal-correlation section, which needs the raw temperature series.

```bash
uv run --with pandas --with pyarrow --with numpy python3 analysis/binning_stats.py
uv run --with pandas --with pyarrow --with numpy python3 analysis/rack_lot.py
```

To regenerate the count tables from scratch, download the M100 ExaData raw archives
([Zenodo](https://doi.org/10.5281/zenodo.7588815), one tar per month), extract the
`plugin=ipmi_pub/metric=p[01]_core*_temp` partitions, then:

```bash
python3 analysis/core_matrix.py <extracted_dir> 20-04
```

## Data source

Borghesi, A., Di Santi, C., Molan, M., Ardebili, M.S., Mauri, A., Guarrasi, M., Galetti, D.,
Cestari, M., Barchi, F., Benini, L., Beneventi, F., Bartolini, A.
"M100 ExaData: a data collection campaign on the CINECA's Marconi100 Tier-0 supercomputer."
*Scientific Data* **10**, 288 (2023). https://doi.org/10.1038/s41597-023-02174-3
