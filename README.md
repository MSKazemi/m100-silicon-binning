# M100 Silicon Binning Study

Recovering per-die **core-harvesting maps** for CINECA's Marconi100 (M100) from out-of-band
BMC/IPMI telemetry in the public [M100 ExaData](https://doi.org/10.1038/s41597-023-02174-3)
release.

> **Status: unpublished working draft.** The paper is [`paper/paper.tex`](paper/paper.tex)
> (compiles to `paper/paper.pdf`). [`PAPER.md`](PAPER.md) is the earlier markdown draft, partly
> superseded.

### Known limitations

Read these before relying on any result here.

- **Single system.** Every number comes from one machine, Marconi100. Whether the lot structure or
  the clustering generalises to other POWER9 installations, or to other vendors, is untested.
- **The lot boundary has no external confirmation.** The changepoint at node 439 is statistically
  sharp (permutation-calibrated *p* < 2.5e-4, bootstrap CI [22,22] in rack terms), but no
  procurement or delivery record was available to check it against. "Procurement lot" remains the
  most parsimonious explanation, not a documented fact.
- **Defect-driven vs policy-driven harvesting is bounded, not settled.** The observed clustering is
  consistent with defect statistics, but a vendor binning *policy* that happens to disable adjacent
  slices is not excluded.
- **The p0/p1 socket labels occasionally swap** between collection periods. Handled by treating the
  node as the unit where it matters, but the underlying cause in the collection stack is unknown.
- **The pooled thermal estimator is noisy.** Reported bands are spreads across core pairs, not
  bootstraps over sockets, wherever the per-pair samples were not retained; figures say which.

## Summary of findings

Marconi100 nodes are IBM AC922 8335-GTG with 2 × 16-core POWER9. A 16-core POWER9 is a **24-core
die with 4 of its 12 slices fused off**. The BMC exposes a sensor per *physical* core position, so
the set of reporting `pX_coreY_temp` metrics reveals which slices were harvested on each die.

Across **1,960 sockets** (980 nodes, 2020-04 and 2021-03):

| Finding | Evidence |
|---|---|
| Every socket reports exactly 16 of 24 cores | 1,960 / 1,960 |
| Disabling is always at slice `(2k, 2k+1)` granularity | 1,960 / 1,960, zero exceptions |
| Map is per-die, not per-SKU | 442 of 495 possible patterns observed |
| Disabled slices cluster spatially (defect signature) | mean index gap z = −20.8 vs curveball null |
| Slice sibling cores are physically abutted | within-slice r = +0.51 vs cross-slice r = +0.29 at equal index gap, 40/40 nodes |
| Sensor index → physical die position **not established** | linear ordering only p = 0.025 vs random; a 3×4 grid fits better |
| Two procurement lots, boundary at rack 22 | permutation-calibrated p < 2.5e-4; bootstrap CI [22,22]; unseen racks assigned to the right lot 95.9% of the time |
| Maps are stable over time | 96.7% identical after 11 months; changes are hardware swaps |
| Changes coincide with **reboots**, measured | 126/132 transitions carry a `boottime` change vs 1.74% of controls |
| Map does **not** predict replacement | out-of-fold C-index 0.448 (null 0.499 ± 0.031) |
| Map's power effect is bounded, not absent | +7.9 W over the feature range, 90% CI [+0.9, +16.6], vs a ±5 W margin |
| The disclosure is nearly free | exact recovery from one 20 s interval; coarser cadence is not a mitigation |

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
