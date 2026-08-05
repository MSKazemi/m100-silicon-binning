# M100 Silicon Binning Study

Recovering per-die **core-harvesting maps** for CINECA's Marconi100 (M100) from out-of-band
BMC/IPMI telemetry in the public [M100 ExaData](https://doi.org/10.1038/s41597-023-02174-3)
release.

> **Status: unpublished working draft.** See [`PAPER.md`](PAPER.md). Sections and references
> marked ⚠️ are not yet verified.

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
| Two procurement lots, boundary at node 440 | Welch t = 27.3; P(slice 0 fused) 89.1% vs 39.4% |
| Maps are stable over time | 96.7% identical after 11 months; changes are hardware swaps |

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
