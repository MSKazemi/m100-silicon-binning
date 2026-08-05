# Fleet-Scale Inference of Silicon Core-Harvesting Patterns from Out-of-Band Telemetry

> **SUPERSEDED — see `paper/paper.tex` (LaTeX, canonical).**
>
> **§4.5 of this file is WRONG.** It claimed the sensor index is monotonic in physical die
> position. A later slice-level test of explicit floorplan hypotheses
> (`analysis/layout_hypotheses.py`) shows the linear ordering beats a random ordering only
> marginally (p = 0.025) and is outperformed by a 3×4 grid. What survives is the narrower claim
> that the *two cores of one slice* are physically abutted. Consequently the *spatial* reading of
> the clustering in §4.4 is a hypothesis, not a result — the index-space clustering itself stands.

**Working draft — v0.1**
Status: results verified on two months (2020-04, 2021-03) of M100 ExaData. Sections marked
⚠️ require additional work before submission.

---

## Abstract (draft)

Processor vendors sell partially-defective dies by permanently disabling faulty units — *core
harvesting*. The resulting per-die disable maps are not published, and are conventionally
invisible to system owners. We show that the out-of-band (BMC/IPMI) sensor telemetry routinely
collected by production supercomputers exposes these maps directly, and we recover them for all
1,960 processor sockets of CINECA's Marconi100 (M100) Tier-0 system from the public M100 ExaData
release. Every socket exposes exactly 16 of 24 possible per-core thermal sensors, and the eight
absent sensors always form four complete *slice* pairs `(2k, 2k+1)` — the POWER9 unit that shares
an L2/L3 block — with zero exceptions across 1,960 sockets. We observe 442 of the 495
combinatorially possible disable patterns, establishing that the map is per-die rather than a
fixed SKU. Using a curveball null model that preserves both per-socket and per-slice marginals,
we show that disabled slices cluster in index space (z = −20.8 for mean index gap), the signature
of spatially-correlated manufacturing defects. We independently validate that the sensor index is
monotonic in physical die position by measuring common-mode-removed thermal correlation between
cores, which decays from r = +0.43 at adjacent indices to r = −0.61 at maximal separation
(40/40 nodes, paired t = 13.9), and which distinguishes within-slice sibling cores (r = +0.54)
from cross-slice neighbours at identical index distance (r = +0.28). Finally, we detect a sharp
procurement-lot boundary at rack 22 (node 440; Welch t = 27.3) separating two populations with
markedly different harvesting statistics. ⚠️ *Implications section to be written.*

---

## 1. Introduction

⚠️ *To be written.* Framing points to develop:

- Core harvesting / binning is standard industry practice but the resulting per-die maps are
  proprietary and undocumented.
- HPC operators increasingly collect holistic out-of-band telemetry [1]. We show this telemetry
  carries an unintended side channel revealing manufacturing outcomes.
- Contribution: (i) a method to recover per-die disable maps from BMC sensor presence alone;
  (ii) the first fleet-scale empirical characterisation of harvesting patterns in a deployed
  Top-10 supercomputer; (iii) evidence of defect clustering and of procurement-lot structure;
  (iv) a thermal-correlation method to recover physical die topology from telemetry.

## 2. Background

### 2.1 POWER9 slice organisation

The POWER9 die used in the IBM Power System AC922 contains 24 SMT4 cores (equivalently 12 SMT8
cores) on a 695 mm² die (25.2 × 27.2 mm), fabricated in GlobalFoundries 14 nm FinFET with 17
metal layers [2,3]. Critically for this work, **each pair of SMT4 cores — or one singleton SMT8
core — comprises a *slice*, and each slice contains its own 512 KB L2 and 10 MB L3 cache** [2].
The slice is therefore the natural granularity for fusing: disabling a single SMT4 core would
strand its half of a shared L2/L3 block.

Marconi100 nodes are AC922 model 8335-GTG, with 2 × 16-core POWER9 at 2.6 GHz nominal / 3.1 GHz
turbo and 4 × NVIDIA V100 GPUs [4,5]. **A 16-core part is a 24-core die with four slices fused
off.** This is the object of our study.

### 2.2 The M100 ExaData dataset

M100 ExaData [1] publishes 49.9 TB of holistic telemetry from Marconi100 covering 2020-03 to
2022-09 across 980+ compute nodes. The IPMI plugin collects BMC sensor data at a 20 s cadence per
node. The dataset documentation describes the core-temperature metric family as
`pX_coreY_temp`, "Temperature of core n. Y in the CPU socket n. X. X=0..1, Y=0..23" [6].

**This description specifies the BMC sensor-ID namespace, not the active core count.** The
distinction is the entry point for this study.

## 3. Method

### 3.1 Recovering the disable map

For each month we enumerate the `plugin=ipmi_pub/metric=pX_coreY_temp` partitions and count
samples per `(node, socket, core)`. A core index is deemed **present** for a socket if it emits
any sample in the observation window. We define the disable map of socket *s* as the binary
vector *M*[*s*, *k*] ∈ {0,1}<sup>12</sup> over slices *k* = ⌊core/2⌋.

The critical validity argument is that **the BMC does not renumber active cores compactly**. If
it did, every socket would report indices 0–15. Instead we observe arbitrary 16-element subsets
of 0–23, so the index is a fixed physical/fused position identifier.

### 3.2 Null model for clustering

Because every socket has exactly four disabled slices, the constraint induces negative dependence
between slices; a naive independence null is invalid. We use the **curveball algorithm**, the
standard null for binary presence–absence matrices, which preserves *both* row sums (4 disabled
per socket) *and* column sums (per-slice marginals) while randomising the association structure.
Row- and column-sum preservation is asserted on every one of the 200 draws.

### 3.3 Recovering physical topology from thermal correlation

Cores on one socket share workload and inlet air, producing a dominant common mode. We remove it
by subtracting the per-timestamp cross-core mean, then double-centre and compute the residual
correlation matrix. Under a thermal-diffusion argument, physically adjacent cores retain positive
residual coupling; distant cores do not. This yields an *independent* test of whether the sensor
index is monotonic in physical position.

⚠️ *Caveat to state explicitly in the paper:* common-mode removal forces each row of the residual
correlation matrix to sum to approximately zero, so negative values at large gaps are partly
mechanical. The robust findings are (a) the monotone decay and (b) the within-slice vs cross-slice
contrast at *identical* index distance, which controls for gap exactly.

## 4. Results

### 4.1 Every socket is 16 cores; the disable granularity is the slice

| Observation | Value |
|---|---|
| Nodes analysed (2020-04) | 980 |
| Sockets | 1,960 |
| Sockets reporting exactly 16 cores | 1,960 (100%) |
| Sockets reporting 24 cores | 0 |
| Sockets whose absent cores form complete `(2k, 2k+1)` pairs | **1,960 / 1,960 (100%)** |

The perfect pair structure is the central mechanical finding: it matches the POWER9 slice
definition [2] exactly and rules out per-core fusing.

### 4.2 The map is per-die, not per-SKU

442 distinct disable patterns are observed among the 495 = C(12,4) possible. The most common
pattern (slices 0,1,2,3) accounts for only 5.7% of sockets. Within a node, the two sockets have
identical maps in only 1.2% of cases, with mean overlap 1.648 slices against a shuffled null of
1.474 ± 0.026 (z = +6.6) — nearly independent, with a small excess attributable to same-lot pairing.

### 4.3 Marginals are strongly non-uniform

| Slice | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(disabled) % | **61.7** | **49.4** | **39.9** | 31.1 | 28.5 | 25.6 | 26.8 | 26.2 | 29.8 | 28.5 | 25.8 | 26.6 |

Uniform expectation is 4/12 = 33.3%; χ² = 836.5 (df = 11, p ≪ 0.001). Low-index slices are
preferentially harvested.

### 4.4 Disabled slices cluster — the defect-clustering signature

Under the curveball null:

- Mean index gap between disabled slices: **observed 4.243 vs null 4.585 ± 0.016, z = −20.8**.
- Co-disable excess is confined to index-adjacent slice pairs:

| Slice pair | Observed | Null | z |
|---|---|---|---|
| 0 & 1 | 722 | 547.9 | **+19.5** |
| 6 & 7 | 183 | 107.6 | +9.5 |
| 8 & 9 | 201 | 131.2 | +8.7 |
| 2 & 3 | 280 | 199.8 | +8.4 |
| 4 & 5 | 172 | 111.0 | +7.9 |
| 1 & 2 | 419 | 335.1 | +7.4 |
| 10 & 11 | 158 | 104.3 | +7.1 |

Every strongly positive term has |Δk| ≤ 2; all negative terms are distant pairs. This is the
classic signature of spatially-correlated killer defects underlying negative-binomial yield
models [7,8].

### 4.5 The sensor index is monotonic in physical position

Residual thermal correlation vs index distance (40 nodes, socket p0, 16 active cores each):

| Gap | 1 | 2 | 3 | 4 | 8 | 12 | 16 | 20 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| r | **+0.434** | +0.095 | −0.084 | −0.204 | +0.143 | −0.069 | +0.050 | −0.292 | **−0.608** |

Per-node contrast near (gap ≤ 2) minus far (gap ≥ 12): +0.453 ± 0.204; **40/40 nodes positive**,
paired t = +13.9.

Decisively, at *identical* index distance of 1:

- within-slice sibling cores (2k, 2k+1), sharing L2/L3: **r = +0.539** (n = 320)
- cross-slice neighbours (2k+1, 2k+2): **r = +0.275** (n = 213)

The two cores of a slice are thermally twice as coupled as equally-numbered neighbours across a
slice boundary — confirming both physical abutment and the slice as a physical unit.

### 4.6 A procurement-lot boundary at node 440

Node IDs map to physical position as rack = ⌊id/20⌋, slot = id mod 20 [9]. A changepoint scan
over rack index maximises Welch t at **rack 22 (node 440), t = 27.3**:

| Population | Nodes | P(slice 0 disabled) | Distinct patterns |
|---|---|---|---|
| Lot A (racks 0–21) | 880 sockets | **89.1%** | 224 |
| Lot B (racks 22–48) | 1,080 sockets | **39.4%** | 414 |

Lot A marginals are steeply decreasing in slice index (89, 65, 46, 30, 21, 19, 21, 21, 25, 21,
20, 22 %); **Lot B marginals are nearly flat (39, 37, 35, 32, 35, 31, 31, 30, 34, 34, 31, 31 %)**.
Permutation tests confirm between-rack heterogeneity for slices 0, 1, 2 and 5 (p < 0.01).
Same-rack sockets share more disabled slices than chance (overlap 1.568 vs null 1.454 ± 0.008,
z = +14.1).

### 4.7 Temporal stability

Comparing 2020-04 with 2021-03 (11 months), 96.7% of sockets have byte-identical active-core
sets. All 65 changed sockets are consistent with hardware replacement: for node 115 socket 0,
slices reporting until 2021-03-03 stop, and a different set appears on 2021-03-30, each set
containing exactly 16 cores.

## 5. Discussion

⚠️ *To be developed.* Two competing explanations for the low-index bias (§4.3, §4.6) must be
separated honestly:

- **H1 — defect-driven.** Harvesting follows wherever killer defects land. Supported by the
  adjacency clustering (§4.4) and the near-uniform Lot B marginals.
- **H2 — policy-driven.** Slice 0 sits adjacent to some shared structure (memory controller,
  on-chip interconnect) making it thermally or electrically disadvantaged, and vendors fuse it
  preferentially. Supported by the extreme Lot A slice-0 rate (89%), hard to explain by defects
  alone.

The lot difference suggests a **change in binning recipe or die source between the two
procurement batches**, which favours H2 for Lot A and H1 for Lot B. Distinguishing them would
require either vendor disclosure or comparison against a second AC922 installation.

⚠️ Practical implications to develop: (i) per-node core count and identity must be discovered,
not assumed, in any telemetry pipeline; (ii) harvesting maps are a confounder for
node-to-node performance-variability studies; (iii) the side channel has procurement-audit value.

## 6. Threats to validity

1. **Sensor presence ≠ core existence.** A slice could be physically present but its sensor
   unreported. Mitigated by: perfect 16/16 consistency across 1,960 sockets and 2 months, exact
   agreement with the published 16-core SKU [4], and perfect slice-pair structure.
2. **Two months only.** ⚠️ Extend to all 31 months to complete the swap timeline (~400 GB read).
3. **Index-to-physical mapping** is established statistically (§4.5), not from a vendor floorplan.
   ⚠️ Obtain an annotated POWER9 die shot to confirm the linear ordering.
4. **Common-mode removal artefact** in §4.5, addressed by the within/cross-slice control.
5. **Single system.** Findings may reflect CINECA's specific procurement, not POWER9 generally.

## 7. Reproducibility

All analysis code in `analysis/`; inputs are the public M100 ExaData Zenodo releases [1].

| Script | Purpose |
|---|---|
| `core_matrix.py` | Build (node, socket, core) sample counts from raw tars |
| `core_pattern.py` | Pair structure, pattern diversity, cross-month stability |
| `binning_stats.py` | Marginals, χ², curveball null, co-disable z-scores |
| `thermal_adjacency.py` | Residual thermal correlation vs index distance |
| `rack_lot.py` | Rack/row clustering, permutation tests, lot changepoint |

`counts_20-04.parquet` and `counts_21-03.parquet` are the derived count tables (117 KB total),
sufficient to reproduce every result except §4.5, which needs the raw temperature series.

---

## References

Verified during this study:

[1] Borghesi, A., Di Santi, C., Molan, M., Ardebili, M.S., Mauri, A., Guarrasi, M., Galetti, D.,
Cestari, M., Barchi, F., Benini, L., Beneventi, F., Bartolini, A. "M100 ExaData: a data collection
campaign on the CINECA's Marconi100 Tier-0 supercomputer." *Scientific Data* **10**, 288 (2023).
https://doi.org/10.1038/s41597-023-02174-3
*(Note: you are a co-author — this is a self-citation and the natural companion paper.)*

[2] WikiChip, "POWER9 — Microarchitectures — IBM."
https://en.wikichip.org/wiki/ibm/microarchitectures/power9
— slice definition (core pair + 512 KB L2 + 10 MB L3), 24 SMT4 / 12 SMT8, die 695 mm², 14 nm.

[3] Wikipedia, "POWER9." https://en.wikipedia.org/wiki/POWER9

[4] IBM, "IBM Power System AC922 Data Sheet."
https://covenco.com/wp-content/uploads/2023/01/AC922-Data-Sheet.pdf

[5] Baccarelli, I., "Introduction to the CINECA Marconi100 HPC system."
https://indico.euro-fusion.org/event/341/attachments/293/664/M100.pdf
— 8335-GTG, 2 × 16-core POWER9 @ 2.6/3.1 GHz, 4 × V100.

[6] M100 ExaData repository, `documentation/plugins/ipmi.md` (local: `/home/mohsen/exadata`).

[9] M100 ExaData repository, `documentation/racks_spatial_distribution.md` — rack/node mapping.

⚠️ **Require verification before submission** (found via search, full text not yet read):

[7] Stapper, C.H. "A unified negative-binomial distribution for yield analysis of defect-tolerant
circuits." *IEEE Trans. Semiconductor Manufacturing* (1993). — confirm authors, volume, pages.

[8] Yield prediction via spatial modeling of clustered defect counts across a wafer map.
*IIE Transactions*, doi:10.1080/07408170701275335. — confirm authors and year.

⚠️ **Still to find:**

- An annotated POWER9 die shot establishing the physical ordering of the 12 slices.
  Lead: https://happytrees.org/dieshots/IBM_-_POWER9_(SMT8_core)_layout
- Prior work on inferring hardware provenance/binning from system telemetry (novelty check).
- OpenPOWER/OpenBMC documentation on how the BMC assigns core sensor IDs (would settle §3.1
  and threat 1 definitively).
- Literature on node-level performance variability in HPC attributable to manufacturing.
