# Limitations and open issues

The register the paper's "Threats to validity" section refers to. Each entry gives what is known,
what would settle it, and where to look. Identifiers (`OI-n`) are cited from `paper/paper.tex`, so
the paper and this file stay in sync. Identifiers are never reused; a gap in the numbering is an
item that does not concern the results.

Status: `RESOLVED` · `BOUNDED` — the ambiguity exists but no result depends on it · `OPEN` — would
need evidence we do not have.

The entries were revised after an external review in 2026-09; `CHANGELOG.md` lists what changed.

---

## OI-1 — Is the IPMI core index the physical die position? `RESOLVED — by IBM's documentation; thermal data confirm the quads only`

The clustering result is measured in *sensor-index* space. Reading it as *spatial* clustering
requires knowing where each index sits on the die.

**IBM documents it.** Each index is a fixed core position, and IBM's POWER9 Processor Registers
Specification, Vol. 1, v1.2 (29 Mar 2021), Fig. 2, p. 33 labels the core chiplets EC00–EC23 and
places the six quads in two bands of three (EQ0/2/4 top, EQ1/3/5 bottom; left, middle, right),
each quad's four cores stacked in one column. Firmware maps core N to chiplet 0x20+N and quad
N>>2 (skiboot `include/xscom.h`, `XSCOM_ADDR_P9_EC/EQ`), and the per-core sensors carry these
numbers (S. Smith, "My POWER9 CPU Core Layout", 2020, on his own POWER9). Slice k is chiplet pair
EX k; quad q holds slices 2q and 2q+1. So index neighbours are die neighbours only **within a
quad**; an earlier version's claim that the index is a "1×12 physical ordering" of slices is
withdrawn as geometry.

**What the thermal data add** (`analysis/thermal_ibm_layout.py`, `analysis/thermal_column_test.py`,
three months, rack-bootstrap intervals):

1. **The quad.** At index distance 1, neighbours in one quad couple at +0.37 (same slice) to
   +0.48 (other slice), neighbours in different quads at +0.10.
2. **Beyond the quad, the core numbering, not the diagram.** Within a die column, slice 4c+1
   couples more with its index neighbour 4c+2 at the far die edge than with the nearer slice 4c+3
   across the nest (by 0.055, 95% CI [0.046, 0.066]; 0.070 and 0.026 in the other two months),
   where heat-source models on IBM's layout predict the opposite sign (−0.002 to −0.14 on the same
   scale); and for the same two slices coupling falls 0.080 [0.073, 0.086] per active slice between
   them (0.094, 0.082 in the other months), against at most 0.032 in those models. Workload placed
   on consecutive logical CPUs is a plausible cause.
3. **Only a small trace of the layout beyond the quad**: same-band slice pairs couple +0.044
   [0.040, 0.049] more at equal index and rank distance, with socket and slice effects (2022-02
   +0.042, 2021-05 +0.059). Column distance is not identifiable apart from slice effects. The
   earlier "natural experiment" (slices a, a+2 with a+1 harvested, +0.143) does not separate
   physical from logical ordering: a dark-source model gives +0.019 to +0.147.

Layout scores, **descriptive only** (Pearson correlation over the 66 slice pairs, with equal
weight, between a layout's slice distance and the slices' thermal similarity; more negative =
better; 1,948 sockets, 2022-08):

| layout | score |
|---|---|
| 1×12 linear (identity index order) | −0.791 (95% CI [−0.800, −0.778]); random-ordering null −0.003 ± 0.123, z = −6.41 (rack-bootstrap [−6.55, −6.24]) |
| exact optimum over all 12!/2 orderings (dynamic programming over prefix sets) | −0.793, `[0,…,9,11,10]` |
| IBM's documented layout, at the diagram's proportions | −0.824 (95% CI [−0.832, −0.813]) |
| the same two, over the 60 pairs in different quads only | index order −0.814, IBM −0.734 |
| 2-D grids of unit spacing (6×2, 4×3, 3×4) | all worse than the index order; see Fig. 6 |

The identity order ranks 2nd of all 239,500,800 orderings in each of three months (an exact
enumeration, not a sampled null), and in all 2,000 rack-bootstrap replicates the optimum stays
within one adjacent swap of it. IBM's layout leads every one-dimensional ordering **only through
the six same-quad pairs**, and its score moves from −0.31 to −0.85 over 90 rescalings of the
diagram (better than the best 1-D ordering in 34 of 90), so it is not robust to the diagram's
proportions. Thermal coupling therefore confirms the quads and cannot recover the layout beyond
them; the physical position of each index rests on IBM's documentation.

**Residual caveat.** IBM's figure is a chip diagram, not a to-scale die photograph, and the
sensor-to-chiplet mapping rests on firmware numbering and one owner's observation of his own
POWER9 (Smith 2020), not on a Marconi100 record. An annotated die shot would add proportions.
Leads: `happytrees.org/dieshots`, `open-power/hostboot` (`src/usr/targeting/`).

**Methodological note.** A three-day sample once scored the linear ordering at −0.283, behind a
3×4 grid, and overstated the within/cross-slice contrast more than threefold (+0.51 vs +0.29,
against +0.372 vs +0.312 over the full month). Two bugs contributed: row-stride downsampling
destroyed cross-core timestamp alignment, and sampling nodes from `core0` biased the population,
because slice 0 is the most-harvested and so present on only ~39% of sockets.

---

## OI-2 — Did a BMC firmware update remap sensor IDs in March 2020? `BOUNDED`

86 of the 235 transitions land on 2020-03-20…21 (89, on 47 nodes, over 2020-03-19…21). A
fleet-wide sensor-ID remap would mimic a configuration change perfectly, but it would impose one
permutation of slice indices on every socket it touches, and neither day admits one, not even with
p0 and p1, or Lot A and Lot B, tested separately; nor does any steady-state date with four or more
changes (0 of 6; `lifecycle_taxonomy_errors.py`, FP1). The cluster is not a single remap.

Maintenance logs are not available, and the result does not depend on them: those days lie inside
the commissioning window, which is located empirically from two independent signals
(`stabilization.py`) and excluded. Re-running everything on 2020-06 onward leaves harvest maps
**byte-identical for all 1,960 sockets present in both windows**, marginals agreeing to 0.1 pp,
the lot boundary unchanged, and no comparable date-clustering.

---

## OI-3 — Defect-driven or policy-driven harvesting? `OPEN — and not identifiable here`

Slice 0 is harvested on 88.2% of Lot A sockets but 39.2% of Lot B (rack cut, 880 and 1,080
sockets; 39.3% over all 1,962 sockets, which puts node 980 in Lot B). Two hypotheses: harvesting
follows killer defects (H1), or slice 0 abuts a shared structure and is preferentially disabled
(H2). What disabled any given position is itself unobserved (OI-18).

**The fitted model cannot separate them, and the paper no longer claims it can.** A correlated
defect process and a binning policy that disables contiguous blocks both predict γ > 0; γ measures
residual spatial association without attributing it. An earlier draft said the decomposition
showed "both"; that overstated the identification and was withdrawn.

Parametric model-replicate checks (`ppc.py`, 10,000 refitted replicates) add a further constraint:
the model is **misspecified**. It reproduces the distinct-pattern count (443 observed vs
436.4 ± 6.3 replicated, p = 0.16) but not the pattern frequencies (955.6 vs 481.5 ± 31.1,
p = 1/(B+1) = 1e-4) nor the compactness of the maps (mean maximum gap 4.469 vs 4.685 ± 0.044,
p = 1e-4), and it misplaces adjacency across the die: over-predicting co-harvest at slices (7,8),
(3,4), (5,6), under-predicting at (0,1), (0,2), (7,10), with simulation-calibrated |z| up to 9.2
and 10 of 66 cells outside the simultaneous 95% band (Pearson residuals understated this; they
were once quoted as "up to 4.8σ"). Whatever drives the clustering is **not position-independent**:
`model_quad.py` finds γ = +0.92 ± 0.04 for adjacency within a POWER9 quad against +0.28 ± 0.06
across quads (rack-clustered z = +13.9 for the difference), and even that split leaves misfit.

**Other routes to a disabled core are documented; one form is excluded.** IBM documents
deconfiguration of cores that are not licensed for use (POWER9 User's Manual v2.1, §23.5.7.10,
p. 311), including a factory *field core override* (IBM Documentation, "Field core override
function overview", POWER9 scale-out servers). Hostboot applies the override by keeping the
lowest-numbered functional core pairs (`open-power/hostboot` 39736a7, `src/usr/hwas/common/hwas.C`,
`restrictECunits`), so on otherwise good dies it would disable the *highest* slices (exactly 8–11
on a fully good die) and spare slice 0. The data run the other way: slice 0 is the most-disabled
slice in both lots, slices 8–11 are disabled no more often than average (22.0% in Lot A, 32.4% in
Lot B, against 33.3%), and the mean disabled slice index is 3.85 in Lot A (rack-bootstrap 95%
interval [3.74, 3.94]) and 5.31 in Lot B ([5.16, 5.44]), against 5.5 for a uniform choice and 9.5
for the override on a good die; only 9 of 1,962 sockets carry {8, 9, 10, 11}
(`tests/test_refs.py`). This assumes the BMC's core index is the firmware's core number (OI-1).
Two vendor-side artefacts support it: OpenBMC's OCC sensor configurations label the per-core
temperature sensors `p0_core0_temp`…`p0_core23_temp` in core order, and skiboot (v7.1) names each
active core in the OS device tree by its physical core number (`include/chip.h`, `P9_PIR2COREID`);
its Witherspoon (AC922, model 8335-GTH) test device tree `hdata/test/op920.wsp.dts` lacks cores
{8, 9, 16, 17} on chip 0 and {6, 7, 18, 19} on chip 8, i.e. whole pairs at different positions on
the two dies. It does not exclude deliberate disabling of good core pairs through partial-good
records written by manufacturing, nor disabling the slowest or leakiest pairs to meet the part's
frequency and power, for which we found no IBM documentation. A published precedent supports the
policy reading of H2: on Intel's Xeon Phi, four tile positions are disabled on 63–71% of dies
against 2–7% for every other position, which McCalpin reads as a default set disabled on
otherwise good dies (TACC TR-2021-04).

**To settle.** Vendor disclosure, or the same analysis on a second AC922 installation.

---

## OI-4 — Reference and novelty check `RESOLVED (references) · OPEN (novelty)`

Every reference was re-checked on 2026-09-11 against its publisher's DOI record (Crossref or
DataCite), the arXiv API, or the primary document, and `tests/test_refs.py` pins the result.
Corrected then: **Koren, Koren & Stapper**, *IEEE T. Computers* 42(6) is pp. **724–734**, 1993 (the
next article in that issue starts on p. 735); SeT-Diff is a **2026** work, cited in its CF '26
version; author lists were completed for HardHarvest, Khan et al., NodeSentry and the arXiv
entries; the M100 ExaData article's fifth author is **Alessio** Mauri, and Bae et al.'s second
author is **Jung Yoon** Hwang; arXiv:2505.14796 analyses Polaris at Argonne, not Marconi100, so
it is no longer cited as M100 ExaData work (PM100 is); the die facts now come from IBM's POWER9
User's Manual, the ISSCC 2017 paper and numbered Hot Chips slides instead of WikiChip; and "slice"
is sourced to the manual's own term for the unit, the *processor-pair cache slice* (pp. 155, 165).
Earlier corrections stand: the negative-binomial model is not Stapper alone in *T. Semiconductor
Manufacturing*, and the wafer-map work is **Bae, Hwang & Kuo**, *IIE Transactions*
39(12):1073–1083, 2007.

The novelty statement rests on search-absence alone, so the paper states it as an **awareness
claim**, not a priority claim: we are not aware of prior work recovering per-die harvest maps from
*published* out-of-band telemetry, without access to the machines, or tracking them daily for
years. The closest prior work we found:

- **McCalpin** (TACC ACELab TR-2021-04, 2021, hdl:2152/89580): running the user-mode CPUID
  instruction on all 4,200 Xeon Phi 7250s of Stampede2, he recovered each die's disabled core
  pairs (over 1,100 patterns), separated four default disable positions from defects, fitted a
  random-defect model, found a 3.7-sigma defect-rate difference between the machine's June 2016
  and April 2017 deployments, and counted 12 processor replacements among 446 nodes by comparing
  snapshots from 2016 and 2020. His companion report (TR-2021-01b, doi:10.26153/tsw/13119) read
  the CAPID6 register with administrator rights on 94% of Stampede2's Xeon Platinum 8160s and found
  120 distinct disabled-tile patterns.
- **Smith** ("My POWER9 CPU Core Layout", 2020) read the enabled cores of his own POWER9 machine
  from the same BMC `pX_coreY_temp` sensors this work uses, and observed an 8-core part running one
  core in each of eight slices, so slice granularity is part-specific.

What remains distinctive here is recovery from public out-of-band telemetry without access to the
machines, at fleet scale, followed daily over 31 months with a reboot-verified change taxonomy,
and a lot boundary found from the maps rather than known in advance. Adjacent literature is cited
so a reader can judge: core salvaging and deliberate disabling, variability in HPC, PUF / inherent
hardware identifiers (which identify *which die*, not *what was disabled on it*), thermal covert
and side channels (which leak sensor *values*, not sensor *presence*), the architectural sense of
"core harvesting" as lending idle cores to other work, node-level anomaly detection, and the
downstream M100 ExaData literature.

---

## OI-5 — Is collection dropout random? `RESOLVED — the lot result is immune`

Monthly coverage ranges from 18% to 99.4%. All set-comparison analysis restricts to *clean*
socket-days, so lot-correlated dropout could in principle manufacture the contrast. It does not
(`robustness.py`):

| clean-day coverage threshold | sockets | Lot A | Lot B | gap |
|---|---|---|---|---|
| all sockets | 1,962 | 88.2% | 39.3% | +48.9 pp |
| ≥90% | 1,847 | 88.0% | 38.9% | **+49.0 pp** |

Clean-day rates are 0.909 (Lot A) vs 0.902 (Lot B), a small difference that is statistically
detectable, and corr(slice-0 harvested, clean-day rate) = **+0.004, p = 0.88**: the filter bears no
relation to the quantity measured.

---

## OI-6, OI-7 — Sub-16 socket-days and same-day transitions `RESOLVED`

Most of the original sub-16 signal was **an extraction bug on our side**: three months had been
written from incomplete tar extractions, so absent metrics looked like disabled cores. The pipeline
now refuses to emit a month unless all 48 metrics are present, and the anomaly count fell from
11,459 to 9. Those nine are the fully-sampled socket-days with an even count below 16, the days of
the three strict guard episodes (OI-12).

The full census of the 256 socket-days that do not report exactly 16 cores
(`lifecycle_os_crosscheck.py`, section D): 209 report **more** than 16 and none of them is fully
sampled (192 are exact unions of the clean maps on either side of a reconfiguration); 11 report 24
cores, again none fully sampled. 39 have an **odd** count, which no baseline map produces: 21 are a
single sensor lost while the OS counted all 128 CPUs (sensor dropouts, nodes 296 and 356), 11 are
single-core deconfigurations the OS confirms (it reports 124 CPUs), 4 are a single-core loss on one
socket before OS collection began (166 p1), 2 are unions across a reconfiguration and 1 has no
earlier clean map. So a BMC core loss can be either a collector fault or a real single-core
deconfiguration, and the OS count tells them apart.

The two nodes appearing to change configuration within a day (315, 342) are **same-day reboots**,
visible once examined at full timestamp resolution: both cut over on 2021-03-02 across a ~28-minute
reporting gap, 16 cores before and 16 after, and `boottime` records a reboot of both that morning.

**Lesson:** any pipeline inferring "hardware absent" from "data absent" must gate on extraction
completeness, or partial ingestion masquerades as hardware failure.

---

## OI-8 — Why do p0/p1 labels swap? `LARGELY SETTLED`

Socket relabelling accounts for 19% of the 105 steady-state node-level map-change events. If the
collector merely exchanged the tags, every per-socket metric should exchange with them. Extending
power collection from six months to all 31 raises usable events from 4 to 23; restricting to
events where the p0−p1 power difference is large enough for its sign to mean anything (>2 W on
both sides), **12 of 15 flip sign** at the relabelling instant (80%, 95% CI [52, 96]). The
producer is `analysis/oi8_and_os_full.py`, which prints that line verbatim (it also prints the
1 W threshold, 14 of 19, so the choice of threshold is visible); the CI `longitudinal` job asserts
it.

The residual minority may be genuine die exchanges that happen to mirror, or differences too small
to resolve. Any per-socket longitudinal study on this dataset should still treat relabelling as a
hazard: it silently mixes two dies.

---

## OI-9 — The lot boundary has no external confirmation `OPEN`

No procurement or installation record was available, so "procurement lot" remains the most
parsimonious explanation rather than a documented fact. Three observations narrow the alternatives:
the boundary falls between adjacent cabinets of one row (middle row: 91.9% vs 39.1% slice-0
harvested, rack rates 87.5–95.0% vs 27.5–47.5%; `lot_geometry.py`); within a lot the rack adds
little (same-rack overlap z = +0.9 under a within-lot shuffle, between-rack variance at most
marginal, smallest p = 0.017; `rack_lot.py`); and dies that arrive later in Lot A racks have slice 0
harvested on 61% (33/54), not Lot A's 88% (`cohort_incoming.py`). The signature travels with the
parts installed at commissioning, so neither the room nor any setting that stays with the rack
explains it; whether it reflects the delivery, vendor-side provisioning or installation-time setup
is not separable from telemetry. Public statements describe the system as acquired within the
PPI4HPC initiative, with installation expected to finish at the end of February 2020, and mention
no delivery phases; 981 of 984 nodes report from the first day of the record (2020-03-09), so
installation order is not observable either.

The *statistics*, however, are no longer a maximised t masquerading as a p-value
(`changepoint.py`, `cohort_boundary.py`):

| | |
|---|---|
| permutation-calibrated max-t | null max\|t\| = 7.08 ± 2.41, never above 17.95 in 4,000 rack permutations, vs observed 26.6 → **p = 2.5e-4**, (b+1)/(B+1), the floor of 4,000 draws |
| breakpoint stability (racks held in place) | rack 22 in 100% (sockets) and 99.6% (nodes) of 2,000 within-rack resamples — a stability check, not a CI |
| rack-level uncertainty | under the fitted two-lot model, rack 22 in 99.7% of rack-block resamples within each side and 98.7% of a beta-binomial parametric bootstrap; profile likelihood ratio 5.6×10⁴ over rack 23 |
| number of populations | chosen by **rack-held-out** likelihood: −7.360 (k=1), **−7.133 (k=2)**, −7.139, −7.157, −7.149 (12-Bernoulli score); k = 2, cut at rack 22, also under conditional-Bernoulli and pairwise models on the 495 valid maps and a Dirichlet-smoothed multinomial except at its lightest smoothing, which prefers k = 1; k = 3, which adds a cut near rack 46, trails by 10–16 nats summed over 1,960 held-out sockets |
| leave-one-rack-out | **49 of 49** fits split the remaining racks exactly as the full data do (the 48 reported earlier compared boundary names: with rack 21 held out, the boundary falls in the gap it leaves and was counted as a miss) |
| held-out lot prediction, nested (boundary re-found without the held-out rack) | **72.9%** of sockets, **47/49 racks (95.9%)**, vs 55.1% majority-class; the target is the inferred lot, not a delivery record |

Note on method: racks cannot be resampled with replacement *across* the scan — that shuffles the
order the changepoint is defined on and returns a CI spanning the whole scan. Resampling sockets or
nodes inside each rack keeps the order but conditions on the racks, so it is a stability check, not
a confidence interval. Rack-level uncertainty comes from resampling whole racks *within each side*
of the boundary and a rack-level beta-binomial parametric bootstrap, both under the fitted two-lot
model, and from the profile likelihood.

Two scan-hygiene points: candidate boundaries must leave ≥100 sockets per side, and **nodes 980–983
sit outside the 49 documented cabinets** and form a degenerate segment that captures the argmax of
an unguarded scan. Both are handled and stated in the paper.

**What would settle it:** delivery or installation records, or processor serial/VPD data for a
sample of sockets in each lot.

---

## OI-10 — Single system `OPEN — the largest remaining weakness`

Every result comes from one machine, one BMC firmware and one procurement history. Whether the
harvest-pattern statistics generalise to POWER9 as a whole, or reflect CINECA's specific
procurement, is untested. The *channel* is not unique to this machine: the same non-renumbered
`pX_coreY_temp` sensors appear on another POWER9 platform (Smith 2020). A replication on a second
AC922 installation (Summit/Sierra telemetry, if obtainable) is the single highest-value addition
and would also settle OI-3.

---

## OI-11 — Pooled thermal estimator and the common-mode artefact `RESOLVED — socket-level now`

Common-mode removal forces each row of the residual matrix to sum to approximately zero, so
negative values at large index gaps are partly mechanical. The paper therefore relies on contrasts
at identical index distance and on ordering tests, which the constraint cannot bias.

The pooled 24×24 matrix of earlier versions discarded the sockets. Per-socket residual matrices are
now published (`analysis/thermal_socket_{meta,c5,c1,lag}_<YM>`, three months) with the socket
support of every cell (`thermal_support_counts*.csv`; minimum 461 sockets, slices 0 & 9; 57 in
Lot A). Every interval resamples the 49 racks: the earlier pair-level CI for the within/cross
contrast understated its variance about five-fold; the contrast is +0.061 [0.047, 0.076]
(Fisher-z pooling: +0.055). Composition controls (position-matched contrasts, post-stratified and
two-way pooled matrices, Fisher-z, per-socket scores) leave the ordering unchanged, but the pooled
within/cross contrast mixes same-quad and cross-quad neighbours (55% / 45%) and changes sign
between months (−0.004 in 2022-02, +0.061 in 2022-08, +0.104 in 2021-05). The position-matched
contrasts keep their sign in every lot, socket position and month: siblings couple less than the
adjacent core of the other slice in their quad (−0.102 [−0.114, −0.089]) and far more than cores
across a quad boundary (+0.246 [0.227, 0.267]).

The lagged-correlation table of earlier versions read lags off the every-fifth-sample series, so its
unit was 100 s, not 20 s (its "±10 samples" was 1,000 s); it has been removed. At true 20 s lags
every kind of neighbour loses 24–52% of its lag-0 coupling within one sample and then changes by
less than 0.02 out to 200 s.

Two bookkeeping notes: the text's rack-bootstrap intervals for the layout scores come from
`thermal_ibm_layout.py` (1,000 replicates), the whiskers of Fig. 6 from `layout_hypotheses.py`
(500 replicates); they differ by at most 0.001. The decimation rule is recorded in `DATA.md`.

---

## OI-12 — Are the guard episodes genuine deconfigurations? `LARGELY SETTLED` · GARD record `OPEN`

Three sockets (347 p1, 411 p1, 946 p0) run on 14 cores for 2–4 days, each losing exactly one slice.
Telemetry alone cannot distinguish a real GARD event from a collection failure that happens to drop
both cores of one slice — so three independent collectors were checked.

1. **Every other sensor kept full cadence.** In all three episodes the two slice sensors read zero
   for the whole window while all ten control metrics on the same socket sustained 4,320
   samples/day.
2. **The OS agrees, node for node.** Of 990 nodes, exactly three ever report 120 logical CPUs
   (= 128 − 8, one slice at four threads), and they are the same three. Under a null flagging three
   arbitrary nodes, the chance of matching is 6.2e-9.
3. **They sit across reboots**, as a hostboot-applied GARD record requires: node 946's episode falls
   between boots on 2020-08-30 and 09-03, node 411's between 2021-11-12 and 11-15/16.

**A failed variant, recorded so others avoid it.** Comparing each day against the socket's *global*
modal map, rather than the adjacent observed day, conflates guard events with permanent
reconfigurations and inflated the count to 30: a socket that changes configuration mismatches its
modal map for hundreds of days.

**Caveat.** The third episode (node 347, 2020-03) cannot be checked against the boot signal at all:
ganglia `boottime` collection begins 2020-05-05. That is absence of coverage, not absence of a
reboot. It also falls inside the excluded commissioning window, so the paper quotes **two
steady-state episodes** under the strict criterion and reports the third separately rather than
folding it into a rate.

**The full census** (`lifecycle_os_crosscheck.py`). The strict criterion (fully-sampled days, even
count) is slice-granular by construction. Checked against every departure of the OS count from 128,
the fleet has **nine** deconfiguration episodes in steady state, all matched node for node and day
for day by a BMC loss of exactly the implied number of cores, each beginning on a boot day: three of
one slice (120 logical CPUs: 347 in 2020-10, 411, 946) and six of one core (124: nodes 39, 145,
166, 216, 584, 742). Each is followed within six days by a new map on that socket, its partner
unchanged, that no other socket gave up that week: the signature of a failing processor replaced.
Two commissioning-era episodes predate OS collection (347 p1 slice 4, 166 p1 core 21); both were
restored with the same map, and both recur on the same slice or core in steady state before the map
changes, eleven episodes in all. The only sustained BMC-side loss the OS does not confirm (node 296
p1, core 23, 19 days, OS at 128) is a sensor dropout, and its map never changes. An earlier
statement that the 124-CPU nodes were OS core offlining invisible to the BMC was wrong and is
withdrawn. Over OS-observed exposure (3,500 socket-years) the steady-state rate is 26 per 10,000
socket-years (95% CI [12, 49]), against 5.0 [0.6, 18.1] per 10,000 over 3,991 observed
socket-years for the strict criterion alone.

**Still open:** the GARD record itself is not in the telemetry; a hostboot, `opal-gard` or service
log for these nodes would confirm the mechanism.

---

## OI-13 — The power result was overstated as a null `RESOLVED — claim withdrawn; the association is bounded, not provably negligible`

An earlier version reported r = +0.039 (p = 0.003) explaining 0.16% of variance and concluded the
harvest map is "operationally invisible". Two defects: "no effect" was argued from a small r rather
than an equivalence test against a stated margin, and the interval treated node-days within a rack
as independent.

The first re-run (steady-state record from 2020-06, rack-clustered, ±5 W margin) reported the mean
active-slice index at +7.9 W [+0.9, +16.6]. External review in 2026-09 showed that it attached one
full-record modal map to every power day and that its covariates were node-level; answering it
exposed two further defects. The three map summaries were fitted jointly and each partial slope was
scaled by its *marginal* range, although mean index and low-half count correlate at −0.89
(VIF 5.1), so the +7.9 W is the effect of a contrast no pair of real maps realises. Evaluated over
the map pairs that do occur, the same joint fit spans 2.2 W from the 5th to the 95th percentile
pair (bootstrap 95% upper 4.3 W) and 4.2 W between the two most extreme pairs (upper 8.1 W; a
maximum over pairs is biased upward). And `cpu_speed`, listed as a core-frequency adjustment, is a
constant 3,800 MHz in this sample.

Current analysis (`equivalence.py`, estimators in `equivalence_core.py`): every node-day carries
the map in force that day (636,026 node-days on 980 nodes, 28 months; under the old join 12,034
carried a non-modal map, 1,166 of them the mirror image after a relabel), one summary per model,
adjusted for cpu_user, load_one, inlet air and month, rack-clustered bootstrap (B = 4,000; CR1
standard errors agree to within about 0.1 W on the bounds). The margin of ±5 W is 7.2% of the
69.0 W mean socket draw:

| summary | slope (W/unit) | effect over full range | 90% CI | per SD | per IQR | smallest margin |
|---|---|---|---|---|---|---|
| mean active-slice index | +0.40 | +2.7 W | [−0.2, +5.6] | +0.46 W | +0.61 W | 5.6 W |
| low-half active count | −0.19 | −1.5 W | [−4.7, +1.8] | −0.25 W | −0.38 W | 4.7 W |
| adjacent active pairs | −0.14 | −1.0 W | [−4.0, +2.1] | −0.17 W | −0.28 W | 4.0 W |

Two summaries are equivalent at ±5 W, all three at ±10 W, none at ±2 W. The mean-index interval
only just includes zero (its lower bound is −0.25 to −0.12 W across bootstrap seeds). In every
sensitivity design (same-day maps only, full power coverage, the 848 nodes whose maps never
change, idle node-days, placement-adjusted) every interval lies within ±7.4 W, but not all within
±5 W: the never-changing nodes give mean index [−0.1, +6.5] and low half [−5.7, +1.8].

**Workload.** The map is fixed at boot and invisible to the scheduler, so an uneven p0/p1 workload
biases the slope only through its correlation with the map difference. The SLURM job table records
per-socket core placement (`cpus_alloc_layout`: SLURM cores 0–15 on socket 0, 16–31 on socket 1)
for jobs ending 2020-05 to 2020-10-13 (`equivalence_alloc.py`; later layouts are empty in the
release). Own-socket allocated fraction correlates with socket power at r = 0.67 (p0) and 0.65
(p1). On the 119 days with complete job records (26,306 node-days), placement explains 40% of the
variance of p0−p1 power (97 W per unit of allocated-fraction difference) but correlates with the
map difference at |r| ≤ 0.021. Because placement moves power so much, that small correlation still
matters: adjusting for it moves the estimates from +0.6, −0.3 and −0.2 W to +1.7, −1.5 and −1.3 W
(mean index, low half, adjacency), about 1 W away from zero, with 90% bounds 4.4, 5.0 and 4.3 W
(the low-half bound sits on the margin: 4.96–5.04 W across bootstrap seeds). In that window,
omitting placement biased the estimates toward zero; whether it does so over the whole record
cannot be checked. In the same window the mean p0−p1 difference (5.1 W) is almost all placement
(4.7 W; socket 0 is filled first) and at equal placement the slot effect is +0.3 W [−0.4, +1.1];
on idle node-days the mean is +0.2 W. The 6.8 W full-record mean, which an earlier version called
a "positional asymmetry" and used to anchor the margin, therefore reflects scheduler placement
rather than socket position. On idle node-days (cpu_user < 0.5%, load_one < 0.5; 30,688; where
placement is recorded, allocated fraction 0.07 against 0.51) the effects are +1.2, −0.9 and −5.0 W;
the adjacency interval [−7.2, −0.5] excludes zero but weights nodes by idle time (one node has 620
idle days), and with one observation per node-epoch it is −2.5 W [−5.1, +0.1]. The idle sockets of
one node differ by 9.2 W (SD of node means): a static die, board or sensor offset.

**Not usable as controls.** Per-socket I/O- and memory-subsystem power were meant as negative
controls, but the I/O difference barely tracks CPU placement (r = +0.10 where both are recorded)
and both are associated with the map (I/O +1.8 W [+0.4, +3.2] over the mean-index range). Either
the die affects them too, or they share an unmeasured influence with the map difference; in both
cases they cannot serve as negative controls, and conditioning on them (mean index −0.5, adjacency
−4.2 [−6.5, −2.0], low half +1.7 W) is not a valid adjustment and is not reported as a result.

**What would settle it.** Per-socket utilisation or GPU placement for the whole record, e.g. from
CINECA's SLURM accounting; the public release has neither after 2020-10.

---

## OI-14 — "Changes occur only across reboots" was inferred, not measured `RESOLVED`

The claim originally rested on every transition coinciding with a **gap in reporting**, which is
consistent with a reboot but does not demonstrate one. The dataset carries
`ganglia_pub/metric=boottime`, which makes reboots directly observable; it had simply never been
extracted (`sweep_covariates.py`).

Over 668,524 node-days on 990 nodes: **126 of 132 steady-state transitions with boot coverage
(95.5%) coincide with an observed reboot**, against **47.6%** expected once control intervals are
matched on length (49.0 ± 4.6 of 103 node-level transitions). The raw per-node-day reboot rate is
1.74% of 1,194,581 non-transition intervals, but a transition sits across a multi-day reporting
gap, so that is not the comparator. Matched on interval length the contrast holds (100% vs 16.7%
at 3–4 days). Controls repeat within
nodes and the two sockets of a node share one boot signal, so the test is run on node-intervals and
permutes transition labels only within node × interval-length strata: 99 of 103 node-level
transitions carry a reboot against 49.0 ± 4.6 expected, exact conditional p = 6e-31 (Monte-Carlo
(0+1)/(200,000+1)); the length-standardised excess is +63 pp, 95% CI [+57, +68] by node-cluster
bootstrap and [+58, +68] by rack (`reboot_evidence.py`, section 6). The earlier shuffle within gap
strata across the whole fleet (+93.7 pp vs +41.3 ± 3.5) ignored that dependence.

**The six without a flagged reboot are resolved.** They are four node-level events (nodes 113, 138,
157, 811) whose Ganglia stream stopped part-way through the interval and resumed only on
2021-05-19. The day-flag rule dates a reboot to the first day a changed epoch is seen, which for
these is 2021-05-19; the boot epoch then reported lies inside every one of the six intervals
(2021-04-22 16:18 to 2021-05-06 12:53). Dating every reboot by its epoch instead, **133 of 136
steady-state transitions contain a boot, none is observed without one, and 3 are unresolved**
(113 p0 2021-05-02→05-04; 412 p0/p1 2020-08-07→08-12: no `boottime` observation brackets them). An
earlier draft called the six out-of-window reboots; that was unsupported and is withdrawn.

---

## OI-15 — Pair-level co-harvest counts `RESOLVED (Monte-Carlo) · BOUNDED (pooled vs per-lot null)`

The pair tests used to rest on a curveball null of 200 chained draws, gave 16–18 significant pairs
depending on the seed, and used a cut (|z| > 3.83) stricter than the two-sided Bonferroni value
for 66 tests (3.37). The trade implementation also dropped the shared slices from the second row
and so executed only trades between sockets sharing no harvested slice (about 10% of attempts).

`analysis/curveball_null.py` replaces this with 100,000 exchangeable draws of a corrected sampler
(Besag–Clifford parallel method), so every p = (b+1)/(B+1) is valid however slowly the chain mixes.
The mean-gap z is −20.7 (p = 1e-5); **20 pairs** are family-wise significant, identically under
Monte-Carlo Bonferroni and max-|z|; **exactly eight positive excesses sit at |Δk| ≤ 2** (the six
same-quad pairs plus 1&2 and 0&2), the ninth, 7&10, at |Δk| = 3; nine of eleven deficits are at
|Δk| ≥ 4. The null's tails are normal to within Monte-Carlo error (null frequency of |z| > 3.37:
7.7e-4, nominal 7.6e-4). A 10,000-draw sequential-sampler run makes the same 66 decisions, and the
legacy sampler's null moments agree with the corrected ones.

**What remains.** The pooled null mixes two lots with different marginals. A null that keeps each
lot's own marginals leaves **11 pairs**: seven of the eight near excesses (0&2 falls to z = +3.1)
and four deficits, all at |Δk| ≥ 6 (mean-gap z = −15.4). Most of the pooled deficits, and part of
the 0&2 excess, therefore reflect the mixing of two lots; the within-quad excesses survive it.

**Seeds** (all fixed in `curveball_null.py`, output independent of the worker count): the legacy
reproduction uses seeds 2024, 1, 2, 3, 4; the main exchangeable null 20260911; the sequential
confirmation 20260912; the lot-stratified null 20260913.

---

## OI-16 — The reliability null is bounded, not absolute `BOUNDED`

No feature of the baseline harvest map is a significant predictor of which dies are later pulled
from their node: held-out risk-set concordance **0.55** (95% CI [0.48, 0.61]) against a
rack-permutation null of 0.50 ± 0.04 (p = 0.12), over 85 events and 53,716 die-months
(`survival.py`: grouped proportional hazards with one baseline effect per calendar month; 7-fold
cross-validation grouped by rack, ten partitions, rarity and scaling learned in-fold). Adding the
map does not improve the held-out survival log-likelihood over the monthly baseline alone
(−2.1 nats, p = 0.14) or the integrated Brier score (+0.13%).

**What this does not say.** The bounds are asymmetric. For the continuous features (adjacent
pairs, dispersion, rarity) the rack-clustered intervals exclude a hazard increase above 21% per SD
but not protective effects down to about 0.5 per SD. For the binary contrasts they exclude little:
slice-0 status 1.25 [0.51, 3.26] and lot B 0.64 [0.33, 1.24], both 0→1. What is excluded is
useful prediction of these events from these features: the upper confidence bound on concordance,
0.61, is below the usual 0.7 bar for acceptable discrimination. Two readings remain consistent with
the data: whatever set the disabled positions is unrelated to what kills a part in the field, or
replacement is dominated by causes unrelated to the die. A fleet with more events, or a longer
record, would tighten this.

**The endpoint is inferred.** Events are replacement-consistent map changes, not FRU records.
Following each die by its map shows how noisy that is: of the 85 dies pulled from their node, 41
reappeared on another node in the same maintenance window and 44 were never seen again. Fleet-wide,
43 departing maps were re-found on another node against about 5 expected from coincident identical
maps; 36 of them form 18 two-way exchanges between the same two nodes, of which only one
(nodes 142/151) moves both sockets, so they read as single-processor exchanges rather than node
renaming. The null holds for dies that left the fleet (46 events; C 0.56 [0.44, 0.66], p = 0.16),
CPU-SWAP-like events alone (72; C 0.51, p = 0.44) and the socket-label endpoint of the first
release (92; C 0.54, p = 0.24), each on one rack partition. The only nominally significant result
is for the broadest endpoint, which also counts relabelling (121 events; C 0.58 [0.49, 0.64],
p = 0.015 from 4,999 permutations, not significant against the Bonferroni threshold of 0.0083 for
six endpoints; 0.57 averaged over ten partitions). Lot alone gives most of it (held-out C 0.575
against 0.580 for all five; Lot A racks carry more map changes of every kind), and its
adjacent-pairs interval (0.73 [0.55, 0.97] per SD) is the only one of 30 endpoint-by-feature
intervals to exclude 1. NODE-SWAP-like events alone (13) are too few to estimate (C 0.70
[0.40, 0.86]). A replacement by a spare with an identical map is invisible, with probability 0.8%
on average and at most 5.7% for the commonest map; that biases the rarity hazard ratio by about 1%
per SD (6% between the commonest and a never-seen map). If a fraction of events were unrelated to
the die (non-differential noise), the continuous-feature bound would widen to [0.34, 1.28] per SD
at 25% noise, and the concordance bound for the die-related events to 0.65 (0.73 at 50% noise:
heavy endpoint noise could hide useful prediction of true failures). Validating the endpoint
against service or FRU records would remove this uncertainty; that has not been done.

**Correction to the first release.** v1.0.0 reported an "out-of-fold concordance index" of 0.448
against a null of 0.499 ± 0.031. That statistic was a pooled person-month AUC with fold intercepts,
not a survival concordance, and its null shuffled fixed scores across rows. Re-running the whole
cross-validation under rack-block permutations centres its null at 0.477 ± 0.050, because a
fold's intercept falls as its held-out event rate rises (r = −0.93), and v1's rack partition was
an unfavourable draw: the same pipeline on 50 other partitions gives 0.495 ± 0.020, and only 2 of
them are lower (`survival_legacy_check.py`). The apparent sub-chance value was therefore an
artefact. v1 also charged 14 of its 92 events either to the partner die (7) or to dies re-found on
the partner socket (7), assumed a constant hazard, and learned pattern rarity on the full fleet.
Learning rarity in-fold moved the v1 statistic to 0.463; in-fold standardisation changed no score
(affine invariance). All are fixed, and none changes the null conclusion.

---

## OI-18 — What disabled each position? `OPEN — not identifiable from telemetry`

Telemetry shows *which* core positions are disabled, not *why*. On POWER9 the active set is applied
by firmware at every boot from three sources (open-power/hostboot, `release-op920`):

1. a **partial-good record** in the processor module's own vital product data (MVPD record `CP00`,
   keyword `PG`, read by `platReadPartialGood()` in `src/usr/hwas/hwasPlat.C`);
2. **GARD records** for units that failed in the field (`deconfigGard.C`, cleared with `opal-gard`);
3. a **field core override** (`processFieldCoreOverride()`), which IBM's factory uses to disable good
   cores for licensing (feature code 2319).

IBM's register specification (POWER9 Registers vol. 1 v1.2, pp. 262–263) has one configuration bit
per core and notes that a bad L2 deconfigures both of its cores; an 8-core POWER9 runs one core per
L2/L3 pair (Smith, 2020). So *disabled* is not synonymous with *defective*, and slice granularity is
not a hardware necessity. The paper therefore defines a **harvest map** as the set of disabled
positions, whatever disabled them, and **harvesting** as disabling units so that a die sells as a
lower-core-count part, whether they are defective or not.

What the telemetry does show: every baseline map disables whole slices; maps are present from a
socket's first observation and change essentially only at reboot; field deconfiguration removes a
whole slice in three episodes, on the only three nodes whose OS ever reports 120 CPUs, and a single
core on each of the six nodes whose OS reports 124 (for whole days on five, from mid-day on node
145); each of the nine is followed within six days by a new map on that socket (OI-12).

**To settle.** The `PG` keyword, `opal-gard list` and field-core-override value for a sample of
Marconi100 processors, or vendor disclosure.

---

## OI-19 — The change taxonomy classifies map changes, not verified hardware events `BOUNDED`

A harvest map is a signature, not a serial number, so RELABEL / CPU-SWAP / NODE-SWAP name the event
each map change is *consistent with*, and every rate derived from them is telemetry-inferred. Their
error mechanisms, quantified from the committed data (`lifecycle_taxonomy_errors.py`):

| mechanism | effect | size |
|---|---|---|
| replacement die carries the same map | die change missed | collision probability 2.07% (Lot A), 0.26% (Lot B): about 1.3 of 96 dies |
| NODE-SWAP whose new dies mirror the old ones | read as RELABEL | 2.6e-4 per node swap |
| firmware / BMC sensor-ID remap | spurious change | no steady-state date with ≥ 4 changes, nor 2020-03-20 or 03-21 (also per socket and per lot), admits the single slice permutation a remap imposes |
| in-place reconfiguration | spurious change | new maps share harvested slices with old ones as independent same-lot dies do: [8, 40, 38, 10] vs [8.8, 35.3, 38.2, 13.8], χ² = 1.73, MC p = 0.60; no excess of one-slice exchanges |
| die moved from another socket | a die change, not a new processor | **50 of 96** new maps are another socket's old map within 7 days (4.4 expected by chance; 42 the same day, 1.2 expected): 43 from another node, 36 of them in 18 two-way exchanges, and 7 from the partner socket; 4 of 11 NODE-SWAPs are a relabel plus one new die |

Rates over the steady state (4,561 calendar socket-years): die changes 96 → **2.10%** per
socket-year, exact Poisson [1.70, 2.57], node-cluster bootstrap [1.62, 2.61], rack-cluster bootstrap
**[1.56, 2.70]** (rack dispersion 1.87); observed clean socket-days as denominator give 2.4%. New
silicon, maps no other socket gave up: 46 within 7 days → **1.01%** [rack 0.68, 1.36]; 54 on the
same day → 1.18% [rack 0.83, 1.56]; about +0.7 die for same-map replacements. The 7-day figures
agree with the die tracking of `survival_core.py` (7 + 43 dies re-found, 46 that left the fleet, 18
two-way exchanges).

Positive evidence: all nine OS-confirmed deconfigurations (OI-12) are followed by a CPU-SWAP whose
new map no other socket gave up that week, the sequence a failing processor replaced by a new one
would produce.

**To settle:** CINECA's FRU or processor-serial history, or maintenance tickets, for even a subset of
the 96 changes (the 18 exchanges and the 9 fault-followed changes first) would give the precision of
each class. Whether the whole-node exchange 142 ↔ 151 is a chassis move or a node-label swap in the
monitoring cannot be told from telemetry; either way no new silicon entered.

---

## OI-20 — The side-channel mitigation numbers were wrong `RESOLVED — corrected and measured; one residual BOUNDED`

The first release modelled random sample deletion incorrectly (`side_channel.py` fed a fractional
count into a slot-placement formula) and reported exact recovery 1.000 after deleting 99.9% of
samples and 0.61 at 99.99%. Found in external review. Independent deletion evaluated on the
empirical per-core counts of 2022-08 (`side_channel.py`; Monte Carlo agrees within its standard
error everywhere; `tests/test_side_channel.py` checks every closed form against simulation):

| deleted | 1 day | 2 days | 7 days | 1 month |
|---|---|---|---|---|
| 99.9% | 0.81 | 0.997 | 1.000 | 1.000 |
| 99.99% | 5×10⁻⁸ | 0.0002 | 0.45 | 0.99997 |
| 99.999% | 1×10⁻²² | 5×10⁻¹⁸ | 5×10⁻¹⁰ | 0.008 |

These use the per-core rule (every active core must keep a sample). Every map is a union of whole
slices, so an attacker who applies the slice rule needs one sample per active slice and does
better: 0.9985 at 99.9% for one day, 0.57 at 99.999% for one month. A failed exact recovery still
discloses: at 99.99% one day reveals on average 4.6 of the 8 active slices, leaving about 63 of
the 495 possible maps. The conclusion survives and is sharper: maps are static, so an attacker
pools days, and holding one-month recovery below one half takes 99.998% deletion (99.9991%
against the slice rule).

Measured on the raw timestamps (`side_channel_raw.py`): one 20 s interval recovers the exact map
in 99.95% of socket-intervals (the daily-count model had said 99.70%), because 99.95% of missing
core-slots fall in slots where the whole socket is silent. The gaps are strongly bursty in time:
most are short (mean 2.6 slots), but 56% of the silent slots lie in outages of an hour or more,
so for windows of 80 s and longer the independent-placement model is slightly optimistic (140 s:
measured 0.99972, model 0.99993). Keeping one slot per six hours still recovers every map;
keeping one per day recovers 99.95%. Deleting whole timestamps protects less than independent
deletion (0.99 at 99.9%, simulated, inside exact analytic bounds). Per-core bursty deletion
matches its closed form (0.411 simulated vs 0.411).

**Residual (BOUNDED).** Compact renumbering removes the presence channel by construction, but the
temperatures keep a weak positional trace: residual coupling between consecutive surviving slices
flags a harvested slice between them with AUC 0.65 (node-clustered 95% CI [0.64, 0.67], 596
sockets, `side_channel_values.py`). Neither of two decoders turns this into maps better than the
per-slice prior alone (3.2% vs 2.9% exact). A stronger attacker might; treat compact renumbering
as closing the presence channel, not every channel. Per-node random renumbering must be redrawn for
every release: a fixed permutation makes the set of reporting pseudonyms a fingerprint unique for
about 99.7% of sockets, against 4.9% for the maps themselves.

---

## Causal structure the paper assumes

```
fab defects, product policy, die source --> partial-good record (module VPD) --+
field core override (factory / BMC) ----------------------------------------+--> active cores at boot
GARD records (field failures) -----------------------------------------------+          |
                                        collector faults --> BMC sensor presence <------+   (observed)
procurement / installation batch --> rack position
room cooling, workload, die properties, socket position --> thermal and power telemetry
component failures, service policy --> map transitions (RELABEL / CPU-SWAP / NODE-SWAP-consistent)
```

Four consequences the paper relies on: rack position cannot cause a map fixed before operation, but
can proxy the installation batch (Sec. IV-G); a power difference cannot be attributed to the map
without controlling workload and socket position (Sec. IV-L); sensor presence has several upstream
causes, so a map says which positions are disabled, not why (OI-18); and a map transition is
evidence of, not the same thing as, a replacement (Sec. IV-I, OI-19).

---

## Analysis choices that moved reported numbers

Kept here because the paper's "Threats to validity" condenses it. Five reported results changed
materially during this work, none because of a new measurement:

1. **Sample size.** A three-day thermal sample overstated the within/cross-slice contrast more than
   threefold and scored the linear ordering behind a 3×4 grid (OI-1).
2. **Preprocessing.** A one-minute alignment grid, adopted to work around an unrelated bug, inflated
   that contrast by 1.3× (2.2× at 15 minutes); native 20 s cadence is now used (Sec. IV-E).
3. **Bookkeeping.** An intermediate revision quoted a corrected thermal figure from a different
   subsample than the one it named.
4. **Uncertainty.** The "operationally invisible" power claim rested on an interval that treated
   node-days within a rack as independent (OI-13).
5. **Specification.** Fitting three collinear map summaries jointly and scaling each partial slope
   by its marginal range nearly tripled the reported power effect (+7.9 against +2.7 W; OI-13).

Four of the five were introduced while fixing or extending something else, and the first four all
ran in the direction the authors would have preferred. The external review of 2026-09 found further
defects in the side-channel deletion model (OI-20), the survival statistic (OI-16), the curveball
sampler (OI-15) and the leave-one-rack-out count (OI-9); each is recorded in its entry.

---

## OI-21 — Artefact items the external review asked for and this release does not do `OPEN — scoped`

The review of 2026-09 made twenty-eight artefact requests. Most are done and are described in
[`DATA.md`](DATA.md), [`docs/reproduce.md`](docs/reproduce.md) and the `Repository` section of
[`CHANGELOG.md`](CHANGELOG.md). These six are **not**, and are recorded here rather than implied
away:

1. **No checksum of our own copy of the raw archives.** The twelve Zenodo records are named by DOI,
   which pins a version, but we did not record a hash per monthly tar as we downloaded it. The chain
   raw&nbsp;→&nbsp;script&nbsp;→&nbsp;committed table is therefore auditable from the record DOI
   down, not byte for byte. Re-deriving `daily/` from a fresh download and diffing it against
   `DATA.sha256` is the check that would close this.
2. **The 48-metric completeness gate writes no machine-readable audit record.** It is a hard
   refusal — `sweep_all_months.py` never emits a month unless all 48 core-temperature partitions
   extracted — and it prints `INCOMPLETE EXTRACTION: <k>/48 metrics` when it fires, but a refusal
   leaves no JSON log to audit later. The 31 committed months are the months that passed it.
3. **The extraction stage has no synthetic fixture.** Everything downstream of `daily/` is unit
   tested and re-derived in CI; the tar-to-`daily/` step is not, because the script runs
   top-to-bottom against a hard-coded archive path. Its rule (UTC day boundary, one row per
   reporting (node, socket, core, day), absence as the signal) is written down in `DATA.md`, and the
   tables it produced are checksummed, but a small fake archive that exercises the code path would
   be a real improvement and is not here.
4. **Nobody has been notified.** The review suggested contacting the M100 ExaData maintainers and
   CINECA, IBM or OpenPOWER, and sites publishing telemetry with comparable sensor namespaces. As of
   this release **no such contact has been made and no disclosure timeline exists.** The finding is
   passive, reads an already-public dataset, recovers manufacturing provenance rather than
   credentials, and is probably also readable in-band on any POWER9 node (`SECURITY.md`), so there
   is no embargo argument; whether to notify anyone before publication is the authors' decision, and
   it is listed as an open question for them rather than quietly assumed either way.
5. **No tag ties the artefact to a manuscript version.** Whether and when to tag a release matching
   the submitted paper is the owner's decision; until then the commit hash is the only pin.
6. **Node identifiers stay at full granularity.** They are the upstream dataset's own pseudonyms and
   the longitudinal analysis needs them, but a node's pair of maps is unique for 93.0% of nodes, so
   the derived tables can be linked to another M100 ExaData-derived release even if it renames nodes.
   Coarsening them would break the lifecycle results, so they are kept and the risk is stated
   (`SECURITY.md`, `DATA.md`).

---

## Operational notes (not paper issues)

- The full sweep must run **alone**: a concurrent pandas job was OOM-killed while the sweep held
  ~18 GB.
- `sweep_all_months.py` and `sweep_covariates.py` are resumable — they skip months whose output
  exists. Launch them as tracked background jobs; a `nohup … &` died once when its parent shell was
  cleaned up.
- Do not write `pgrep -f <pattern>` inside a shell whose own command line contains `<pattern>`: it
  matches itself and any `until ! pgrep …` loop never terminates.
