# Limitations and open issues

The register the paper's "Threats to validity" section refers to. Each entry gives what is known,
what would settle it, and where to look. Identifiers (`OI-n`) are cited from `paper/paper.tex`, so
the paper and this file stay in sync.

Status: `RESOLVED` · `BOUNDED` — the ambiguity exists but no result depends on it · `OPEN` — would
need evidence we do not have.

---

## OI-1 — Is the IPMI core index the physical die position? `RESOLVED — yes, thermally`

The clustering result is measured in *sensor-index* space. Reading it as *spatial* clustering —
the whole link to the yield literature — requires the index to track physical layout.

Measured over the full month 2022-08, both sockets, all 1,948 fully-configured sockets, at native
20 s cadence, scoring candidate floorplans by corr(predicted distance, thermal coupling):

| floorplan | fit (more negative = better) |
|---|---|
| **1×12 linear (identity index order)** | **−0.791** (null +0.001 ± 0.121, z = −6.55, p < 1e-4) |
| 6×2 | −0.742 |
| 4×3 | −0.578 |
| 3×4 | −0.481 |
| 2×6 | −0.238 |
| unconstrained hill-climb over all 12!/2 orderings | −0.793, converging to `[10,11,9,8,7,6,5,4,3,2,1,0]` |

The hill-climb answer is the identity **reversed** — physically identical, since distance is
reflection-invariant — differing by one adjacent transposition, |ρ| = 0.993.

**Residual caveat.** This is an inference from thermal coupling, not a vendor floorplan. An
annotated die shot, or the hostboot sensor-ID map, would confirm it directly. Leads:
`happytrees.org/dieshots`, `open-power/hostboot` (`src/usr/targeting/`).

**Methodological note.** A three-day sample *inverted* this conclusion and overstated the
within/cross-slice contrast threefold. Two bugs contributed: row-stride downsampling destroyed
cross-core timestamp alignment, and sampling nodes from `core0` biased the population, because
slice 0 is the most-harvested and so present on only ~39% of sockets.

---

## OI-2 — Did a BMC firmware update remap sensor IDs in March 2020? `BOUNDED`

86 of 235 transitions land in 2020-03-19…21 across 33+ nodes. A fleet-wide sensor-ID remap would
mimic a configuration change perfectly.

Maintenance logs are not available, but they are not needed: those days lie inside the
commissioning window, which is located empirically from two independent signals (`stabilization.py`)
and excluded. Re-running everything on 2020-06 onward leaves harvest maps **byte-identical for all
1,960 sockets present in both windows**, marginals agreeing to 0.1 pp, the lot boundary unchanged,
and no comparable date-clustering. The ambiguity is confined to a window no result depends on.

---

## OI-3 — Defect-driven or policy-driven harvesting? `OPEN — and not identifiable here`

Slice 0 is harvested on 88.2% of Lot A sockets but 39.3% of Lot B. Two hypotheses: harvesting
follows killer defects (H1), or slice 0 abuts a shared structure and is preferentially fused (H2).

**The fitted model cannot separate them, and the paper no longer claims it can.** A correlated
defect process and a binning policy that fuses contiguous blocks both predict γ > 0; γ measures
residual spatial association without attributing it. An earlier draft said the decomposition showed
"both" — that overstated the identification and was withdrawn.

Posterior-predictive checks (`ppc.py`) add a further constraint: the model is **misspecified**. It
reproduces the distinct-pattern count (443 observed vs 436.6 ± 6.2 replicated, p = 0.17) but not
the pattern frequencies (955.6 vs 494.8 ± 32.9), and it misplaces adjacency across the die —
over-predicting co-harvest at slices (7,8) and (3,4), under-predicting at (7,10) and (0,1), up to
4.8σ. Whatever drives the clustering is **not position-independent**.

**To settle.** Vendor disclosure, or the same analysis on a second AC922 installation.

---

## OI-4 — Reference and novelty check `RESOLVED (references) · OPEN (novelty)`

All references verified against their original sources. Two were materially wrong and corrected:
the negative-binomial model is **Koren, Koren & Stapper**, *IEEE T. Computers* 42(6):724–737, 1993
(not Stapper alone in *T. Semiconductor Manufacturing*); and the wafer-map work is **Bae, Hwang &
Kuo**, *IIE Transactions* 39(12):1073–1083, 2007.

The novelty statement rests on search-absence alone, so the paper states it as an **awareness
claim**, not a priority claim. Adjacent literature is cited explicitly so a reader can judge: PUF /
inherent hardware identifiers (which identify *which die*, not *what was disabled on it*), the
architectural sense of "core harvesting" as reclaiming idle *cycles*, node-level anomaly detection,
and the downstream M100 ExaData literature.

---

## OI-5 — Is collection dropout random? `RESOLVED — the lot result is immune`

Monthly coverage ranges from 18% to 99.4%. All set-comparison analysis restricts to *clean*
socket-days, so lot-correlated dropout could in principle manufacture the contrast. It does not
(`robustness.py`):

| clean-day coverage threshold | sockets | Lot A | Lot B | gap |
|---|---|---|---|---|
| all sockets | 1,962 | 88.2% | 39.3% | +48.9 pp |
| ≥90% | 1,847 | 88.0% | 38.9% | **+49.0 pp** |

Clean-day rates are 0.909 (Lot A) vs 0.902 (Lot B), and corr(slice-0 harvested, clean-day rate) =
**+0.004, p = 0.88** — the filter has no relationship to the quantity measured.

---

## OI-6, OI-7 — Sub-16 socket-days and same-day transitions `RESOLVED`

Most of the original sub-16 signal was **an extraction bug on our side**: three months had been
written from incomplete tar extractions, so absent metrics looked like disabled cores. The pipeline
now refuses to emit a month unless all 48 metrics are present, and the anomaly count fell from
11,459 to 9. Those 9 are the guard episodes.

The two nodes appearing to change configuration within a day (315, 342) are **same-day reboots**,
visible once examined at full timestamp resolution: both cut over on 2021-03-02 across a ~28-minute
reporting gap, 16 cores before and 16 after.

**Lesson:** any pipeline inferring "hardware absent" from "data absent" must gate on extraction
completeness, or partial ingestion masquerades as hardware failure.

---

## OI-8 — Why do p0/p1 labels swap? `LARGELY SETTLED`

Socket relabelling accounts for 19% of steady-state map changes. If the collector merely exchanged
the tags, every per-socket metric should exchange with them. Extending power collection from six
months to all 31 raises usable events from 4 to 23; restricting to events where the p0−p1 power
difference is large enough for its sign to mean anything (>2 W on both sides), **12 of 15 flip sign**
at the relabelling instant (80%, 95% CI [52, 96]).

The residual minority may be genuine die exchanges that happen to mirror, or differences too small
to resolve. Any per-socket longitudinal study on this dataset should still treat relabelling as a
hazard: it silently mixes two dies.

---

## OI-9 — The lot boundary has no external confirmation `OPEN`

No procurement or installation record was available, so "procurement lot" remains the most
parsimonious explanation rather than a documented fact.

The *statistics*, however, are no longer a maximised t masquerading as a p-value
(`changepoint.py`):

| | |
|---|---|
| permutation-calibrated max-t | null max\|t\| = 7.08 ± 2.41, never above 17.95 in 4,000 rack permutations, vs observed 26.6 → **p < 2.5e-4** |
| bootstrap CI for the breakpoint | **[22, 22]** — 100% of 2,000 within-rack resamples |
| number of populations | chosen by **rack-held-out** likelihood: −7.360 (k=1), **−7.133 (k=2)**, −7.139, −7.157, −7.149 |
| leave-one-rack-out | rack 22 recovered in **48 of 49** fits |
| held-out lot prediction | **72.9%** of sockets, **47/49 racks (95.9%)**, vs 55.1% majority-class |

Note on method: racks **cannot** be resampled with replacement here — that shuffles the order the
changepoint is defined on and returns a CI spanning the whole scan. The correct cluster bootstrap
keeps each rack in place and resamples the sockets inside it.

Two scan-hygiene points: candidate boundaries must leave ≥100 sockets per side, and **nodes 980–983
sit outside the 49 documented cabinets** and form a degenerate segment that captures the argmax of
an unguarded scan. Both are handled and stated in the paper.

---

## OI-10 — Single system `OPEN — the largest remaining weakness`

Every result comes from one machine. Whether the harvest-pattern statistics generalise to POWER9 as
a whole, or reflect CINECA's specific procurement, is untested. A replication on a second AC922
installation (Summit/Sierra telemetry, if obtainable) is the single highest-value addition and
would also settle OI-3.

---

## OI-11 — Pooled thermal estimator `OPEN — minor`

The pooled 24×24 matrix averages per-node matrices whose common-mode removal uses each node's own
16-core subset. Per-node standardisation of off-diagonal entries before pooling was not tried.
Where per-pair samples were not retained, reported bands are spreads **across core pairs**, not
bootstraps over sockets; the figures say which.

---

## OI-12 — Are the guard episodes genuine deconfigurations? `LARGELY SETTLED`

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

**Caveat.** The third episode (node 347, 2020-03) cannot be checked against the boot signal at all:
ganglia `boottime` collection begins 2020-05-05. That is absence of coverage, not absence of a
reboot. It also falls inside the excluded commissioning window, so the paper quotes **two
steady-state episodes** and reports the third separately rather than folding it into a rate.

---

## OI-13 — The power result was overstated as a null `RESOLVED — claim withdrawn`

An earlier version reported r = +0.039 (p = 0.003) explaining 0.16% of variance and concluded the
harvest map is "operationally invisible". Two defects: "no effect" was argued from a small r rather
than an equivalence test against a stated margin, and the interval treated node-days within a rack
as independent.

Re-run over the full 31 months (636,223 paired node-days, 980 nodes), adjusted for utilisation,
frequency, inlet air and month, with rack-clustered bootstrap intervals, against a ±5 W margin
(`equivalence.py`):

| feature | effect over range | 90% CI | TOST |
|---|---|---|---|
| mean active-slice index | +7.9 W | [+0.9, +16.6] | **not equivalent** |
| low-half active count | +5.8 W | [−2.2, +14.5] | inconclusive |
| adjacent active pairs | −1.4 W | [−4.5, +1.7] | equivalent |

Only one feature is bounded below the margin. The paper now reports a **bounded, not negligible**
effect: 7.9 W is 11% of the 69.0 W mean socket draw, against a p0–p1 positional asymmetry of 6.8 W
(10%) and a socket-to-socket spread of 19.9 W (29%).

---

## OI-14 — "Changes occur only across reboots" was inferred, not measured `RESOLVED`

The claim originally rested on every transition coinciding with a **gap in reporting**, which is
consistent with a reboot but does not demonstrate one. The dataset carries
`ganglia_pub/metric=boottime`, which makes reboots directly observable; it had simply never been
extracted (`sweep_covariates.py`).

Over 668,524 node-days on 990 nodes: **126 of 132 steady-state transitions with boot coverage
(95.5%) coincide with an observed reboot**, against **1.74%** of 1,194,581 non-transition intervals.
Matched on interval length the contrast holds (100% vs 16.7% at 3–4 days), and a gap-stratified
permutation test gives +93.7 pp against a null of +41.3 ± 3.5 pp, p < 2.5e-4.

The remaining 4.5% are transitions whose reboot fell outside what day-granularity can resolve, not
counterexamples attributable to in-operation change.

---

## OI-16 — The reliability null is bounded, not absolute `BOUNDED`

No feature of the baseline harvest map predicts subsequent processor replacement: out-of-fold
concordance is **0.448** against a permutation null of 0.499 ± 0.031 (p = 0.95), over 92 sockets
with an event and 53,597 socket-months at risk (`survival.py`).

**What this does not say.** With 92 events the study is not powered to exclude a small effect. The
rack-clustered intervals exclude hazard ratios beyond roughly ±1.5 per standard deviation — a
large effect is ruled out, a modest one is not. Two readings remain consistent with the data: the
defects that drive fusing at manufacture are not the ones that kill a part in the field, or
replacement is dominated by causes unrelated to the die. A fleet with more events, or a longer
record, would tighten this.

---

## OI-15 — Pair-level co-harvest counts are Monte-Carlo sensitive `BOUNDED`

The Bonferroni-corrected pair tests are computed against a curveball null estimated from 200 draws.
Repeating the whole null under five seeds gives **16–18 significant pairs**, of which 8–9 are
positive excesses and 8–10 deficits — individual pairs cross the threshold and back.

What is stable across every run: **exactly eight positive-excess pairs at |Δk| ≤ 2**, always the six
adjacent pairs plus 1&2 and 0&2, with 0&1 strongest at z ≈ +15 to +17. Deficits are overwhelmingly
distant, though one near pair (2&4) crosses in some runs. The paper reports the range rather than a
single run's counts, and quotes the mean-gap z to the nearest integer (≈ −20) for the same reason.

---

## Operational notes (not paper issues)

- The full sweep must run **alone**: a concurrent pandas job was OOM-killed while the sweep held
  ~18 GB.
- `sweep_all_months.py` and `sweep_covariates.py` are resumable — they skip months whose output
  exists. Launch them as tracked background jobs; a `nohup … &` died once when its parent shell was
  cleaned up.
- Do not write `pgrep -f <pattern>` inside a shell whose own command line contains `<pattern>`: it
  matches itself and any `until ! pgrep …` loop never terminates.
