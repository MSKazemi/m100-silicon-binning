---
title: "FAQ — recovering CPU core-harvest maps from telemetry"
description: "Does this apply to my fleet? Is it a security vulnerability? How is it different from PUF fingerprinting? What can an operator actually do with a harvest map?"
---

# FAQ

## What is silicon harvesting?

A die with faulty units can still be sold once those units are disabled, and vendors also disable
working units to fill cheaper products; either way the die ships as a lower-core-count part. A
16-core POWER9 is a 24-core die with eight cores disabled; on Marconi100 they always come as four
of its twelve *slices*, a slice being what IBM's manual calls a *processor-pair cache slice*, two
cores with their L2 and L3. The practice is openly acknowledged; what vendors do not publish is the
**per-die map** of which units were disabled. Telemetry shows *which* positions are disabled, not
*why*.

## Is this the same as "core harvesting" in cloud computing?

No, and the collision is unfortunate. In current architecture literature "core harvesting" almost
always means reclaiming **idle CPU cycles** for co-located work. This project is about **silicon**:
which physical cores are disabled on each die. The two are unrelated. We use *silicon
harvesting* or *die harvesting* to disambiguate.

## How does the recovery actually work?

A baseboard management controller exposes one thermal sensor per core position on the die —
`p0_core0_temp` through `p1_core23_temp` on an AC922 — and does not renumber the survivors
compactly. A disabled core never emits a sample, so the sensors that never report mark exactly the
disabled positions: the harvest map. (Stewart Smith read the same sensors on a single POWER9
machine in 2020; the new part is doing it for a whole fleet from published data.)

The validity argument is internal and simple: if the BMC renumbered survivors, every socket would
report indices 0–15 and there would be exactly one pattern. Instead we observe arbitrary 16-element
subsets of 0–23 and 443 distinct patterns. See [Method](method.md).

## Does this apply to my fleet?

It needs one thing: a management controller that exposes **one sensor per core position** and
does not renumber survivors. Check it in a minute:

- If every socket reports cores `0..15` — your BMC renumbers. The channel is closed; there is
  nothing to recover.
- If sockets report **arbitrary** subsets of `0..23` — the channel is open and the method applies
  directly.

A useful sanity check: on Marconi100 the disabled units always form complete pairs at
`(2k, 2k+1)`. An **odd** active-core count is not a baseline map: it is either a collector fault or
a single core deconfigured in the field. The operating system's CPU count tells them apart.

## How much telemetry does an attacker need?

Almost none. Under healthy collection each core reports in essentially every sampling interval,
and the collector's gaps almost always silence a whole socket rather than single cores, so,
measured on the raw timestamps of 2022-08, **a single 20-second interval recovers the exact map in
99.95% of socket-intervals**, and in all but 5 in 10 million of those in which the socket reports
at all. Modelling the degraded regime, where the number of samples surviving from each core is
Poisson with mean λ, independently across cores, exact recovery is `(1 - e^-λ)^16`: negligible at
λ ≤ 1, 0.44 at λ = 3, 0.99 at λ = 7. An attacker who knows that cores are disabled in pairs needs
only half as many samples, one per active slice.

The reason it is so cheap is that a harvested core can never emit a sample, so the channel has **no
false positives**. The only way to fail is for an active core to stay silent for the whole window,
and sixteen independent chances to miss is a demanding condition.

## I publish a telemetry dataset — what should I do?

Measured, not asserted:

**Does not work**

- *Reducing sampling cadence.* Keeping one 20 s slot in every six hours still recovers every map
  from a single day, and one slot per day recovers 99.95%. A single sample names its core just as
  well as 4,320 do.
- *Deleting samples at random.* Independent deletion of 99.9% of samples leaves one-day exact
  recovery at 0.81, and at 0.9985 for an attacker who uses the pairing of cores into slices. At
  99.99% a single day rarely gives the exact map yet still reveals about 4.6 of the 8 active slices,
  and because the maps are static for months an attacker pools days: one month recovers 0.99997.
  Holding one-month recovery below one half takes 99.998% deletion (99.9991% against the
  pair-aware attacker) — by which point the data is useless for its intended purpose. Deleting
  whole timestamps protects even less (0.99 at 99.9%).

**Works**

- *Publishing per-socket aggregates* instead of per-core series. Nothing per-core survives, so
  there is nothing to recover.
- *Renumbering survivors compactly* to `0..n-1`. Every socket then reports the same set by
  construction — which is precisely what this BMC does **not** do. The temperatures still carry a
  weak positional trace (AUC 0.65 for a harvested slice between two consecutive survivors) that the
  two decoders we tried could not turn into maps.
- *Per-node random sensor renumbering* destroys the map, though the active-core count still leaks,
  and the permutation must be redrawn for every release: one fixed over time makes the set of
  reporting pseudonyms a fingerprint unique for about 99.7% of sockets (against 4.9% for the maps
  themselves) and still exposes every die change.

The principle: the channel is carried by **which sensors exist**, not by what they report. Only
mitigations that remove per-core *position* work.

## Is this a security vulnerability?

Not in the usual sense, and we are careful not to overstate it. The information recovered is
**manufacturing provenance** — a vendor-undisclosed configuration attribute — not credentials or
user data, and it grants no access to any system. Recovery is entirely passive — it reads an
already-public dataset. Its direct severity is modest; it matters for telemetry publication policy
and supply-chain confidentiality. See [`SECURITY.md`](https://github.com/MSKazemi/m100-silicon-binning/blob/main/SECURITY.md).

We measured how identifying a map is: the pattern distribution carries 8.08 bits (plug-in;
8.24–8.33 bias-corrected) and only **4.9%** of sockets have a map unique in the fleet. Across a
whole fleet a single map is only a **partial** identifier — strong evidence about which
*population* a part came from, weak evidence about which *part* — but side information changes
that: with the rack known 78% of sockets are unique, and a node's two maps together are unique
for 93% of nodes.

## How is this different from PUF or hardware fingerprinting?

PUFs and inherent hardware identifiers answer *which die is this*, for traceability and counterfeit
detection, using designed-in circuitry or physical access. We answer *what did the vendor disable on
it*, from telemetry alone. Different question, different mechanism, and — as the 4.9% figure above
shows — much weaker as an identifier of a single part across a fleet, though not of a node.

## What can an operator actually do with this?

Three things, in decreasing confidence:

1. **Detect latent hardware cohorts without vendor cooperation or a procurement database.** Test
   whether the installed parts are one population. On Marconi100 a harvest map places **95.9%** of
   racks the model has never seen in their inferred lot, with the boundary re-found without them.
2. **Infer processor replacements with no asset database.** A change in a socket's map is most
   plausibly a change of die. Sockets change die at 2.1% per socket-year, recovered without any
   maintenance log; because a map travels with its die, 50 of the 96 changes are traced to dies
   moved from another socket, leaving an inferred 1.0–1.2% per socket-year of new processors. A
   mirror test separates socket *relabelling*, 19% of changes, in which no die moves.
3. **Spot field core-guard events.** Deconfigurations that would otherwise be visible only in
   service logs: nine in steady state, each confirmed by the OS's own CPU count.

## What can it *not* do?

- **Predict which dies fail.** No feature of the baseline map is a significant predictor of
  subsequent replacement: held-out risk-set concordance 0.55 (95% CI [0.48, 0.61]) against a
  permutation null of 0.50 ± 0.04. Even the upper bound is poor discrimination, so the map is no
  use as a reliability covariate; it is still a bounded null, and 85 events cannot exclude a modest
  effect.
- **Tell you whether harvesting is defect-driven or policy-driven.** A correlated defect process and
  a binning policy that disables contiguous blocks both predict the clustering we measure. The
  fitted model cannot attribute it.
- **Generalise beyond one machine.** Every result comes from Marconi100. See
  [Limitations](limitations.md).

## Is the sensor index really the physical position on the die?

Each index is a fixed core position, and IBM documents where: its POWER9 Registers Specification
(Vol. 1, v1.2, Fig. 2) labels the core chiplets EC00–EC23, the core numbers the per-core sensors
carry, and places the six quads in two bands of three, each quad's four cores stacked. So the
index is not a left-to-right line: index neighbours are die neighbours only within a quad. Thermal
coupling confirms the quads, but beyond them it follows how the surviving cores are numbered, so it
does not measure the geometry — the best one-dimensional ordering it finds (the index order, 2nd of
all 12!/2) is the numbering, not a row of slices on the die. The clustering is spatial at the scale
of a quad, by IBM's diagram.

## Can I reuse the code?

Yes — MIT licensed. Please cite the manuscript; `CITATION.cff` in the repository is
machine-readable. If you replicate on another system, we would genuinely like to hear about it,
including if you **disagree**. See [Limitations](limitations.md).
