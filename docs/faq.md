---
title: "FAQ — recovering CPU core-harvest maps from telemetry"
description: "Does this apply to my fleet? Is it a security vulnerability? How is it different from PUF fingerprinting? What can an operator actually do with a harvest map?"
---

# FAQ

## What is silicon harvesting?

Every large processor die ships with defects. Rather than discard it, the vendor permanently fuses
off the faulty units and sells the remainder as a lower-core-count part. A 16-core POWER9 is a
24-core die with four of its twelve *slices* fused off. The practice is universal and openly
acknowledged; what is proprietary is the **per-die map** of which units were disabled.

## Is this the same as "core harvesting" in cloud computing?

No, and the collision is unfortunate. In current architecture literature "core harvesting" almost
always means reclaiming **idle CPU cycles** for co-located work. This project is about **silicon**:
which physical cores were fused off at manufacture. The two are unrelated. We use *silicon
harvesting* or *die harvesting* to disambiguate.

## How does the recovery actually work?

A baseboard management controller exposes one thermal sensor per **physical** core position —
`p0_core0_temp` through `p1_core23_temp` on an AC922 — and does not renumber the survivors
compactly. A core that was fused off at manufacture never emits a sample. So the set of sensors
that report *is* the harvest map.

The validity argument is internal and simple: if the BMC renumbered survivors, every socket would
report indices 0–15 and there would be exactly one pattern. Instead we observe arbitrary 16-element
subsets of 0–23 and 443 distinct patterns. See [Method](method.md).

## Does this apply to my fleet?

It needs one thing: a management controller that exposes **one sensor per physical core position**
and does not renumber survivors. Check it in a minute:

- If every socket reports cores `0..15` — your BMC renumbers. The channel is closed; there is
  nothing to recover.
- If sockets report **arbitrary** subsets of `0..23` — the channel is open and the method applies
  directly.

A useful sanity check: harvested units should form complete pairs at `(2k, 2k+1)`. An **odd**
active-core count cannot arise from slice-granular fusing, so if you see one, you are looking at a
collector fault rather than at hardware.

## How much telemetry does an attacker need?

Almost none. Under healthy collection each core reports in essentially every sampling interval, so
**a single 20-second interval recovers the exact map for 99.7% of sockets**. Modelling the degraded
regime, exact recovery is `(1 - e^-λ)^16` where λ is samples per core in the window: negligible at
λ ≤ 1, 0.44 at λ = 3, 0.99 at λ = 7. The knee is around **two minutes** of ordinary telemetry.

The reason it is so cheap is that a harvested core can never emit a sample, so the channel has **no
false positives**. The only way to fail is for an active core to stay silent for the whole window,
and sixteen independent chances to miss is a demanding condition.

## I publish a telemetry dataset — what should I do?

Measured, not asserted:

**Does not work**

- *Reducing sampling cadence.* Recovery stays at 1.000 down to one sample per core per day. A
  single sample names its core just as well as 4,320 do.
- *Dropping samples at random.* Recovery is still 1.000 after discarding 99.9% of samples, and only
  falls to 0.61 at 99.99% — by which point the data is useless for its intended purpose.

**Works**

- *Publishing per-socket aggregates* instead of per-core series. Nothing per-core survives, so
  there is nothing to recover.
- *Renumbering survivors compactly* to `0..n-1`. Every socket then reports the same set by
  construction — which is precisely what this BMC does **not** do, and the whole reason the channel
  exists.
- *Per-node random sensor renumbering* destroys the map, though the active-core count still leaks.

The principle: the channel is carried by **which sensors exist**, not by what they report. Only
mitigations that remove per-core *position* work.

## Is this a security vulnerability?

Not in the usual sense, and we are careful not to overstate it. The information recovered is
**manufacturing provenance**, not credentials or user data, and it grants no access to any system.
Recovery is entirely passive — it reads an already-public dataset.

We measured how identifying a map is, precisely so this is not overstated: the pattern distribution
carries 8.08 bits and only **4.9%** of sockets have a map unique in the fleet. A harvest map is a
**partial** identifier — strong evidence about which *population* a part came from, weak evidence
about which *part* it is.

## How is this different from PUF or hardware fingerprinting?

PUFs and inherent hardware identifiers answer *which die is this*, for traceability and counterfeit
detection, using designed-in circuitry or physical access. We answer *what did the vendor disable on
it*, from telemetry alone. Different question, different mechanism, and — as the 4.9% figure above
shows — much weaker as an identifier.

## What can an operator actually do with this?

Three things, in decreasing confidence:

1. **Audit a procurement without vendor cooperation.** Detect whether a delivery is homogeneous. On
   Marconi100 a harvest map assigns a rack the model has never seen to the correct delivery
   **95.9%** of the time.
2. **Detect processor replacements with no asset database.** A change in a socket's map is a change
   of die. This yields a 2.1% annualised socket replacement rate, recovered without any maintenance
   log — and a taxonomy that separates genuine replacements from socket *relabelling*, which moves
   no silicon and accounts for 19% of changes.
3. **Spot field core-guard events.** Deconfigurations that would otherwise be visible only in
   service logs.

## What can it *not* do?

- **Predict which dies fail.** No feature of the baseline map predicts subsequent replacement:
  out-of-fold C-index 0.448 against a permutation null of 0.499 ± 0.031. That is a real null, and
  it is bounded rather than absolute — 92 events cannot exclude a small effect.
- **Tell you whether harvesting is defect-driven or policy-driven.** A correlated defect process and
  a binning policy that fuses contiguous blocks both predict the clustering we measure. The fitted
  model cannot attribute it.
- **Generalise beyond one machine.** Every result comes from Marconi100. See
  [Limitations](limitations.md).

## Is the sensor index really the physical position on the die?

It is recovered as such from thermal coupling, independently of any vendor document. Scoring
candidate floorplans by how well predicted physical distance explains measured thermal coupling,
the identity 1×12 linear ordering wins at −0.791 against a random-ordering null of +0.001 ± 0.121
(z = −6.55), and an unconstrained search over all 12!/2 orderings converges to the same arrangement
reversed — physically identical, since distance is reflection-invariant.

This licenses reading the clustering *spatially*. It remains an inference from thermal coupling
rather than a vendor floorplan; an annotated die shot would confirm it directly.

## Can I reuse the code?

Yes — MIT licensed. Please cite the manuscript; `CITATION.cff` in the repository is
machine-readable. If you replicate on another system, we would genuinely like to hear about it,
including if you **disagree**. See [Limitations](limitations.md).
