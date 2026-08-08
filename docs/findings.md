---
title: "Findings — what BMC telemetry reveals about POWER9 core harvesting"
description: "Slice-granular fusing with zero exceptions, per-die rather than per-SKU maps, spatial defect clustering, a procurement-lot boundary, field core-guard episodes, and two bounded nulls."
---

# Findings

All figures are over 1,962 POWER9 sockets with a well-defined map, across 1.67 million socket-days
(2020-03 to 2022-09).

## Harvesting operates exactly at the slice

A POWER9 die carries 24 SMT4 cores organised as **12 slices**, each slice being two cores plus
their shared L2 and L3. The slice is the natural fusing granularity, because disabling one SMT4
core would strand its half of a shared cache block.

That is exactly what the telemetry shows: **1,962 of 1,962 sockets** have their harvested cores
forming complete `(2k, 2k+1)` pairs. Zero exceptions in 1.67 million socket-days. This rules out
per-core fusing.

## The map is per-die, not per-SKU

**443** distinct patterns appear among the 495 possible. The most common covers only 5.5% of
sockets. Within a node the two sockets match in just 1.2% of cases. Two processors of the same part
number are not the same object.

## Harvested slices cluster spatially

Under the curveball null, the mean index gap between harvested slices is 4.243 versus 4.585 ± 0.016
(z ≈ −20). At pair level, **exactly eight positive-excess pairs sit at |Δk| ≤ 2 in every run** —
the six adjacent pairs plus 1&2 and 0&2.

The effect is real but moderate, and we give the size as well as the significance, because a z
grows with √n and so reports sample size as much as effect: averaged over pairs at |Δk| ≤ 2 the
excess is 1.12× the null, against 0.94× for more distant pairs, with the strongest individual pairs
reaching 1.7–1.8×.

## The sensor index is the physical die ordering

Scoring candidate floorplans by how well predicted physical distance explains measured thermal
coupling — coupling should fall with distance, so a better floorplan gives a more negative
correlation:

| Floorplan | Fit |
|---|---|
| **1×12 linear (identity index order)** | **−0.791** |
| 6×2 | −0.742 |
| 4×3 | −0.578 |
| 3×4 | −0.481 |
| 2×6 | −0.238 |
| unconstrained hill-climb over all 12!/2 orderings | −0.793 |

Against a random-ordering null of +0.001 ± 0.121 (z = −6.55). The hill-climb answer is the identity
**reversed** — physically identical, since distance is reflection-invariant — differing by one
adjacent transposition, |ρ| = 0.993.

This is what licenses reading the clustering *spatially* rather than merely in index space.

## A procurement-lot boundary

A changepoint scan puts the boundary at **rack 22**, and it survives proper calibration — a
maximised Welch *t* is not itself a p-value:

- Rack-permutation null: max|t| = 7.08 ± 2.41, never above 17.95 in 4,000 draws, against an
  observed 26.6 → **calibrated p < 2.5e-4**
- Bootstrap CI for the breakpoint: **[22, 22]** (100% of 2,000 within-rack resamples)
- Number of populations chosen by *rack-held-out* likelihood: **k = 2** wins
- Leave-one-rack-out: rack 22 recovered in **48 of 49** fits
- Held-out prediction: **47 of 49 racks (95.9%)** assigned to the correct delivery

The two lots differ enormously — slice 0 is harvested on 88.2% of one and 39.3% of the other — and
the geometric alternative is excluded: once the lot is accounted for, correlations with machine-room
position vanish. The causal direction is fixed *a priori* anyway, since fusing happens at
manufacture, months before a part is installed.

## Configurations are static, and change at reboot

**89.5%** of sockets never change configuration. Out of sample, a map derived from the first half of
the record predicts **96.8%** of 839,067 held-out socket-days more than a year later.

When they do change, it is at reboot — and this is **measured**, not inferred from gaps in
reporting. Using the OS's own `boottime` metric, **126 of 132** steady-state transitions coincide
with an observed reboot, against **1.74%** of 1,194,581 non-transition intervals. Matched on
interval length the contrast holds: 100% versus 16.7% at 3–4 days.

## Field core guarding, corroborated three ways

Three sockets run on 14 cores for 2–4 days, each losing exactly one slice. Telemetry alone cannot
distinguish a real GARD event from a collection failure, so three independent collectors were
checked:

1. **Every other sensor kept full cadence** — all ten control metrics on the same socket sustained
   4,320 samples/day while the two slice sensors read zero.
2. **The OS agrees, node for node** — of 990 nodes, exactly three ever report 120 logical CPUs
   (= 128 − 8, one slice at four threads), and they are the same three. Under a null flagging three
   arbitrary nodes, the chance of matching is 6.2e-9.
3. **They sit across reboots**, as a hostboot-applied GARD record requires.

## Hardware changes, without an asset database

A taxonomy separates what physically happened: **RELABEL** (p0 loses exactly what p1 gains — no
silicon moved), **CPU-SWAP** (one processor replaced), **NODE-SWAP**, and **GUARD**.

This yields an annualised socket replacement rate of **2.1%** (95% CI [1.7, 2.6]) recovered with no
maintenance log. It also shows that **19% of steady-state map changes move no silicon at all** — a
study reading every change as a replacement would overcount by that much.

## Two bounded nulls

Negative results, reported as carefully as the positive ones.

**The map does not predict which dies get replaced.** Out-of-fold concordance is **0.448** against
a permutation null of 0.499 ± 0.031 (p = 0.95). No coefficient's interval excludes zero. This is
bounded rather than absolute: 92 events exclude hazard ratios beyond roughly ±1.5 per standard
deviation, but not a small effect.

**Its effect on power is bounded, not negligible.** An earlier version of this work called harvest
maps "operationally invisible"; that claim was **withdrawn**. Over the full 31 months with
rack-clustered intervals, the strongest feature moves socket power +7.9 W across its range, 90% CI
[+0.9, +16.6], against a ±5 W margin — comparable to which slot a processor occupies (6.8 W), and
small against the socket-to-socket spread (19.9 W), but not bounded below relevance.

---

*Next: [Reproduce](reproduce.md) · [FAQ](faq.md) · [Limitations](limitations.md)*
