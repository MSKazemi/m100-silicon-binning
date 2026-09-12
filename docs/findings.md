---
title: "Findings — what BMC telemetry reveals about POWER9 core harvesting"
description: "Slice-granular disable maps with zero exceptions, per-die rather than per-SKU maps, clustering of disabled slices within quads, a rack-aligned lot boundary, field core-guard episodes, and two bounded nulls."
---

# Findings

All figures are over 1,962 POWER9 sockets with a well-defined map, across 1.67 million socket-days
(2020-03 to 2022-09). A **harvest map** is the set of disabled core positions on a die; the
telemetry shows which positions are disabled, not why.

## Every map disables whole slices

A POWER9 die carries 24 SMT4 cores organised as **12 slices**, each slice being two cores plus
their shared L2 and L3 (IBM's manual calls it a *processor-pair cache slice*). The hardware can
also deconfigure a single core (POWER9 User's Manual v2.1, p. 311), and an 8-core POWER9 part runs
one core per slice, so slice granularity is not automatic.

Yet it is exactly what the telemetry shows: **1,962 of 1,962 sockets** have their disabled cores
forming complete `(2k, 2k+1)` pairs, with no exception on 1.52 million clean socket-days. Slice
granularity is a property of these 16-core parts, not a constraint of the die.

## The map is per-die, not per-SKU

**443** distinct patterns appear among the 495 possible. The most common covers only 5.5% of
sockets. Within a node the two sockets match in just 1.2% of cases. Two processors of the same part
number are not the same object.

## Harvested slices cluster, above all within quads

Under the curveball null, the mean index gap between harvested slices is 4.242 versus 4.582 ± 0.016
(z = −20.7; none of 100,000 exchangeable null draws is as small). At pair level, 20 of the 66 pairs
are family-wise significant, and **exactly eight positive-excess pairs sit at |Δk| ≤ 2** — the six
slice pairs (2m, 2m+1) that share a POWER9 *quad*, plus 1&2 and 0&2. Of the neighbours that
straddle two quads only 1&2 shows an excess; 3&4, 5&6, 7&8 and 9&10 show none. With each lot's
marginals kept apart, 11 pairs remain, including seven of the eight near excesses.

The effect is real but moderate, and we give the size as well as the significance, because a z
grows with √n and so reports sample size as much as effect: averaged over pairs at |Δk| ≤ 2 the
excess is 1.12× the null, against 0.94× for more distant pairs, and the six largest excesses,
1.32–1.73× (strongest 6&7), are exactly the six same-quad pairs. IBM's chip diagram places the two
slices of each quad side by side, so this part of the clustering is spatial, at the scale of a
quad. Spatially clustered defects and disabling at quad granularity would both produce it, and the
telemetry cannot tell them apart.

## Thermal coupling follows the quad, not the slice

Over every socket with both halves fully configured — **1,948 sockets** — in the month 2022-08, at
native 20 s cadence: at identical index distance 1, within-slice siblings couple more strongly than
cross-slice neighbours on average (r = +0.372 against +0.312; contrast +0.061, 95% CI
[0.047, 0.076], resampling the 49 racks). That average mixes two kinds of neighbour: cross-slice
neighbours within a quad couple at +0.480, those across a quad boundary at +0.104. Matched within
each socket, siblings couple *less* than the adjacent core of the other slice in their quad and far
more than cores across a quad boundary. The thermal unit is the quad; the slice is the unit of
disabling.

## The sensor index is IBM's core numbering

IBM's POWER9 Processor Registers Specification (Vol. 1, v1.2, Fig. 2) labels the core chiplets
EC00–EC23 — the core numbers the per-core sensors carry — and places the six quads in two bands of
three, each quad's four cores stacked. Index neighbours are therefore die neighbours only within a
quad. Scoring candidate layouts by the Pearson correlation, over the 66 slice pairs with equal
weight, between layout distance and thermal similarity — coupling should fall with distance, so a
better layout gives a more negative score:

| Layout | Score |
|---|---|
| 1×12 linear (identity index order) | −0.791 |
| exact optimum over all 12!/2 orderings (dynamic programming) | −0.793, `[0,…,9,11,10]` |
| IBM's documented layout, at the diagram's proportions | −0.824 |

Against a random-ordering null of −0.003 ± 0.123 (z = −6.41, rack-bootstrap [−6.55, −6.24]); the index order is 2nd of 239,500,800
orderings. IBM's layout scores best on all 66 pairs, but only through the six same-quad pairs: over
the 60 pairs in different quads the index order fits better (−0.814 vs −0.734). Beyond the quad the
coupling follows how the surviving cores are numbered: within a die column a slice couples more
with its index neighbour at the far edge than with the nearer slice across the nest (by 0.055, 95%
CI [0.046, 0.066], where physical models on IBM's layout predict the opposite sign), and the
coupling of the same two slices falls 0.080 per active slice between them — plausibly workload
placed on consecutive logical CPUs. So the 1-D ordering it recovers is the numbering, not the
geometry; **the layout comes from IBM's documentation, not from telemetry**, and the scores are
descriptive.

## A lot boundary at rack 22

A changepoint scan puts the boundary at **rack 22**, and it survives proper calibration — a
maximised Welch *t* is not itself a p-value:

- Rack-permutation null: max|t| = 7.08 ± 2.41, never above 17.95 in 4,000 draws, against an
  observed 26.6 → **calibrated p = 2.5e-4** ((b+1)/(B+1), the floor of 4,000 draws)
- Breakpoint stability: rack 22 in 100% of 2,000 within-rack socket resamples and 99.6% of node
  resamples (conditional on the racks, so a stability check, not a CI); at the rack level and
  under the fitted two-lot model, 99.7% of rack-block resamples and 98.7% of a beta-binomial
  parametric bootstrap
- Number of populations chosen by *rack-held-out* likelihood: **k = 2** wins, also under three
  models defined on the 495 valid maps
- Leave-one-rack-out: **49 of 49** fits split the remaining racks exactly as the full data do
- Nested held-out prediction (boundary re-found without the held-out rack): **47 of 49 racks
  (95.9%)** assigned to their inferred lot

The two lots differ enormously — slice 0 is harvested on 88.2% of one and 39.3% of the other — and
the geometric alternative is excluded: once the lot is accounted for, correlations with machine-room
position vanish, and the boundary falls between adjacent cabinets of one row. Dies that arrive
later in Lot A racks do not take on the Lot A signature (61% vs 88% slice-0 harvested), so it
travels with the dies. Procurement batches are the most plausible reading; no delivery record
confirms it.

## Configurations are static, and change at reboot

**89.5%** of sockets never change configuration. Out of sample, a map derived from the first half of
the record predicts **96.8%** of 839,067 held-out socket-days more than a year later.

When they do change, it is at reboot — and this is **measured**, not inferred from gaps in
reporting. Using the OS's own `boottime` metric, **126 of 132** steady-state transitions coincide
with an observed reboot, against **1.74%** of 1,194,581 non-transition intervals. Matched on
interval length the contrast holds: 100% versus 16.7% at 3–4 days, and it survives an exact test
stratified by node (p = 6e-31). The six without a flagged reboot fall in a Ganglia outage; the boot
epoch reported when collection resumed lies inside every one of their intervals, so dated by epoch
**133 of 136** transitions contain a boot and none is seen without one.

## Field core guarding, corroborated four ways

Three sockets run on 14 cores for 2–4 days, each losing exactly one slice. Telemetry alone cannot
distinguish a real GARD event from a collection failure, so independent collectors were checked:

1. **Every other sensor kept full cadence** — all ten control metrics on the same socket sustained
   4,320 samples/day while the two slice sensors read zero.
2. **The OS agrees, node for node** — of 990 nodes, exactly three ever report 120 logical CPUs
   (= 128 − 8, one slice at four threads), and they are the same three. Under a null flagging three
   arbitrary nodes, the chance of matching is 6.2e-9.
3. **They sit across reboots**, as a hostboot-applied GARD record requires.
4. **Every OS departure from 128 is seen by the BMC** — six further nodes report 124 logical CPUs
   (one core short), and on each the BMC loses exactly one core sensor from the same boot. All nine
   OS departures in the record are matched, and each is followed within six days by a new map on
   that socket.

The telemetry shows the deconfiguration, with the signature of a GARD record applied at boot, not
the GARD record itself.

## Hardware changes, without an asset database

A taxonomy names each class of map change by the event it is consistent with: **RELABEL** (p0 loses
exactly what p1 gains — tags swapped, no die moved), **CPU-SWAP** (one socket receives another
processor), **NODE-SWAP** (both do), and **GUARD** (a core or slice lost at a boot).

Sockets change die at **2.1%** per socket-year (rack-clustered 95% CI [1.6, 2.7]), recovered with no
maintenance log. Because a map travels with its die, it also traces dies that move: **50 of the 96**
die changes bring in exactly the map another socket gave up within a week (4.4 expected by chance),
most of them in two-way exchanges between nodes, so new processors enter at an inferred
**1.0–1.2%** per socket-year. And **19% of steady-state map changes move no die at all** — a study
reading every change as a replacement would overcount. Only service records could measure each
class's precision.

## Two bounded nulls

Negative results, reported as carefully as the positive ones.

**The map does not usefully predict which dies get replaced.** Held-out risk-set concordance is
**0.55** (95% CI [0.48, 0.61]) against a rack-permutation null of 0.50 ± 0.04 (p = 0.12), and
adding the map does not improve the held-out survival likelihood. No hazard-ratio interval excludes
1. This is bounded rather than absolute: a hazard increase above 21% per SD of a continuous map
feature is excluded, but protective effects, and effects of up to about threefold for slice-0
status or lot, are not.

**Its association with power is bounded and small, but not provably negligible.** An earlier
version of this work called harvest maps "operationally invisible"; that claim was **withdrawn**.
With each day's power joined to the map in force that day and rack-clustered intervals, the three
map summaries shift the p0−p1 socket-power difference by +2.7, −1.5 and −1.0 W across their full
observed ranges (90% CIs within ±5.6 W, and within ±7.4 W in every sensitivity design; under
0.5 W per standard deviation). Two of three are equivalent at a ±5 W margin and all three at
±10 W. The 6.8 W mean p0−p1 difference reflects scheduler placement rather than socket position,
and sockets differ from one another by 19.9 W (SD).

---

*Next: [Reproduce](reproduce.md) · [FAQ](faq.md) · [Limitations](limitations.md)*
