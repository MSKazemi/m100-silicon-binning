---
title: "Method — how per-die harvest maps are recovered from BMC telemetry"
description: "The recovery rule, its validity argument, the four failure modes each with a detector, and the pipeline that makes it reliable at 1.67 million socket-days."
---

# Method

## The recovery rule

For each month, enumerate the `metric=pX_coreY_temp` partitions and count samples per
(node, socket, core). **A core index is *present* for a socket if it emits any sample in the
window.** The disable map of socket *s* is then `M[s,k] ∈ {0,1}` over the twelve slices
`k = ⌊core/2⌋`.

That is the whole method. Everything else is making it reliable.

## Why it is valid

The key argument is that **the BMC does not renumber active cores compactly**. Were that so, every
socket would report indices 0–15 and there would be exactly one observed pattern. Instead we
observe arbitrary 16-element subsets of {0,…,23} and **443 distinct patterns**, so the index is a
fixed physical-position identifier.

That argument is internal. It is also confirmed externally, by two collectors that never touch the
IPMI sensor family:

- **The operating system.** POWER9 runs four SMT threads per core, so two sockets of 16 cores
  should present 128 logical CPUs. Across 847,297,478 samples on 990 nodes, the OS reports exactly
  128 on **99.9853%** of samples.
- **The scheduler.** SLURM's configured-CPU totals across four partitions sum to 125,696 = 982×128.

The single OS-level anomaly in the month used for thermal work is itself a confirmation: exactly
one node deviates, and that window contains the only BMC-visible reconfiguration on that node. Two
collectors with nothing in common flag the same node in the same 48 hours.

## Four things that could make this wrong

Each has a detector, and one of them caught a real defect in our own pipeline.

| Failure mode | Detector |
|---|---|
| **Partial ingestion** — extraction silently truncates, so absent metrics look like harvested cores | A completeness gate refusing to emit a month unless all 48 metrics are present |
| **Compact renumbering** — a BMC that renumbers survivors would yield nothing but a constant | Arbitrary subsets and 443 patterns; a renumbering BMC produces exactly one |
| **Per-metric collector faults** — a dropped stream mimics deconfiguration | Odd active-core counts, which slice-granular fusing cannot produce; and control sensors on the same socket keeping full cadence |
| **Socket-tag swaps** — exchanged p0/p1 tags silently mix two dies in a longitudinal study | A mirror test, which identifies 19% of steady-state changes as relabelling rather than replacement |

The first of these was found the hard way: three months had been written from incomplete
extractions, one missing all 24 sensors of socket p1, which would have read as a socket with no
active cores at all. **Any pipeline that infers "hardware absent" from "data absent" must gate on
extraction completeness**, or partial ingestion masquerades as hardware failure.

## The pipeline

1. **Extraction.** Each monthly archive is scanned once; only the 48 `p{0,1}_core{0..23}_temp`
   partitions are extracted, behind the completeness gate above.
2. **Daily aggregation.** Samples reduce to a count per (node, socket, core, day) — a 15 MB table
   from a 400 GB scan. Every structural and longitudinal result is computed from that table.
3. **The clean-day filter.** A socket-day is *clean* when exactly 16 cores report **and** each
   reports more than 90% of the 4,320 samples expected at 20 s cadence. This retains 1,517,473 of
   1,671,297 socket-days (90.8%).
4. **Per-socket map.** A socket is summarised by the modal harvest map over its clean days (median
   775 days supporting each map).

The clean-day threshold is the load-bearing discretionary choice, so its effect is stated plainly:
moving it from 0.50 to 0.99 discards 18% of socket-days and changes nothing that matters — the same
1,962 sockets, 442–443 patterns, an identical 61.2% slice-0 rate and an identical lot contrast.

## Thermal probing of die topology

Cores on a socket share workload and inlet air, producing a dominant common mode. We remove it by
subtracting the per-timestamp cross-core mean, double-centre, and compute residual correlations at
the sensors' **native 20 s cadence** — no resampling.

That last point matters more than it sounds. An earlier version aligned onto a one-minute grid, a
step introduced to work around an unrelated bug and never revisited. It is not neutral: coarser
averaging suppresses high-frequency noise and inflates every correlation, within-slice more than
cross-slice, growing the contrast from +0.063 at 20 s to +0.140 at 15 min. We quote the
native-resolution figure, which is the most conservative.

Because common-mode removal forces each row of the residual matrix to sum to approximately zero,
negative values at large index gaps are partly mechanical. We therefore rely on contrasts **at
identical index distance**, which control for this exactly.

## Null model

Every socket has exactly four harvested slices, so the constraint induces negative dependence and a
naive independence null is invalid. We use the **curveball algorithm**, which preserves *both* row
sums (4 per socket) and column sums (per-slice marginals) while randomising association structure,
and assert both on every draw.

Monte-Carlo noise is real and reported rather than hidden: repeating the whole null at 25–400 draws
moves the resulting z by about ±1, which is why we quote it to the nearest integer (≈ −20) and give
pair-level counts as a range.

---

*Next: [Findings](findings.md) · [Reproduce](reproduce.md) · [Limitations](limitations.md)*
