---
title: "Limitations — what is not settled, and what would settle it"
description: "The single-system limitation, the unidentified defect-vs-policy question, the withdrawn power claim, and the full open-issues register."
---

# Limitations

This project publishes its open issues rather than burying them. The full register — sixteen
entries, each with what is known, what would settle it, and where to look — is
[`LIMITATIONS.md`](https://github.com/MSKazemi/m100-silicon-binning/blob/main/LIMITATIONS.md) in
the repository.

## The largest one: a single system

Every number comes from one machine, Marconi100. Whether the lot structure or the clustering
generalises to other POWER9 installations, or to other vendors, is **untested**. A replication on a
second AC922 installation is the single highest-value addition anyone could make to this work, and
it would also settle the defect-versus-policy question below.

If you have per-core BMC telemetry from any comparable fleet, we would like to hear from you —
including if your results **disagree**. A disagreement would delimit exactly when the technique
works, which is worth as much as a confirmation.

## Defect-driven or policy-driven? Not identified

Slice 0 is harvested on 88.2% of one lot's sockets but 39.3% of the other's. Two hypotheses:
harvesting follows killer defects, or slice 0 abuts a shared structure and is preferentially fused
for reasons of binning policy.

**The fitted model cannot separate them, and we no longer claim it can.** A correlated defect
process and a binning policy that fuses contiguous blocks both predict the clustering we measure.
An earlier draft said the decomposition showed "both"; that overstated the identification and was
withdrawn.

Posterior-predictive checks add a further constraint: the model is **misspecified**. It reproduces
the distinct-pattern count but not the pattern frequencies, and it misplaces adjacency across the
die — over-predicting co-harvest at some slice pairs and under-predicting at others, up to 4.8σ.
Whatever drives the clustering is **not position-independent**.

## The power claim was withdrawn

An earlier version reported a small correlation and concluded that harvest maps are "operationally
invisible". That does not survive two corrections: "no effect" was argued from a small *r* rather
than an equivalence test against a stated margin, and the interval treated node-days within a rack
as independent.

Re-run over the full record with covariate adjustment and rack-clustered intervals, only one of
three map features is bounded below a ±5 W margin. The strongest moves socket power +7.9 W across
its range, 90% CI [+0.9, +16.6]. The honest summary is **bounded, not negligible**.

We record this because it is the most transferable thing we learned: **analysis choices moved our
numbers more often than the data did**, four times, and every one of them ran in the direction we
would have preferred.

## Other open items

- **The lot boundary has no external confirmation.** Statistically sharp, but no procurement or
  delivery record was available to check it against. "Procurement lot" is the most parsimonious
  explanation, not a documented fact.
- **The floorplan is inferred, not documented.** The sensor index is recovered as the physical
  ordering from thermal coupling alone. An annotated POWER9 die shot, or the hostboot sensor-ID
  map, would confirm it directly.
- **The reliability null is bounded.** 92 events cannot exclude a small effect.
- **Pair-level co-harvest counts are Monte-Carlo sensitive** — 16–18 significant pairs depending on
  seed. The invariant is exactly eight positive excesses at |Δk| ≤ 2, every run.
- **The novelty claim is an awareness claim**, not a priority claim. It rests on a systematic but
  necessarily incomplete search. If you know of earlier work recovering per-die disable maps from
  deployed telemetry, please tell us.

---

*Back to [home](index.md) · [FAQ](faq.md) · [full register](https://github.com/MSKazemi/m100-silicon-binning/blob/main/LIMITATIONS.md)*
