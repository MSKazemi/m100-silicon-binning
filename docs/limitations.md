---
title: "Limitations — what is not settled, and what would settle it"
description: "The single-system limitation, the unidentified defect-vs-policy question, the bounded power association, the documented die layout, and the full open-issues register."
---

# Limitations

This project publishes its open issues rather than burying them. The full register — nineteen
issues, each with what is known, what would settle it, and where to look — is
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
harvesting follows killer defects, or slice 0 abuts a shared structure and is preferentially disabled
for reasons of binning policy. Telemetry shows which positions are disabled, not why: a partial-good
record, a field failure or a licensing override can each disable a core.

**The fitted model cannot separate them, and we no longer claim it can.** A correlated defect
process and a binning policy that disables contiguous blocks both predict the clustering we measure.
An earlier draft said the decomposition showed "both"; that overstated the identification and was
withdrawn.

Parametric model-replicate checks add a further constraint: the model is **misspecified**. It
reproduces the distinct-pattern count but not the pattern frequencies, and it misplaces adjacency
across the die — over-predicting co-harvest at some slice pairs and under-predicting at others, with
simulation-calibrated residuals up to |z| = 9.2 and 10 of 66 cells outside the simultaneous 95%
band. Whatever drives the clustering is **not position-independent**: it sits mainly on slice pairs
that share a POWER9 quad.

## The power claim was withdrawn, and replaced by a bound

An earlier version reported a small correlation and concluded that harvest maps are "operationally
invisible". That does not survive two corrections: "no effect" was argued from a small *r* rather
than an equivalence test against a stated margin, and the interval treated node-days within a rack
as independent.

Re-run over the steady-state record with contemporaneous maps, one map summary per model and
rack-clustered intervals, the three summaries shift p0−p1 socket power by +2.7, −1.5 and −1.0 W
across their full observed ranges, 90% CIs within ±5.6 W. Two are equivalent at ±5 W; in every
sensitivity design, including idle node-days and a window adjusted for SLURM socket placement,
every interval stays within ±7.4 W. The summary is **bounded and small, but not provably
negligible**. (An intermediate version reported +7.9 W; that came from fitting three collinear
summaries jointly.)

We record this because it is the most transferable thing we learned: **analysis choices moved our
numbers more often than the data did**, five times, and the first four ran in the direction we
would have preferred.

## Other open items

- **The lot boundary has no external confirmation.** Statistically sharp, but no procurement or
  delivery record was available to check it against. "Procurement lot" is the most parsimonious
  explanation, not a documented fact.
- **The floorplan comes from a diagram, not a die photograph.** IBM's chip diagram documents the
  layout (quads in two bands of three) but not to a stated scale. Thermal coupling confirms the
  quads but beyond them follows the numbering of the surviving cores, so it cannot check the rest
  of the layout.
- **Map-change classes are inferred.** RELABEL, CPU-SWAP and NODE-SWAP name the event a change is
  consistent with; the derived die-change and new-processor rates are telemetry-inferred, and only
  service records could measure each class's precision.
- **The reliability null is bounded.** 85 replacement-consistent events exclude useful prediction by
  the map features (concordance upper bound 0.61) but not a modest effect, and binary contrasts up
  to about threefold remain possible. The endpoint is inferred from map changes, not service
  records.
- **Pair-level co-harvest counts depend on the null** — against 100,000 exchangeable curveball
  draws 20 pairs are family-wise significant; a null that keeps each lot's own marginals leaves 11.
  The near excesses (eight pooled, seven per lot) are the robust part.
- **Compact renumbering closes the presence channel, not necessarily every channel.** The
  temperatures keep a weak positional trace (AUC 0.65 for a harvested slice between two consecutive
  survivors) that two decoders could not turn into maps; a stronger attacker might. The first
  release's random-deletion numbers were also wrong and are corrected (OI-20 in the full register).
- **The novelty claim is an awareness claim**, not a priority claim. It rests on a systematic but
  necessarily incomplete search. The closest prior work we know of recovered disabled core pairs
  by running CPUID on every Xeon Phi processor of TACC's Stampede2 (McCalpin, TR-2021-04), and
  disabled-tile maps from a privileged register on its Xeons (TR-2021-01b); Smith read the enabled
  cores of his own POWER9 machine from the same BMC sensors (2020). If you know of earlier work
  recovering per-die disable maps from published out-of-band telemetry, please tell us.

---

*Back to [home](index.md) · [FAQ](faq.md) · [full register](https://github.com/MSKazemi/m100-silicon-binning/blob/main/LIMITATIONS.md)*
