# Changelog

## v1.0.0 — 2026-08-08

First public release, accompanying the preprint *"Which 16 of 24? Per-Die Silicon Harvest Maps,
Procurement Lots and Core Guarding from Supercomputer Telemetry"*.

### What this release contains

- The recovery method and every analysis behind the manuscript, in `analysis/`.
- Derived tables sufficient to reproduce every headline result **without** downloading the
  49.9 TB source: per-(node, socket, core, day) counts for all 31 months, per-socket power, OS
  logical-CPU counts, and the covariate sweep.
- CI that re-derives the headline numbers on every push, on Python 3.10 and 3.12.
- `LIMITATIONS.md` — sixteen open issues, each with what would settle it.

### Results, in brief

- Harvesting operates exactly at the POWER9 slice: 1,962 / 1,962 sockets, zero exceptions in
  1.67 M socket-days.
- The map is per-die, not per-SKU: 443 of 495 possible patterns.
- The sensor index **is** the physical die ordering, recovered from thermal coupling alone.
- A procurement-lot boundary at rack 22, permutation-calibrated *p* < 2.5e-4, placing unseen racks
  in the right delivery 95.9% of the time.
- Configuration changes coincide with reboots — measured against an independent boot signal,
  126 of 132, versus 1.74% of controls.
- Recovery costs about two minutes of ordinary telemetry, and reducing cadence does not mitigate it.

### Results that are null, and reported as such

- The harvest map does **not** predict which dies get replaced (out-of-fold C-index 0.448).
- Its effect on socket power is **bounded, not negligible** — an earlier draft called it
  "operationally invisible"; that claim was withdrawn.

### Known limitations

Every result comes from one machine. See `LIMITATIONS.md`; a replication on a second installation
is the most valuable contribution anyone could make.
