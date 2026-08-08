# Contributing

This is a research artefact, so the most valuable contributions are not the usual ones.

## What helps most

**1. Replication on another machine.** The single largest open limitation is that every result
comes from one system (see [`LIMITATIONS.md`](LIMITATIONS.md), OI-10). If you have per-core BMC or
OpenBMC telemetry from any AC922, POWER9, or other fleet whose management controller exposes one
sensor per *physical* core position, the recovery method should transfer directly. A replication —
agreeing **or disagreeing** — is more useful to us than any code change. Disagreement is
especially useful: it would delimit exactly when the technique works.

**2. Telling us a number is wrong.** Every figure in the paper is derived by a script in
[`analysis/`](analysis/) from the public M100 ExaData release. If you re-run one and get something
different, open an issue with the script, the command, and both numbers. This project has already
corrected four of its own results that way; see the "Threats to validity" section of the paper.

**3. Pointing at prior work we missed.** The novelty statement is an *awareness* claim, not a
priority claim — it rests on a systematic but necessarily incomplete search. If you know of
earlier work recovering per-die disable maps from deployed telemetry, please say so.

**4. Checking the floorplan inference.** The sensor index is recovered as the physical slice
ordering from thermal coupling alone. An annotated POWER9 die shot, or the hostboot sensor-ID map,
would confirm it directly and settle OI-1's residual caveat.

## What to expect from the code

The scripts are research code: single-purpose, run top-to-bottom, print their results. They are
not a library and there is no API to keep stable. Each one carries a docstring explaining *why* it
exists and what it corrected — those are worth reading before changing anything.

Two hard-won rules the code enforces, and any change should preserve:

- **Never infer "hardware absent" from "data absent"** without gating on extraction completeness.
  A partial ingestion is indistinguishable downstream from a socket with disabled cores.
- **Cluster uncertainty at the level the data are actually dependent** — sockets within a rack are
  not independent, and treating them as such once produced a headline claim that had to be
  withdrawn.

## Running things

```bash
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow numpy scipy matplotlib
.venv/bin/python analysis/binning_stats.py
```

Most results need only the derived tables committed here. The thermal, reboot, and power sections
need a pass over the raw M100 ExaData archives; the scripts that perform it are included.

## Reporting problems

Open an issue. For anything security-relevant, see [`SECURITY.md`](SECURITY.md) instead.
