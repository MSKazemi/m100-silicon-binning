# Contributing

This is a research artefact, so the most valuable contributions are not the usual ones.

## What helps most

**1. Replication on another machine.** The single largest open limitation is that every result
comes from one system (see [`LIMITATIONS.md`](LIMITATIONS.md), OI-10). If you have per-core BMC or
OpenBMC telemetry from any AC922, POWER9, or other fleet whose management controller exposes one
sensor per core position without renumbering the survivors, the recovery method should transfer
directly. A replication —
agreeing **or disagreeing** — is more useful to us than any code change. Disagreement is
especially useful: it would delimit exactly when the technique works.

**2. Telling us a number is wrong.** Every figure in the paper is derived by a script in
[`analysis/`](analysis/) from the public M100 ExaData release. If you re-run one and get something
different, open an issue with the script, the command, and both numbers. This project has
corrected several of its own results, most recently after an external review in 2026-09; see
[`CHANGELOG.md`](CHANGELOG.md) and the "Threats to validity" section of the paper.

**3. Pointing at prior work we missed.** The novelty statement is an *awareness* claim, not a
priority claim — it rests on a systematic but necessarily incomplete search. The closest prior
work we know of is McCalpin's surveys of disabled cores on TACC's Stampede2 and Smith's reading of
one POWER9 machine (see `LIMITATIONS.md`, OI-4). If you know of earlier work recovering per-die
disable maps from published out-of-band telemetry, please say so.

**4. Checking the die layout.** Where each sensor index sits on the die rests on IBM's chip
diagram (POWER9 Registers Specification v1.2, Fig. 2), which is not to scale, and on the sensor
index being the firmware core number. An annotated POWER9 die photograph, or an AC922 BMC sensor
map tying `pX_coreY_temp` to core chiplets, would settle OI-1's residual caveat.

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
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/          # about 20 s; CI runs the same
.venv/bin/python analysis/binning_stats.py
```

`requirements-lock.txt` reproduces the exact environment of the published numbers. If a change
alters a reported number, update every surface that quotes it (paper, `README.md`, `docs/`,
`LIMITATIONS.md`, `CHANGELOG.md`); `tests/test_consistency.py` checks the headline ones.

Most results need only the derived tables committed here. The thermal, reboot, and power sections
need a pass over the raw M100 ExaData archives; the scripts that perform it are included.

## Reporting problems

Open an issue. For anything security-relevant, see [`SECURITY.md`](SECURITY.md) instead.
