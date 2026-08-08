# Security

## What this repository is, in security terms

This project documents an **unintended disclosure in published telemetry**: an out-of-band
management controller exposes one thermal sensor per *physical* core position and does not
renumber the surviving cores, so the set of sensors that report is the die's core-harvest map. The
disclosure is a property of how monitoring data is collected and published, not a defect in any
piece of software here.

The information recovered is **manufacturing provenance** — which units a vendor fused off a given
die. It is not credentials, not user data, and not a path to code execution or privilege on any
system. Recovery is entirely passive: it reads an already-public dataset and requires no access to
any machine.

We measured how identifying a map actually is, precisely so this is not overstated: over 1,962
sockets the pattern distribution carries 8.08 bits, and only 4.9% of sockets have a map unique in
the fleet. A harvest map is a **partial** identifier — strong evidence about which population a
part came from, weak evidence about which part it is.

## If you publish fleet telemetry

The mitigations that work, and the ones that don't, are measured in the paper and reproducible
from [`analysis/side_channel.py`](analysis/side_channel.py):

- **Does not work:** reducing sampling cadence, or dropping samples at random. Recovery stays at
  1.000 after discarding 99.9% of samples, because a single sample names its core just as well as
  4,320 do. The channel is carried by *which sensors exist*, not by what they report.
- **Works:** publishing per-socket aggregates instead of per-core series, or renumbering surviving
  cores compactly to 0…n-1. Per-node random renumbering destroys the map while still leaking the
  active-core count.

If you maintain a public telemetry dataset and want help assessing whether it carries this
channel, open an issue or contact the maintainers — we would rather help than have it discovered
quietly.

## Reporting a vulnerability

If you find a genuine security problem in the code in this repository — for example something in
the extraction pipeline that could be exploited by a malicious archive — please report it
privately rather than opening a public issue:

**mohsen.seyedkazemi@unibo.it**

Please include what you found, how to reproduce it, and how you would like to be credited. We will
acknowledge within a reasonable period and keep you informed.

## Scope

In scope: the analysis and extraction code in [`analysis/`](analysis/).

Out of scope: the upstream M100 ExaData dataset (report to its maintainers), the Marconi100 system
itself, and the existence of the disclosure described above — that is the published finding, not a
vulnerability to be reported.
