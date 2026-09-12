# Security

## What this repository is, in security terms

This project documents an **unintended disclosure in published telemetry**: an out-of-band
management controller exposes one thermal sensor per core position and does not renumber the
surviving cores, so the sensors that never report mark the die's disabled cores, its core-harvest
map. The disclosure is a property of how monitoring data is collected and published, not a defect
in any piece of software here.

The information recovered is **manufacturing provenance** — which cores are disabled on a given
die, a vendor-undisclosed configuration attribute — and the fleet's cohort structure. It is not
credentials, not user data, and not a path to code execution or privilege on any system. Recovery
is entirely passive: it reads an already-public dataset and requires no access to any machine. Its
direct severity is modest; it matters for **telemetry publication policy** and for
**supply-chain confidentiality** (which parts a site received, and when dies are moved or
replaced), not as an intrusion technique. On the machine itself the active cores are probably no
secret: skiboot names each active core in the operating system's device tree by its physical core
number, so a user of a POWER9 Linux node can likely read the same map in-band (inferred from the
skiboot source, not checked on Marconi100).

We measured how identifying a map actually is, precisely so this is neither overstated nor
understated: over 1,962 sockets the pattern distribution carries 8.08 bits (plug-in; 8.24–8.33 bits
bias-corrected), and only 4.9% of sockets have a map unique in the fleet. A single map is a
**partial** identifier across a whole fleet, but uniqueness rises steeply with side information:
78% of sockets are unique once the rack is known, and a node's two maps together are unique for
93% of nodes. Anyone publishing per-core telemetry under pseudonymised node names should treat the
pair of maps as a quasi-identifier that can link a node across releases. The derived tables in this
repository carry only the dataset's own node identifiers, which the public release already exposes.

## If you publish fleet telemetry

The mitigations that work, and the ones that don't, are measured in the paper and reproducible
from [`analysis/side_channel.py`](analysis/side_channel.py) (committed counts) and
[`analysis/side_channel_raw.py`](analysis/side_channel_raw.py) (raw 20 s timestamps):

- **Does not work:** reducing sampling cadence, or deleting samples at random. A **single 20 s
  interval** already gives the exact map for **99.95%** of socket-intervals, measured on the raw
  timestamps of 2022-08, and keeping one 20 s slot in every six hours still recovers every map from
  a single day, because a single sample names its core just as well as 4,320 do. Deleting each sample independently with probability
  99.9% leaves one-day exact recovery at 0.81, and at 0.9985 for an attacker who uses the fact that
  cores are disabled in pairs (one surviving sample per active slice is enough). At 99.99% a single
  day rarely gives the exact map but still reveals about 4.6 of the 8 active slices, and because
  maps are static for months an attacker pools days: one month recovers 0.99997. Holding
  one-month recovery below one half takes 99.998% deletion (99.9991% against the pair-aware
  attacker), and over the 2.5-year record more than 99.9999%. Deleting whole timestamps protects
  even less (one-day recovery 0.99 at 99.9%). The channel is carried by *which sensors exist*, not
  by what they report.
- **Works:** publishing per-socket aggregates instead of per-core series, or renumbering surviving
  cores compactly to 0…n-1. Compact renumbering removes the presence channel by construction; the
  temperatures still carry a weak positional trace (AUC 0.65 for a harvested slice between two
  consecutive survivors) that the two decoders we tried could not turn into maps any better than
  guessing from slice frequencies alone
  ([`analysis/side_channel_values.py`](analysis/side_channel_values.py)). Per-node random
  renumbering destroys the map while still leaking the active-core count, but the permutation must
  be **redrawn for every release**: one fixed over time makes the set of reporting pseudonyms a
  fingerprint unique for about 99.7% of sockets (against 4.9% for the maps themselves), which
  re-identifies a socket across releases even if node names change and still exposes every die
  change.

The first public release overstated how well random deletion fails (it reported exact recovery
1.000 after deleting 99.9% of samples); the figures above are the corrected ones
([`LIMITATIONS.md`](LIMITATIONS.md), OI-20).

If you maintain a public telemetry dataset and want help assessing whether it carries this
channel, open an issue or contact the maintainers — we would rather help than have it discovered
quietly.

## Reporting a vulnerability

If you find a genuine security problem in the code in this repository — for example something in
the extraction pipeline that could be exploited by a malicious archive — please report it
privately rather than opening a public issue:

**mohsen.seyedkazemi@gmail.com**

Please include what you found, how to reproduce it, and how you would like to be credited. We will
acknowledge within a reasonable period and keep you informed.

## Scope

In scope: the analysis and extraction code in [`analysis/`](analysis/) and the tests in
[`tests/`](tests/).

Out of scope: the upstream M100 ExaData dataset (report to its maintainers), the Marconi100 system
itself, and the existence of the disclosure described above — that is the published finding, not a
vulnerability to be reported.
