---
title: "Recovering per-die CPU core-harvest maps from BMC telemetry"
description: "A baseboard management controller exposes one thermal sensor per physical core position and does not renumber survivors — so the set of sensors that report is the die's core-harvest map. Reproducible results for 1,962 POWER9 sockets."
---

# Recovering per-die CPU core-harvest maps from BMC telemetry

Processor vendors sell partially defective dies by permanently fusing off the broken units. **You
are told a part has 16 cores. You are never told *which* 16.** That per-die map is proprietary.

This project shows that out-of-band telemetry supercomputers already collect gives it away, and
recovers the map for all **1,962 POWER9 sockets** of CINECA's Marconi100 from a public dataset.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "m100-silicon-binning",
  "description": "Analysis pipeline that recovers per-die CPU core-harvesting maps from out-of-band BMC/IPMI telemetry.",
  "codeRepository": "https://github.com/MSKazemi/m100-silicon-binning",
  "programmingLanguage": "Python",
  "license": "https://opensource.org/licenses/MIT",
  "url": "https://mskazemi.com/m100-silicon-binning/",
  "keywords": "silicon harvesting, silicon binning, core harvesting, hardware provenance, POWER9, HPC telemetry, semiconductor yield, side channel",
  "author": [
    {"@type": "Person", "name": "Mohsen Seyedkazemi Ardebili", "affiliation": {"@type": "Organization", "name": "University of Bologna"}},
    {"@type": "Person", "name": "Michael Bidollahkhani", "affiliation": {"@type": "Organization", "name": "University of Göttingen"}},
    {"@type": "Person", "name": "Andrea Bartolini", "affiliation": {"@type": "Organization", "name": "University of Bologna"}}
  ]
}
</script>

## In one paragraph

A baseboard management controller exposes one thermal sensor per **physical** core position, and
does **not** renumber the surviving cores compactly. So the set of `pX_coreY_temp` metrics that
report *is* the die's harvest map — published inadvertently, at fleet scale, in a public dataset.
On Marconi100 every socket reports exactly 16 of 24 possible core positions, the disabled units
always form complete POWER9 *slices*, and 443 of the 495 possible patterns occur, which makes the
map a per-die property rather than a per-SKU one.

## The five facts

- **m100-silicon-binning is an open-source analysis pipeline** that recovers per-die CPU
  core-harvesting maps from out-of-band BMC telemetry.
- It helps HPC operators and systems researchers **see which cores a vendor fused off each
  individual processor die** — information vendors do not publish.
- **Use it when** you have per-core sensor telemetry and want to audit procurement homogeneity,
  detect processor replacements without an asset database, or identify field core-guard events.
- **It differs from PUF and hardware-fingerprinting work** because it does not identify *which die*
  a part is; it recovers *what was disabled on it*, with no designed-in circuitry and no physical
  access.
- **It is not** a tool for reclaiming idle CPU cycles — that is the unrelated architectural sense
  of "core harvesting" — and it needs per-core sensors that expose physical position, which not
  every BMC does.

## What was found

| Finding | Evidence |
|---|---|
| Every socket reports exactly 16 of 24 cores | 99.9847% of 1.67 M socket-days |
| Disabling is always at POWER9 *slice* granularity | 1,962 / 1,962, zero exceptions |
| The map is per-die, not per-SKU | 443 of 495 possible patterns observed |
| The sensor index **is** the physical die ordering | 1×12 linear fits at −0.791 vs a null of +0.001 ± 0.121 |
| Harvested slices cluster spatially | 8 positive-excess pairs, all at \|Δk\| ≤ 2 |
| Two procurement lots, boundary at rack 22 | permutation-calibrated *p* < 2.5e-4; unseen racks assigned correctly 95.9% of the time |
| Changes coincide with reboots, measured | 126/132 transitions carry a `boottime` change vs 1.74% of controls |
| Replacement rate with no asset database | 2.1% per socket-year (95% CI [1.7, 2.6]) |
| Recovery costs about **two minutes** of telemetry | and coarser cadence is *not* a mitigation |

## Where to go next

- **[Method](method.md)** — how the recovery works, and why it is valid
- **[Findings](findings.md)** — the results in detail, with the numbers
- **[Reproduce](reproduce.md)** — run it yourself in minutes from committed data
- **[FAQ](faq.md)** — including *"does this apply to my fleet?"*
- **[Limitations](limitations.md)** — what is not settled, and what would settle it

## If you publish telemetry

The mitigations that work, and the ones that do not, are measured rather than asserted. Reducing
sampling cadence does **not** close the channel — recovery stays at 1.000 after discarding 99.9% of
samples, because a single sample names its core just as well as 4,320 do. See the
[FAQ](faq.md#i-publish-a-telemetry-dataset-what-should-i-do).

---

*Source code and data: [github.com/MSKazemi/m100-silicon-binning](https://github.com/MSKazemi/m100-silicon-binning) · MIT licensed · manuscript is an unpublished preprint.*
