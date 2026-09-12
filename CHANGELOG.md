# Changelog

## Unreleased

Not tagged: whether and when to tag a release matching the submitted manuscript is the owner's
decision. Figure numbers below are the paper's (the source file is given in parentheses).

### Manuscript revision of 2026-09-08

- **Authors:** Julian Martin Kunkel (University of Göttingen and GWDG) joins the manuscript;
  Andrea Bartolini moves to last author. `CITATION.cff` and the docs metadata follow.
- **Fig. 6 (`f5_layout_hypotheses`) fixed:** it was drawn from the old three-day thermal matrix and
  contradicted the text. `layout_hypotheses.py` now defaults to the fleet census
  (`thermal_corr_allnodes.npy`), and the figure matches the numbers quoted in the paper (linear
  −0.791, z = −6.41). *(An earlier version of this entry called it Fig. 5, the file's number, and
  quoted z = −6.55, which is the end of that z's rack-bootstrap interval rather than the estimate.)*
- **Fig. 2 (`f1_slice_schematic`)** labels L2 and L3 as separate rows with units; **Fig. 10
  (`f13_model_check`)** is enlarged to full column width with a single-line colour-bar label;
  "Ganglia" is capitalised throughout, including Fig. 13 (`f9_guard_episodes`). *(This entry
  previously numbered them Figs. 1, 13 and 9, by file name.)*
- **Manuscript prose:** em-dash asides rewritten as commas, colons or parentheses; the abstract now
  explains where the 495 possible patterns come from.

### External review of 2026-09-11

An external review raised 189 comments. Each was checked against the code and data; the
corrections below changed analysis code, results, the paper, and every public surface of this
repository. `LIMITATIONS.md` carries the detail for each item.

**What the die layout rests on (OI-1, Secs. IV-D–F, Figs. 2, 5, 6).** The sensor index is IBM's
core numbering, and IBM's POWER9 Registers Specification v1.2 (Fig. 2, p. 33) places the six quads
in two bands of three, each quad's cores stacked. The earlier claim that thermal coupling
"recovers the sensor index as the physical 1×12 ordering of the die" is **withdrawn**: coupling
confirms the quads (index neighbours in one quad +0.37 to +0.48, across a quad boundary +0.10) and
beyond them follows how the surviving cores are numbered (die-column test 0.055 [0.046, 0.066];
rank slope 0.080 per active slice). IBM's layout scores −0.824 against −0.791 for the index order,
but only through the six same-quad pairs; the scores are descriptive. The within/cross-slice
contrast is +0.061 [0.047, 0.076] with the 49 racks resampled (the old pair-level interval was
about five-fold too narrow) and mixes two kinds of neighbour. New `thermal_socket_build.py`,
`thermal_socket_stats.py`, `thermal_ibm_layout.py`, `thermal_column_test.py`, `thermal_lib.py`;
per-socket matrices for three months are published. The exact optimum over all 12!/2 orderings
(−0.793) replaces the hill climb; the index order is 2nd of 239,500,800. The lag table is removed:
its lag unit was 100 s, not 20 s. Fig. 2 (`make_fig_schematic.py`) adds quad brackets.

**Clustering and the pairwise model (OI-3, OI-15, Secs. III-E, IV-D, IV-J, Figs. 4, 10).** The
curveball trade ran only between sockets sharing no harvested slice, and the Bonferroni cut for 66
tests is 3.37, not 3.83. `curveball_null.py` draws 100,000 null matrices exchangeable with the data
(Besag–Clifford): mean-gap z = −20.7, p = 1/(B+1); **20** family-wise significant pairs (was
16–18), 11 with each lot's marginals kept; the eight near positive excesses are unchanged and six
of them are the same-quad pairs. The pairwise model is fitted by Newton with a checked gradient; the
lot difference in γ is z = +4.16 with racks resampled (i.i.d. +4.47, not +5.22; rack SEs
0.047/0.058/0.047). Model checks are parametric model-replicate checks with simulation-calibrated
residuals (|z| up to 9.2, 10 of 66 cells outside the band); adjacency within a quad carries
γ = +0.92 against +0.28 across quads. Marginal uniformity is tested with the four-per-socket
constraint (Monte-Carlo p = 5e-5, rack-clustered p = 3e-6) and monthly stability on a fixed cohort
of 1,900 sockets over the 28 post-commissioning months (largest SD 0.22 pp), against 0.25 pp for
the population over the 30 months with clean data.

**Lot boundary (OI-9, Sec. IV-G, Figs. 7–9).** Lots are described as rack-aligned hardware
populations, procurement being their most plausible reading. The within-rack resampling is a
stability check, not a CI ("[22, 22]" withdrawn); rack-level uncertainty is added (rack 22 in 99.7%
of rack-block and 98.7% of beta-binomial draws; likelihood ratio 5.6×10⁴ over rack 23). The
leave-one-rack-out count is **49/49**, not 48/49 (the old count compared boundary names, not
partitions); held-out classification is nested (still 47/49 racks); k = 2 is re-selected under
three likelihoods on the 495 valid maps. New same-row and arriving-die tests: adjacent cabinets 21
and 22 differ 90.0% vs 42.5%, and dies arriving in Lot A racks have slice 0 disabled on 61%, not
88%. Monte-Carlo p-values are reported as (b+1)/(B+1): the calibrated boundary p is 2.5e-4.
Identifiability now states its entropy estimator (8.08 bits plug-in, 8.24–8.33 bias-corrected) and
uniqueness with side information (rack 78.4%, a node's two maps 93.0%).

**Lifecycle (OI-12, OI-14, OI-19, Secs. IV-H, IV-I, Figs. 1, 13).** Reboots are also dated by boot
epoch: **133 of 136** steady-state transitions contain a boot and none is seen without one (the six
day-flag misses were a Ganglia outage); the reboot test is node-stratified (exact p = 6e-31,
excess +63 pp [+57, +68]). All nine OS departures from 128 logical CPUs are matched by a BMC core
loss (the claim that the 124-CPU nodes were invisible to the BMC is withdrawn), giving nine
steady-state core-guard episodes, three of them meeting the strict slice criterion. Taxonomy
classes are named by the event they are consistent with; 50 of the 96 die changes are dies moved
from another socket, so sockets change die at 2.1% per socket-year (rack-clustered CI [1.6, 2.7])
but new processors enter at an inferred **1.0–1.2%**. "0 sockets ever report 24 cores" is corrected
(11 partially sampled socket-days do; no fully-sampled day exceeds 16).

**Power (OI-13, Sec. IV-L, Table IX, Fig. 15).** Each power day carries the map in force that day;
each map summary is fitted alone (the joint fit of collinear summaries produced the +7.9 W); the
constant `cpu_speed` is no longer counted as an adjustment; SLURM per-socket placement from the job
table (`equivalence_alloc.py`), idle-day and never-changing-node designs are added. Result over
636,026 paired node-days: +2.7, −1.5 and −1.0 W over each summary's full range, 90% CIs within
±5.6 W (±7.4 W in every design); two of three equivalent at ±5 W, all at ±10 W. The 6.8 W mean
p0−p1 difference reflects scheduler placement, not socket position.

**Reliability (OI-16, Sec. IV-M, Fig. 11).** Dies are followed by their map through relabelling (85
events over 53,716 die-months, was 92 over 53,597); one baseline hazard per calendar month; rarity
and scaling learned inside the training folds; binary hazard ratios reported 0→1. The first
release's "C-index 0.448" was a pooled person-month AUC whose sub-chance value was a
cross-validation artefact (`survival_legacy_check.py`). Held-out risk-set concordance is **0.55**
(95% CI [0.48, 0.61], p = 0.12): still a null, now with honest bounds.

**Side channel (OI-20, Sec. IV-N, Fig. 11).** `side_channel.py` modelled random deletion by feeding
a fractional count into a slot-placement formula and reported exact recovery 1.000 after deleting
99.9% of samples and 0.61 at 99.99%. Independent deletion on the empirical counts gives **0.81**
for one day at 99.9% (0.9985 for an attacker using slice granularity), yet one month still recovers
0.99997 at 99.99%. New `side_channel_raw.py` measures one 20 s interval on raw timestamps: exact
map in **99.95%** of socket-intervals (modelled before as 99.70%); "about two minutes of telemetry"
is dropped. New `side_channel_values.py` tests whether compact renumbering leaks through the
temperatures (AUC 0.65, not decodable into maps by two decoders). A fixed per-node pseudonym is a
fingerprint unique for 99.7% of sockets.

**References and wording (OI-4, OI-18).** References re-verified against DOI records and primary
documents and pinned by `tests/test_refs.py` (Koren et al. pp. 724–734; SeT-Diff 2026; A. Mauri and
J. Y. Hwang corrected; arXiv:2505.14796 no longer cited as M100 ExaData work; WikiChip replaced by
IBM's manual, ISSCC 2017 and Hot Chips slides). New related work: McCalpin's Stampede2 surveys of
disabled core pairs and tiles and Smith's single-machine POWER9 reading, the closest prior work; the
novelty statement is narrowed accordingly. "Fused off" is replaced by "disabled" wherever the
mechanism is not observed: telemetry shows *which* positions are disabled, not why, and a harvest
map is defined as that set. Slice granularity is stated as a property of these 16-core parts, not of
the die. The abstract is rewritten (248 words).

### Audit and repair pass of 2026-09-12

Three independent audits (numbers, reader, completeness) were run against the revised manuscript and
the round's own logs. Everything they found is fixed here, or recorded with the evidence for why it
is not a defect.

- **Two blockers.** The layout score's random-ordering null was quoted with the wrong sign on its
  mean and with the extreme end of a bootstrap interval in place of the estimate: it is
  **−0.003 ± 0.123, z = −6.41** (rack-bootstrap [−6.55, −6.24]), not +0.001 ± 0.121, z = −6.55.
  And Table III's Lot B rows named ranges that do not contain the sockets counted: racks 22–48 hold
  **1,080** sockets (39.2%, 416 patterns) and nodes 439–979 hold **1,082** (39.1%, 416); the two
  extra were node 980, which has no documented cabinet and which Sec. IV-A excludes from every
  rack-indexed result. Both corrections run against the authors' interest.
- **The reboot comparator.** 126 of 132 transitions carrying a reboot were set against **1.74%**, a
  per-node-*day* rate applied to multi-day intervals. The comparator is the length-matched
  **47.6%** (49.0 ± 4.6 of 103 node-level transitions), which the next sentence already gave.
- **The abstract** now states the power result as a bound ("an equivalence analysis bounds …"),
  matching the Introduction and Sec. IV-K, quotes the 47 of 49 held-out racks that make the lot
  boundary "predictable", and no longer says the guard episodes match the OS count "day for day":
  the OS window brackets the sensor one, because a BMC episode counts only fully-sampled days.
- **Smaller number corrections.** The robustness envelope is **±7.4 W**, not ±7.3 W (measured
  maximum 7.3192 W). The 39 odd-count socket-days are itemised in full (11 + 4 + 21 + 3), with a
  footnote on why 39, 209 and nine do not partition the 256. The lag-0 decay is given as its
  measured range (24–52%). Slices 8–11 are "not conspicuously above average" with the rates, since
  Lot B's slices 8 and 9 sit just above 33.3%. The within-node overlap null is **1.470 ± 0.026
  (z = +6.6)** from its new committed producer. Sec. IV-L now lists the broadest replacement
  endpoint (0.64, nominal *p* = 0.015, not significant across six endpoints), the only nominally
  significant sensitivity, which the paper had omitted while LIMITATIONS.md disclosed it.
- **Fig. 7's caption** claimed the lot boundary changes "all twelve slices at once". Slice 3 is
  flat and slices 4–11 move the *other* way (Lot B takes them 9–14 points more often), which is the
  informative half; the caption now says so.
- **Producers and provenance.** `analysis/within_node_overlap.py` is new and owns Sec. IV-C's
  overlap numbers (asserted in CI); `oi8_and_os_full.py` now thresholds at 2 W and prints OI-8's
  "12/15 (80%), 95% CI [52, 96]%" verbatim (asserted in CI, and named in LIMITATIONS.md);
  `full_record_structure.py` says in its docstring that its 200-draw curveball chain is superseded
  by `curveball_null.py` and which script owns each duplicated statistic; Fig. 6's error bars for
  the index order and IBM's layout now come from the same 1,000-replicate bootstrap the text
  quotes. `curveball_null.py`, `side_channel_raw.py` and `side_channel_values.py` print their
  runtime instead of writing it into JSON files pinned in `DATA.sha256`, which made the checksum
  gate fail on every re-run with every number bit-identical.
- **Figures and floats.** Fig. 4's legend gains a swatch for the eleven family-wise significant
  *deficits*; Fig. 8's caption explains how a Chao–Shen effective count can exceed the observed
  richness; colour is no longer the only cue in Figs. 12, 13 and 15. The fifteen floats were moved
  next to the text that cites them: mean distance from first citation to placement falls from 5.4
  pages to 2.3, with figure and table numbering unchanged.
- **Floats and cross-references.** Table I gains a label and every table and figure is now cited by
  number at least once; Sec. VII says that OI-14 and OI-16 are tracked in LIMITATIONS.md only; item
  12 carries its [OI-13] tag. A duplicated RELABEL definition, a duplicated Discussion paragraph
  and one of three statements of the thermal/quad conclusion are cut. The paper is **20 pages**,
  0 overfull boxes, 0 undefined references.
- **Artefact.** `pytest.ini` scopes collection to `tests/`, so a bare `pytest` from the repository
  root runs the 246 tests instead of aborting on an analysis script it mistook for a test module.
  `analysis/oi8_and_os_full.py` and `analysis/within_node_overlap.py` are listed in
  `docs/reproduce.md` and run in CI.
- **Length, decided rather than inherited.** The revision of 2026-09-11 took the paper from 18
  pages to 20. This pass took the four cuts that lose no result (a duplicated Discussion
  paragraph, a second definition of RELABEL, one of three statements of the thermal/quad
  conclusion, and OI-1's restatement of Sec. IV-F) and **kept 20 pages**: the remaining levers
  all cost evidence — deleting Fig. 14, which is now cited, or compressing Sec. IV-H — and under
  P4 detail moves to `LIMITATIONS.md` rather than being deleted. A venue with a 14-page limit
  would need a decision from the authors, not another editorial pass. The abstract is 279 words
  (248 before this pass; the three corrections to it cost 31).

### Repository

- **CI** (`.github/workflows/reproduce.yml`) now runs the test suite and the corrected analyses —
  side channel, the constrained marginal test, the pairwise model and its replicate checks,
  the thermal layout scores, the lot boundary, reboots and the OS cross-check, the taxonomy error
  budget, power equivalence, survival and the legacy-statistic check — asserting their new outputs
  on every push, and asserts the committed outputs of the curveball null and the per-socket thermal
  summaries, which need either too much time or the raw archive. It verifies `DATA.sha256` first and
  ends with `tests/test_consistency.py` against the freshly written result files. The 3.12 legs
  install the exact environment of the results; the 3.10 leg installs the newest versions the floors
  allow. What CI does not re-derive is listed in `docs/reproduce.md`, and a green badge means the
  committed tables still give the asserted numbers, not that every paper number was recomputed from
  raw data.
- **`requirements-lock.txt`** pins the environment that produced the results (Python 3.12.3);
  `requirements.txt` stays as floors; **`requirements-dev.txt`** adds pytest 9.1.
- **`DATA.md`** documents the schema, units, day boundaries (UTC) and lineage of every derived
  table, the twelve Zenodo records used, and the thermal decimation rule; **`DATA.sha256`**
  checksums the derived tables CI reads.
- **Tests** (`tests/`, 8 modules from the review plus `tests/test_consistency.py`): every `OI-n` the
  paper cites resolves in `LIMITATIONS.md`, headline numbers in the README match the result files,
  and withdrawn claims no longer appear on the public pages.
- `curveball_null.py` starts its worker processes with `spawn` instead of `fork` (forking a
  multi-threaded process can deadlock; Python 3.12 warns). Null draws are bit-identical.
- The hero and social-preview images drop "fused off", "procurement boundary" and "~2 minutes".
- `SECURITY.md`, `README.md`, `docs/` and `CITATION.cff` follow the corrected results; `SECURITY.md`
  gives a contact address that is not being retired, frames the disclosure as passive and of modest
  direct severity, and now carries the measured single-interval recovery (99.95%).
- **What the review asked for and this release does not do** is written down, not implied away:
  `LIMITATIONS.md` OI-21 lists the six items — no checksum of our own copy of the raw tars, no
  machine-readable log of a completeness-gate refusal, no synthetic fixture for the extraction stage,
  **no notification of anyone** (no contact has been made and no disclosure timeline exists), no tag
  tying the artefact to a manuscript version, and node identifiers kept at full granularity with the
  linkage risk stated.

## v1.0.0 — 2026-08-08

First public release, accompanying the preprint *"Which 16 of 24? Per-Die Silicon Harvest Maps,
Procurement Lots and Core Guarding from Supercomputer Telemetry"*. *Kept as released; several of
the results below were corrected after the external review of 2026-09 (see Unreleased).*

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
