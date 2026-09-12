# DATA — what is committed, what it means, and where it came from

Every headline number in the paper, the README and `docs/` is derived from the tables committed
here: 169 files, 103 MB, listed with their SHA-256 in [`DATA.sha256`](DATA.sha256). Nothing in this
file is needed to *run* the analysis — `docs/reproduce.md` covers that. It is here so that a reader
can check what each column is, in what unit, on which clock, and which script turned the 49.9 TB
public archive into it.

Verify the committed tables before trusting a result:

```bash
sha256sum -c DATA.sha256      # 169 files OK
```

CI runs exactly that line before anything else.

---

## 1. Lineage: from the public archive to a number in the paper

```
M100 ExaData, twelve Zenodo records, v1.0.0, one tar per month, 49.9 TB
  |
  |  analysis/sweep_all_months.py        ipmi_pub/metric=p{0,1}_core{0..23}_temp
  +->  daily/daily_<YM>.parquet          per (node, socket, core, day) sample counts
  |
  |  analysis/core_matrix.py             the same for one month, not aggregated to days
  +->  analysis/counts_<YM>.parquet      per (node, socket, core) counts, single month
  |
  |  analysis/sweep_power_os_all.py      ipmi_pub/metric=p{0,1}_power, ganglia_pub/cpu_num
  +->  power_all/pw_<YM>.parquet          per (node, socket, day) socket power
  +->  os_all/os_<YM>.parquet             per (node, day) logical-CPU count the OS reports
  |
  |  analysis/sweep_covariates.py        ganglia_pub boottime, cpu_speed, cpu_user, load_one
  |                                      ipmi_pub ambient, total_power
  +->  cov_all/cov_<YM>.parquet           per (node, day) covariates, incl. the boot signal
  |
  |  analysis/a1_sweep_power.py          ipmi_pub power metrics at finer granularity
  +->  power/power_<YM>.parquet           per (node, socket, day, metric), six months only
  |
  |  analysis/equivalence_alloc.py       job_table/metric=job_info_marconi100
  +->  analysis/equivalence_alloc.parquet per (node, day) SLURM core/GPU seconds per socket
  |
  |  analysis/thermal_socket_build.py    the raw per-core temperature series (~400 GB)
  +->  analysis/thermal_socket_*         per-socket residual correlation matrices, three months
  |
  +->  analysis/full_record_structure.py -> harvest_maps_full.npy, harvest_keys_full.npy,
                                            per_month_marginals.csv
```

Everything downstream of that reads only committed files. The recovery rule itself — a core
position is *disabled* when its `pX_coreY_temp` sensor never reports on a socket-day that is
otherwise fully sampled — is in `docs/method.md`, with its four failure modes and their detectors.

**The twelve source records** (M100 ExaData v1.0.0, CC BY 4.0): Zenodo
[7588815](https://doi.org/10.5281/zenodo.7588815) (2020), then 7589131, 7589320, 7589630, 7589942,
7590061, 7590308, 7590547, 7590555, 7590565, 7590574, 7590583 (2021-01 to 2022-09). The records are
named by DOI, which is what pins a version; **we did not record a per-archive checksum of our own
download**, so the chain from tar to committed table is auditable only from the record DOI down.
Re-deriving `daily/` from a fresh download and diffing against `DATA.sha256` is the check that
closes that gap, and it is the single most useful artefact contribution after a replication on
another machine.

---

## 2. Clocks, days and identifiers

- **Timestamps.** The archive stores `timestamp` as a timezone-aware UTC instant (ms resolution).
  No local time, DST rule or site calendar enters anywhere.
- **Day boundary.** Every `day` column is **UTC midnight**: `timestamp.dt.tz_convert('UTC')` then
  `.dt.floor('D')` in `sweep_all_months.py`, and `.dt.floor('D')` on the already-UTC column in the
  other sweeps. A "socket-day" is therefore a UTC calendar day, and a 24 h day is 4,320 slots of
  20 s. Marconi100 stood in Bologna (CET/CEST), so a UTC day is offset 1–2 h from the operators'
  local day; nothing in the analysis depends on which, because every comparison is day-to-day
  within the same convention.
- **Sampling grid.** IPMI metrics arrive on a 20 s grid. `thermal_socket_build.py` asserts that the
  month's first slot is phase-aligned and counts any sample off the grid rather than silently
  rounding it.
- **`node`.** The dataset's own node identifier (0…983), and the only identifier here. It is a
  *stable pseudonym*: the same integer is the same chassis across all 31 months, which is what
  makes a longitudinal map history possible — and is also a linkage risk. A node's pair of harvest
  maps is unique for 93.0% of nodes, so anyone joining these tables to another M100 ExaData-derived
  release can re-identify most nodes even if that release renames them. The identifiers are not new
  exposure (the upstream public release already carries them) but the *combination* is a
  quasi-identifier; `SECURITY.md` says what a publisher should do about it. There is no user, job
  owner, account or project field in any committed table.
- **`socket`.** 0 or 1, the BMC's `p0`/`p1` namespace. The two labels occasionally swap between
  collection periods (`LIMITATIONS.md`, OI-8); analyses that care follow the die, not the label.
- **`core`.** 0…23, the BMC sensor index, which is IBM's core (EC chiplet) number. Slice
  `k` = cores `(2k, 2k+1)` = an EX; quad `q` = slices `2q, 2q+1`. The index is *not* a
  two-dimensional position: the floorplan comes from IBM's POWER9 Registers Specification, not from
  this data (`LIMITATIONS.md`, OI-1).

---

## 3. Schema of each derived table

Types are as written by pandas/pyarrow; `datetime64[ms, UTC]` below means a tz-aware UTC timestamp.

### `daily/daily_<YM>.parquet` — 31 files, the spine of the whole analysis

| column | type | unit / meaning |
|---|---|---|
| `node` | `int32` | node pseudonym, 0…983 |
| `day` | `datetime64[ms, UTC]` | UTC midnight of the day |
| `socket` | `int8` | 0 or 1 |
| `core` | `int8` | BMC sensor index 0…23 |
| `n` | `int32` | number of temperature samples that (node, socket, core) reported that day; a full day is 4,320 |

One row per reporting (node, socket, core, day). **A disabled core has no row** — absence is the
signal, so the table must never be reindexed to a full grid without keeping that distinction. No
filter to 16 active cores is applied: sockets with *fewer* active cores are the field
deconfiguration signal.

### `analysis/counts_<YM>.parquet` — single-month snapshot (2020-04, 2021-03)

| column | type | meaning |
|---|---|---|
| `node` | `str` | node pseudonym as a string |
| `socket` | `int64` | 0 or 1 |
| `core` | `int64` | 0…23 |
| `n` | `int64` | samples in the whole month |

The fast path used by `binning_stats.py`. It is representative of the full record but not identical
to it: `docs/reproduce.md` says by how much.

### `power_all/pw_<YM>.parquet` — per-socket power

| column | type | unit |
|---|---|---|
| `node` | `int32` | |
| `day` | `datetime64[ms, UTC]` | |
| `socket` | `int8` | 0 or 1 |
| `mean` | `float64` | mean socket power, **watts** (`ipmi_pub/metric=pX_power`) |
| `count` | `int64` | samples behind the mean |

### `power/power_<YM>.parquet` — six months, finer granularity

As above plus `std` (`float64`, watts) and `metric` (`str`, the source metric name).

### `os_all/os_<YM>.parquet` — what the operating system thinks it has

| column | type | meaning |
|---|---|---|
| `node` | `int32` | |
| `day` | `datetime64[ms, UTC]` | |
| `value` | `int32` | logical CPUs the OS reports (`ganglia_pub/metric=cpu_num`); 128 on a healthy node (2 × 16 cores × 4 SMT threads) |
| `n` | `int64` | samples behind it |

29 files: the `cpu_num` stream begins in 2020-05. Every departure from 128 in the record is matched
by a BMC core loss on one socket (`lifecycle_os_crosscheck.py`).

### `cov_all/cov_<YM>.parquet` — covariates, **schema varies by month**

Always present: `node` (`int32`), `day` (`datetime64[ms, UTC]`), and for each of `ambient`
(degrees Celsius) and `total_power` (watts) a `_mean` (`float32`/`float64`), `_max` and `_n`.

From 2020-05, when the Ganglia streams begin, additionally: `cpu_speed_{mean,max,n}` (MHz, constant
on this fleet, which is why `equivalence.py` drops it as an adjustment), `cpu_user_{mean,max,n}`
(percent), `load_one_{mean,max,n}` (runnable processes), and the boot signal
`boot_min`, `boot_max` (**Unix epoch seconds** of the kernel boot, from `ganglia_pub/boottime`),
`boot_ndistinct` (how many distinct boot epochs were seen that day — the reboot flag) and `boot_n`.
Code that reads covariates must tolerate the missing columns of 2020-03 and 2020-04; a reboot is
*unobservable*, not absent, before 2020-05-05.

### `analysis/equivalence_alloc.parquet` — SLURM placement

| column | type | unit |
|---|---|---|
| `node` | `int32` | |
| `day` | `datetime64[ms, UTC]` | |
| `core_s0`, `core_s1` | `float32` | allocated core-seconds on socket 0 / socket 1 that day |
| `gpu_s` | `float32` | allocated GPU-seconds |
| `nolay_s` | `float32` | core-seconds of jobs whose layout could not be resolved to a socket |
| `njobs` | `int32` | jobs touching the node that day |

Needs the raw `job_table` partitions, so CI does not rebuild it; it is committed and checksummed.

### `analysis/harvest_maps_full.npy`, `harvest_keys_full.npy` — the maps themselves

`harvest_maps_full.npy` is `(1962, 12) int8`: one row per socket with a well-defined map, 1 where
slice *k* is disabled, 0 where it is active; every row sums to 4. `harvest_keys_full.npy` is
`(1962, 2) int64` with the matching `(node, socket)`, in the same row order. Rack = `node // 20`.

### `analysis/per_month_marginals.csv` — 30 rows, one per month with clean data

`month` (`YY-MM`), `socketdays`, then `s0`…`s11`: the fraction of clean socket-days on which slice
*k* is disabled. This is the population stability series behind Fig. 12.

### `analysis/thermal_socket_*` — per-socket thermal summaries, 2021-05, 2022-02, 2022-08

Published so that no thermal claim rests on a single pooled matrix (`LIMITATIONS.md`, OI-11).
Per month `<YM>`:

| file | shape / type | meaning |
|---|---|---|
| `thermal_socket_meta_<YM>.csv` | 1,948–1,958 rows | per socket: `node`, `socket`, `active` (active core count), `rows5`/`rows1` (valid time rows at decimated / native cadence), `mean_temp` (°C), `n_within`, `n_samequad`, `n_crossquad` (pairs contributing), `rack` |
| `thermal_socket_c5_<YM>.npy` | `(nsock, 120) float32` | residual correlations of the upper triangle of the 16 active cores, **decimated** series |
| `thermal_socket_c1_<YM>.npy` | `(nsock, 120) float32` | the same at **native 20 s** cadence |
| `thermal_socket_lag_<YM>.npy` | `(nsock, 2, 9, 4) float32` | lagged means. Axis 1: 0 = native, lag in real 20 s steps; 1 = decimated rows, lag in rows. Axis 3: within-slice, same-quad, cross-quad, autocorrelation |

**The decimation rule, exactly.** A socket enters a month's census only if both halves are fully
configured in that month and each of its 16 active cores has more than 20,000 samples. Temperatures
are then read on a fixed grid of 20 s slots from the month's first slot, and **every fifth slot is
kept** (`DECIM = 5`, i.e. one sample per 100 s); `thermal_socket_build.py` asserts the phase of the
first kept slot so the grid matches the paper's. The `c1` files keep every slot instead, and exist
precisely so that the decimation cannot be the source of a result: the lag-0 contrast is +0.0606
decimated and +0.0607 native. The separate axis-1 distinction matters because an earlier draft
reported a "lag profile" whose unit was 100 s decimated rows, not 20 s — that table is withdrawn
(`CHANGELOG.md`).

### `analysis/thermal_corr_allnodes.npy`, `thermal_corr_24x24.npy`, `thermal_corr_v2.npy`

`(24, 24) float64` pooled residual correlation matrices: the fleet census, the pooled estimate kept
for history, and the three-day subsample of the first release. `thermal_socket_stats.py` reproduces
the census matrix from the per-socket files to 9.6e-10. `make_figures.py` overwrites
`thermal_corr_24x24.npy` with zeros when run without the raw data — **do not run it**
(`docs/reproduce.md`).

### Machine-readable results

Every corrected analysis writes its numbers as JSON next to its script, which is what
`tests/test_consistency.py` compares the README and docs against: `side_channel_results.json`,
`side_channel_raw.json`, `side_channel_values.json`, `model_fit.json`, `model_gamma_se.json`,
`model_quad.json`, `model_ppc.json`, `model_marginals.json`, `model_curveball.json`,
`model_balanced_panel.json`, `cohort_boundary.json`, `cohort_diversity.json`,
`cohort_incoming.json`, `lifecycle_reboot.json`, `lifecycle_os_crosscheck.json`,
`lifecycle_taxonomy_errors.json`, `equivalence_fit.json`, `equivalence_alloc_days.json`,
`survival_fit.json`, `survival_cindex.json`, `survival_sensitivity.json`,
`survival_legacy_check.json`, `thermal_socket_stats*.json`, `thermal_ibm_layout.json`,
`thermal_column_test.json`, `thermal_month_summary.json`. Several carry a `schema` string naming
what changed and when.

---

## 4. What is and is not asserted

`DATA.sha256` covers two kinds of file, and deliberately not a third:

1. **Inputs** — `daily/`, `cov_all/`, `os_all/`, `power_all/`, `power/`, `analysis/counts_*`,
   `harvest_*`, `per_month_marginals.csv`. Nothing in the repository rewrites these, so a changed
   checksum means the data changed.
2. **Committed outputs CI cannot recompute** — the per-socket thermal summaries and pooled
   matrices, the raw-timestamp side-channel measurements, the SLURM placement table, the
   100,000-draw curveball null and the slower thermal and cohort results. CI asserts their
   checksums and then asserts numbers read out of them. None of them carries a wall-clock timing
   field: a script that embedded one would break this gate on every re-run with every number
   bit-identical, so `curveball_null.py`, `side_channel_raw.py` and `side_channel_values.py` print
   their runtime instead of writing it.
3. **Not in the manifest: the files CI regenerates** (`model_fit.json`, `model_ppc.json`,
   `model_gamma_se.json`, `model_quad.json`, `model_marginals.json`, `side_channel_results.json`,
   `lifecycle_*.json`, `equivalence_fit.json`, `survival_*.json`, `changepoint_*.npy`,
   `ppc_resid_*.npy`). Every one of them is bit-identical when re-run on the machine that produced
   it — that was checked for all 64 committed result files — but *not* across BLAS builds and CPU
   microarchitectures, so a checksum would be a false alarm on another machine. CI re-derives them
   and asserts the values instead, which is the stronger check.

**Determinism and seeds.** Every random procedure fixes its seed in source, so each script is
deterministic given its inputs. The main ones: `curveball_null.py` splits its 100,000 draws into
`MODEL_CHUNKS` (default 12) independently seeded chunks, so the process count changes the speed and
not the numbers (`tests/test_model.py` asserts that); `changepoint.py` 11; `ppc.py` 29;
`gamma_clustered_se.py` 2026; `model_quad.py` 2027; `side_channel.py` 17; `survival.py` 15;
`equivalence.py` 37; `cohort_boundary.py` 22 and 55; `layout_hypotheses.py` 5 and 11;
`thermal_socket_stats.py` 0, 4, 5, 7, 8, 19 and 20260911 for its separate resamplings. The
superseded "five seeds" sensitivity of the first release is preserved in `model_curveball.json`
under `legacy_runs`: seeds 2024, 1, 2, 3, 4, giving gap *z* −20.23, −21.03, −20.89, −19.87, −20.08
and 17, 18, 17, 18, 16 pairs past the old 3.83 cut. The exchangeable null that replaced it reports
(*b*+1)/(*B*+1) against B = 100,000 rather than a seed-dependent count.

**The completeness gate.** `sweep_all_months.py` refuses to write a month's output unless all 48
core-temperature metric partitions (`p{0,1}_core{0..23}_temp`) extracted, because a partial
extraction is indistinguishable from disabled cores. The gate is a hard refusal and prints
`INCOMPLETE EXTRACTION: <k>/48 metrics` with the tar's return code; it does **not** yet write a
machine-readable audit record of a refusal, and the 31 committed months are the months that passed
it. Turning that print into a JSON audit log is an open artefact item, noted in the review log; it
is not claimed here.

---

## 5. Licence

The derived tables are CC BY 4.0, matching the upstream M100 ExaData release they come from; cite
Borghesi et al., *Scientific Data* **10**, 288 (2023), https://doi.org/10.1038/s41597-023-02174-3.
The code that produced them is MIT.
