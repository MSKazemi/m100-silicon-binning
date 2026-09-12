"""Socket-level workload for the power-equivalence analysis, from the SLURM job table.

The paired contrast p0 - p1 removes everything two sockets of a node share, but not a workload
that the scheduler places unevenly between them. Node-level utilisation cannot measure that
asymmetry. The M100 ExaData job table can: `cpus_alloc_layout` lists, per node, the SLURM core
indices allocated to each job, 0-15 on socket 0 and 16-31 on socket 1 (socket-major numbering,
16 cores per socket). Integrating each job's allocation over its run time gives, per node-day,
the allocated core-seconds on each socket. This is a *placement* decision made by the scheduler,
which sees 16 identical cores per socket and never the harvest map, so it is a pre-treatment
covariate rather than a consequence of socket power.

Output: analysis/equivalence_alloc.parquet with one row per (node, day) that had any job:
    node, day, core_s0, core_s1   allocated core-seconds on socket 0 / socket 1 that day
            gpu_s               allocated GPU-seconds on the node that day (node-level)
            njobs               distinct jobs overlapping the day
            nolay_s             node-seconds of overlapping jobs whose layout is EMPTY
The job table is filed by the month in which a job ENDED, so a job appears once; jobs spanning
midnight are split across days by exact overlap. Months 2020-03, 2020-04 and 2021-03 carry no
job table in the public release. More importantly, the per-core layout is populated only for
jobs that ended by 2020-10; from 2020-11 on the release records `{node: []}`. A node-day is
therefore usable for socket placement only if nolay_s == 0 and it falls on or before
2020-10-31 (a day with no job at all is a genuine zero before that date, unknown after it).
The table also has holes (no job ends at all on 2020-08-02..04, a few dozen on some days), so the
number of jobs ending on each day is written to analysis/equivalence_alloc_days.json and the
analysis keeps a node-day only if the job records for that day and the next two are complete.

Needs the raw monthly tars (RAW below); the derived parquet is what the analysis reads.
"""
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
RAW = Path('/home/mohsen/exadata/raw')
OUT = root / 'analysis' / 'equivalence_alloc.parquet'
OUT_DAYS = root / 'analysis' / 'equivalence_alloc_days.json'
PAIR = re.compile(r'(\d+)\s*:\s*\[([^\]]*)\]')
GPU = re.compile(r'gres/gpu=(\d+)')
DAY = 86400


def parse_layout(s):
    """'{668: [20], 669: [0, 1]}' -> list of (node, cores on socket 0, cores on socket 1, empty).

    SLURM numbers the 32 cores of an AC922 node socket-major: 0-15 socket 0, 16-31 socket 1.
    `empty` is True when the release recorded the node but no core list ('{668: []}').
    Entries whose node is None ('{None: []}') are skipped by the regex.
    """
    out = []
    if not isinstance(s, str):
        return out
    for node, cores in PAIR.findall(s):
        c = [int(x) for x in cores.split(',') if x.strip()]
        n0 = sum(1 for k in c if k < 16)
        out.append((int(node), n0, len(c) - n0, len(c) == 0))
    return out


def split_days(start, end):
    """Seconds of [start, end) falling on each UTC day: returns (day_start_epoch, seconds)."""
    d0 = (start // DAY) * DAY
    days, secs = [], []
    d = d0
    while d < end:
        ov = min(end, d + DAY) - max(start, d)
        if ov > 0:
            days.append(d)
            secs.append(ov)
        d += DAY
    return days, secs


def job_end_days(j):
    """Number of jobs ending on each UTC day: a day with few or none is a gap in the job table."""
    e = pd.to_datetime(j.end_time, utc=True).dropna().dt.floor('D')
    return e.value_counts()


def read_month(tar_path, ym):
    member = f'year_month={ym}/plugin=job_table/metric=job_info_marconi100/a_0.parquet'
    with tarfile.open(tar_path, 'r:') as tf:
        try:
            m = tf.getmember(member)
        except KeyError:
            return None
        buf = tf.extractfile(m).read()
    return pd.read_parquet(io.BytesIO(buf), columns=['job_id', 'start_time', 'end_time',
                                                      'cpus_alloc_layout', 'tres_alloc_str',
                                                      'num_nodes'])


def reduce_jobs(j):
    """Explode jobs to (node, day) allocated core-seconds per socket."""
    j = j.dropna(subset=['start_time', 'end_time', 'cpus_alloc_layout'])
    epoch = pd.Timestamp(0, tz='UTC')
    st = ((pd.to_datetime(j.start_time, utc=True) - epoch) // pd.Timedelta('1s')).to_numpy()
    en = ((pd.to_datetime(j.end_time, utc=True) - epoch) // pd.Timedelta('1s')).to_numpy()
    rows = []
    for jid, s, e, lay, tres, nn in zip(j.job_id, st, en, j.cpus_alloc_layout,
                                         j.tres_alloc_str, j.num_nodes):
        if e <= s:
            continue
        g = GPU.search(tres) if isinstance(tres, str) else None
        gpn = (int(g.group(1)) / max(int(nn), 1)) if g else 0.0     # GPUs per node
        days, secs = split_days(int(s), int(e))
        for node, n0, n1, empty in parse_layout(lay):
            for d, sec in zip(days, secs):
                rows.append((node, d, n0 * sec, n1 * sec, gpn * sec, sec if empty else 0, jid))
    if not rows:
        return None
    r = pd.DataFrame(rows, columns=['node', 'day', 'core_s0', 'core_s1', 'gpu_s', 'nolay_s',
                                    'job_id'])
    return r


def main():
    if not RAW.exists():
        sys.exit(f'raw archive not found at {RAW}; the derived {OUT.name} is committed')
    parts, ends = [], []
    for tp in sorted(RAW.glob('*.tar')):
        ym = tp.stem
        j = read_month(tp, ym)
        if j is None:
            print(f'[{ym}] no job table in the release')
            continue
        ends.append(job_end_days(j))
        r = reduce_jobs(j)
        print(f'[{ym}] jobs {len(j):>7,}  node-day pieces {0 if r is None else len(r):>9,}',
              flush=True)
        if r is not None:
            parts.append(r)
    R = pd.concat(parts, ignore_index=True)
    # a job is filed once (by end month), but guard against a re-filed job anyway
    R = R.drop_duplicates(['job_id', 'node', 'day'])
    A = (R.groupby(['node', 'day'])
          .agg(core_s0=('core_s0', 'sum'), core_s1=('core_s1', 'sum'),
               gpu_s=('gpu_s', 'sum'), nolay_s=('nolay_s', 'sum'),
               njobs=('job_id', 'nunique'))
          .reset_index())
    A['day'] = pd.to_datetime(A['day'], unit='s', utc=True)
    A['node'] = A['node'].astype('int32')
    for c in ['core_s0', 'core_s1', 'gpu_s', 'nolay_s']:
        A[c] = A[c].astype('float32')
    A['njobs'] = A['njobs'].astype('int32')
    A.to_parquet(OUT, index=False)
    E = pd.concat(ends).groupby(level=0).sum().sort_index()
    json.dump({str(d.date()): int(n) for d, n in E.items()},
              open(OUT_DAYS, 'w'), indent=0)
    print(f'wrote {OUT_DAYS.relative_to(root)}: job ends per day, {len(E)} days')
    cap = 16 * DAY
    print(f'\nwrote {OUT.relative_to(root)}: {len(A):,} node-days on {A.node.nunique()} nodes, '
          f'{A.day.min().date()} .. {A.day.max().date()}')
    full = A[A.nolay_s == 0]
    print(f'  node-days whose every job carries a core layout: {len(full):,} '
          f'(last such day {full[full.core_s0 + full.core_s1 > 0].day.max().date()})')
    print(f'  mean allocated fraction on those: socket 0 {full.core_s0.mean() / cap:.3f}, '
          f'socket 1 {full.core_s1.mean() / cap:.3f}')
    over = ((A.core_s0 > 1.001 * cap) | (A.core_s1 > 1.001 * cap)).mean()
    print(f'  node-days with a socket allocated beyond 16 cores x 24 h: {100 * over:.2f}% '
          f'(overlapping records; clipped at use)')


if __name__ == '__main__':
    main()
