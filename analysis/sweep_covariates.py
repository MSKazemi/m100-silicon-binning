"""One pass over the 31 raw monthly archives for everything still missing.

Extracts, per month:
  ganglia_pub : boottime, cpu_speed, cpu_user, load_one
  ipmi_pub    : ambient, total_power

and reduces each to a per-(node, day) summary written to cov_all/cov_<ym>.parquet.

`boottime` is the important one: it is the epoch at which the node last booted, so a
*change* in boottime is a reboot, observed directly rather than inferred from a gap in
reporting. That replaces the paper's "changes coincide with a reporting gap" argument
with a real boot signal.

The rest are covariates for the workload-adjusted power analysis: node utilisation
(cpu_user), core frequency (cpu_speed), queue depth (load_one) and inlet air (ambient).

Resumable: skips months whose output already exists. Run it alone -- the extraction
holds a monthly partition in memory and a concurrent pandas job has been OOM-killed
before.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
RAW = Path('/home/mohsen/exadata/raw')
OUT = root / 'cov_all'
TMP = root / '.cov_tmp'
OUT.mkdir(exist_ok=True)

GANGLIA = ['boottime', 'cpu_speed', 'cpu_user', 'load_one']
IPMI = ['ambient', 'total_power']


def members(ym):
    """The exact tar members we want, so the archive is read once."""
    out = [f'year_month={ym}/plugin=ganglia_pub/metric={m}/a_0.parquet' for m in GANGLIA]
    out += [f'year_month={ym}/plugin=ipmi_pub/metric={m}/a_0.parquet' for m in IPMI]
    return out


def load(path):
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    # ExaMon schema: timestamp (already datetime64), value (float), node (str).
    if not pd.api.types.is_datetime64_any_dtype(d['timestamp']):
        d['timestamp'] = pd.to_datetime(d['timestamp'], unit='s', utc=True, errors='coerce')
    d = d.dropna(subset=['timestamp'])
    d['day'] = d['timestamp'].dt.floor('D')
    d['node'] = pd.to_numeric(d['node'], errors='coerce')
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    return d.dropna(subset=['node', 'value'])


def summarise(d, name):
    """Per (node, day). boottime needs distinct values; the rest need moments."""
    g = d.groupby(['node', 'day'])['value']
    if name == 'boottime':
        s = g.agg(['min', 'max', 'nunique', 'count'])
        s.columns = [f'boot_{c}' for c in ['min', 'max', 'ndistinct', 'n']]
        return s
    s = g.agg(['mean', 'max', 'count'])
    s.columns = [f'{name}_{c}' for c in ['mean', 'max', 'n']]
    return s


def do_month(ym):
    dst = OUT / f'cov_{ym}.parquet'
    if dst.exists():
        print(f'[{ym}] exists, skip', flush=True)
        return
    tar = RAW / f'{ym}.tar'
    if not tar.exists():
        print(f'[{ym}] no archive', flush=True)
        return
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    # A single sequential pass; missing members are not fatal (metric coverage varies).
    subprocess.run(['tar', 'xf', str(tar), '-C', str(TMP), *members(ym)],
                   stderr=subprocess.DEVNULL, check=False)

    parts = []
    for plug, mets in (('ganglia_pub', GANGLIA), ('ipmi_pub', IPMI)):
        for m in mets:
            d = load(TMP / f'year_month={ym}' / f'plugin={plug}' / f'metric={m}' / 'a_0.parquet')
            if d is None or d.empty:
                print(f'[{ym}]   {m}: absent', flush=True)
                continue
            parts.append(summarise(d, m))
            print(f'[{ym}]   {m}: {len(d):,} samples', flush=True)
            del d

    if not parts:
        print(f'[{ym}] nothing extracted', flush=True)
        shutil.rmtree(TMP, ignore_errors=True)
        return
    out = pd.concat(parts, axis=1).reset_index()
    out['node'] = out['node'].astype('int32')
    out.to_parquet(dst, index=False)
    print(f'[{ym}] wrote {len(out):,} node-days -> {dst.name}', flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    months = sys.argv[1:] or sorted(p.stem for p in RAW.glob('*.tar'))
    for ym in months:
        do_month(ym)
    print('sweep complete', flush=True)
