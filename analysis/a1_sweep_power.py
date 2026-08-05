"""A1 step 1 -- collect per-socket power/thermal telemetry for the operational-impact study.

Six well-covered months spread across the record (stated as a sampling decision, not a
silent cap). Aggregates to per (node, socket, day) means so the analysis is cheap.
"""
import subprocess, shutil, time
import pandas as pd
from pathlib import Path

RAW = Path('/home/mohsen/exadata/raw')
OUT = Path('/home/mohsen/exadata-silicon-study/power'); OUT.mkdir(exist_ok=True)
TMP = Path('/home/mohsen/exadata-silicon-study/.a1_tmp')
MONTHS = ['20-07', '21-01', '21-06', '21-12', '22-05', '22-08']
SOCK = ['p{s}_power', 'p{s}_vdd_temp', 'p{s}_io_power', 'p{s}_mem_power']
NODE = ['total_power', 'ambient']

for ym in MONTHS:
    dest = OUT / f'power_{ym}.parquet'
    if dest.exists(): print(f'[{ym}] skip', flush=True); continue
    t0 = time.time()
    if TMP.exists(): shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    pats = [f'year_month={ym}/plugin=ipmi_pub/metric={m.format(s=s)}'
            for s in (0, 1) for m in SOCK]
    pats += [f'year_month={ym}/plugin=ipmi_pub/metric={m}' for m in NODE]
    r = subprocess.run(['tar', '-xf', str(RAW / f'{ym}.tar'), '-C', str(TMP), '--wildcards',
                        '--touch', '--warning=no-timestamp'] + pats,
                       capture_output=True, text=True)
    base = TMP / f'year_month={ym}' / 'plugin=ipmi_pub'
    found = sorted(p.name for p in base.glob('metric=*')) if base.exists() else []
    if len(found) != len(pats):
        print(f'[{ym}] INCOMPLETE {len(found)}/{len(pats)} -- skipping. {r.stderr[:150]}', flush=True)
        shutil.rmtree(TMP, ignore_errors=True); continue

    parts = []
    for s in (0, 1):
        for m in SOCK:
            mm = m.format(s=s)
            d = pd.read_parquet(base / f'metric={mm}' / 'a_0.parquet',
                                columns=['timestamp', 'value', 'node'])
            d['day'] = d.timestamp.dt.floor('D')
            g = d.groupby(['node', 'day'])['value'].agg(['mean', 'std', 'count']).reset_index()
            g['socket'] = s; g['metric'] = m.format(s='X')
            parts.append(g)
    for m in NODE:
        d = pd.read_parquet(base / f'metric={m}' / 'a_0.parquet',
                            columns=['timestamp', 'value', 'node'])
        d['day'] = d.timestamp.dt.floor('D')
        g = d.groupby(['node', 'day'])['value'].agg(['mean', 'std', 'count']).reset_index()
        g['socket'] = -1; g['metric'] = m
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    df['node'] = df['node'].astype('int32'); df['socket'] = df['socket'].astype('int8')
    df.to_parquet(dest, index=False)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'[{ym}] rows={len(df):,} nodes={df.node.nunique()} '
          f'days={df.day.nunique()} t={time.time()-t0:.0f}s', flush=True)
shutil.rmtree(TMP, ignore_errors=True)
print('A1 SWEEP COMPLETE', flush=True)
