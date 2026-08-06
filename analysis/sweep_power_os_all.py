"""One pass over all 31 archives collecting BOTH remaining external checks.

  (a) per-socket power for every month  -> settles OI-8, the p0/p1 relabelling mechanism,
      which needs ~20 events and had only 4 from the six-month sample.
  (b) ganglia cpu_num for every month   -> extends the OS ground-truth check from one month
      to the whole record.

Resumable: months whose output exists are skipped. Bounded disk: extract, reduce, delete.
"""
import subprocess, shutil, time, sys
import pandas as pd
from pathlib import Path

RAW = Path('/home/mohsen/exadata/raw')
OUT = Path('/home/mohsen/exadata-silicon-study/power_all'); OUT.mkdir(exist_ok=True)
GT = Path('/home/mohsen/exadata-silicon-study/os_all'); GT.mkdir(exist_ok=True)
TMP = Path('/home/mohsen/exadata-silicon-study/.pw_tmp')
EXPECT = 4320

months = sorted(p.stem for p in RAW.glob('*.tar'))
print(f'{len(months)} months', flush=True)
for ym in months:
    dp, dg = OUT / f'pw_{ym}.parquet', GT / f'os_{ym}.parquet'
    if dp.exists() and dg.exists():
        print(f'[{ym}] skip', flush=True); continue
    t0 = time.time()
    if TMP.exists(): shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    pats = [f'year_month={ym}/plugin=ipmi_pub/metric=p0_power',
            f'year_month={ym}/plugin=ipmi_pub/metric=p1_power',
            f'year_month={ym}/plugin=ganglia_pub/metric=cpu_num']
    r = subprocess.run(['tar', '-xf', str(RAW / f'{ym}.tar'), '-C', str(TMP), '--wildcards',
                        '--touch', '--warning=no-timestamp'] + pats,
                       capture_output=True, text=True)
    ip = TMP / f'year_month={ym}' / 'plugin=ipmi_pub'
    gp = TMP / f'year_month={ym}' / 'plugin=ganglia_pub'

    # (a) per-socket power -> daily mean per (node, socket)
    parts = []
    for s in (0, 1):
        f = ip / f'metric=p{s}_power' / 'a_0.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        d['day'] = d.timestamp.dt.floor('D')
        g = d.groupby(['node', 'day'])['value'].agg(['mean', 'count']).reset_index()
        g['socket'] = s
        parts.append(g); del d
    if parts:
        pw = pd.concat(parts, ignore_index=True)
        pw['node'] = pw['node'].astype('int32'); pw['socket'] = pw['socket'].astype('int8')
        pw.to_parquet(dp, index=False)
    else:
        print(f'[{ym}] no power metrics', flush=True)

    # (b) OS logical-CPU count -> value distribution per (node, day)
    f = gp / 'metric=cpu_num' / 'a_0.parquet'
    if f.exists():
        d = pd.read_parquet(f, columns=['timestamp', 'value', 'node'])
        d['day'] = d.timestamp.dt.floor('D')
        g = d.groupby(['node', 'day', 'value']).size().reset_index(name='n')
        g['node'] = g['node'].astype('int32')
        g.to_parquet(dg, index=False); del d
    else:
        print(f'[{ym}] no ganglia cpu_num', flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'[{ym}] done in {time.time()-t0:.0f}s', flush=True)
shutil.rmtree(TMP, ignore_errors=True)
print('SWEEP COMPLETE', flush=True)
