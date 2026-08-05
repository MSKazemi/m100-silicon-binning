"""Full-dataset sweep: per (node, socket, core, day) sample counts for every month.

Processes one month at a time -- extract only the core-temperature partitions from the tar,
aggregate to daily counts, delete the extracted files -- so peak disk stays a few GB.
Resumable: months whose output parquet already exists are skipped.

IMPORTANT: does NOT filter to sockets with exactly 16 active cores. Sockets with FEWER
active cores are the signal for field deconfiguration (core guard), so they must survive.
"""
import subprocess, sys, shutil, time
import pandas as pd
from pathlib import Path

RAW = Path('/home/mohsen/exadata/raw')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/mohsen/exadata-silicon-study/daily')
TMP = Path('/home/mohsen/exadata-silicon-study/.sweep_tmp')
OUT.mkdir(parents=True, exist_ok=True)

months = sorted(p.stem for p in RAW.glob('*.tar'))
print(f'{len(months)} months: {months[0]} .. {months[-1]}', flush=True)

for ym in months:
    dest = OUT / f'daily_{ym}.parquet'
    if dest.exists():
        print(f'[{ym}] skip (done)', flush=True); continue
    t0 = time.time()
    if TMP.exists(): shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    r = subprocess.run(
        ['tar', '-xf', str(RAW / f'{ym}.tar'), '-C', str(TMP), '--wildcards',
         f'year_month={ym}/plugin=ipmi_pub/metric=p0_core*_temp',
         f'year_month={ym}/plugin=ipmi_pub/metric=p1_core*_temp'],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[{ym}] TAR FAILED rc={r.returncode}: {r.stderr[:300]}', flush=True)
        continue
    t_tar = time.time() - t0

    parts = []
    base = TMP / f'year_month={ym}' / 'plugin=ipmi_pub'
    for sock in (0, 1):
        for core in range(24):
            f = base / f'metric=p{sock}_core{core}_temp' / 'a_0.parquet'
            if not f.exists():
                continue
            d = pd.read_parquet(f, columns=['timestamp', 'node'])
            if not len(d):
                continue
            d['day'] = d['timestamp'].dt.tz_convert('UTC').dt.floor('D')
            g = d.groupby(['node', 'day'], observed=True).size().reset_index(name='n')
            g['socket'] = sock; g['core'] = core
            parts.append(g)
    if not parts:
        print(f'[{ym}] NO CORE METRICS FOUND', flush=True)
        shutil.rmtree(TMP, ignore_errors=True); continue

    df = pd.concat(parts, ignore_index=True)
    df['node'] = df['node'].astype('int32'); df['socket'] = df['socket'].astype('int8')
    df['core'] = df['core'].astype('int8'); df['n'] = df['n'].astype('int32')
    df.to_parquet(dest, index=False)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'[{ym}] rows={len(df):>9,}  nodes={df.node.nunique():>4}  '
          f'days={df.day.nunique():>3}  tar={t_tar:5.0f}s  total={time.time()-t0:5.0f}s',
          flush=True)

shutil.rmtree(TMP, ignore_errors=True)
print('SWEEP COMPLETE', flush=True)
