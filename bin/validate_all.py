#!/usr/bin/env python3
"""Run the random-seed ensemble, independent linear theory, and figures."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--julia', default=shutil.which('julia'))
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--jobs', type=int, default=3, help='Independent Julia workers')
parser.add_argument('--size', type=int, help='Points per axis; default 32 (8 in smoke mode)')
parser.add_argument('--worker-threads', type=int, default=1)
parser.add_argument('--integrator', choices=('rk4', 'leapfrog'), default='rk4')
parser.add_argument('--timestep-factor', type=float, default=0.01)
parser.add_argument('--output', type=Path)
parser.add_argument('--master-seed', type=int)
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
if not args.julia:
    parser.error('Julia not found; supply --julia /path/to/julia')
if not 1 <= args.jobs <= args.runs:
    parser.error('Require 1 <= jobs <= runs')
if args.worker_threads < 1:
    parser.error('worker-threads must be positive')
prefix = [args.julia, '--project='+str(project), '--compiled-modules=existing', '--threads=1']
cmd = prefix + [str(project/'bin/ensemble.jl'), '--runs', str(args.runs)]
cmd += ['--worker-threads', str(args.worker_threads)]
cmd += ['--integrator', args.integrator, '--timestep-factor', str(args.timestep_factor)]
if args.size is not None:
    cmd += ['--size', str(args.size)]
if args.output is not None:
    cmd += ['--output', str(args.output.resolve())]
if args.master_seed is not None:
    cmd += ['--master-seed', str(args.master_seed)]
if args.smoke:
    cmd += ['--smoke']
driver = subprocess.Popen(cmd, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
root = None
helper = None
helper_log = None
pipeline_stage = 'starting'
def progress_report():
    while True:
        if root is not None:
            try:
                lines = ['# 128³を含む格子検証：実行状況', '',
                         f'更新 UTC: {datetime.now(timezone.utc).isoformat()}', '',
                         f'工程: **{pipeline_stage}** / 予定試行数: {args.runs}', '',
                         '|試行|状態|保存済み e-fold|ステップ|', '|---|---|---:|---:|']
                for d in sorted(root.glob('trial_*')):
                    if not d.is_dir():
                        continue
                    complete = (d/'status.toml').exists()
                    row = {}
                    files = sorted(d.glob('simulation/*/background.csv'))
                    if files:
                        with files[-1].open() as f:
                            rows = list(csv.DictReader(f))
                        if rows:
                            row = rows[-1]
                    state = '完了' if complete else ('実行中' if (d/'.worker-lock').exists() else '待機')
                    lines.append(f"|{d.name}|{state}|{row.get('efolds','—')}|{row.get('step','—')}|")
                lines += ['', '進捗は保存済みの出力時刻です。出力間隔の途中でも計算は進みます。', '',
                          '全試行の後、独立線形理論との比較・PNG/PDF画像・日本語レポートを自動生成します。',
                          '終了条件は ε_H ≥ 1。理論は計量摂動なし、各試行の実際の初期場・初期微分から計算します。']
                if (root/'REPORT_JA.md').exists():
                    lines += ['', f'[結果レポート]({root/"REPORT_JA.md"})',
                              f'[曲率パワー比較]({root/"figures/curvature_power_comparison.png"})']
                tmp = root/'RUN_STATUS.md.tmp'
                tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
                tmp.replace(root/'RUN_STATUS.md')
            except (OSError, csv.Error):
                pass # A concurrently written output will be read on the next refresh.
        if pipeline_stage in ('completed','failed'):
            return
        time.sleep(60)

def status(stage, **details):
    global pipeline_stage
    pipeline_stage = stage
    if root is not None:
        data = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                    runs_requested=args.runs, lattice_size=args.size or (8 if args.smoke else 32),
                    jobs=args.jobs, worker_threads=args.worker_threads, **details)
        tmp = root/'PIPELINE_STATUS.json.tmp'
        tmp.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
        tmp.replace(root/'PIPELINE_STATUS.json')
        if stage in ('completed','failed'):
            progress_report()
for line in driver.stdout:
    print(line, end='', flush=True)
    if line.startswith('OUTPUT_DIR='):
        root = Path(line.strip().split('=',1)[1])
        status('simulations')
        threading.Thread(target=progress_report,daemon=True).start()
    if line.startswith('DONE trial='):
        status('simulations', last_driver_update=line.strip())
    if line.startswith('START trial=1 ') and args.jobs > 1 and helper is None:
        helper_log = (root/'parallel_dispatch.log').open('w')
        helper = subprocess.Popen(prefix+[str(project/'bin/assist_batch.jl'),str(root),str(args.jobs-1)],
                                  cwd=project,stdout=helper_log,stderr=subprocess.STDOUT)
code = driver.wait()
helper_code = helper.wait() if helper is not None else 0
if helper_log:
    helper_log.close()
if code or helper_code:
    status('failed', driver_exit=code, helper_exit=helper_code)
    raise SystemExit(f'Ensemble failed (driver={code}, helper={helper_code}); inspect trial logs in {root}')
theory_prefix = [args.julia, '--project='+str(project), '--compiled-modules=existing',
                '--threads='+str(args.worker_threads)]
for script, command in [('linear_theory', theory_prefix+[str(project/'bin/linear_theory.jl'),str(root)]),
                        ('plot_validation', [sys.executable,str(project/'bin/plot_validation.py'),str(root)]),
                        ('backreacted_theory', theory_prefix+[str(project/'bin/backreacted_theory.jl'),str(root)]),
                        ('plot_backreacted', [sys.executable,str(project/'bin/plot_backreacted.py'),str(root)])]:
    print(f'{script}: {root}', flush=True)
    status(script)
    with (root/(script+'.log')).open('w') as log:
        result = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status('failed', failed_stage=script, exit_code=result.returncode)
        raise SystemExit(f'{script} failed; inspect {root/(script+".log")}')
manifest = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','bin') for p in (project/folder).rglob('*')
            if p.suffix in ('.jl','.py')}
(root/'analysis_code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
status('completed', report=str(root/'REPORT_JA.md'),
       curvature_figure=str(root/'figures/backreacted_trial_01.png'))
print('REPORT:',root/'REPORT_JA.md')
print('CURVATURE FIGURE:',root/'figures/backreacted_trial_01.png')
