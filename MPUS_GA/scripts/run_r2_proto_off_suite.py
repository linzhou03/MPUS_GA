"""Two-GPU relay for the R2 prototype-subsystem-off diagnostic."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_six_direction_final_suite import _code_hashes, resolve_gpu_uuid

SUBJECT_COUNTS = {'A': 16, 'B': 20, 'C': 16}
SWITCH_INTERVAL_SECONDS = 1.0


def make_plan(run_name, output_root, gpus, seeds):
    result_root = output_root / f'results_{run_name}'
    log_root = output_root / 'logs' / run_name
    jobs = [dict(direction=d, seed=seed, seeds=[seed], folds=SUBJECT_COUNTS[d],
                 result_root=str(result_root), log_dir=str(log_root))
            for d in ('A', 'B', 'C') for seed in seeds]
    queues = {gpu: jobs[index::len(gpus)] for index, gpu in enumerate(gpus)}
    summary = dict(method='r2_prototype_off', directions=['A', 'B', 'C'], seeds=seeds,
                   direction_seed_combinations=6, folds=sum(j['folds'] for j in jobs),
                   switch_interval_seconds=SWITCH_INTERVAL_SECONDS,
                   gpu_queues={gpu: [f"{j['direction']}/seed{j['seed']}" for j in q] for gpu, q in queues.items()})
    return jobs, queues, summary, result_root, log_root


def train_command(python, job, data_dir):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train', '--method', 'r2_prototype_off',
            '--experiment', job['direction'], '--data-dir', str(data_dir), '--result-root', job['result_root'],
            '--random-seeds', *map(str, job['seeds']), '--target-subjects', 'all',
            '--source-batch-size', '24', '--target-batch-size', '16', '--learning-rate', '0.0005',
            '--evaluation-protocol', 'fixed_final', '--device', 'cuda:0']


def run_queue(queue, uuid, python, package, data_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package.parent))
    for job in queue:
        log_path = Path(job['log_dir']) / f"{job['direction']}_seed{job['seed']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{datetime.now().isoformat(timespec='seconds')} START GPU={uuid} {job['direction']} seeds={job['seeds']}", flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            code = subprocess.call(train_command(python, job, data_dir), cwd=package.parent,
                                   env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(f"{datetime.now().isoformat(timespec='seconds')} END GPU={uuid} {job['direction']} exit={code}", flush=True)
        if code:
            return code
        time.sleep(SWITCH_INTERVAL_SECONDS)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus', nargs=2, default=['0', '1'])
    p.add_argument('--random-seeds', nargs=2, type=int, default=[42, 43])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.run_name.startswith('r2_proto_off_') or Path(args.run_name).name != args.run_name:
        p.error('run-name must start with r2_proto_off_')
    if len(set(args.gpus)) != 2 or any(not g.isdigit() for g in args.gpus):
        p.error('two distinct GPU indices are required')
    if len(set(args.random_seeds)) != 2:
        p.error('two distinct random seeds are required')
    package = Path(__file__).resolve().parents[1]
    jobs, queues, summary, result_root, log_root = make_plan(args.run_name, args.output_root, args.gpus, args.random_seeds)
    if args.dry_run:
        print(json.dumps({'summary': summary, 'jobs': jobs,
                          'commands': {g: [train_command(sys.executable, j, args.data_dir) for j in q]
                                       for g, q in queues.items()}}, indent=2))
        return
    result_root.mkdir(parents=True, exist_ok=True); log_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / '.suite.lock'
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('this relay already owns the result root')
    data_files = sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
    if not data_files:
        raise SystemExit('processed data not found')
    manifest = dict(summary=summary, data_dir=str(args.data_dir), gpu_indices=args.gpus,
                    gpu_uuids={g: resolve_gpu_uuid(g) for g in args.gpus}, code_sha256=_code_hashes(package),
                    protocol='fixed_final_1000_no_target_label_selection',
                    data_identity=[(str(f.relative_to(args.data_dir)), f.stat().st_size, f.stat().st_mtime_ns) for f in data_files])
    path = result_root / 'suite_manifest.json'
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise SystemExit('existing result root has a different manifest')
    path.write_text(serialized)
    uuids = {g: resolve_gpu_uuid(g) for g in args.gpus}
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda item: run_queue(item[1], uuids[item[0]], sys.executable,
                                                         package, args.data_dir), queues.items()))
    if any(statuses):
        raise SystemExit(1)
    (result_root / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('R2 PROTOTYPE-OFF RELAY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
