"""Independent GPU relays: GPU 0 A/C/E, GPU 1 B/D/F, one-second handover."""
import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from MPUS_GA.trial_temporal.full_pseudo_alignment import R4_VARIANTS, FullPseudoConfig
from .run_six_direction_final_suite import _code_hashes, resolve_gpu_uuid

MODULE = 'MPUS_GA.scripts.run_r4_suite'
COUNTS = {'A': 16, 'B': 20, 'C': 16, 'D': 15, 'E': 20, 'F': 15}
QUEUES = {'0': ('A', 'C', 'E'), '1': ('B', 'D', 'F')}


def build_plan(run_name, output_root, gpu, variants=R4_VARIANTS):
    return [{'variant': variant, 'direction': direction,
             'seeds': [43, 42] if variant == 'full' else [43],
             'folds': COUNTS[direction] * (2 if variant == 'full' else 1),
             'result_root': str(output_root / f'results_{run_name}_{variant}'),
             'log_dir': str(output_root / 'logs' / f'{run_name}_{variant}')}
            for variant in variants for direction in QUEUES[gpu]]


def command(python, item, data_dir):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train', '--method', 'r4',
            '--r4-variant', item['variant'], '--experiment', item['direction'],
            '--data-dir', str(data_dir), '--result-root', item['result_root'],
            '--random-seeds', *map(str, item['seeds']), '--target-subjects', 'all',
            '--source-batch-size', '24', '--target-batch-size', '16',
            '--adaptation-warmup-iterations', '0', '--adaptation-ramp-end', '0',
            '--evaluation-protocol', 'fixed_final', '--device', 'cuda:0']


class Stopped(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=QUEUES, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=R4_VARIANTS, default=list(R4_VARIANTS))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not args.run_name.startswith('r4_') or Path(args.run_name).name != args.run_name:
        parser.error('Use a unique r4_ run name')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Variants must be unique')
    package = Path(__file__).resolve().parents[1]
    plan = build_plan(args.run_name, args.output_root, args.gpu, args.variants)
    summary = {'gpu': args.gpu, 'directions': QUEUES[args.gpu], 'direction_jobs': len(plan),
               'direction_seed_combinations': sum(len(x['seeds']) for x in plan),
               'folds': sum(x['folds'] for x in plan), 'switch_interval_seconds': 1.0}
    if args.dry_run:
        print(json.dumps({'summary': summary, 'plan': plan}, indent=2))
        return
    gpu_uuid = resolve_gpu_uuid(args.gpu)
    files = sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
    if not files:
        raise SystemExit('Processed data missing')
    root = args.output_root / f'results_{args.run_name}'
    logs = args.output_root / 'logs' / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {'method': 'r4', 'summary': summary, 'plan': plan, 'gpu_uuid': gpu_uuid,
                'config': {v: asdict(FullPseudoConfig(v)) for v in args.variants},
                'code_sha256': _code_hashes(package), 'python': sys.executable,
                'data_identity': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                'protocol': 'fixed_final_1000; current_target_subject_only; no_target_truth_selection'}
    manifest_path = root / f'suite_manifest_gpu{args.gpu}.json'
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise SystemExit('Code/config/data changed; use a new run name')
    if not args.worker:
        with (logs / f'suite_gpu{args.gpu}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, *sys.argv[1:], '--worker'],
                                     cwd=package.parent, env=dict(os.environ, PYTHONPATH=str(package.parent)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'worker_pid': child.pid, **summary,
                          'suite_log': str(logs / f'suite_gpu{args.gpu}.log')}, indent=2))
        return
    with (root / f'.gpu{args.gpu}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This GPU relay is already running')
        if not manifest_path.exists() and any(list((Path(x['result_root']) / x['direction']).glob('seed_*_subject_*.json')) for x in plan):
            raise SystemExit('Refusing results without a manifest')
        manifest_path.write_text(serialized)
        for item in plan:
            variant_root = Path(item['result_root'])
            variant_root.mkdir(parents=True, exist_ok=True)
            (variant_root / manifest_path.name).write_text(serialized)
        def stopped(signum, frame):
            raise Stopped()
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        print('SUITE START ' + json.dumps(summary), flush=True)
        try:
            for item in plan:
                path = Path(item['log_dir']) / f"{item['direction']}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as log:
                    child = None
                    try:
                        child = subprocess.Popen(command(sys.executable, item, args.data_dir), cwd=package.parent,
                                                 env=env, stdin=subprocess.DEVNULL, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        print(f"{datetime.now().isoformat()} START {item['variant']}/{item['direction']} pid={child.pid}", flush=True)
                        code = child.wait()
                    finally:
                        if child is not None and child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                print(f"{datetime.now().isoformat()} END {item['variant']}/{item['direction']} exit={code}", flush=True)
                if code:
                    raise SystemExit(code)
                time.sleep(1.)
        except Stopped:
            print('SUITE STOPPED; remaining queue cancelled', flush=True)
            raise SystemExit(143)
        print('SUITE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
