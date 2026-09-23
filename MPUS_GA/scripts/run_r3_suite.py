"""Two parallel GPU relays: A/C/E and B/D/F; two full seeds, one ablation seed."""
from __future__ import annotations

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

from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig
from MPUS_GA.trial_temporal.style_augmentation import STYLE_VARIANTS, style_config
from .run_r2_subgroup_suite import command as r2_command
from .run_six_direction_final_suite import DIRECTIONS, _code_hashes, resolve_gpu_uuid

MODULE = 'MPUS_GA.scripts.run_r3_suite'
SUBJECT_COUNTS = {'A': 16, 'B': 20, 'C': 16, 'D': 15, 'E': 20, 'F': 15}
SWITCH_INTERVAL_SECONDS = 1.0


def build_plan(run_name, output_root, variants, full_seeds, ablation_seed, subjects='all'):
    plan = []
    for variant in variants:
        seeds = list(full_seeds) if variant == 'full' else [ablation_seed]
        for direction in DIRECTIONS:
            subject_count = SUBJECT_COUNTS[direction] if subjects == 'all' else len(set(map(int, subjects.split(','))))
            plan.append({'variant': variant, 'direction': direction, 'seeds': seeds,
                         'folds': subject_count * len(seeds),
                         'result_root': str(output_root / f'results_{run_name}_{variant}'),
                         'log_dir': str(output_root / 'logs' / f'{run_name}_{variant}')})
    return plan


def training_command(python, item, data_dir, subjects, config):
    return r2_command(python, item['direction'], data_dir, item['result_root'],
                      item['seeds'], subjects, config, method='r3') + ['--r3-variant', item['variant']]


class SuiteStopped(Exception):
    pass


def split_plan(plan, gpus):
    """Keep each direction on one physical GPU across every variant and seed."""
    assignment = {direction: gpus[index % len(gpus)] for index, direction in enumerate(DIRECTIONS)}
    return {gpu: [item for item in plan if assignment[item['direction']] == gpu] for gpu in gpus}


def plan_summary(plan):
    return {'direction_jobs': len(plan), 'direction_seed_combinations': sum(len(item['seeds']) for item in plan),
            'folds': sum(item['folds'] for item in plan)}


def scheduler_only_change(old, new):
    """Resume existing folds only when training code, data and protocol are identical."""
    scheduling = {'physical_gpu', 'gpu_uuid', 'physical_gpus', 'gpu_uuids', 'gpu_queues', 'switch_interval_seconds',
                  'code_package', 'code_sha256'}
    # JSON normalization handles tuples in freshly computed data identities.
    normalized = json.loads(json.dumps(new))
    if {k: v for k, v in old.items() if k not in scheduling} != {
            k: v for k, v in normalized.items() if k not in scheduling}:
        return False
    previous, current = old.get('code_sha256', {}), normalized.get('code_sha256', {})
    allowed = {'scripts/run_r3_suite.py'}
    return bool(previous) and {k: v for k, v in previous.items() if k not in allowed} == {
        k: v for k, v in current.items() if k not in allowed}


def validate_manifest(path, serialized, allow_scheduler_update):
    if path.exists() and path.read_text() != serialized:
        if not allow_scheduler_update or not scheduler_only_change(json.loads(path.read_text()), json.loads(serialized)):
            raise SystemExit('Training code/config/data changed or scheduler update not authorized; use a new run name')


def run_parallel_plans(queues, python, package, data_dir, subjects, config, gpu_uuids,
                       launch=subprocess.Popen, pause=time.sleep, clock=time.monotonic,
                       build_command=training_command):
    """One child per GPU; any failure/cancellation stops both relays and their children."""
    remaining = {gpu: iter(items) for gpu, items in queues.items()}
    active = {}
    ready_at = {}

    def start_next(gpu):
        item = next(remaining[gpu], None)
        if item is None:
            return
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuids[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package.parent))
        log_path = Path(item['log_dir']) / f"{item['direction']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('a', encoding='utf-8')
        try:
            child = launch(build_command(python, item, data_dir, subjects, config), cwd=package.parent,
                           env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        active[gpu] = (child, log, item)
        print(f"{datetime.now().isoformat(timespec='seconds')} START GPU={gpu} pid={child.pid} "
              f"{item['variant']}/{item['direction']} seeds={item['seeds']}", flush=True)

    try:
        for gpu in queues:
            start_next(gpu)
        while active or ready_at:
            finished = []
            for gpu, (child, log, item) in active.items():
                code = child.poll()
                if code is not None:
                    print(f"{datetime.now().isoformat(timespec='seconds')} END GPU={gpu} "
                          f"{item['variant']}/{item['direction']} exit={code}", flush=True)
                    if code:
                        return code
                    finished.append(gpu)
            # Check every running child for failure before submitting any next job.
            for gpu in finished:
                _, log, _ = active.pop(gpu)
                log.close()
                ready_at[gpu] = clock() + SWITCH_INTERVAL_SECONDS
            for gpu in list(ready_at):
                if clock() >= ready_at[gpu]:
                    del ready_at[gpu]
                    start_next(gpu)
            if active or ready_at:
                pause(min(1.0, max(0.0, min(ready_at.values()) - clock())) if ready_at else 1.0)
        return 0
    finally:
        for child, _, _ in active.values():
            if child.poll() is None:
                child.terminate()
        for child, log, _ in active.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            finally:
                log.close()


def run_plan(plan, python, package, data_dir, subjects, config, gpu_uuid, launch=subprocess.Popen):
    """Never submit the next direction if a child fails or the supervisor is stopped."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package.parent))
    for item in plan:
        print(f"{datetime.now().isoformat(timespec='seconds')} START {item['variant']}/{item['direction']} seeds={item['seeds']}", flush=True)
        log_path = Path(item['log_dir']) / f"{item['direction']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            child = None
            try:
                child = launch(training_command(python, item, data_dir, subjects, config), cwd=package.parent,
                               env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
        print(f"{datetime.now().isoformat(timespec='seconds')} END {item['variant']}/{item['direction']} exit={code}", flush=True)
        if code:
            return code
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpus', nargs=2, help='Two physical nvidia-smi GPU indices; default: 0 1')
    gpu_group.add_argument('--gpu', help='Optional single-GPU relay')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-root', type=Path, help='Parent of results_<run> and logs/<run>')
    parser.add_argument('--full-seeds', nargs=2, type=int, default=[43, 42])
    parser.add_argument('--ablation-seed', type=int, default=43)
    parser.add_argument('--variants', nargs='+', choices=STYLE_VARIANTS, default=list(STYLE_VARIANTS))
    parser.add_argument('--target-subjects', default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--allow-scheduler-update', action='store_true',
                        help='Preserve completed folds after a verified scheduling-only change')
    args = parser.parse_args()
    gpus = [args.gpu] if args.gpu is not None else (args.gpus or ['0', '1'])
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus) or len(set(args.full_seeds)) != 2:
        parser.error('Provide distinct GPU indices and two distinct full-model seeds')
    if not args.run_name.startswith('r3_') or Path(args.run_name).name != args.run_name:
        parser.error('Run name must be r3_<unique-name>')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Variants must be unique')
    if args.target_subjects != 'all':
        try:
            subjects = list(map(int, args.target_subjects.split(',')))
            if not subjects or len(set(subjects)) != len(subjects) or min(subjects) < 1 or max(subjects) > min(SUBJECT_COUNTS.values()):
                raise ValueError()
        except ValueError:
            parser.error('Subjects must be all or distinct comma-separated integers 1..15')
    package = Path(__file__).resolve().parents[1]
    output_root = (args.output_root or package).resolve()
    data_dir = (args.data_dir or output_root / 'data_processed').resolve()
    config = CoTeachingConfig()
    plan = build_plan(args.run_name, output_root, args.variants, args.full_seeds, args.ablation_seed, args.target_subjects)
    summary = {'configurations': len(args.variants), **plan_summary(plan)}
    queues = split_plan(plan, gpus)
    gpu_queues = {gpu: {'directions': list(dict.fromkeys(item['direction'] for item in items)),
                        **plan_summary(items)} for gpu, items in queues.items()}
    if args.dry_run:
        print(json.dumps({'summary': summary, 'physical_gpus': gpus, 'gpu_queues': gpu_queues,
                          'switch_interval_seconds': SWITCH_INTERVAL_SECONDS,
                          'order': 'parallel_GPUs_each_variant_then_assigned_directions',
                          'plan': [{**item, 'command': training_command(sys.executable, item, data_dir, args.target_subjects, config)} for item in plan]}, indent=2))
        return
    uuids = {gpu: resolve_gpu_uuid(gpu) for gpu in gpus}
    if len(set(uuids.values())) != len(gpus):
        raise SystemExit('GPU indices resolved to duplicate UUIDs')
    files = []
    for domain in ('seed_iv', 'seed_v', 'seed_vii'):
        found = sorted((data_dir / domain).glob('window_*/*.npz'))
        if not found:
            raise SystemExit(f'Missing processed data: {domain}')
        files.extend(found)
    result_root = output_root / f'results_{args.run_name}'
    log_dir = output_root / 'logs' / args.run_name
    result_root.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'method': 'r3', 'summary': summary, 'plan': plan, 'physical_gpus': gpus, 'gpu_uuids': uuids,
                'gpu_queues': gpu_queues, 'switch_interval_seconds': SWITCH_INTERVAL_SECONDS,
                'full_seeds': args.full_seeds, 'ablation_seed': args.ablation_seed, 'subjects': args.target_subjects,
                'subgroup_config': asdict(config), 'variants': {v: asdict(style_config(v)) for v in args.variants},
                'python': sys.executable, 'code_package': str(package), 'code_sha256': _code_hashes(package),
                'data_dir': str(data_dir), 'data_identity': [(str(p.relative_to(data_dir)), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                'protocol': 'fixed_final_1000_no_target_truth_selection'}
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
    manifest_path = result_root / 'suite_manifest.json'
    validate_manifest(manifest_path, serialized, args.allow_scheduler_update)
    if not manifest_path.exists() and any(Path(item['result_root']).exists() for item in plan):
        raise SystemExit('Refusing unmanifested variant results; use a new run name')
    if not args.worker:
        # Fail early if an existing worker owns the same batch.
        with (result_root / '.suite.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('This R3 batch already has an active worker')
        with (log_dir / 'suite.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--worker',
                                      *(['--gpu', *gpus] if len(gpus) == 1 else ['--gpus', *gpus]),
                                      '--run-name', args.run_name,
                                      '--data-dir', str(data_dir), '--output-root', str(output_root),
                                      '--full-seeds', *map(str, args.full_seeds), '--ablation-seed', str(args.ablation_seed),
                                      '--variants', *args.variants, '--target-subjects', args.target_subjects,
                                      *(['--allow-scheduler-update'] if args.allow_scheduler_update else [])],
                                     cwd=package.parent, env=dict(os.environ, PYTHONPATH=str(package.parent)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'worker_pid': child.pid, 'summary': summary, 'gpu_queues': gpu_queues, 'suite_log': str(log_dir / 'suite.log'),
                          'manifest': str(manifest_path)}, indent=2))
        return
    with (result_root / '.suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This R3 batch already has an active worker')
        validate_manifest(manifest_path, serialized, args.allow_scheduler_update)
        if manifest_path.exists() and manifest_path.read_text() != serialized:
            backup = result_root / f"suite_manifest.before_scheduler_update_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
            backup.write_text(manifest_path.read_text())
            print(f'SCHEDULER UPDATE; original manifest saved to {backup}', flush=True)
        manifest_path.write_text(serialized)
        for item in plan:
            variant_root = Path(item['result_root'])
            variant_root.mkdir(parents=True, exist_ok=True)
            (variant_root / 'suite_manifest.json').write_text(serialized)
        def stopped(signum, frame):
            raise SuiteStopped()
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        print(f'SUITE START {args.run_name} {summary} GPU_QUEUES={gpu_queues}', flush=True)
        try:
            code = run_parallel_plans(queues, sys.executable, package, data_dir, args.target_subjects, config, uuids)
        except SuiteStopped:
            print('SUITE STOPPED; no subsequent tasks will start', flush=True)
            raise SystemExit(143)
        print(f'SUITE END exit={code}', flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
