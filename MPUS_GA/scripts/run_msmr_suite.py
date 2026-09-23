"""Six main MSMR directions x seeds 42/43/44; two independent GPU relays."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from MPUS_GA.trial_temporal.masked_multiscale import MSMRConfig
from .run_r3_suite import build_plan, split_plan, plan_summary, run_parallel_plans, SuiteStopped
from .run_six_direction_final_suite import _code_hashes, resolve_gpu_uuid

MODULE = 'MPUS_GA.scripts.run_msmr_suite'
SEEDS = [42, 43, 44]


def training_command(python, item, data_dir, subjects, config):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_msmr',
            '--experiment', item['direction'], '--data-dir', str(data_dir),
            '--result-root', item['result_root'], '--random-seeds', *map(str, item['seeds']),
            '--target-subjects', subjects, '--device', 'cuda:0']


def main_plan(run_name, output_root, gpus):
    plan = build_plan(run_name, output_root, ['full'], SEEDS, 42)
    queues = split_plan(plan, gpus)
    summary = {'configurations': 1, **plan_summary(plan), 'seeds': SEEDS,
               'gpu_queues': {g: {'directions': [p['direction'] for p in q], **plan_summary(q)}
                              for g, q in queues.items()}, 'switch_interval_seconds': 1.}
    return plan, queues, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs=2, default=['0', '1'])
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error('Provide two distinct physical GPUs')
    if not args.run_name.startswith('msmr_') or Path(args.run_name).name != args.run_name:
        parser.error('Use a unique msmr_ run name')
    if args.worker != (args.lock_fd is not None):
        parser.error('Worker requires the inherited suite lock')
    config = MSMRConfig()
    package = Path(__file__).resolve().parents[1]
    plan, queues, summary = main_plan(args.run_name, args.output_root, args.gpus)
    if args.dry_run:
        print(json.dumps({'summary': summary, 'config': asdict(config),
                          'commands': [training_command(sys.executable, p, args.data_dir, 'all', config) for p in plan]}, indent=2))
        return
    root = args.output_root / f'results_{args.run_name}'
    logs = args.output_root / 'logs' / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.suite.lock'
    lock = os.fdopen(args.lock_fd, 'a') if args.worker else lock_path.open('a')
    with lock:
        if args.worker:
            expected, inherited = lock_path.stat(), os.fstat(lock.fileno())
            if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
                raise SystemExit('Inherited lock does not belong to this suite')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This MSMR batch is already running; no duplicate was started')
        uuids = {g: resolve_gpu_uuid(g) for g in args.gpus}
        if len(set(uuids.values())) != 2:
            raise SystemExit('GPU identities must differ')
        files = sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
        if not files:
            raise SystemExit('Processed data is missing')
        manifest = {'method': 'msmr', 'config': asdict(config), 'summary': summary, 'plan': plan,
                    'gpu_uuids': uuids, 'code_sha256': _code_hashes(package), 'python': sys.executable,
                    'data_identity': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                    'protocol': 'fixed_final_1000_no_target_label_selection; no ablations'}
        serialized = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
        path = root / 'suite_manifest.json'
        if path.exists() and path.read_text() != serialized:
            raise SystemExit('Code/config/data changed; use a new run name')
        if not path.exists():
            if any(list(Path(p['result_root']).glob('*/seed_*_subject_*.json')) for p in plan):
                raise SystemExit('Refusing results without an experiment manifest')
            temp = path.with_suffix('.json.tmp')
            temp.write_text(serialized)
            temp.replace(path)
        if not args.worker:
            environment = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (logs / 'suite.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, *sys.argv[1:],
                                          '--worker', '--lock-fd', str(lock.fileno())],
                                         cwd=package.parent, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock.fileno(),))
            print(json.dumps({'worker_pid': child.pid, 'suite_log': str(logs / 'suite.log'), **summary}, indent=2))
            return
        variant_root = Path(plan[0]['result_root'])
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / 'suite_manifest.json').write_text(serialized)

        def stop(signum, frame):
            raise SuiteStopped()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print('SUITE START ' + json.dumps(summary), flush=True)
        try:
            code = run_parallel_plans(queues, sys.executable, package, args.data_dir, 'all', config, uuids,
                                      build_command=training_command)
        except SuiteStopped:
            print('SUITE STOPPED; subsequent tasks cancelled', flush=True)
            raise SystemExit(143)
        print('SUITE END exit=' + str(code), flush=True)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
