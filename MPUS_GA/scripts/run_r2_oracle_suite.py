"""One physical GPU, one inherited lock, matched reference -> R2 -> Oracle relay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .run_r3_suite import SuiteStopped, run_parallel_plans
from .run_six_direction_final_suite import resolve_gpu_uuid
from ..trial_temporal.oracle_study import PROTOCOL
from ..trial_temporal.pcdiag import code_identity, save_json

MODULE = 'MPUS_GA.scripts.run_r2_oracle_suite'


def make_plan(run_name, output_root, gpu='0'):
    root, logs = Path(output_root) / f'results_{run_name}', Path(output_root) / 'logs' / run_name
    plan = []
    for direction in 'BCE':
        modes = ['reference', 'R2', 'P', 'C', 'PC'] + (['C_truth', 'PC_truth'] if direction == 'E' else [])
        for subject in range(1, 6):
            for mode in modes:
                plan.append({'direction': direction, 'mode': mode, 'subject': subject,
                    'variant': mode + f'/subject_{subject:02d}', 'seeds': [42],
                    'result_root': str(root), 'log_dir': str(logs / mode)})
    summary = {'gpu': gpu, 'core_conditions': 60, 'extra_E_truth_coverage_conditions': 10,
               'reported_conditions': 70, 'reference_generation_runs': 15, 'total_training_invocations': 85,
               'training_steps': 15 * 1000 + 70 * 700,
               'subjects_per_direction': 5, 'seed': 42, 'switch_interval_seconds': 1,
               'memory_limit_gib': None, 'order': 'B -> C -> E; each subject: reference -> exact R2 gate -> P/C/PC -> E extras'}
    return {gpu: plan}, summary


def command(python, item, data_dir, subjects=None, config=None):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_oracle_study',
            '--direction', item['direction'], '--mode', item['mode'], '--subject', str(item['subject']),
            '--result-root', item['result_root'], '--data-dir', str(data_dir)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpu', choices=['0'], default='0')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.run_name.startswith('r2_oracle_') or Path(args.run_name).name != args.run_name:
        p.error('Use a new r2_oracle_<unique> run name')
    if args.worker != (args.lock_fd is not None):
        p.error('Worker requires an inherited lock')
    queues, summary = make_plan(args.run_name, args.output_root, args.gpu)
    if args.dry_run:
        print(json.dumps({'summary': summary, 'protocol': PROTOCOL, 'queues': queues}, indent=2))
        return
    package = Path(__file__).resolve().parents[1]
    root, logs = args.output_root / f'results_{args.run_name}', args.output_root / 'logs' / args.run_name
    root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    with (os.fdopen(args.lock_fd, 'a') if args.worker else (root / '.suite.lock').open('a')) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('This study already has an active relay; no duplicate started', flush=True)
            return
        uuid = resolve_gpu_uuid(args.gpu)
        files = sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
        if not files:
            raise RuntimeError('Missing processed data')
        import torch
        import torch_geometric
        manifest = {'summary': summary, 'protocol': PROTOCOL, 'queues': queues,
                    'frozen': json.loads((package / 'trial_temporal/pcdiag_frozen.json').read_text()),
                    'code_sha256': code_identity(package), 'gpu_uuid': uuid,
                    'environment': {'python': sys.executable, 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
                                    'cuda': torch.version.cuda, 'deterministic_algorithms': True, 'tf32': False},
                    'data_identity': [[str(f), f.stat().st_size, f.stat().st_mtime_ns] for f in files]}
        path = root / 'suite_manifest.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError('Code/config/data/environment changed; use a new run name')
        if not path.exists():
            if list(root.glob('*/[BCE]/seed_*.json')):
                raise RuntimeError('Refusing results without a manifest')
            save_json(path, manifest)
        if not args.worker:
            env = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLCONFIGDIR=str(logs / '.matplotlib'))
            with (logs / 'suite.log').open('a') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, *sys.argv[1:], '--worker', '--lock-fd', str(lock.fileno())],
                    cwd=package.parent, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock.fileno(),))
            save_json(root / 'launch.json', {'worker_pid': child.pid, 'gpu_uuid': uuid, 'suite_log': str(logs / 'suite.log')})
            print(json.dumps({'worker_pid': child.pid, 'suite_log': str(logs / 'suite.log'), **summary}, indent=2))
            return
        def stopped(signum, frame):
            raise SuiteStopped()
        signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
        try:
            print('SUITE START ' + args.run_name, flush=True)
            status = run_parallel_plans(queues, sys.executable, package, args.data_dir, None, None,
                                        {args.gpu: uuid}, build_command=command)
            if status:
                save_json(root / 'suite_failed.json', {'exit_code': status, 'subsequent_jobs_cancelled': True})
                raise SystemExit(status)
            from ..trial_temporal.oracle_study_report import report_study
            completed = report_study(root, require_complete=True)
            save_json(root / 'all_complete.json', {'reported_conditions': completed, 'training_invocations': 85})
            print('ALL COMPLETE: 70/70 reported conditions', flush=True)
        except SuiteStopped:
            print('STOPPED; all subsequent jobs cancelled', flush=True)
            raise SystemExit(143)


if __name__ == '__main__':
    main()
