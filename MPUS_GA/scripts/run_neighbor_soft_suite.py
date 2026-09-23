"""Frozen R2 + neighbor/SoftMatch study: csu dual relay, xju GPU 1 relay."""
import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .run_r3_suite import SuiteStopped, run_parallel_plans
from .run_six_direction_final_suite import resolve_gpu_uuid
from ..trial_temporal.neighbor_soft import NeighborConfig
from ..trial_temporal.pcdiag import code_identity, save_json

COUNTS = dict(A=16, B=20, C=16, D=15, E=20, F=15)
MODULE = 'MPUS_GA.scripts.run_neighbor_soft_suite'


def make_plan(host, run_name, output_root):
    root = Path(output_root) / f'results_{run_name}'
    logs = Path(output_root) / 'logs' / run_name
    variants = ('full', 'r2') if host == 'csu' else ('raw_ce', 'raw_soft', 'neighbor_ce')
    phases = []
    for variant in variants:
        queues = {'0': [], '1': []} if host == 'csu' else {'1': []}
        for index, direction in enumerate('ABCDEF' if host == 'csu' else 'ABC'):
            gpu = str(index % 2) if host == 'csu' else '1'
            seeds = [42, 43, 44] if variant == 'full' else [42]
            queues[gpu].append(dict(direction=direction, variant=variant, seeds=seeds,
                folds=COUNTS[direction] * len(seeds), result_root=str(root / variant),
                log_dir=str(logs / variant)))
        phases.append((variant, queues))
    items = [item for _, queues in phases for queue in queues.values() for item in queue]
    return phases, dict(host=host, variants=list(variants), direction_seed_jobs=sum(len(x['seeds']) for x in items),
        folds=sum(x['folds'] for x in items), switch_interval_seconds=1,
        full_seeds=[42, 43, 44], other_seeds=[42], iterations=1000,
        phases=[name for name, _ in phases], physical_gpus=['0', '1'] if host == 'csu' else ['1'])


def command(python, item, data_dir, subjects, config):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_neighbor_soft',
            '--direction', item['direction'], '--variant', item['variant'],
            '--seeds', *map(str, item['seeds']), '--data-dir', str(data_dir),
            '--result-root', item['result_root'], '--subjects', 'all']


def dataset_identity(data_dir):
    result = {}
    for domain in ('seed_iv', 'seed_v', 'seed_vii'):
        files = sorted((data_dir / domain).glob('window_*/*.npz'))
        if not files:
            raise RuntimeError('Missing processed dataset: ' + domain)
        for path in files:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            result[str(path.relative_to(data_dir))] = h.hexdigest()
    return result


def environment():
    import torch, numpy, scipy, sklearn, torch_geometric
    return dict(python=sys.version, executable=sys.executable, cuda=torch.version.cuda,
        packages={x.__name__: x.__version__ for x in (torch, numpy, scipy, sklearn, torch_geometric)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=['csu', 'xju'], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run_name.startswith('neighbor_soft_') or Path(args.run_name).name != args.run_name:
        parser.error('Use a unique neighbor_soft_ run name')
    if args.worker != (args.lock_fd is not None):
        parser.error('Worker requires inherited lock')
    phases, summary = make_plan(args.host, args.run_name, args.output_root)
    root, logs = args.output_root / f'results_{args.run_name}', args.output_root / 'logs' / args.run_name
    if args.dry_run:
        print(json.dumps(dict(summary=summary, phases=phases), indent=2))
        return
    if args.status:
        counts = {variant: len(list((root / variant).glob('*/seed_*_subject_*.json'))) for variant in summary['variants']}
        counts = {variant: len([p for p in (root / variant).glob('*/seed_*_subject_*.json') if p.name.count('.') == 1]) for variant in counts}
        print(json.dumps(dict(summary=summary, completed_folds=counts,
                             suite_state=json.loads((root / 'suite_state.json').read_text()) if (root / 'suite_state.json').exists() else None), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.suite.lock'
    lock = os.fdopen(args.lock_fd, 'a') if args.worker else lock_path.open('a')
    with lock:
        if args.worker:
            a, b = lock_path.stat(), os.fstat(lock.fileno())
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise RuntimeError('Inherited suite lock mismatch')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALREADY RUNNING: no duplicate queue started')
            return
        package = Path(__file__).resolve().parents[1]
        gpu_uuids = {gpu: resolve_gpu_uuid(gpu) for gpu in summary['physical_gpus']}
        if len(set(gpu_uuids.values())) != len(gpu_uuids):
            raise RuntimeError('GPU identities must be distinct')
        manifest = dict(summary=summary, phases=phases, environment=environment(),
            code_sha256=code_identity(package), code_package=str(package),
            data_sha256=dataset_identity(args.data_dir), gpu_uuids=gpu_uuids,
            config={v: asdict(NeighborConfig(v)) for v in summary['variants']},
            protocol='original_R2_fixed1000; auxiliary uses existing 300/600 ramp; final target evaluation only')
        manifest = json.loads(json.dumps(manifest))
        path = root / 'suite_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('Code/data/config/environment changed; use a new run name')
        if not path.exists():
            save_json(path, manifest)
        if not args.worker:
            env = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLCONFIGDIR=str(logs / '.matplotlib'),
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with (logs / 'suite.log').open('a') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, *sys.argv[1:],
                    '--worker', '--lock-fd', str(lock.fileno())], cwd=package.parent, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock.fileno(),))
            print(json.dumps(dict(worker_pid=child.pid, suite_log=str(logs / 'suite.log'), **summary), indent=2))
            return
        def stop(signum, frame):
            raise SuiteStopped()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            for phase, queues in phases:
                print(f'{datetime.now().isoformat()} PHASE START {phase}', flush=True)
                save_json(root / 'suite_state.json', dict(status='running', phase=phase, pid=os.getpid()))
                status = run_parallel_plans(queues, sys.executable, package, args.data_dir, 'all', None,
                                            gpu_uuids, build_command=command)
                if status:
                    save_json(root / 'suite_state.json', dict(status='failed', phase=phase, exit_code=status))
                    raise SystemExit(status)
                save_json(root / f'{phase}_complete.json', dict(complete=True))
        except SuiteStopped:
            save_json(root / 'suite_state.json', dict(status='stopped', phase=phase))
            print('STOPPED: queued successors cancelled', flush=True)
            raise SystemExit(143)
        save_json(root / 'suite_state.json', dict(status='complete', summary=summary))
        print('ALL PHASES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
