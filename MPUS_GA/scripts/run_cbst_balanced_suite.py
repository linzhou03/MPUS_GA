"""Paired strict-quota versus independent-class CBST, seed 43, on xju."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import mean
import sys

from .run_r3_suite import SuiteStopped, run_parallel_plans
from .run_six_direction_final_suite import _code_hashes, resolve_gpu_uuid

COUNTS = dict(A=16, B=20, C=16, D=15, E=20, F=15)
VARIANTS = ('strict_quota', 'independent')
SELECTION = 'post300_bal_best'


def plan(run_name, output_root, gpus=('0', '1')):
    output_root = Path(output_root)
    queues = {gpu: [] for gpu in gpus}
    for index, direction in enumerate(COUNTS):
        gpu = gpus[index % len(gpus)]
        for variant in VARIANTS:
            queues[gpu].append(dict(
                variant=variant, selection=SELECTION, direction=direction,
                seeds=[43], folds=COUNTS[direction],
                result_root=str(output_root / f'results_{run_name}_{variant}'),
                log_dir=str(output_root / 'logs' / f'{run_name}_{variant}' / 'main' / 'seed_43'),
            ))
    summary = dict(variants=list(VARIANTS), directions=list(COUNTS), seeds=[43],
                   selection=SELECTION, iterations=1000, evaluation_interval=1,
                   jobs=sum(map(len, queues.values())), folds=2*sum(COUNTS.values()),
                   gpu_queues={gpu:[f"{item['direction']}/{item['variant']}" for item in items]
                               for gpu,items in queues.items()})
    return queues, summary


def command(python, item, data_dir, subjects=None, config=None):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_cbst',
            '--direction', item['direction'], '--variant', item['variant'],
            '--data-dir', str(data_dir), '--result-root', item['result_root'],
            '--seed', '43', '--subjects', 'all', '--selection', SELECTION]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def report(run_name, output_root):
    rows = {}
    for variant in VARIANTS:
        root = Path(output_root) / f'results_{run_name}_{variant}'
        rows[variant] = {}
        for direction, count in COUNTS.items():
            values = []
            for subject in range(1, count+1):
                path = root / direction / f'seed_43_subject_{subject:02d}.offline.json'
                if path.exists():
                    values.append(json.loads(path.read_text())['primary'])
            rows[variant][direction] = dict(completed=len(values), expected=count,
                metrics={key:mean(v[key] for v in values) for key in
                         ('accuracy','balanced_accuracy','macro_f1')} if values else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if not args.run_name.startswith('r3_balanced_') or Path(args.run_name).name != args.run_name:
        parser.error('Use a unique r3_balanced_ run name')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not all(x.isdigit() for x in args.gpus):
        parser.error('GPU indices must be distinct integers')
    queues, summary = plan(args.run_name, args.output_root, tuple(args.gpus))
    master = args.output_root / f'results_{args.run_name}'
    if args.dry_run:
        print(json.dumps(dict(summary=summary, queues=queues), indent=2)); return
    if args.report:
        print(json.dumps(report(args.run_name, args.output_root), indent=2)); return
    if args.status:
        state_path = master / 'suite_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('status') == 'running':
            try: os.kill(state['pid'], 0)
            except ProcessLookupError: state = {**state, 'status': 'not_alive'}
        print(json.dumps(dict(summary=summary, state=state,
                              progress=report(args.run_name,args.output_root)), indent=2)); return

    master.mkdir(parents=True, exist_ok=True)
    with (master / '.suite.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALREADY RUNNING'); return
        from .run_neighbor_soft_suite import environment
        from ..trial_temporal.train_cbst import cbst_config, complete
        package = Path(__file__).resolve().parents[1]
        uuids = {gpu:resolve_gpu_uuid(gpu) for gpu in args.gpus}
        if len(set(uuids.values())) != len(uuids): raise RuntimeError('Duplicate physical GPU')
        files = []
        for dataset in ('seed_iv','seed_v','seed_vii'):
            found = sorted((args.data_dir/dataset).glob('window_*/*.npz'))
            if not found: raise RuntimeError('Missing processed dataset '+dataset)
            files.extend(found)
        manifest = dict(summary=summary, queues=queues, gpu_uuids=uuids,
            environment=environment(), code_package=str(package), code_sha256=_code_hashes(package),
            config={variant:asdict(cbst_config(variant)) for variant in VARIANTS},
            frozen_profile_sha256=hashlib.sha256((package/'trial_temporal/pcdiag_frozen.json').read_bytes()).hexdigest(),
            data_identity=[[str(path.relative_to(args.data_dir)),path.stat().st_size,path.stat().st_mtime_ns]
                           for path in files],
            protocol='target-test BalAcc maximum, steps 301..1000, earliest tie')
        path = master/'suite_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('Code/config/data manifest changed; use a new run name')
        if not path.exists() and any(Path(item['result_root']).exists()
                                     for queue in queues.values() for item in queue):
            raise RuntimeError('Refusing unmanifested variant results')
        save(path, manifest)
        save(master/'experiment_list.json', dict(summary=summary, queues=queues))
        def stop(signum, frame): raise SuiteStopped()
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        save(master/'suite_state.json', dict(status='running',pid=os.getpid(),started=datetime.now().isoformat()))
        print('EXPERIMENT PLAN '+json.dumps(summary), flush=True)
        try:
            code = run_parallel_plans(queues, sys.executable, package, args.data_dir,
                                      'all', None, uuids, build_command=command)
            if code: raise RuntimeError(f'Training failed with exit {code}')
            for variant in VARIANTS:
                root = args.output_root/f'results_{args.run_name}_{variant}'
                for direction,count in COUNTS.items():
                    for subject in range(1,count+1):
                        result = root/direction/f'seed_43_subject_{subject:02d}.json'
                        if not complete(result, SELECTION, variant) or not result.with_suffix('.offline.json').exists():
                            raise RuntimeError(f'Missing complete fold: {result}')
        except SuiteStopped:
            save(master/'suite_state.json',dict(status='stopped',progress=report(args.run_name,args.output_root)))
            raise SystemExit(143)
        except BaseException as exc:
            save(master/'suite_state.json',dict(status='failed',error=str(exc),progress=report(args.run_name,args.output_root)))
            raise
        summary_report = report(args.run_name,args.output_root)
        save(master/'primary_summary.json',summary_report)
        save(master/'suite_state.json',dict(status='complete',progress=summary_report))
        print('ALL VARIANTS AND DIRECTIONS COMPLETE', flush=True)


if __name__ == '__main__': main()
