"""Positive pseudo-label gate on CSU: B and E together, then C, all on GPU 0."""
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

COUNTS = {'B': 20, 'C': 16, 'E': 20}
VARIANT = 'positive_gate'
SELECTION = 'post300_bal_best'


def plan(run_name, output_root):
    output_root = Path(output_root)
    result_root = output_root / f'results_{run_name}'
    log_dir = output_root / 'logs' / run_name / 'main' / 'seed_43'
    queues = {}
    for worker, gpu, direction in (('gpu0_B', '0', 'B'), ('gpu0_E', '0', 'E'), ('gpu0_C', '0', 'C')):
        queues[worker] = [dict(variant=VARIANT, direction=direction, selection=SELECTION,
            seeds=[43], folds=COUNTS[direction], result_root=str(result_root), log_dir=str(log_dir),
            physical_gpu=gpu)]
    summary = dict(variant=VARIANT, selection=SELECTION, seeds=[43], directions=list(COUNTS),
        folds=sum(COUNTS.values()), jobs=3, iterations=1000, evaluation_interval=1,
        phases=[{'physical_gpu': '0', 'concurrent_directions': ['B', 'E']},
                {'physical_gpu': '0', 'concurrent_directions': ['C'],
                 'starts_after': 'B and E both complete'}])
    return queues, summary


def command(python, item, data_dir, subjects=None, config=None):
    return [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_cbst',
        '--direction', item['direction'], '--variant', VARIANT, '--data-dir', str(data_dir),
        '--result-root', item['result_root'], '--seed', '43', '--subjects', 'all',
        '--selection', SELECTION]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def report(root):
    rows = {}
    for direction, expected in COUNTS.items():
        values = []
        for subject in range(1, expected + 1):
            path = root / direction / f'seed_43_subject_{subject:02d}.offline.json'
            if path.exists():
                values.append(json.loads(path.read_text())['primary'])
        rows[direction] = dict(completed=len(values), expected=expected,
            metrics={key: mean(row[key] for row in values) for key in
                     ('accuracy', 'balanced_accuracy', 'macro_f1')} if values else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if not args.run_name.startswith('r3_posgate_') or Path(args.run_name).name != args.run_name:
        parser.error('Use a unique r3_posgate_ run name')
    queues, summary = plan(args.run_name, args.output_root)
    root = args.output_root / f'results_{args.run_name}'
    if args.dry_run:
        print(json.dumps(dict(summary=summary, queues=queues), indent=2)); return
    if args.report:
        print(json.dumps(report(root), indent=2)); return
    if args.status:
        state_path = root / 'suite_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('status') == 'running':
            try: os.kill(state['pid'], 0)
            except ProcessLookupError: state = {**state, 'status': 'not_alive'}
        print(json.dumps(dict(summary=summary, state=state, progress=report(root)), indent=2)); return

    root.mkdir(parents=True, exist_ok=True)
    with (root / '.suite.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALREADY RUNNING'); return
        from .run_neighbor_soft_suite import environment
        from ..trial_temporal.train_cbst import cbst_config, complete
        package = Path(__file__).resolve().parents[1]
        physical_uuids = {'0': resolve_gpu_uuid('0')}
        worker_uuids = {worker: physical_uuids[item[0]['physical_gpu']]
                        for worker, item in queues.items()}
        files = []
        for dataset in ('seed_iv', 'seed_v', 'seed_vii'):
            found = sorted((args.data_dir / dataset).glob('window_*/*.npz'))
            if not found: raise RuntimeError('Missing processed dataset ' + dataset)
            files.extend(found)
        manifest = dict(summary=summary, queues=queues, physical_gpu_uuids=physical_uuids,
            worker_gpu_uuids=worker_uuids, environment=environment(), code_package=str(package),
            code_sha256=_code_hashes(package), config=asdict(cbst_config(VARIANT)),
            frozen_profile_sha256=hashlib.sha256(
                (package / 'trial_temporal/pcdiag_frozen.json').read_bytes()).hexdigest(),
            data_identity=[[str(path.relative_to(args.data_dir)), path.stat().st_size,
                            path.stat().st_mtime_ns] for path in files],
            protocol='target-test BalAcc maximum, steps 301..1000, earliest tie')
        path = root / 'suite_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('Code/config/data manifest changed; use a new run name')
        if not path.exists() and any(root.glob('*/seed_43_subject_*.json')):
            raise RuntimeError('Refusing unmanifested results')
        save(path, manifest)
        save(root / 'experiment_list.json', dict(summary=summary, queues=queues))
        def stop(signum, frame): raise SuiteStopped()
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        save(root / 'suite_state.json', dict(status='running', pid=os.getpid(),
            started=datetime.now().isoformat()))
        print('EXPERIMENT PLAN ' + json.dumps(summary), flush=True)
        try:
            for phase in (('gpu0_B', 'gpu0_E'), ('gpu0_C',)):
                phase_queues = {worker: queues[worker] for worker in phase}
                phase_uuids = {worker: worker_uuids[worker] for worker in phase}
                print('START PHASE ' + json.dumps(list(phase)), flush=True)
                code = run_parallel_plans(phase_queues, sys.executable, package, args.data_dir,
                                          'all', None, phase_uuids, build_command=command)
                if code: raise RuntimeError(f'Training failed with exit {code}')
            for direction, count in COUNTS.items():
                for subject in range(1, count + 1):
                    result = root / direction / f'seed_43_subject_{subject:02d}.json'
                    if (not complete(result, SELECTION, VARIANT) or
                            not result.with_suffix('.offline.json').exists()):
                        raise RuntimeError(f'Missing complete fold: {result}')
        except SuiteStopped:
            save(root / 'suite_state.json', dict(status='stopped', progress=report(root)))
            raise SystemExit(143)
        except BaseException as exc:
            save(root / 'suite_state.json', dict(status='failed', error=str(exc),
                progress=report(root)))
            raise
        summary_report = report(root)
        save(root / 'primary_summary.json', summary_report)
        save(root / 'suite_state.json', dict(status='complete', progress=summary_report))
        print('ALL THREE DIRECTIONS COMPLETE', flush=True)


if __name__ == '__main__': main()
