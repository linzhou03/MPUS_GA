"""Two locked GPU relays: original R2, offline audit, replay gate, then Oracle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .run_r3_suite import run_parallel_plans, SuiteStopped
from .run_six_direction_final_suite import resolve_gpu_uuid
from ..trial_temporal.pcdiag import code_identity, frozen_profile, save_json
from ..trial_temporal.pcdiag_observe import aggregate, subject_summary

MODULE = 'MPUS_GA.scripts.run_r2_pcdiag_suite'
COUNTS = dict(A=16, B=20, C=16, D=15, E=20, F=15)


def make_plan(run_name, output_root, gpus=('0', '1')):
    root = Path(output_root) / f'results_{run_name}'
    logs = Path(output_root) / 'logs' / run_name
    baseline, replay, oracle = ({g: [] for g in gpus} for _ in range(3))
    def item(direction, mode, stage, result_root, log_dir):
        return {'direction': direction, 'variant': mode + '_' + stage, 'mode': mode, 'stage': stage,
                'seeds': [42, 43, 44] if mode == 'baseline' else [42],
                'folds': COUNTS[direction] * 3 if mode == 'baseline' else 5,
                'result_root': str(result_root), 'log_dir': str(log_dir), 'baseline_root': str(root)}
    for gpu, directions in zip(gpus, ('CEA', 'DFB'), strict=True):
        for d in directions:
            baseline[gpu].extend([item(d, 'baseline', stage, root, logs) for stage in ('train', 'observe')])
    for gpu, d in zip(gpus, ('C', 'B'), strict=True):
        replay[gpu].append(item(d, 'replay', 'train', root / 'engineering_replay', logs / 'engineering_replay'))
        for mode in ('P', 'C', 'PC'):
            oracle[gpu].extend([item(d, mode, stage, root / 'oracle_pcdiag' / mode, logs / 'oracle_pcdiag' / mode)
                                for stage in ('train', 'observe')])
    summary = {'baseline_folds': 306, 'baseline_direction_seed_combinations': 18,
               'new_oracle_tails': 30, 'oracle_conditions_including_reused_baseline': 40,
               'engineering_replay_tails': 10, 'seeds': [42, 43, 44], 'steps': 1000,
               'oracle_resume_step': 300, 'gpu_queues': {gpus[0]: 'C -> E -> A (156)', gpus[1]: 'D -> F -> B (150)'},
               'switch_interval_seconds': 1, 'allocator_budget_gib': None,
               'phases': ['baseline+offline observations', 'offline aggregate', 'replay equivalence gate', 'P/C/PC+offline observations', 'paired Oracle report']}
    return baseline, replay, oracle, summary


def command(python, item, data_dir, subjects=None, config=None):
    if item['stage'] == 'observe':
        return [python, '-u', '-m', 'MPUS_GA.trial_temporal.pcdiag_observe', '--direction', item['direction'],
                '--data-dir', str(data_dir), '--result-root', item['result_root']]
    result = [python, '-u', '-m', 'MPUS_GA.trial_temporal.train_pcdiag', '--direction', item['direction'],
              '--data-dir', str(data_dir), '--result-root', item['result_root'], '--mode', item['mode']]
    if item['mode'] != 'baseline':
        result.extend(['--baseline-root', item['baseline_root']])
    return result


def oracle_report(root):
    from ..trial_temporal.train_pcdiag import fold_paths, verify_complete
    from ..trial_temporal.pcdiag_observe import load_truth
    import torch
    root = Path(root)
    rows = []
    for d in 'BC':
        for subject in range(1, 6):
            conditions = {}
            for mode in ('baseline', 'replay', 'P', 'C', 'PC'):
                folder = root if mode == 'baseline' else root / 'engineering_replay' if mode == 'replay' else root / 'oracle_pcdiag' / mode
                result, audit = fold_paths(folder, d, 42, subject)
                if not verify_complete(result, audit):
                    raise RuntimeError('Missing Oracle condition ' + str(result))
                row = json.loads(result.read_text())
                conditions[mode] = row['evaluation']['fused']
            # C and PC must consume exactly the same baseline candidate additions.
            for step in range(400, 1001, 100):
                cdir = fold_paths(root / 'oracle_pcdiag' / 'C', d, 42, subject)[1]
                pdir = fold_paths(root / 'oracle_pcdiag' / 'PC', d, 42, subject)[1]
                left = torch.load(cdir / f'batches_{step:04d}.pt', weights_only=False)
                right = torch.load(pdir / f'batches_{step:04d}.pt', weights_only=False)
                for a, b in zip(left, right, strict=True):
                    if not torch.equal(a['added'], b['added']) or not torch.equal(a['target_ids'], b['target_ids']):
                        raise RuntimeError('Coverage additions differ between C and PC')
            rows.append({'direction': d, 'subject': subject, 'seed': 42, 'conditions': conditions})
    comparisons = []
    for d in 'BC':
        for left, right in [('P', 'baseline'), ('C', 'baseline'), ('PC', 'P'), ('PC', 'C')]:
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                values = [(r['subject'], r['conditions'][left][metric] - r['conditions'][right][metric]) for r in rows if r['direction'] == d]
                comparisons.append({'direction': d, 'contrast': left + ' - ' + right, 'metric': metric, **subject_summary(values)})
    save_json(root / 'oracle_pcdiag' / 'paired_results.json', {'rows': rows, 'comparisons': comparisons})
    lines = ['# Oracle diagnostic results', '', 'These branches use target truth in training and are not UDA scores.',
             'Five predeclared subjects in each of B/C, seed 42. Baseline results are reused; replay is an engineering gate.',
             'PC minus P is the primary coverage contrast. C minus baseline changes both precision and coverage.', '',
             '|Direction|Contrast|Metric|Mean paired difference|Subject bootstrap 95% CI|', '|---|---|---|---:|---|']
    for c in comparisons:
        lines.append(f'|{c["direction"]}|{c["contrast"]}|{c["metric"]}|{c["mean"]:.4f}|{c["ci95"]}|')
    (root / 'oracle_pcdiag' / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus', nargs=2, default=['0', '1'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.run_name.startswith('r2_original_pcdiag_') or Path(args.run_name).name != args.run_name:
        p.error('Use a unique r2_original_pcdiag_ run name')
    if len(set(args.gpus)) != 2 or any(not g.isdigit() for g in args.gpus):
        p.error('Two distinct physical GPUs are required')
    if args.worker != (args.lock_fd is not None):
        p.error('Worker requires inherited lock')
    baseline, replay, oracle, summary = make_plan(args.run_name, args.output_root, args.gpus)
    if args.dry_run:
        print(json.dumps({'summary': summary, 'baseline': baseline, 'replay': replay, 'oracle': oracle}, indent=2))
        return
    package = Path(__file__).resolve().parents[1]
    root, logs = args.output_root / f'results_{args.run_name}', args.output_root / 'logs' / args.run_name
    root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.suite.lock'
    lock = os.fdopen(args.lock_fd, 'a') if args.worker else lock_path.open('a')
    with lock:
        if args.worker:
            a, b = lock_path.stat(), os.fstat(lock.fileno())
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise SystemExit('Incorrect inherited suite lock')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This diagnostic suite is already running; no duplicate started')
        uuids = {g: resolve_gpu_uuid(g) for g in args.gpus}
        if len(set(uuids.values())) != 2:
            raise SystemExit('GPU UUIDs must differ')
        files = sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
        if not files:
            raise SystemExit('Missing processed data')
        import torch
        import torch_geometric
        frozen = json.loads((package / 'trial_temporal/pcdiag_frozen.json').read_text())
        from ..preprocessing.dataset_metadata import EMOTION_TO_THREE_CLASS
        manifest = {'summary': summary, 'frozen': frozen, 'baseline': baseline, 'replay': replay, 'oracle': oracle,
                    'label_mapping': EMOTION_TO_THREE_CLASS,
                    'code_sha256': code_identity(package), 'gpu_uuids': uuids,
                    'environment': {'python': sys.executable, 'torch': torch.__version__, 'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda},
                    'data_identity': [[str(f), f.stat().st_size, f.stat().st_mtime_ns] for f in files]}
        path = root / 'suite_manifest.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise SystemExit('Code/config/data/environment changed; use a new run name')
        if not path.exists():
            if list(root.glob('*/seed_*_subject_*.json')):
                raise SystemExit('Refusing unmanifested results')
            save_json(path, manifest)
        if not args.worker:
            environment = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLCONFIGDIR=str(logs / '.matplotlib'))
            with (logs / 'suite.log').open('a') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, *sys.argv[1:], '--worker', '--lock-fd', str(lock.fileno())],
                                         cwd=package.parent, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
            print(json.dumps({'worker_pid': child.pid, 'suite_log': str(logs / 'suite.log'), **summary}, indent=2))
            return
        def stop(signum, frame):
            raise SuiteStopped()
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            for name, queues in [('baseline', baseline), ('replay', replay), ('oracle', oracle)]:
                print('PHASE START ' + name, flush=True)
                status = run_parallel_plans(queues, sys.executable, package, args.data_dir, 'all', None, uuids, build_command=command)
                if status:
                    raise SystemExit(status)
                if name == 'baseline':
                    if aggregate(root) != 306:
                        raise RuntimeError('Expected 306 complete baseline observations before Oracle')
                if name == 'oracle':
                    for mode in ('P', 'C', 'PC'):
                        aggregate(root / 'oracle_pcdiag' / mode)
                    oracle_report(root)
                save_json(root / f'{name}_complete.json', {'complete': True})
        except SuiteStopped:
            print('SUITE STOPPED; all subsequent tasks cancelled', flush=True)
            raise SystemExit(143)
        print('ALL PHASES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
