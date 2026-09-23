"""Frozen original-R2 training or isolated, predeclared Oracle tails."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from . import train
from .data import prepare_sources
from .pcdiag import Audit, frozen_profile, save_json


def arguments(direction, data_dir, result_root, subjects='all', seeds=(42, 43, 44), device='cuda:0'):
    spec, frozen = frozen_profile(direction)
    args = train.build_parser().parse_args(['--experiment', 'A_R2'])
    for k, v in frozen.items():
        setattr(args, k, v)
    args.experiment, args.data_dir, args.result_root = direction, Path(data_dir), Path(result_root)
    args.target_subjects, args.random_seeds, args.device = subjects, list(seeds), device
    train.validate_args(args, spec)
    return args, spec


def fold_paths(root, direction, seed, subject):
    result = Path(root) / direction / f'seed_{seed}_subject_{subject:02d}.json'
    return result, result.with_suffix('.audit')


def verify_complete(result, directory):
    if result.exists():
        if not (directory / 'complete.json').exists():
            raise RuntimeError(f'Existing result has incomplete diagnostic artifacts: {result}')
        state = json.loads((directory / 'complete.json').read_text())
        if not (directory / 'formal_final_context.pt').exists():
            raise RuntimeError('Missing formal final context')
        for node in state['nodes']:
            if node > state['start_iteration'] or (node == 0 and state['start_iteration'] == 0):
                if not (directory / f'step_{node:04d}.pt').exists():
                    raise RuntimeError(f'Missing snapshot: {directory}, step {node}')
        final = state['nodes'][-1]
        for step in range((state['start_iteration'] // 100 + 1) * 100, final + 1, 100):
            if not (directory / f'batches_{step:04d}.pt').exists():
                raise RuntimeError(f'Missing batch audit at {step}: {directory}')
        return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=list('ABCDEF'), required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path)
    p.add_argument('--mode', choices=['baseline', 'replay', 'P', 'C', 'PC'], default='baseline')
    cli = p.parse_args()
    if cli.mode != 'baseline' and (cli.direction not in 'BC' or cli.baseline_root is None):
        p.error('Oracle is fixed to directions B/C and requires --baseline-root')
    args, spec = arguments(cli.direction, cli.data_dir, cli.result_root,
                           subjects='all' if cli.mode == 'baseline' else '1,2,3,4,5',
                           seeds=(42, 43, 44) if cli.mode == 'baseline' else (42,))
    device = torch.device('cuda:0')
    args.cuda_memory_budget_gib = 0.
    torch.cuda.set_device(device)
    save_json(cli.result_root / cli.direction / 'frozen_config.json',
              {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'spec': train.asdict(spec), 'mode': cli.mode, 'allocator_budget_gib': None,
               'torch': torch.__version__, 'cuda': torch.version.cuda})
    prepared = None
    for seed in args.random_seeds:
        for subject in args.target_subjects:
            result, directory = fold_paths(cli.result_root, cli.direction, seed, subject)
            if verify_complete(result, directory):
                print('SKIP COMPLETE ' + str(result), flush=True)
                continue
            baseline_directory = None
            if cli.mode != 'baseline':
                baseline_result, baseline_directory = fold_paths(cli.baseline_root, cli.direction, seed, subject)
                if not verify_complete(baseline_result, baseline_directory):
                    raise RuntimeError('Baseline must finish before any Oracle')
            if prepared is None:
                prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
            args._diagnostic = Audit(directory, mode=cli.mode, baseline_directory=baseline_directory)
            print(f'START {cli.mode} {cli.direction} seed={seed} subject={subject}', flush=True)
            try:
                train.run_fold(args, spec, prepared, seed, subject, device)
            finally:
                del args._diagnostic
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(1.)


if __name__ == '__main__':
    main()
