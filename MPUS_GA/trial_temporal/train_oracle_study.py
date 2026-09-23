"""Run one predeclared paired R2/Oracle fold on the one visible CUDA device."""
import argparse
import json
from pathlib import Path

import torch

from . import train
from .data import prepare_sources
from .oracle_study import MODES, PROTOCOL, StudyAudit, configure_determinism
from .pcdiag import save_json
from .train_pcdiag import arguments, fold_paths, verify_complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=list('BCE'), required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--subject', type=int, choices=range(1, 6), required=True)
    p.add_argument('--result-root', type=Path, required=True, help='Entire study root')
    p.add_argument('--data-dir', type=Path, required=True)
    cli = p.parse_args()
    if cli.mode.endswith('_truth') and cli.direction != 'E':
        p.error('The supplementary truth-coverage conditions are predeclared for E only')
    configure_determinism()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU by UUID')
    root = cli.result_root / cli.mode
    result, directory = fold_paths(root, cli.direction, 42, cli.subject)
    args, spec = arguments(cli.direction, cli.data_dir, root, str(cli.subject), (42,))
    args.cuda_memory_budget_gib = 0.
    reference = None
    if cli.mode != 'reference':
        ref_result, reference = fold_paths(cli.result_root / 'reference', cli.direction, 42, cli.subject)
        if not verify_complete(ref_result, reference):
            raise RuntimeError('A complete deterministic R2 reference is required')
        if cli.mode != 'R2':
            control_result, control = fold_paths(cli.result_root / 'R2', cli.direction, 42, cli.subject)
            if not verify_complete(control_result, control) or not (control / 'exact_replay_passed.json').exists():
                raise RuntimeError('Oracle blocked: exact matched R2 replay has not passed')
    if not verify_complete(result, directory):
        save_json(root / cli.direction / 'frozen_config.json', {
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'spec': train.asdict(spec), 'study': PROTOCOL, 'mode': cli.mode,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'tf32': False,
            'deterministic_algorithms': True, 'allocator_budget_gib': None})
        prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
        args._diagnostic = StudyAudit(directory, cli.mode, reference)
        print(f'START {cli.mode} {cli.direction} subject={cli.subject} seed=42', flush=True)
        train.run_fold(args, spec, prepared, 42, cli.subject, device)
        del args._diagnostic, prepared
    else:
        if json.loads(result.read_text()).get('study_protocol') != PROTOCOL:
            raise RuntimeError('Existing fold does not match this study protocol')
        print('SKIP COMPLETE ' + str(result), flush=True)
    if cli.mode != 'reference':
        import gc
        gc.collect(); torch.cuda.empty_cache()
        from .pcdiag_observe import observe_fold
        from .oracle_study_report import write_fold_exposures, report_study
        if not (directory / 'offline/complete.json').exists():
            observe_fold(result, cli.data_dir, device)
        write_fold_exposures(result, cli.data_dir)
        report_study(cli.result_root, require_complete=False)
    print('COMPLETE ' + str(result), flush=True)


if __name__ == '__main__':
    main()
