"""Prototype-free R2 + CBST with Confusion Matrix Improvement modules."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import torch
from . import train
from .cbst import CBSTConfig
from .data import prepare_sources
from .oracle_study import configure_determinism
from .pcdiag import save_json
from .train_cbst import (
    cbst_config, install_passive_evaluation, Recorder, offline_report, complete
)
from .train_pcdiag import arguments, fold_paths


def confusion_fix_config(ablation: str):
    """Parse ablation string into module enable flags."""
    parts = set(ablation.split('_'))
    return {
        'use_asymmetric_loss': 'asy' in parts,
        'use_calibration': 'cal' in parts,
        'use_ca_cbst': 'cbst' in parts,
    }


def settings(direction, data_dir, result_root, ablation='baseline', subjects='all',
             seed=43, device='cuda:0', selection='post300_bal_best', variant='positive_gate'):
    """Configure experiment with confusion fix modules."""
    args, spec = arguments(direction, data_dir, result_root, subjects, (seed,), device)

    # Parse ablation
    conf_fix = confusion_fix_config(ablation)
    ablation_suffix = f"cbst_{variant}_{selection}_conffix_{ablation}"

    spec = replace(
        spec,
        ablation=ablation_suffix,
        description=f'{direction}: R2+CBST with confusion fix {ablation}; {variant}; {selection}',
        use_prototypes=False,
        use_source_prototype_memory=False,
        prototype_weight=0.,
        use_source_multiview_anchor=False
    )

    assert not spec.use_subgroup_alignment and not spec.use_multiscale_coteaching

    # Attach CBST config
    args._cbst_config = cbst_config(variant)
    args.source_balance_alpha = 1.0

    # Attach confusion fix configs
    if conf_fix['use_asymmetric_loss']:
        from .losses.asymmetric_confusion import AsymmetricConfusionConfig
        args._asymmetric_loss_config = AsymmetricConfusionConfig()

    if conf_fix['use_calibration']:
        from .calibration.multiscale_balance import MultiscaleBalanceConfig
        args._calibration_config = MultiscaleBalanceConfig()

    if conf_fix['use_ca_cbst']:
        # Replace CBST config with CA-CBST
        from .pseudo_label.confusion_aware_cbst import ConfusionAwareCBSTConfig
        args._cbst_config = ConfusionAwareCBSTConfig(
            base_selection_mode=variant,
            use_negative_protection=True,
            use_neutral_rescue=True
        )

    # Selection protocol
    if selection == 'post300_bal_best':
        args._target_selection_metric = 'balanced_accuracy'
        args._target_selection_min_iteration = args.adaptation_warmup_iterations + 1

    args.evaluation_protocol = train.EVALUATION_PROTOCOL_CAGA_TARGET_BEST
    args.target_eval_interval = 1

    return args, spec, conf_fix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=list('ABCDEF'), required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    p.add_argument('--ablation', required=True,
                   help='Ablation config: baseline, asy, cal, cbst, asy_cal, asy_cbst, cal_cbst, full')
    p.add_argument('--selection', choices=['fixed_final', 'test_best', 'post300_bal_best'],
                   default='post300_bal_best')
    p.add_argument('--variant', choices=['strict_quota', 'independent', 'positive_gate'],
                   default='positive_gate')
    p.add_argument('--target-normalization', choices=['source', 'std_only', 'domain'],
                   default='domain')
    p.add_argument('--seed', type=int, default=43, choices=[43])
    p.add_argument('--subjects', default='all')

    cli = p.parse_args()

    # Validate ablation
    valid_ablations = [
        'baseline', 'asy', 'cal', 'cbst',
        'asy_cal', 'asy_cbst', 'cal_cbst', 'full'
    ]
    if cli.ablation not in valid_ablations:
        raise ValueError(f'Unknown ablation: {cli.ablation}. Valid: {valid_ablations}')

    configure_determinism()
    torch.set_num_threads(4)
    install_passive_evaluation()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one physical GPU per worker')

    args, spec, conf_fix = settings(
        cli.direction, cli.data_dir, cli.result_root,
        ablation=cli.ablation, subjects=cli.subjects, seed=cli.seed,
        selection=cli.selection, variant=cli.variant
    )
    args._target_normalization = cli.target_normalization

    # Save config
    config_dict = dict(
        spec=asdict(spec),
        cbst_config=asdict(args._cbst_config),
        confusion_fix=conf_fix,
        selection=cli.selection,
        variant=cli.variant,
        ablation=cli.ablation,
        target_normalization=cli.target_normalization,
        args={k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items() if not k.startswith('_')}
    )

    if conf_fix['use_asymmetric_loss']:
        config_dict['asymmetric_loss'] = asdict(args._asymmetric_loss_config)
    if conf_fix['use_calibration']:
        config_dict['calibration'] = asdict(args._calibration_config)

    save_json(
        cli.result_root / cli.direction / f'frozen_config_{cli.ablation}.json',
        config_dict
    )

    prepared = None
    for subject in args.target_subjects:
        path, _ = fold_paths(cli.result_root, cli.direction, cli.seed, subject)

        # Check completion
        if not complete(path, cli.selection, cli.variant, cli.target_normalization):
            if prepared is None:
                prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)

            args._diagnostic = Recorder(path, cli.selection)
            print(
                f'START CBST_CONFFIX/{cli.ablation}/{cli.variant}/{cli.selection}/'
                f'{cli.direction}/seed{cli.seed}/subject{subject:02d}',
                flush=True
            )

            train.run_fold(args, spec, prepared, cli.seed, subject, device)

            del args._diagnostic
            gc.collect()
            torch.cuda.empty_cache()

        if not path.with_suffix('.offline.json').exists():
            offline_report(path, args.data_dir)

        print('COMPLETE ' + str(path), flush=True)
        time.sleep(1.)


if __name__ == '__main__':
    main()
