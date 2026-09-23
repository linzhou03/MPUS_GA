"""One short CPU integration smoke, with no EEG data or target truth."""
from copy import deepcopy
from pathlib import Path
import json
import torch

from MPUS_GA.tests.test_training import _small_model
from MPUS_GA.tests.test_multiscale_coteaching import batch
from MPUS_GA.trial_temporal.full_pseudo_alignment import (FullPseudoAlignment, FullPseudoConfig,
                                                        attach_semantic_projection, R4_VARIANTS)
from MPUS_GA.trial_temporal.train import train_step, r4_experiment, build_parser, validate_args
from MPUS_GA.scripts.run_r4_suite import build_plan, command


def main():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    model = _small_model(scales=(1., 2., 4.), num_domains=2, dropout=.1)
    attach_semantic_projection(model)
    source, target = batch(True, 6), batch(False, 6)
    class Evidence:
        dataset = range(6)
        def __iter__(self):
            yield deepcopy(target)
    controller = FullPseudoAlignment(model)
    controller.evidence_loader = Evidence()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    trace = []
    for iteration in (1, 2, 25):
        row = train_step(model, [source], target, optimizer, scheduler, torch.device('cpu'),
                         iteration, r4_experiment('A'), torch.ones(1, 3)/3, None, .1, 5., 0, 0, .6,
                         full_pseudo_alignment=controller)
        assert torch.isfinite(torch.tensor(row['total']))
        assert len(controller.rows) == 6 and all(not r['q'].requires_grad for r in controller.rows.values())
        assert model.r4_semantic_projection[0].weight.grad.abs().sum() > 0
        trace.append({'iteration': iteration, 'total': row['total'], **row['full_pseudo_alignment']})
    assert not hasattr(controller, 'teacher')
    json.dumps(controller.state())
    for variant in R4_VARIANTS:
        args = build_parser().parse_args(['--method', 'r4', '--r4-variant', variant, '--experiment', 'A',
                                         '--result-root', '/private/tmp/r4_smoke_results'])
        validate_args(args, r4_experiment('A', variant))
        assert args.adaptation_warmup_iterations == args.adaptation_ramp_end == 0
    plans = [build_plan('r4_test', Path('/tmp'), gpu) for gpu in ('0', '1')]
    assert sum(x['folds'] for plan in plans for x in plan) == 714
    assert sum(len(x['seeds']) for plan in plans for x in plan) == 42
    assert {x['direction'] for x in plans[0]}.isdisjoint({x['direction'] for x in plans[1]})
    assert '--r4-variant' in command('python', plans[0][0], Path('/tmp'))
    # Removing confidence must remove target supervision without normalization undoing it.
    for row in controller.rows.values():
        row['reliability'] = 0.
    out_s = model(source['x'], source['mask'])
    out_t = model(target['x'], target['mask'])
    for variant in R4_VARIANTS:
        controller.config = FullPseudoConfig(variant)
        loss, info = controller.loss([out_s], [source], out_t, target)
        assert info['pseudo'] == info['align'] == 0
        assert torch.isfinite(loss)
        if variant in ('source_only', 'no_compact'):
            assert float(loss) == 0
    try:
        controller.refresh(model, [source], 26)
        raise AssertionError('Target labels were accepted')
    except RuntimeError as error:
        assert 'unlabeled' in str(error)
    print(json.dumps({'status': 'PASS', 'cpu_steps': 3, 'all_target_trials': 6,
                      'direction_seed_combinations': 42, 'folds': 714, 'trace': trace}, indent=2))


if __name__ == '__main__':
    main()
