"""Local synthetic CPU smoke; deliberately does not read EEG target truth or GPUs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from .test_style_augmentation import step_fixture
from .test_multiscale_coteaching import batch
from MPUS_GA.trial_temporal.train import train_step, r3_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, choices=(20, 350), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(43)
    student, alignment, style, optimizer, scheduler, bank, memory = step_fixture('full')
    source = batch(True, 18)
    for value in source['x'].values():
        # Synthetic learnable source signal; the target loader still has no labels.
        value.add_(source['y'][:, None, None, None] * .7)
    target = deepcopy(source); del target['y']; target['domain_id'].fill_(1)
    for value in target['x'].values():
        value.mul_(1.15).add_(.1)
    started = time.monotonic()
    trace, frozen, active = [], None, 0
    for iteration in range(1, args.iterations + 1):
        record = train_step(student, [source], target, optimizer, scheduler, torch.device('cpu'),
                            iteration, r3_experiment('A'), torch.ones(1, 3)/3, bank, .1, 5., 300, 600, .6,
                            source_prototype_memory=memory, subgroup_alignment=alignment, style_augmentation=style)
        assert torch.isfinite(torch.tensor(record['total']))
        style_record = record['style_augmentation']
        if iteration <= 300:
            assert style_record['loss'] == 0
        else:
            torch.testing.assert_close(memory.memory, frozen, atol=0, rtol=0)
            active += int(style_record['active_style_samples'] > 0)
        if iteration == 300:
            frozen = memory.memory.clone()
        if iteration in (1, 20, 299, 300, 301, 350) or iteration % 50 == 0:
            row = {'iteration': iteration, 'total_loss': record['total'], 'style': style_record}
            trace.append(row); print(json.dumps(row), flush=True)
    if args.iterations > 300:
        assert active == args.iterations - 300
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'synthetic_cpu_runtime_only', 'iterations': args.iterations, 'seed': 43,
              'embedding_dimension': 32, 'batch_size': 18, 'seconds': time.monotonic()-started,
              'target_labels_available_to_training': False, 'post_warmup_active_steps': active,
              'source_memory_update_calls': memory.update_calls,
              'source_memory_frozen_after_300': args.iterations > 300,
              'style_state': style.state(), 'trace': trace}
    args.output.write_text(json.dumps(report, indent=2))
    torch.save(style.state_dict(), args.output.with_suffix('.style.pt'))
    print(f'PASS synthetic {args.iterations}-step smoke; report={args.output}', flush=True)


if __name__ == '__main__':
    main()
