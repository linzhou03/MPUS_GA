"""Bounded, quantitative CUDA replay investigation. Never writes old results."""
import argparse
import gc
import json
from pathlib import Path

import torch

from ..trial_temporal import train
from ..trial_temporal.data import prepare_sources
from ..trial_temporal.pcdiag import Audit, STATE_OBJECTS, cpu, rng_state, save_json, save_pt
from ..trial_temporal.train_pcdiag import arguments, fold_paths


def difference(actual, expected):
    a, b = actual.cpu(), expected.cpu()
    if a.shape != b.shape:
        return {'shape_mismatch': [list(a.shape), list(b.shape)]}
    if a.is_floating_point():
        diff = (a.double() - b.double()).abs()
        return {'max_abs': float(diff.max()) if diff.numel() else 0.,
                'max_relative': float((diff / b.double().abs().clamp_min(1e-12)).max()) if diff.numel() else 0.,
                'outside_original_tolerance': int((diff > 2e-6 + 1e-5 * b.double().abs()).sum()),
                'unequal': int((a != b).sum())}
    return {'unequal': int((a != b).sum())}


def tree_differences(actual, expected, prefix=''):
    result = {}
    if isinstance(actual, torch.Tensor):
        d = difference(actual, expected)
        if d.get('unequal') or d.get('shape_mismatch'):
            result[prefix] = d
    elif isinstance(actual, dict):
        for key in actual:
            result.update(tree_differences(actual[key], expected[key], prefix + '.' + str(key)))
    elif isinstance(actual, (tuple, list)):
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            result.update(tree_differences(a, b, prefix + '.' + str(i)))
    else:
        import numpy as np
        if not np.array_equal(actual, expected):
            result[prefix] = {'unequal': True}
    return result


def deterministic(enabled):
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False if enabled else True


class ProbeDone(Exception):
    pass


class Probe(Audit):
    def __init__(self, directory, baseline, stop_step):
        super().__init__(directory, 'replay', baseline)
        self.stop_step, self.differences = stop_step, []

    def ready(self):
        super().ready()
        r = self.runtime
        restored = {'model': cpu(r['model'].state_dict()), 'optimizer': cpu(r['optimizer'].state_dict()),
                    'scheduler': cpu(r['scheduler'].state_dict()), 'rng': rng_state(self.device),
                    'objects': {k: cpu(vars(r[k])) for k in STATE_OBJECTS}}
        self.restore_differences = tree_differences(restored, {k: self.resume[k] for k in restored})

    def compare_evidence(self, iteration, key, actual, expected):
        self.differences.append({'step': iteration, 'key': key, **difference(actual, expected)})

    def after_step(self, iteration, record):
        ref = self.reference(iteration)
        for key in ('total', 'classification', 'domain', 'prototype'):
            self.differences.append({'step': iteration, 'key': 'loss.' + key,
                **difference(torch.tensor(record[key], dtype=torch.float64), torch.tensor(ref['losses'][key], dtype=torch.float64))})
        if iteration == 301:
            self.snapshot(iteration)
        self.pending.clear()
        if iteration == self.stop_step:
            self.snapshot(iteration)
            save_json(self.directory / 'differences.json', {'restore_differences': self.restore_differences,
                      'comparisons': self.differences, 'deterministic': torch.are_deterministic_algorithms_enabled()})
            raise ProbeDone()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--direction', choices=['B', 'C'], default='C')
    p.add_argument('--stop-step', type=int, default=340)
    args = p.parse_args()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    prepared = None
    for setting in ('historical', 'deterministic'):
        for replicate in (1, 2):
            deterministic(setting == 'deterministic')
            root = args.result_root / f'{setting}_{replicate}'
            if (fold_paths(root, args.direction, 42, 1)[1] / 'differences.json').exists():
                continue
            config, spec = arguments(args.direction, args.data_dir, root, '1', (42,))
            if prepared is None:
                prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
            audit = Probe(fold_paths(root, args.direction, 42, 1)[1],
                          fold_paths(args.baseline_root, args.direction, 42, 1)[1], args.stop_step)
            config._diagnostic = audit
            try:
                train.run_fold(config, spec, prepared, 42, 1, device)
            except ProbeDone:
                print('PROBE COMPLETE ' + str(audit.directory), flush=True)
            del config._diagnostic, audit
            gc.collect(); torch.cuda.empty_cache()
    report = {}
    for setting in ('historical', 'deterministic'):
        dirs = [fold_paths(args.result_root / f'{setting}_{i}', args.direction, 42, 1)[1] for i in (1, 2)]
        a, b = [torch.load(d / f'step_{args.stop_step:04d}.pt', weights_only=False, map_location='cpu') for d in dirs]
        compare = ('model', 'objects', 'rng')
        report[setting] = {'replicate_differences': tree_differences({k: a[k] for k in compare}, {k: b[k] for k in compare}),
                           'against_historical': [json.loads((d / 'differences.json').read_text()) for d in dirs]}
    save_json(args.result_root / 'replay_investigation.json', report)
    print('INVESTIGATION SAVED ' + str(args.result_root / 'replay_investigation.json'), flush=True)


if __name__ == '__main__':
    main()
