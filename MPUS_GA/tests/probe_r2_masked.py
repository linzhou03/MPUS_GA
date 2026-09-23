"""Three production CUDA steps on longest real trials; no experiment results."""
import argparse
from itertools import repeat
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.data import prepare_sources, collate_multiscale
from MPUS_GA.trial_temporal.train_msmr import validate_trial_lengths


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=list('ABCDEF'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    torch.set_num_threads(4)
    args = train.build_parser().parse_args([
        '--method', 'r2_msmr', '--experiment', opts.direction, '--data-dir', str(opts.data_dir),
        '--result-root', '/tmp/r2_msmr_memory_probe_no_results', '--source-batch-size', '24',
        '--target-batch-size', '16', '--learning-rate', '0.0005', '--cuda-memory-budget-gib', '8'])
    spec = train.r2_masked_experiment(opts.direction)
    train.validate_args(args, spec)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    maximum, subject = 0, None
    for path in sorted((opts.data_dir / spec.target_dataset / 'window_1s').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            count = int(np.unique(z['trial_id'], return_counts=True)[1].max())
            if count > maximum:
                maximum, subject = count, int(z['subject_id'][0])
    assert subject is not None
    prepared = prepare_sources(opts.data_dir, spec.source_domains, spec.scales)
    longest = {}

    def batch(dataset, domain, size, labeled):
        validate_trial_lengths(dataset)
        index = max(range(len(dataset)), key=lambda i: len(dataset.groups[1.][dataset.keys[i]]))
        item = dataset._item(index, include_label=labeled)
        longest[domain] = {'trial_key': dataset.keys[index],
                           'windows': {s: len(v) for s, v in item['x'].items()}}
        return collate_multiscale([item] * size)

    def source_loader(dataset, *unused):
        return repeat(batch(dataset, spec.source_domains[0], 24, True), 3)

    def target_loaders(dataset, *unused):
        return repeat(batch(dataset, spec.target_dataset, 16, False), 3), (), ()

    train._source_loader, train._target_loaders = source_loader, target_loaders
    actual_step, rows = train.train_step, []

    def step(*pos, **kw):
        pos = list(pos)
        pos[6] = (1, 350, 650)[len(rows)]
        record = actual_step(*pos, **kw)
        torch.cuda.synchronize()
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'], text=True)
        memory = [int(line.split(',')[1]) for line in output.splitlines()
                  if line.split(',')[0].strip() == str(os.getpid())]
        reconstruction = record['r2_masked_reconstruction']
        assert reconstruction['source_reconstruction'] > 0 and reconstruction['target_reconstruction'] > 0
        assert reconstruction['applied_auxiliary_gradient_norm'] > 0
        assert 0 < reconstruction['applied_gradient_ratio'] <= .20001
        assert memory and max(memory) < 10 * 1024
        rows.append({'iteration': pos[6], 'total_loss': record['total'],
                     'reconstruction': reconstruction,
                     'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                     'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                     'nvidia_smi_mib': max(memory)})
        print('PROBE_STEP ' + json.dumps(rows[-1]), flush=True)
        if len(rows) == 3:
            opts.output.parent.mkdir(parents=True, exist_ok=True)
            opts.output.write_text(json.dumps({'status': 'PASS', 'direction': opts.direction,
                                               'source_target_batch': [24, 16], 'auxiliary_batch': 4,
                                               'longest': longest, 'steps': rows,
                                               'scope': 'three production steps, longest real trials repeated; '
                                                        'iterations forced to 1/350/650 to exercise adaptation; '
                                                        'no target evaluation or formal results'}, indent=2))
            raise ProbeComplete()
        return record

    train.train_step = step
    try:
        train.run_fold(args, spec, prepared, 42, subject, device)
    except ProbeComplete:
        pass


if __name__ == '__main__':
    main()
