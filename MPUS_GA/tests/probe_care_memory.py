"""Three-step CUDA memory stress probe; never writes experiment results.

Pads batch=8 to each dataset's longest observed trial at every scale. Runs the
production R2 model/optimizer/teacher and forces an auxiliary CE on all target
rows to measure the extra backward path even before CARE has stable evidence.
"""
import argparse
from itertools import repeat
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.care_pseudo_label import CareController
from MPUS_GA.trial_temporal.data import prepare_sources, collate_multiscale, scale_key


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', choices=list('ABCDEF'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    torch.set_num_threads(4)
    args = train.build_parser().parse_args([
        '--method', 'care', '--experiment', opts.direction, '--data-dir', str(opts.data_dir),
        '--result-root', '/tmp/care_memory_probe_no_results', '--source-batch-size', '8',
        '--target-batch-size', '8', '--cuda-memory-budget-gib', '8'])
    spec = train.care_experiment(opts.direction)
    train.validate_args(args, spec)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    lengths = {}
    for domain in (*spec.source_domains, spec.target_dataset):
        lengths[domain] = {}
        for scale in spec.scales:
            maximum = 0
            for path in sorted((opts.data_dir / domain / ('window_' + scale_key(scale))).glob('*.npz')):
                with np.load(path, allow_pickle=False) as z:
                    maximum = max(maximum, int(np.unique(z['trial_id'], return_counts=True)[1].max()))
            lengths[domain][scale_key(scale)] = maximum
    print('MAX_WINDOWS', json.dumps(lengths), flush=True)
    prepared = prepare_sources(opts.data_dir, spec.source_domains, spec.scales)

    def padded(dataset, domain, labeled):
        index = max(range(len(dataset)), key=lambda i: len(dataset.groups[1.][dataset.keys[i]]))
        item = dataset._item(index, include_label=labeled)
        batch = collate_multiscale([item] * 8)
        for key, windows in lengths[domain].items():
            old = batch['x'][key]
            new = torch.zeros(8, windows, *old.shape[2:])
            new[:, :old.shape[1]] = old
            batch['x'][key] = new
            batch['mask'][key] = torch.ones(8, windows, dtype=torch.bool)
        return batch

    def source_loader(dataset, *unused):
        return repeat(padded(dataset, spec.source_domains[0], True), 3)

    def target_loaders(dataset, *unused):
        loader = repeat(padded(dataset, spec.target_dataset, False), 3)
        return loader, (), ()

    train._source_loader, train._target_loaders = source_loader, target_loaders
    actual_loss = CareController.loss

    def stress_loss(self, alignment, output, batch, iteration):
        loss, record = actual_loss(self, alignment, output, batch, iteration)
        logits = output['scale_logits']
        labels = alignment.pending[6].mean(1).argmax(-1)
        stress_ce = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(-1, 3).reshape(-1))
        return loss + .05 * stress_ce, record

    CareController.loss = stress_loss
    actual_step = train.train_step
    rows = []

    def step(*pos, **kw):
        pos = list(pos); pos[6] = (1, 350, 650)[len(rows)]
        record = actual_step(*pos, **kw)
        torch.cuda.synchronize()
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'], text=True)
        mine = [int(line.split(',')[1]) for line in output.splitlines()
                if line.split(',')[0].strip() == str(os.getpid())]
        rows.append({'iteration': pos[6], 'total_loss': record['total'],
                     'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                     'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
                     'nvidia_smi_mib': max(mine) if mine else None})
        print(json.dumps(rows[-1]), flush=True)
        if len(rows) == 3:
            opts.output.parent.mkdir(parents=True, exist_ok=True)
            opts.output.write_text(json.dumps({'status': 'PASS', 'direction': opts.direction,
                                               'batch': [8, 8], 'max_windows': lengths,
                                               'scope': 'maximum-padding + forced CE; 3 steps, not a full run',
                                               'steps': rows}, indent=2))
            raise ProbeComplete()
        return record

    train.train_step = step
    try:
        train.run_fold(args, spec, prepared, 43, 1, device)
    except ProbeComplete:
        pass


if __name__ == '__main__': main()
