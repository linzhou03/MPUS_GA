"""Bounded real-data CUDA smoke test; never produces a formal experimental result."""
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
from MPUS_GA.trial_temporal.pcdiag import Audit, save_json
from MPUS_GA.trial_temporal.train_pcdiag import arguments


class Finished(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=list('ABCDEF'), default='B')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    cli = p.parse_args()
    torch.set_num_threads(4)
    args, spec = arguments(cli.direction, cli.data_dir, Path('/tmp/pcdiag_probe_no_formal_results'), subjects='1', seeds=(42,))
    args.cuda_memory_budget_gib = 22.0
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(22.0 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    prepared = prepare_sources(cli.data_dir, spec.source_domains, spec.scales)
    maximum, subject = 0, 1
    for path in sorted((cli.data_dir / spec.target_dataset / 'window_1s').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            count = int(np.unique(z['trial_id'], return_counts=True)[1].max())
            if count > maximum:
                maximum, subject = count, int(z['subject_id'][0])
    lengths, records = {}, []
    def batch(dataset, domain, size, labeled):
        index = max(range(len(dataset)), key=lambda i: len(dataset.groups[1.][dataset.keys[i]]))
        item = dataset._item(index, include_label=labeled)
        lengths[domain] = {s: len(v) for s, v in item['x'].items()}
        return collate_multiscale([item] * size)
    train._source_loader = lambda data, *unused: repeat(batch(data, spec.source_domains[0], 24, True), 3)
    train._target_loaders = lambda data, *unused: (repeat(batch(data, spec.target_dataset, 16, False), 3), (), ())
    audit = Audit(cli.output.parent / 'probe_audit', nodes=(0, 3))
    args._diagnostic = audit
    actual = train.train_step
    def step(*pos, **kw):
        pos = list(pos)
        pos[6] = (1, 350, 650)[len(records)]
        result = actual(*pos, **kw)
        torch.cuda.synchronize()
        used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        memory = [int(row.split(',')[1]) for row in used.splitlines() if row.split(',')[0].strip() == str(os.getpid())]
        records.append({'iteration': pos[6], 'loss': result['total'], 'allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                        'reserved_mib': torch.cuda.max_memory_reserved() / 1024**2, 'process_mib': max(memory) if memory else None})
        print('PROBE ' + json.dumps(records[-1]), flush=True)
        if len(records) == 3:
            save_json(cli.output, {'status': 'PASS', 'direction': cli.direction, 'longest_trials': lengths, 'steps': records,
                                   'scope': 'three steps at forced stages; no final evaluation or formal experiment'})
            raise Finished()
        return result
    train.train_step = step
    try:
        train.run_fold(args, spec, prepared, 42, subject, device)
    except Finished:
        pass


if __name__ == '__main__':
    main()
