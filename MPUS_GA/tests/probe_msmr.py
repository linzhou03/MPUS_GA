"""Three real-data CUDA steps on worst-length trials; no formal experiment output."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from MPUS_GA.trial_temporal.data import ALL_SCALES, prepare_sources, prepare_target, collate_multiscale
from MPUS_GA.trial_temporal.masked_multiscale import MSMRConfig, MaskedMultiScaleModel
from MPUS_GA.trial_temporal.train_msmr import configure_runtime, train_step, inputs, validate_trial_lengths, DIRECTION_DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direction', choices=DIRECTION_DATASETS, default='E')
    args = parser.parse_args()
    config = MSMRConfig()
    device = torch.device('cuda:0')
    configure_runtime(device, 42, config)
    # Find the longest actual source and target trials using only window metadata.
    maxima = {}
    subjects = {}
    source_domain, target_domain = DIRECTION_DATASETS[args.direction]
    for domain in (source_domain, target_domain):
        maximum = 0
        for p in (args.data_dir / domain / 'window_1s').glob('*.npz'):
            with np.load(p, allow_pickle=False) as z:
                count = int(np.unique(z['trial_id'], return_counts=True)[1].max())
                if count > maximum:
                    maximum = count
                    subjects[domain] = int(z['subject_id'][0])
        maxima[domain] = maximum
    prepared = prepare_sources(args.data_dir, (source_domain,), ALL_SCALES)
    data = prepared.datasets[0]
    validate_trial_lengths(data)
    index = max(range(len(data)), key=lambda i: len(data.groups[1.][data.keys[i]]))
    source = collate_multiscale([data._item(index, True)] * config.source_batch_size)
    target_data = prepare_target(args.data_dir, target_domain, subjects[target_domain], prepared, ALL_SCALES)
    validate_trial_lengths(target_data)
    target_index = max(range(len(target_data)), key=lambda i: len(target_data.groups[1.][target_data.keys[i]]))
    target = collate_multiscale([target_data._item(target_index, False)] * config.target_batch_size)
    model = MaskedMultiScaleModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    records = []
    for iteration in range(1, 4):
        record = train_step(model, optimizer, source, target, device, config)
        torch.cuda.synchronize()
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'], text=True)
        memory = [int(line.split(',')[1]) for line in output.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        record.update(iteration=iteration, nvidia_smi_mib=max(memory) if memory else None,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
        print(json.dumps(record), flush=True)
        records.append(record)
    optimizer.zero_grad(set_to_none=True)
    target_loss, _ = model.reconstruct(*inputs(target, device))
    target_loss.backward()
    gradients = {'spatial': float(model.spatial.raw.weight.grad.norm()),
                 'temporal': float(model.temporal.layers[0].self_attn.in_proj_weight.grad.norm())}
    assert all(v > 0 for v in gradients.values())
    assert all(r['nvidia_smi_mib'] is not None and r['nvidia_smi_mib'] < 10 * 1024 for r in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'PASS', 'config': asdict(config), 'max_windows': maxima,
                                     'scope': 'longest real source/target trials repeated; cross-domain memory/gradient probe only',
                                     'direction': args.direction, 'source_trial_key': data.keys[index],
                                     'target_trial_key': target_data.keys[target_index], 'steps': records,
                                     'target_encoder_gradient_norms': gradients}, indent=2))


if __name__ == '__main__':
    main()
