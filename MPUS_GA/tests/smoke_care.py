"""Short actual R2+CARE training on synthetic EEG; no target truth."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from MPUS_GA.tests.test_muse_integration import fixture, with_ids, batch
from MPUS_GA.trial_temporal.train import train_step, care_experiment
from MPUS_GA.trial_temporal.care_pseudo_label import CareController, care_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(43)
    m, a, o, s, b, memory = fixture(enabled=False)
    care = CareController(care_config(), a.config)
    source = with_ids(batch(True, 18))
    for x in source['x'].values():
        x.mul_(.03)
        for i, c in enumerate(source['y']): x[i, :, :, int(c)] += 3.
    target = deepcopy(source); del target['y']; target['domain_id'].fill_(1)
    for x in target['x'].values(): x.add_(torch.randn_like(x) * .10)
    started = time.monotonic(); trace = []; active = 0
    for iteration in range(1, 651):
        rec = train_step(m, [source], target, o, s, torch.device('cpu'), iteration,
                         care_experiment('A'), torch.ones(1, 3)/3, b, .1, 5., 300, 600, .6,
                         source_prototype_memory=memory, subgroup_alignment=a, care=care)
        row = rec['care']
        assert torch.isfinite(torch.tensor(rec['total']))
        if iteration <= 300: assert row['added_loss'] == 0
        active += row['added_loss'] > 0
        if iteration == 1 or iteration % 50 == 0:
            trace.append(row); print(json.dumps(row), flush=True)
    assert active > 0, 'No CARE supervision activated'
    args.output.write_text(json.dumps({'status': 'PASS', 'iterations': 650,
                                      'seconds': time.monotonic()-started,
                                      'active_care_steps': active, 'trace': trace}, indent=2))


if __name__ == '__main__': main()
