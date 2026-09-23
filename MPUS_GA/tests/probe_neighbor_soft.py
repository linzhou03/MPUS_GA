"""Three real-data optimizer updates; no formal score, no queued training."""
import argparse
import json
from pathlib import Path

import torch

from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.data import prepare_sources
from MPUS_GA.trial_temporal.neighbor_soft import NeighborConfig
from MPUS_GA.trial_temporal.oracle_study import configure_determinism
from MPUS_GA.trial_temporal.pcdiag import save_json
from MPUS_GA.trial_temporal.train_pcdiag import arguments


class Done(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', default='full')
    cli = p.parse_args()
    configure_determinism()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    args, spec = arguments('A', cli.data_dir, cli.output.parent / 'no_formal_results', '1', (42,))
    args._neighbor_config = NeighborConfig(cli.variant)
    args.cuda_memory_budget_gib = 0.
    prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
    records = []
    original = train.train_step
    def step(*pos, **kw):
        pos = list(pos)
        pos[6] = (1, 350, 650)[len(records)]
        record = original(*pos, **kw)
        torch.cuda.synchronize()
        records.append(dict(iteration=pos[6], total=record['total'], auxiliary=record['neighbor_learning'],
                            allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2))
        print('PROBE ' + json.dumps(records[-1]), flush=True)
        if len(records) == 3:
            assert all(r['auxiliary']['added_loss'] > 0 for r in records[1:])
            save_json(cli.output, dict(status='PASS', variant=cli.variant, steps=records,
                                      scope='Three optimizer updates at forced stages; no formal result'))
            raise Done()
        return record
    train.train_step = step
    try:
        train.run_fold(args, spec, prepared, 42, 1, device)
    except Done:
        pass


if __name__ == '__main__':
    main()
