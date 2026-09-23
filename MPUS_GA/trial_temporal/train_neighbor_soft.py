"""Frozen R2 and its four target-classification component variants."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch

from . import train
from .data import prepare_sources
from .neighbor_soft import NeighborConfig, VARIANTS, collect_bank, refine_bank
from .oracle_study import configure_determinism
from .pcdiag import STATE_OBJECTS, cpu, identities, inference_context, move, save_json, save_pt
from .train_pcdiag import arguments, fold_paths


class Recorder:
    """Passive batch records; final inference and truth join occur after training."""
    start_iteration = 0

    def __init__(self, result, config):
        self.result, self.config = Path(result), config
        self.batches = []

    def bind(self, runtime):
        self.runtime = {k: runtime[k] for k in ('args', 'spec', 'prepared', 'model', 'device',
                         'seed', 'subject', 'target_evidence_loader', 'neighbor_learning', *STATE_OBJECTS)}

    def ready(self):
        pass

    def capture(self, iteration, sources, target, output, consensus, active, source_labels, threshold):
        if 'y' in target:
            raise RuntimeError('Target labels in training recorder')
        self.batches.append(cpu(dict(iteration=iteration, ids=identities(target),
            pseudo_label=consensus.pseudo_label, accepted=consensus.valid_mask,
            confidence=consensus.confidence, votes=consensus.vote_count, jsd=consensus.js_divergence,
            fused_prediction=output['probability'].detach().argmax(-1), active=active)))
        return consensus, {}

    def after_step(self, iteration, record):
        pass

    def finish(self, result, final_evidence):
        r = self.runtime
        context = inference_context(r, 1000, final_evidence)
        bank = collect_bank(r['model'], r['target_evidence_loader'], r['device'], move(context, r['device']))
        q, neighbors = refine_bank(bank, self.config.neighbors)
        bank.update(refined_probability=q, neighbors=neighbors)
        learner = r['neighbor_learning']
        diagnostic = dict(config=asdict(self.config), r2_batches=self.batches, final_bank=bank,
                          auxiliary=learner.state() if learner is not None else None,
                          target_truth_used=False)
        save_pt(self.result.with_suffix('.pseudo.pt'), diagnostic)
        save_pt(self.result.with_suffix('.pt'), dict(model=cpu(r['model'].state_dict()),
            context=context, source_stats=cpu(r['prepared'].stats), spec=asdict(r['spec']),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(r['args']).items() if not k.startswith('_')},
            seed=r['seed'], subject=r['subject'], iteration=1000, neighbor_config=asdict(self.config),
            scope='Final inference checkpoint; interrupted folds restart from fixed seed'))
        result['neighbor_study'] = dict(config=asdict(self.config), target_truth_used=False,
            classification_start=301 if learner is not None else None,
            classification_ramp_end=600 if learner is not None else None, full_target_refresh_interval=50,
            pseudo_file=self.result.with_suffix('.pseudo.pt').name,
            checkpoint_file=self.result.with_suffix('.pt').name,
            numerics='deterministic; TF32 disabled', formal_evaluation='original fused prediction; no kNN postprocessing')


def offline_report(result_path, data_dir):
    """Truth is loaded only after the formal result has been atomically committed."""
    from .pcdiag_observe import load_truth, ratio
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    state = torch.load(result_path.with_suffix('.pseudo.pt'), map_location='cpu', weights_only=False)
    spec = result['experiment_spec']
    subject = int(result_path.stem.rsplit('_', 1)[1])
    truth = load_truth(data_dir, spec['target_dataset'], subject)
    def labels(ids):
        return torch.tensor([truth[tuple(key)][0] for key in ids.tolist()])
    final = state['final_bank']
    y = labels(final['ids'])
    raw, refined = final['raw_probability'].argmax(-1), final['refined_probability'].argmax(-1)
    reported = result['evaluation']['fused']['accuracy']
    if abs(float((raw == y).float().mean()) - reported) > 1e-6:
        raise RuntimeError('Saved final prediction does not reproduce formal R2 evaluation')
    rows = []
    for snapshot in (state['auxiliary'] or {}).get('snapshots', []):
        actual = labels(snapshot['ids'])
        original = snapshot['raw_probability'].argmax(-1)
        predicted = snapshot['labels']
        weight = snapshot['weights']
        classes = []
        for c, name in enumerate(train.CLASS_NAMES):
            chosen, true = predicted == c, actual == c
            correct = chosen & true
            classes.append(dict(class_name=name, predicted_count=int(chosen.sum()),
                precision=ratio(int(correct.sum()), int(chosen.sum())),
                correct_trial_coverage=ratio(int(correct.sum()), int(true.sum())),
                correct_weight_mass=float(weight[correct].sum()), wrong_weight_mass=float(weight[chosen & ~true].sum()),
                weighted_precision=ratio(float(weight[correct].sum()), float(weight[chosen].sum()))))
        rows.append(dict(iteration=snapshot['iteration'], classes=classes,
            wrong_to_correct=int(((original != actual) & (predicted == actual)).sum()),
            correct_to_wrong=int(((original == actual) & (predicted != actual)).sum())))
    save_json(result_path.with_suffix('.offline.json'), dict(
        scope='post-training diagnostic only; never used for training or model selection',
        final=dict(raw_accuracy=float((raw == y).float().mean()),
                   refined_accuracy=float((refined == y).float().mean()),
                   wrong_to_correct=int(((raw != y) & (refined == y)).sum()),
                   correct_to_wrong=int(((raw == y) & (refined != y)).sum())),
        snapshots=rows))


def complete(result, config):
    if not result.exists():
        return False
    row = json.loads(result.read_text())
    if row.get('neighbor_study', {}).get('config') != asdict(config):
        raise RuntimeError('Existing result has a different study configuration')
    if any(not result.with_suffix(suffix).exists() for suffix in ('.pt', '.pseudo.pt')):
        raise RuntimeError('Completed result is missing required checkpoint/evidence')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=list('ABCDEF'), required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--subjects', default='all')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    cli = p.parse_args()
    configure_determinism()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one physical GPU by UUID')
    config = NeighborConfig(cli.variant)
    args, spec = arguments(cli.direction, cli.data_dir, cli.result_root, cli.subjects, cli.seeds)
    args.cuda_memory_budget_gib = 0.
    if config.variant != 'r2':
        args._neighbor_config = config
    save_json(cli.result_root / cli.direction / 'frozen_config.json', dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')},
        spec=asdict(spec), neighbor=asdict(config), torch=torch.__version__, cuda=torch.version.cuda))
    prepared = None
    for seed in cli.seeds:
        for subject in args.target_subjects:
            result, _ = fold_paths(cli.result_root, cli.direction, seed, subject)
            if not complete(result, config):
                if prepared is None:
                    prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
                args._diagnostic = Recorder(result, config)
                print(f'START {cli.variant} {cli.direction} seed={seed} subject={subject}', flush=True)
                train.run_fold(args, spec, prepared, seed, subject, device)
                del args._diagnostic
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print('SKIP COMPLETE ' + str(result), flush=True)
            if not result.with_suffix('.offline.json').exists():
                offline_report(result, args.data_dir)
            print('COMPLETE ' + str(result), flush=True)
            time.sleep(1.)


if __name__ == '__main__':
    main()
