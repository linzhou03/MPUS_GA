"""Independent MSMR training; fixed 1000 updates and one final target evaluation."""
import argparse
from dataclasses import asdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from MPUS_GA.protocols import FIXED_UDA_PROTOCOL
from MPUS_GA.scripts.run_six_direction_final_suite import _code_hashes
from .data import (ALL_SCALES, DATASET_LAYOUT, UnlabeledMultiScaleView,
                   collate_multiscale, prepare_sources, prepare_target)
from .masked_multiscale import MSMRConfig, MaskedMultiScaleModel

DIRECTION_DATASETS = {
    'A': ('seed_vii', 'seed_v'), 'B': ('seed_v', 'seed_vii'),
    'C': ('seed_iv', 'seed_v'), 'D': ('seed_v', 'seed_iv'),
    'E': ('seed_iv', 'seed_vii'), 'F': ('seed_vii', 'seed_iv'),
}
TARGET_TRIALS = {'seed_v': 45, 'seed_vii': 80, 'seed_iv': 72}
CLASS_NAMES = ('positive', 'neutral', 'negative')


def validate_temporal_artifacts(data_dir, domains):
    """Validate saved timestamps rather than infer alignment from trial counts."""
    files = []
    for domain in sorted(set(domains)):
        for scale in (1, 2, 4):
            paths = sorted((data_dir / domain / f'window_{scale}s').glob('*.npz'))
            if len(paths) != DATASET_LAYOUT[domain]['files']:
                raise ValueError(f'Missing {domain}/{scale}s feature artifacts')
            for path in paths:
                with np.load(path, allow_pickle=False) as archive:
                    if 'start_second' not in archive or not np.allclose(
                            archive['start_second'], archive['window_id'] * scale, atol=1e-6):
                        raise ValueError(f'Nonzero origin or overlapping/shifted windows: {path}')
                files.append((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns))
    return files


def validate_trial_lengths(dataset):
    for key in dataset.keys:
        counts = {scale: len(dataset.groups[scale][key]) for scale in ALL_SCALES}
        if counts[1.] // 2 != counts[2.] or counts[1.] // 4 != counts[4.]:
            raise ValueError(f'Unaligned temporal windows for trial {key}: {counts}')


def configure_runtime(device, seed, config):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., config.cuda_memory_budget_gib * 1024**3 / total), device)
        torch.cuda.reset_peak_memory_stats(device)


def loader(dataset, batch_size, *, training=False, source=False, seed=42, iterations=1000, alpha=.4):
    sampler = None
    generator = torch.Generator().manual_seed(seed)
    if training:
        if source:
            counts = torch.bincount(dataset.labels.long(), minlength=3)
            if (counts == 0).any():
                raise ValueError('Source dataset must contain all three classes')
            weights = counts.float().pow(-alpha)[dataset.labels.long()].double()
            sampler = WeightedRandomSampler(weights, batch_size * iterations, replacement=True, generator=generator)
        else:
            dataset = UnlabeledMultiScaleView(dataset)
            sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * iterations, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                      drop_last=training, collate_fn=collate_multiscale, generator=generator)


def inputs(batch, device):
    return ({key: value.to(device) for key, value in batch['x'].items()},
            {key: value.to(device) for key, value in batch['mask'].items()})


def train_step(model, optimizer, source, target, device, config):
    if any(key in target for key in ('y', 'target_y', 'labels', 'target_labels', 'target_label')):
        raise ValueError('Target reconstruction must not receive labels')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sx, sm = inputs(source, device)
    tx, tm = inputs(target, device)
    output = model(sx, sm)
    classification = F.cross_entropy(output['logits'], source['y'].to(device),
                                     label_smoothing=config.label_smoothing)
    if not torch.isfinite(classification):
        raise FloatingPointError('Non-finite source classification loss')
    classification.backward()
    del output
    source_reconstruction, source_info = model.reconstruct(sx, sm)
    if not torch.isfinite(source_reconstruction):
        raise FloatingPointError('Non-finite source reconstruction loss')
    (config.source_reconstruction_weight * source_reconstruction).backward()
    target_reconstruction, target_info = model.reconstruct(tx, tm)
    if not torch.isfinite(target_reconstruction):
        raise FloatingPointError('Non-finite target reconstruction loss')
    (config.target_reconstruction_weight * target_reconstruction).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    ce, src, tgt = (float(value.detach()) for value in (classification, source_reconstruction, target_reconstruction))
    return {'classification': ce, 'source_reconstruction': src, 'target_reconstruction': tgt,
            'total': ce + config.source_reconstruction_weight * src + config.target_reconstruction_weight * tgt,
            'source_mask': source_info, 'target_mask': target_info, 'gradient_norm': float(norm)}


def classification_metrics(labels, predictions):
    matrix = np.bincount(np.asarray(labels) * 3 + np.asarray(predictions), minlength=9).reshape(3, 3)
    support, predicted = matrix.sum(1), matrix.sum(0)
    recall = np.diag(matrix) / np.maximum(support, 1)
    precision = np.diag(matrix) / np.maximum(predicted, 1)
    f1 = 2 * np.diag(matrix) / np.maximum(support + predicted, 1)
    return {'accuracy': float(np.trace(matrix) / matrix.sum()),
            'balanced_accuracy': float(recall[support > 0].mean()), 'macro_f1': float(f1.mean()),
            'worst_class_recall': float(recall.min()), 'recall_gap': float(recall.max() - recall.min()),
            'confusion_matrix': matrix.tolist(), 'trials': int(matrix.sum()),
            'per_class_recall': dict(zip(CLASS_NAMES, recall.tolist())),
            'per_class_precision': dict(zip(CLASS_NAMES, precision.tolist())),
            'per_class_f1': dict(zip(CLASS_NAMES, f1.tolist())),
            'true_counts': dict(zip(CLASS_NAMES, support.tolist())),
            'prediction_counts': dict(zip(CLASS_NAMES, predicted.tolist()))}


@torch.no_grad()
def evaluate(model, evaluation_loader, device):
    model.eval()
    labels, predictions, records = [], [], []
    for batch in evaluation_loader:
        probability = model(*inputs(batch, device))['logits'].softmax(-1).cpu()
        chosen = probability.argmax(-1)
        labels.extend(batch['y'].tolist())
        predictions.extend(chosen.tolist())
        for i in range(len(chosen)):
            records.append({'trial_key': [int(batch[key][i]) for key in ('subject_id', 'session_id', 'trial_id')],
                            'label': int(batch['y'][i]), 'prediction': int(chosen[i]),
                            'probability': probability[i].tolist()})
    return {'fused': classification_metrics(labels, predictions)}, records


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def write_summary(directory):
    results = [json.loads(p.read_text()) for p in sorted(directory.glob('seed_*_subject_*.json'))]
    metrics = {'trial_accuracy': 'accuracy', 'trial_balanced_accuracy': 'balanced_accuracy',
               'trial_macro_f1': 'macro_f1', 'worst_class_recall': 'worst_class_recall', 'recall_gap': 'recall_gap'}
    rows = []
    for seed in [None, *sorted({r['random_seed'] for r in results})]:
        selected = [r['evaluation']['fused'] for r in results if seed is None or r['random_seed'] == seed]
        if not selected:
            continue
        rows = []
        for metric in [*metrics, *('recall_' + c for c in CLASS_NAMES)]:
            values = np.array([r[metrics[metric]] if metric in metrics else r['per_class_recall'][metric[7:]] for r in selected])
            rows.append({'metric': metric, 'mean': values.mean(), 'std': values.std(ddof=1) if len(values) > 1 else 0., 'folds': len(values)})
        name = 'summary.csv' if seed is None else f'summary_seed_{seed}.csv'
        temp = directory / (name + '.tmp')
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=('metric', 'mean', 'std', 'folds'))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(directory / name)


def make_scheduler(optimizer, iterations):
    def multiplier(step):
        if step < 50:
            return (step + 1) / 50
        return .5 * (1 + math.cos(math.pi * min(1., (step - 50) / max(iterations - 50, 1))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_fold(args, prepared, subject, seed, config, identity):
    source_name, target_name = DIRECTION_DATASETS[args.experiment]
    directory = args.result_root / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'seed_{seed}_subject_{subject:02d}.json'
    checkpoint_path = path.with_suffix('.pt')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity or saved['random_seed'] != seed or saved['target_subject'] != subject:
            raise ValueError('Existing fold has a different identity; use a new run name')
        if not checkpoint_path.exists():
            raise ValueError(f'Completed fold is missing its checkpoint: {checkpoint_path}')
        print(f'SKIP {path}', flush=True)
        write_summary(directory)
        return
    iterations = FIXED_UDA_PROTOCOL.training_iterations
    target = prepare_target(args.data_dir, target_name, subject, prepared, ALL_SCALES)
    validate_trial_lengths(target)
    FIXED_UDA_PROTOCOL.validate_fold(target_trials=len(target), expected_target_trials=TARGET_TRIALS[target_name],
                                    training_iterations=iterations)
    device = torch.device(args.device)
    configure_runtime(device, seed, config)
    source = prepared.datasets[0]
    source_loader = loader(source, config.source_batch_size, training=True, source=True, seed=seed,
                           iterations=iterations, alpha=config.source_balance_alpha)
    target_loader = loader(target, config.target_batch_size, training=True, seed=seed + 1001, iterations=iterations)
    model = MaskedMultiScaleModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = make_scheduler(optimizer, iterations)
    history = []
    activity = {'source_reconstruction_steps': 0, 'target_reconstruction_steps': 0}
    target_trials_seen = set()
    start = time.monotonic()
    progress = tqdm(zip(source_loader, target_loader), total=iterations,
                    desc=f'{args.experiment} | seed={seed} | target={subject:02d}')
    with path.with_suffix('.trace.jsonl').open('w', buffering=1) as trace:
        for iteration, (sb, tb) in enumerate(progress, 1):
            target_trials_seen.update(zip(tb['subject_id'].tolist(), tb['session_id'].tolist(), tb['trial_id'].tolist()))
            record = train_step(model, optimizer, sb, tb, device, config)
            scheduler.step()
            activity['source_reconstruction_steps'] += int(record['source_reconstruction'] > 0)
            activity['target_reconstruction_steps'] += int(record['target_reconstruction'] > 0)
            if iteration == 1 or iteration % 50 == 0:
                record.update(iteration=iteration, learning_rate=optimizer.param_groups[0]['lr'])
                history.append(record)
                trace.write(json.dumps(record) + '\n')
                tqdm.write('MSMR ' + json.dumps({'direction': args.experiment, 'seed': seed, 'subject': subject, **record}))
                progress.set_postfix(cls=f"{record['classification']:.3f}", src_mask=f"{record['source_reconstruction']:.3f}",
                                     tgt_mask=f"{record['target_reconstruction']:.3f}")
    # Target truth is first passed to the model-evaluation path after all updates.
    evaluation, predictions = evaluate(model, loader(target, config.target_batch_size), device)
    source_evaluation, _ = evaluate(model, loader(source, config.source_batch_size), device)
    checkpoint = {'method': 'msmr', 'config': asdict(config), 'identity': identity,
                  'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  'source_normalization': {str(k): [torch.from_numpy(v) for v in values] for k, values in prepared.stats.items()},
                  'random_seed': seed, 'target_subject': subject, 'iteration': iterations,
                  'class_names': CLASS_NAMES, 'scope': 'final_model_for_inference; interrupted_folds_restart_from_seed'}
    checkpoint_tmp = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(checkpoint, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    result = {'method': 'msmr', 'variant': 'msmr_v1_main', 'experiment': args.experiment,
              'random_seed': seed, 'target_subject': subject, 'config': asdict(config), 'identity': identity,
              'protocol': {**FIXED_UDA_PROTOCOL.as_dict(), 'selected_iteration': iterations,
                           'source_domains': [source_name], 'target_dataset': target_name,
                           'target_trials': len(target), 'target_labels_used_for_training': False,
                           'target_scope': 'current_subject_all_trials_only', 'normalization': 'source_only_per_scale_channel_band'},
              'evaluation': evaluation, 'predictions': predictions,
              'source_training_evaluation': source_evaluation,
              'source_training_evaluation_scope': 'seen source training set; not held-out validation',
              'training_trace': history, 'activity': {**activity, 'unique_target_trials_seen': len(target_trials_seen)},
              'model_parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_file': checkpoint_path.name, 'elapsed_seconds': time.monotonic() - start,
              'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None,
              'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == 'cuda' else None}
    write_json(path, result)
    write_summary(directory)
    print('FINAL ' + json.dumps({'direction': args.experiment, 'seed': seed, 'subject': subject, **evaluation['fused']}), flush=True)
    del model, optimizer, scheduler, checkpoint, target, source_loader, target_loader, progress, sb, tb
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=DIRECTION_DATASETS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--random-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--target-subjects', default='all')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error('Seeds must be distinct')
    source_name, target_name = DIRECTION_DATASETS[args.experiment]
    config = MSMRConfig()
    data_identity = validate_temporal_artifacts(args.data_dir, (source_name, target_name))
    code = _code_hashes(Path(__file__).resolve().parents[1])
    identity = hashlib.sha256(json.dumps({'config': asdict(config), 'data': data_identity, 'code': code,
                                          'direction': args.experiment, 'torch': torch.__version__}, sort_keys=True).encode()).hexdigest()
    subjects = list(range(1, DATASET_LAYOUT[target_name]['subjects'] + 1)) if args.target_subjects == 'all' else [int(s) for s in args.target_subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or min(subjects) < 1 or max(subjects) > DATASET_LAYOUT[target_name]['subjects']:
        parser.error('Invalid target subjects')
    print('Protocol: fixed 1000 iterations; source CE + 0.1 source mask + 0.1 target mask; final target evaluation only', flush=True)
    print('MSMR CONFIG ' + json.dumps(asdict(config)), flush=True)
    prepared = prepare_sources(args.data_dir, (source_name,), ALL_SCALES)
    validate_trial_lengths(prepared.datasets[0])
    for seed in args.random_seeds:
        for subject in subjects:
            run_fold(args, prepared, subject, seed, config, identity)
            # Leave one second between subject processes' work as well as directions.
            time.sleep(1.)


if __name__ == '__main__':
    main()
