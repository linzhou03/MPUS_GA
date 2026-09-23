"""Offline-only full-target observations, truth joins, rejected-set audit and plots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train
from .data import PreparedMultiSource, UnlabeledMultiScaleView, collate_multiscale, prepare_target
from .pcdiag import OUTPUTS, cpu, identities, move, save_json, save_pt

CLASSES = ('positive', 'neutral', 'negative')
BINS = (0., .5, .6, .7, .8, .9, 1.000001)


def ratio(n, d):
    return float(n / d) if d else None


def metric_rows(observation, truth):
    pred, accepted = observation['pseudo_label'].numpy(), observation['valid_mask'].numpy().astype(bool)
    conf, truth = observation['confidence'].numpy(), np.asarray(truth)
    fused = observation['fused_probability'].argmax(-1).numpy()
    scales = observation['calibrated_scale_probability'].argmax(-1).numpy()
    raw = observation['raw_scale_probability'].argmax(-1).numpy()
    active = observation.get('adaptation_active', observation['step'] > 300)
    rows = []
    for c, name in enumerate(CLASSES):
        predicted, actual = pred == c, truth == c
        selected = predicted & accepted
        correct = selected & actual
        weighted = correct & (conf > .6) & active
        rows.append({'class': name, 'predicted_count': int(predicted.sum()), 'true_count': int(actual.sum()),
                     'accepted_count': int(selected.sum()), 'correct_accepted': int(correct.sum()),
                     'wrong_accepted': int((selected & ~actual).sum()),
                     'A': ratio(selected.sum(), predicted.sum()), 'P': ratio(correct.sum(), selected.sum()),
                     'R': ratio(correct.sum(), actual.sum()), 'recall': ratio((actual & (fused == c)).sum(), actual.sum()),
                     'consensus_recall': ratio((actual & predicted).sum(), actual.sum()),
                     'scale_recall': [ratio((actual & (scales[:, s] == c)).sum(), actual.sum()) for s in range(3)],
                     'raw_scale_recall': [ratio((actual & (raw[:, s] == c)).sum(), actual.sum()) for s in range(3)],
                     'adaptation_active': active, 'eligible_correct_coverage': ratio(weighted.sum(), actual.sum()),
                     'eligible_effective_mass': float((((conf - .6) / .4).clip(0, 1) * selected * active).sum())})
    return rows


def rejected_rows(observation, truth, subclasses):
    pred = observation['pseudo_label'].numpy()
    confidence = observation['confidence'].numpy()
    rejected = ~observation['valid_mask'].numpy().astype(bool)
    truth, subclasses = np.asarray(truth), np.asarray(subclasses)
    result = []
    for grouping, categories in [('predicted_class', pred), ('true_class', truth)]:
        for c, name in enumerate(CLASSES):
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                mask = rejected & (categories == c) & (confidence >= lo) & (confidence < hi)
                n, correct = int(mask.sum()), int((mask & (pred == truth)).sum())
                reason_counts = {}
                for flag in ('confidence', 'votes', 'jsd'):
                    reason_counts[flag] = int((mask & ~observation['passes_' + flag].numpy()).sum())
                combinations = Counter()
                for i in np.flatnonzero(mask):
                    reason = '+'.join(k for k in ('confidence', 'votes', 'jsd') if not bool(observation['passes_' + k][i]))
                    combinations[reason] += 1
                result.append({'grouping': grouping, 'class': name, 'bucket': f'[{lo:g},{min(hi, 1):g}' + (']' if hi > 1 else ')'),
                               'count': n, 'correct': correct, 'precision': ratio(correct, n),
                               'true_class_fraction': ratio(n, (truth == c).sum()) if grouping == 'true_class' else None,
                               'subclasses': dict(Counter(subclasses[mask])), 'rejection_reasons': reason_counts,
                               'rejection_combinations': dict(combinations)})
    return result


def load_truth(data_dir, dataset, subject):
    from ..preprocessing import dataset_metadata as metadata
    mapping = getattr(metadata, 'EMOTION_TO_ORIGINAL_' + dataset.upper())
    names = {v: k for k, v in mapping.items()}
    result = {}
    for path in sorted((Path(data_dir) / dataset / 'window_1s').glob(f'subject_{subject:02d}_session_*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            if 'label_original' not in archive:
                raise ValueError(f'Original emotion labels unavailable: {path}')
            for s, session, trial, y, original in zip(*(archive[k] for k in
                    ('subject_id', 'session_id', 'trial_id', 'label_3class', 'label_original')), strict=True):
                key, value = (int(s), int(session), int(trial)), (int(y), names[int(original)])
                if key in result and result[key] != value:
                    raise ValueError('Conflicting trial truth metadata')
                result[key] = value
    return result


@torch.no_grad()
def observe_snapshot(checkpoint, target, device, formal_context=None):
    args = SimpleNamespace(**checkpoint['args'])
    raw_spec = dict(checkpoint['spec'])
    raw_spec['scales'], raw_spec['source_domains'] = tuple(raw_spec['scales']), tuple(raw_spec['source_domains'])
    spec = train.ExperimentSpec(**raw_spec)
    prepared = PreparedMultiSource(tuple(checkpoint['source_domains']), (), checkpoint['source_stats'])
    model = train.build_fold_model(args, spec, prepared, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    context = move(formal_context if formal_context is not None else checkpoint['context'], device)
    loader = DataLoader(UnlabeledMultiScaleView(target), batch_size=args.target_batch_size,
                        shuffle=False, collate_fn=collate_multiscale)
    collected = defaultdict(list)
    for batch in loader:
        x, mask = train._batch_to_device(batch, device)
        output = model(x, mask, **context)
        consensus = train.independent_scale_consensus(output['calibrated_scale_logits'], args.pseudo_confidence_threshold,
                                                       args.consensus_jsd_threshold, args.consensus_minimum_votes)
        for k in OUTPUTS:
            if k != 'probability':
                collected[k].append(cpu(output[k]))
        for k, v in asdict(consensus).items():
            collected[k].append(cpu(v))
        collected['target_ids'].append(identities(batch))
        collected['fused_probability'].append(cpu(output['probability']))
        collected['raw_scale_probability'].append(cpu(output['scale_logits'].softmax(-1)))
        collected['calibrated_scale_probability'].append(cpu(output['calibrated_scale_logits'].softmax(-1)))
    result = {k: torch.cat(v) for k, v in collected.items()}
    ids = [tuple(x) for x in result['target_ids'].tolist()]
    if len(set(ids)) != len(target) or ids != list(target.keys):
        raise ValueError('Observer must cover each target trial exactly once in order')
    result.update(step=checkpoint['step'], mode='full_target_eval', teacher_present=False, teacher=None,
                  adaptation_active=checkpoint['step'] > args.adaptation_warmup_iterations,
                  context_kind='formal_final_refreshed' if formal_context is not None else 'post_update',
                  passes_confidence=result['confidence'] >= args.pseudo_confidence_threshold,
                  passes_votes=result['vote_count'] >= args.consensus_minimum_votes,
                  passes_jsd=result['js_divergence'] <= args.consensus_jsd_threshold)
    return result


def audit_training(directory, truth_map):
    """Sampled exposures are separate from unique-trial snapshot coverage."""
    blocks = []
    for path in sorted(directory.glob('batches_*.pt')):
        records = torch.load(path, map_location='cpu', weights_only=False)
        byclass = [Counter() for _ in CLASSES]
        seen, accepted_ids = set(), [set() for _ in CLASSES]
        for r in records:
            for i, key in enumerate(r['target_ids'].tolist()):
                key = tuple(key)
                seen.add(key)
                y = truth_map[key][0]
                c = int(r['pseudo_label'][i])
                counts = byclass[c]
                counts['predicted_exposures'] += 1
                counts['accepted_exposures'] += int(r['valid_mask'][i])
                if r['valid_mask'][i]:
                    accepted_ids[c].add(key)
                    counts['correct_accepted_exposures'] += int(c == y)
                for flag in ('domain_used', 'prototype_used', 'memory_used'):
                    counts[flag + '_exposures'] += int(r[flag][i])
                counts['effective_mass'] += float(r['effective_weight'][i]) * int(r['valid_mask'][i]) * r['adaptation_active']
                counts['added_exposures'] += int(r['added'][i])
        blocks.append({'first_step': records[0]['step'], 'last_step': records[-1]['step'],
                       'unique_observed_trials': len(seen), 'counting_unit': 'sampled_exposures_not_unique_coverage',
                       'classes': [{'class': c, **dict(counts), 'unique_accepted_trials': len(ids)}
                                   for c, counts, ids in zip(CLASSES, byclass, accepted_ids, strict=True)]})
    return blocks


def observe_fold(result_path, data_dir, device):
    result = json.loads(result_path.read_text())
    directory = result_path.with_suffix('.audit')
    output = directory / 'offline'
    if (output / 'complete.json').exists():
        return json.loads((output / 'summary.json').read_text())
    if not (directory / 'complete.json').exists():
        raise RuntimeError('Cannot join truth before training completes')
    paths = sorted(directory.glob('step_*.pt'))
    last = torch.load(paths[-1], map_location='cpu', weights_only=False)
    target_name, subject = last['spec']['target_dataset'], last['subject']
    truth_map = load_truth(data_dir, target_name, subject)
    prepared = PreparedMultiSource(tuple(last['source_domains']), (), last['source_stats'])
    target = prepare_target(data_dir, target_name, subject, prepared, last['spec']['scales'])
    ordered = [truth_map[tuple(k)] for k in target.keys]
    truth, subclasses = np.array([v[0] for v in ordered]), [v[1] for v in ordered]
    save_json(output / 'truth_join.json', [{'subject_id': int(k[0]), 'session_id': int(k[1]), 'trial_id': int(k[2]),
                                          'coarse_label': int(y), 'emotion': e}
                                         for k, (y, e) in zip(target.keys, ordered, strict=True)])
    summary = {'direction': result['experiment'], 'seed': last['seed'], 'subject': subject,
               'dataset': target_name, 'oracle': result.get('oracle', False), 'nodes': [], 'stability': []}
    previous = None
    for path in paths:
        cp = torch.load(path, map_location='cpu', weights_only=False)
        observation = observe_snapshot(cp, target, device)
        node = {'step': cp['step'], 'kind': 'post_update', 'metrics': metric_rows(observation, truth),
                'rejected': rejected_rows(observation, truth, subclasses)}
        summary['nodes'].append(node)
        if previous is not None:
            stable = observation['pseudo_label'] == previous['pseudo_label']
            correct = observation['pseudo_label'] == torch.as_tensor(truth)
            rejected_both = ~observation['valid_mask'] & ~previous['valid_mask']
            summary['stability'].append({'from': previous['step'], 'to': cp['step'],
                 'classes': [{'class': c, 'stable_rejected': int((stable & rejected_both & (observation['pseudo_label'] == i)).sum()),
                              'stable_correct_rejected': int((stable & correct & rejected_both & (observation['pseudo_label'] == i)).sum())}
                             for i, c in enumerate(CLASSES)]})
        # Truth appears only under offline/, after the complete marker exists.
        save_pt(output / f'observation_{cp["step"]:04d}.pt', {**observation, 'true_label': truth, 'emotion': subclasses})
        previous = observation
        del cp
    formal = observe_snapshot(last, target, device, torch.load(directory / 'formal_final_context.pt', weights_only=False))
    summary['formal_final'] = metric_rows(formal, truth)
    accuracy = float((formal['fused_probability'].argmax(-1).numpy() == truth).mean())
    if not np.isclose(accuracy, result['evaluation']['fused']['accuracy'], atol=1e-8):
        raise RuntimeError('Offline formal-final accuracy does not reproduce original evaluation')
    save_pt(output / 'formal_final.pt', {**formal, 'true_label': truth, 'emotion': subclasses})
    summary['train_exposures'] = audit_training(directory, truth_map)
    save_json(output / 'summary.json', summary)
    save_json(output / 'complete.json', {'complete': True, 'target_truth_used_only_offline': True})
    return summary


def subject_summary(values):
    """Average repeated seeds within subject before subject bootstrap."""
    bysubject = defaultdict(list)
    for subject, value in values:
        if value is not None:
            bysubject[subject].append(value)
    means = np.array([np.mean(v) for v in bysubject.values()])
    if not len(means):
        return {'mean': None, 'ci95': None, 'subjects': 0}
    ci = None
    if len(means) >= 2:
        rng = np.random.default_rng(20260914)
        boot = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {'mean': float(means.mean()), 'ci95': ci, 'subjects': len(means)}


def aggregate(root):
    root = Path(root)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob('*/seed_*.audit/offline/summary.json'))]
    curves, buckets = defaultdict(list), defaultdict(list)
    for r in reports:
        for node in r['nodes']:
            for m in node['metrics']:
                for metric in ('A', 'P', 'R', 'recall', 'wrong_accepted', 'accepted_count', 'eligible_correct_coverage'):
                    curves[(r['direction'], node['step'], m['class'], metric)].append((r['subject'], m[metric]))
            for bucket in node['rejected']:
                key = (r['direction'], node['step'], bucket['grouping'], bucket['class'], bucket['bucket'])
                buckets[key].append((r['subject'], r['seed'], bucket))
    output = root / 'diagnostics'
    rows = [{'direction': d, 'step': s, 'class': c, 'metric': m, **subject_summary(values)}
            for (d, s, c, m), values in sorted(curves.items())]
    rejected = []
    for (d, step, grouping, c, bucket), observations in sorted(buckets.items()):
        subclass, reasons, combos = Counter(), Counter(), Counter()
        for _, _, row in observations:
            subclass.update(row['subclasses']); reasons.update(row['rejection_reasons']); combos.update(row['rejection_combinations'])
        rejected.append({'direction': d, 'step': step, 'grouping': grouping, 'class': c, 'bucket': bucket,
                         'observations_across_seeds': sum(v['count'] for _, _, v in observations),
                         'correct_observations_across_seeds': sum(v['correct'] for _, _, v in observations),
                         'support_subjects': len({s for s, _, v in observations if v['count']}),
                         'precision_subject_mean': subject_summary([(s, v['precision']) for s, _, v in observations]),
                         'candidate_count_subject_mean': subject_summary([(s, v['count']) for s, _, v in observations]),
                         'true_class_fraction_subject_mean': subject_summary([(s, v['true_class_fraction']) for s, _, v in observations]),
                         'subclass_observations': dict(subclass), 'rejection_reasons': dict(reasons), 'rejection_combinations': dict(combos)})
    save_json(output / 'curves.json', rows)
    save_json(output / 'rejected_audit.json', rejected)
    save_json(output / 'stability.json', [{k: r[k] for k in ('direction', 'seed', 'subject', 'stability')} for r in reports])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for direction in sorted({r['direction'] for r in reports}):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharex=True)
        for ax, metric in zip(axes, ('A', 'P', 'R', 'recall')):
            for c in CLASSES:
                selected = [r for r in rows if r['direction'] == direction and r['class'] == c and r['metric'] == metric]
                line, = ax.plot([r['step'] for r in selected], [np.nan if r['mean'] is None else r['mean'] for r in selected], '.-', label=c)
                ax.fill_between([r['step'] for r in selected], [np.nan if r['ci95'] is None else r['ci95'][0] for r in selected],
                                [np.nan if r['ci95'] is None else r['ci95'][1] for r in selected], color=line.get_color(), alpha=.12)
            ax.axvline(300, linestyle=':', color='gray'); ax.set_title(metric); ax.set_ylim(-.02, 1.02); ax.set_xlabel('step')
        axes[0].legend(fontsize=8); fig.suptitle(direction + ' original R2: subject mean after seed averaging')
        fig.tight_layout(); fig.savefig(output / f'{direction}_precision_coverage.png', dpi=160); plt.close(fig)
    report = [f'# Original R2 diagnostics\n\nCompleted folds: {len(reports)}.\n',
              'Curves use unique target trials per checkpoint. Three seeds are averaged within each subject before the subject bootstrap. Undefined denominators are null.\n',
              'Steps 0/250/300 precede adaptation; their zero active coverage is warmup. Intermediate observations use eval mode and do not represent actual train-mode sampled exposure.\n',
              'Formal final predictions use the original complete-target refresh and are stored separately. Rejected candidate reliability is retrospective evidence, not a threshold tuning signal.\n',
              '|Direction|Subject/seed folds|Final Accuracy|Final macro-F1|\n|---|---:|---:|---:|']
    for d in sorted({r['direction'] for r in reports}):
        results = [json.loads(p.read_text()) for p in sorted((root / d).glob('seed_*_subject_*.json'))]
        report.append(f'|{d}|{len(results)}|{np.mean([r["evaluation"]["fused"]["accuracy"] for r in results]):.4f}|{np.mean([r["evaluation"]["fused"]["macro_f1"] for r in results]):.4f}|')
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    return len(reports)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-root', type=Path, required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--direction', choices=list('ABCDEF'))
    p.add_argument('--aggregate', action='store_true')
    args = p.parse_args()
    if args.aggregate:
        print('Aggregated folds:', aggregate(args.result_root), flush=True)
        return
    if args.direction is None or args.data_dir is None:
        p.error('Observation requires direction and data directory')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    for path in sorted((args.result_root / args.direction).glob('seed_*_subject_*.json')):
        observe_fold(path, args.data_dir, device)
        torch.cuda.empty_cache()
        print('OBSERVED ' + str(path), flush=True)


if __name__ == '__main__':
    main()
