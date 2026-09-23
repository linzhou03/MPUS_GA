"""Offline intervention exposure audit and paired subject-level causal contrasts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .oracle_study import FOCUS, PROTOCOL
from .pcdiag import save_json
from .pcdiag_observe import CLASSES, load_truth, ratio, subject_summary
from .train_pcdiag import fold_paths


def summarize_exposures(records, truth_map):
    counts = [Counter() for _ in CLASSES]
    correct_ids = [set() for _ in CLASSES]
    used_ids = [set() for _ in CLASSES]
    unique_truth = Counter(y for y, _ in truth_map.values())
    for r in records:
        if not r['adaptation_active']:
            continue
        for i, key in enumerate(r['target_ids'].tolist()):
            key = tuple(key)
            y, _ = truth_map[key]
            c = int(r['pseudo_label'][i])
            accepted, added = bool(r['valid_mask'][i]), bool(r['added'][i])
            weight = float(r['effective_weight'][i])
            counts[y]['true_exposures'] += 1
            counts[c]['predicted_exposures'] += 1
            if 'baseline_accepted' in r:
                bc, ba = int(r['baseline_pseudo_label'][i]), bool(r['baseline_accepted'][i])
                counts[bc]['reference_accepted'] += int(ba)
                counts[bc]['reference_correct_accepted'] += int(ba and bc == y)
                counts[bc]['originally_correct_rejected_candidates'] += int(not ba and bc == y)
                counts[y]['true_class_rejected_candidates'] += int(not ba)
            if accepted:
                counts[c]['accepted'] += 1
                counts[c]['correct_accepted'] += int(c == y)
                counts[c]['wrong_accepted'] += int(c != y)
                counts[c]['effective_mass'] += weight
                counts[c]['correct_effective_mass'] += weight * (c == y)
                if c == y:
                    correct_ids[c].add(key)
            counts[c]['added'] += int(added)
            counts[c]['correct_added'] += int(added and c == y)
            for consumer in ('domain', 'prototype', 'memory'):
                used = bool(r[consumer + '_used'][i])
                counts[c][consumer + '_used'] += int(used)
                counts[c][consumer + '_correct_used'] += int(used and c == y)
                if consumer == 'prototype' and used and c == y:
                    used_ids[c].add(key)
    required = ('true_exposures', 'predicted_exposures', 'accepted', 'correct_accepted', 'wrong_accepted',
                'effective_mass', 'correct_effective_mass', 'added', 'correct_added', 'reference_accepted',
                'reference_correct_accepted', 'originally_correct_rejected_candidates', 'true_class_rejected_candidates',
                'domain_used', 'domain_correct_used', 'prototype_used', 'prototype_correct_used', 'memory_used', 'memory_correct_used')
    result = []
    for c, name in enumerate(CLASSES):
        v = {k: counts[c][k] for k in required}
        result.append({'class': name, **v, 'unique_true_trials': unique_truth[c],
            'A': ratio(v['accepted'], v['predicted_exposures']),
            'P': ratio(v['correct_accepted'], v['accepted']),
            'R_exposure': ratio(v['correct_accepted'], v['true_exposures']),
            'unique_correct_accepted': len(correct_ids[c]),
            'unique_correct_coverage': ratio(len(correct_ids[c]), unique_truth[c]),
            'unique_prototype_correct_coverage': ratio(len(used_ids[c]), unique_truth[c]),
            'prototype_correct_exposure_coverage': ratio(v['prototype_correct_used'], v['true_exposures']),
            'weighted_precision': ratio(v['correct_effective_mass'], v['effective_mass'])})
    return result


def write_fold_exposures(result_path, data_dir):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    directory = result_path.with_suffix('.audit')
    start = result['diagnostic']['start_iteration']
    cp = torch.load(directory / f'step_{start:04d}.pt', map_location='cpu', weights_only=False)
    truth = load_truth(data_dir, cp['spec']['target_dataset'], cp['subject'])
    blocks, all_records = [], []
    digests = {k: hashlib.sha256() for k in ('added', 'valid_mask', 'effective_weight')}
    for path in sorted(directory.glob('batches_*.pt')):
        records = [r for r in torch.load(path, weights_only=False, map_location='cpu') if r['step'] > start]
        if not records:
            continue
        # Retain only compact fields required for statistics, never all embeddings.
        keys = ('step', 'adaptation_active', 'target_ids', 'pseudo_label', 'valid_mask', 'effective_weight',
                'added', 'domain_used', 'prototype_used', 'memory_used', 'baseline_accepted', 'baseline_pseudo_label')
        compact = [{k: r[k] for k in keys if k in r} for r in records]
        blocks.append({'first_step': records[0]['step'], 'last_step': records[-1]['step'],
                       'classes': summarize_exposures(compact, truth),
                       'mean_losses': {k: float(np.mean([r['losses'][k] for r in records])) for k in records[0]['losses']}})
        all_records.extend(compact)
        for r in compact:
            for key, digest in digests.items():
                digest.update(str(r['step']).encode())
                digest.update(r['target_ids'].numpy().tobytes())
                digest.update(r[key].numpy().tobytes())
    payload = {'direction': result['experiment'], 'subject': cp['subject'], 'seed': cp['seed'],
               'condition': result['diagnostic']['condition'], 'blocks': blocks,
               'adaptation': summarize_exposures(all_records, truth),
               'policy_digests': {k: d.hexdigest() for k, d in digests.items()},
               'fork': json.loads((directory / 'fork.json').read_text()),
               'counting': 'R_exposure uses sampled true-class exposures; unique coverage uses all true-class target trials.',
               'consumer_note': 'Original R2 domain CE is class balanced and unweighted; .25 is the added prototype/memory weight.'}
    save_json(directory / 'offline/intervention_exposures.json', payload)
    return payload


def report_study(root, require_complete=False):
    root = Path(root)
    rows = []
    for d in 'BCE':
        modes = ['R2', 'P', 'C', 'PC'] + (['C_truth', 'PC_truth'] if d == 'E' else [])
        for subject in range(1, 6):
            conditions = {}
            for mode in modes:
                result, audit = fold_paths(root / mode, d, 42, subject)
                exposure = audit / 'offline/intervention_exposures.json'
                if result.exists() and exposure.exists() and (audit / 'offline/complete.json').exists():
                    value = json.loads(result.read_text())
                    conditions[mode] = {'evaluation': value['evaluation']['fused'],
                                        'exposures': json.loads(exposure.read_text())}
            if require_complete and set(conditions) != set(modes):
                raise RuntimeError(f'Incomplete conditions: {d} S{subject}: {list(conditions)}')
            if not conditions:
                continue
            forks = {v['exposures']['fork']['sha256'] for v in conditions.values()}
            if len(forks) != 1:
                raise RuntimeError('Unmatched fork checkpoint')
            for left, right in [('C', 'PC'), ('C_truth', 'PC_truth')]:
                if left in conditions and right in conditions:
                    if conditions[left]['exposures']['policy_digests']['added'] != conditions[right]['exposures']['policy_digests']['added']:
                        raise RuntimeError(f'{left}/{right} additions differ')
            if 'P' in conditions and 'R2' in conditions:
                for key in ('valid_mask', 'effective_weight'):
                    if conditions['P']['exposures']['policy_digests'][key] != conditions['R2']['exposures']['policy_digests'][key]:
                        raise RuntimeError('P did not preserve original selection/weights')
            rows.append({'direction': d, 'subject': subject, 'seed': 42, 'conditions': conditions})
    comparisons = []
    for d in 'BCE':
        contrasts = [('P', 'R2'), ('C', 'R2'), ('PC', 'P'), ('PC', 'C')]
        if d == 'E':
            contrasts += [('C_truth', 'R2'), ('PC_truth', 'P'), ('PC_truth', 'PC')]
        for left, right in contrasts:
            pairs = [r for r in rows if r['direction'] == d and left in r['conditions'] and right in r['conditions']]
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                values = [(r['subject'], r['conditions'][left]['evaluation'][metric] - r['conditions'][right]['evaluation'][metric]) for r in pairs]
                comparisons.append({'direction': d, 'contrast': left + ' - ' + right, 'metric': metric, **subject_summary(values)})
            for c, name in enumerate(CLASSES):
                for metric in ('recall', 'P', 'R_exposure', 'unique_correct_coverage', 'prototype_correct_exposure_coverage',
                               'accepted', 'wrong_accepted', 'effective_mass', 'correct_effective_mass', 'added'):
                    values = []
                    for r in pairs:
                        def value(mode):
                            cond = r['conditions'][mode]
                            return cond['evaluation']['per_class_recall'][name] if metric == 'recall' else cond['exposures']['adaptation'][c][metric]
                        a, b = value(left), value(right)
                        values.append((r['subject'], None if a is None or b is None else a - b))
                    comparisons.append({'direction': d, 'class': name, 'focus': c == FOCUS[d], 'contrast': left + ' - ' + right,
                                        'metric': metric, **subject_summary(values)})
    completed = sum(len(r['conditions']) for r in rows)
    save_json(root / 'paired_results.json', {'protocol': PROTOCOL, 'completed_conditions': completed,
                                           'expected_conditions': 70, 'rows': rows, 'comparisons': comparisons})
    lines = ['# R2 配对 Oracle 机制诊断', '', f'已完成 {completed}/70 个条件；B/C/E 各 5 名被试，seed 42。',
             '历史 306 份 baseline 保留。本批以新的确定性 R2 作为配对对照，不能和历史分数混用。',
             'P/C/PC 使用目标真值，不属于正式 UDA 结果。单种子小样本结果只作为探索性机制证据。', '',
             'P 保留全局接收数量与权重，但修正标签会改变类别分配和正确覆盖，两个干预因素并非完全正交。',
             'P 无接收样本时 precision 记为 NA。R_exposure 为采样次数比率；unique coverage 为整个目标集合中被正确监督过的独立 trial 比例。',
             '先确认各分支确实改变了正确监督及消费者参与，再解读性能差。E 的普通 C 若无增量，只能说明该干预未实施到位。', '',
             '|方向|对比|指标|类别|被试数|平均配对差|被试 bootstrap 95% CI|', '|---|---|---|---|---:|---:|---|']
    for c in comparisons:
        if c.get('class') and not c['focus']:
            continue
        if c['metric'] not in ('macro_f1', 'recall', 'P', 'R_exposure', 'correct_effective_mass', 'added'):
            continue
        mean = 'NA' if c['mean'] is None else f'{c["mean"]:.6f}'
        lines.append(f'|{c["direction"]}|{c["contrast"]}|{c["metric"]}|{c.get("class", "overall")}|{c["subjects"]}|{mean}|{c["ci95"]}|')
    lines.extend(['', '完整的每被试结果、各类别、每 100 步消费者使用统计在 paired_results.json 及各 audit/offline/intervention_exposures.json。',
                  '只有干预确实增加可靠监督，并在配对性能上有一致改善时，才支持相应机制；不能由单次 F1 上升直接宣称普遍因果成立。'])
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return completed
