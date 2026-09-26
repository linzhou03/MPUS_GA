"""Offline error-edge audit of the frozen CSU CBST run; never used in training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ..trial_temporal.pcdiag_observe import load_truth

TARGET = {'B': 'seed_vii', 'C': 'seed_v', 'E': 'seed_vii'}
PERIODS = {'early': (301, 450), 'middle': (451, 700), 'late': (701, 1000)}


def matrix():
    return [[0, 0, 0] for _ in range(3)]


def add_prediction(cm, true, predicted):
    for y, p in zip(true, predicted, strict=True):
        cm[y][p] += 1


def audit(baseline_root, data_dir):
    result = {'scope': 'Offline truth join of repeated pseudo-label refresh events; '
                       'target truth never used to train or set positive gate', 'directions': {}}
    for direction, dataset in TARGET.items():
        accepted = {period: matrix() for period in PERIODS}
        accepted['all'] = matrix()
        snapshots = {step: [] for step in (300, 500, 750, 1000)}
        negative_emotions = {step: Counter() for step in snapshots}
        files = sorted((baseline_root / direction).glob('seed_43_subject_[0-9][0-9].pseudo.pt'))
        for path in files:
            subject = int(path.name.split('_subject_')[1].split('.')[0])
            truth = load_truth(data_dir, dataset, subject)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            for row in saved['cbst_state']['rounds']:
                step = int(row['iteration'])
                ids = [tuple(map(int, values)) for values in row['ids'].tolist()]
                selected = row['accepted'].tolist()
                true = [truth[key][0] for key, accept in zip(ids, selected) if accept]
                predicted = [pred for pred, accept in zip(row['pseudo_label'].tolist(), selected) if accept]
                add_prediction(accepted['all'], true, predicted)
                for name, (first, last) in PERIODS.items():
                    if first <= step <= last:
                        add_prediction(accepted[name], true, predicted)
            for step in snapshots:
                row = saved['snapshots'][step]
                ids = [tuple(map(int, values)) for values in row['ids'].tolist()]
                true = [truth[key][0] for key in ids]
                predicted = row['fused'].argmax(-1).tolist()
                cm = matrix()
                add_prediction(cm, true, predicted)
                snapshots[step].append(cm)
                for key, y, p in zip(ids, true, predicted, strict=True):
                    if y == 2:
                        emotion = truth[key][1]
                        negative_emotions[step][f'{emotion}:total'] += 1
                        if p == 0: negative_emotions[step][f'{emotion}:to_positive'] += 1
        result['directions'][direction] = dict(folds=len(files), accepted_confusion=accepted,
            snapshot_confusions={str(step): rows for step, rows in snapshots.items()},
            negative_emotions={str(step): dict(counts) for step, counts in negative_emotions.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline_root, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output),
        'folds': {direction: row['folds'] for direction, row in result['directions'].items()}}),
        flush=True)


if __name__ == '__main__': main()
