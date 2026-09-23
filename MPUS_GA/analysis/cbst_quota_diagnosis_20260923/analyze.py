"""Read completed CBST fold artifacts and summarize quota starvation."""
import argparse
import json
from pathlib import Path

import torch


def analyze(root):
    result = {}
    for direction in 'ABCDEF':
        folds = sorted((root/direction).glob('seed_43_subject_*.offline.json'))
        empty_steps = steps = empty_rounds = rounds = accepted = 0
        class_accepted = [0, 0, 0]
        class_correct = [0, 0, 0]
        initial_support = []
        per_fold_empty = []
        for path in folds:
            offline = json.loads(path.read_text())
            pseudo = torch.load(path.with_name(path.name.replace('.offline.json','.pseudo.pt')),
                                map_location='cpu', weights_only=False)
            batches = pseudo['training_batches']
            assert len(batches) == 700
            missing = sum(not bool(batch['accepted'].any()) for batch in batches)
            empty_steps += missing
            steps += len(batches)
            per_fold_empty.append(missing / len(batches))
            refreshes = pseudo['cbst_state']['rounds']
            initial_support.append(refreshes[0]['raw_class_support'].tolist())
            rounds += len(refreshes)
            empty_rounds += sum(not bool(row['accepted'].any()) for row in refreshes)
            accepted += sum(int(row['accepted'].sum()) for row in refreshes)
            for row in offline['rounds']:
                for index, cls in enumerate(row['classes']):
                    class_accepted[index] += cls['accepted']
                    class_correct[index] += cls['correct_accepted']
        result[direction] = dict(
            folds=len(folds), empty_target_batch_fraction=empty_steps/steps if steps else None,
            min_fold_empty_fraction=min(per_fold_empty) if per_fold_empty else None,
            max_fold_empty_fraction=max(per_fold_empty) if per_fold_empty else None,
            zero_quota_refresh_fraction=empty_rounds/rounds if rounds else None,
            accepted_per_refresh=accepted/rounds if rounds else None,
            initial_raw_support_mean=[sum(row[i] for row in initial_support)/len(initial_support)
                                      for i in range(3)] if initial_support else None,
            pseudo_precision_by_label=[class_correct[i]/class_accepted[i] if class_accepted[i] else None
                                       for i in range(3)],
            accepted_round_observations_by_label=class_accepted,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(analyze(args.root), indent=2))


if __name__ == '__main__':
    main()
