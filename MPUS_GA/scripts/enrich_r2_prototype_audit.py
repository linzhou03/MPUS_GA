"""Post-prediction semantic audit of EMA prototypes, using target truth offline."""
import argparse
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from diagnose_r2_prototypes import flips, metrics


def matrix(proto, initialized, z, categories, keys):
    centres, support = [], []
    for key in keys:
        mask = torch.tensor([x == key for x in categories], dtype=torch.bool)
        support.append(int(mask.sum()))
        centres.append(F.normalize(z[mask].mean(0), dim=-1) if mask.any() else torch.zeros_like(z[0]))
    centres = torch.stack(centres, 1)
    cosine = torch.einsum('scd,sjd->scj', F.normalize(proto, dim=-1), centres)
    out = []
    for c in range(proto.shape[1]):
        valid = initialized[:, c]
        out.append(cosine[valid, c].mean(0).tolist() if valid.any() else [None]*len(keys))
    return dict(columns=keys, supports=support, mean_cosine_by_named_prototype=out,
                scope='Cosines to TRUE target class/subclass centroids in the CURRENT frozen representation; diagnostic only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    a = p.parse_args()
    from MPUS_GA.trial_temporal.pcdiag_observe import load_truth
    torch.set_num_threads(2)
    total = 0
    for path in sorted(a.root.glob('*/subject_*/step_*.json')):
        if not re.fullmatch(r'step_\d{4}\.json', path.name):
            continue
        report = json.loads(path.read_text())
        output = path.with_name(path.stem + '.semantics.json')
        if output.exists():
            continue
        cp = torch.load(report['checkpoint'], map_location='cpu', weights_only=False)
        pred = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        z = pred['predictions']['original']['scale_embeddings']
        ids = pred['predictions']['original']['ids'].tolist()
        truth = load_truth(a.data_dir, cp['spec']['target_dataset'], report['subject'])
        labels = [truth[tuple(key)][0] for key in ids]
        emotions = [truth[tuple(key)][1] for key in ids]
        bank = cp['objects']['prototype_bank']
        # Preserve non-initialized prototype entries as null, never as zero evidence.
        content = dict(direction=report['direction'], subject=report['subject'], step=report['step'],
            class_names=['positive', 'neutral', 'negative'], target_initialized=bank['target_initialized'].tolist(),
            target_ema_class=matrix(bank['target'], bank['target_initialized'], z, labels, [0, 1, 2]),
            target_ema_subclass=matrix(bank['target'], bank['target_initialized'], z, emotions, sorted(set(emotions))))
        y = torch.tensor(labels)
        consensus = pred['predictions']['original']['pseudo_label']
        accepted = pred['predictions']['original']['accepted']
        content['consensus_class_counts'] = []
        for c, name in enumerate(content['class_names']):
            selected = accepted & (consensus == c)
            content['consensus_class_counts'].append(dict(class_name=name,
                predicted=int((consensus == c).sum()), accepted=int(selected.sum()),
                correct_accepted=int((selected & (y == c)).sum()), true_count=int((y == c).sum())))
        content['geometry_vs_consensus'] = {}
        for name, geometric in pred['geometric'].items():
            prediction = geometric['prediction']
            comparisons = dict(all=flips(prediction, consensus, y))
            for partition, mask in [('accepted', accepted), ('rejected', ~accepted)]:
                comparisons[partition] = dict(count=int(mask.sum()),
                    flips=flips(prediction[mask], consensus[mask], y[mask]))
                if mask.any():
                    comparisons[partition]['consensus_accuracy'] = metrics(consensus[mask], y[mask])['accuracy']
                    comparisons[partition]['geometry_accuracy'] = metrics(prediction[mask], y[mask])['accuracy']
            content['geometry_vs_consensus'][name] = comparisons
        output.write_text(json.dumps(content, indent=2) + '\n')
        total += 1
    print(json.dumps(dict(enriched=total, status='complete')))


if __name__ == '__main__':
    main()
