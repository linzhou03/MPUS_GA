"""Read-only checkpoint interventions for original R2; never trains a model.

The only target truth access is after all predictions have been produced.
Source support/holdout IDs and step-300 slot assignments are fixed in advance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import datetime
import fcntl
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

CLASSES = ('positive', 'neutral', 'negative')
CONDITIONS = ('original', 'memory_off', 'rebuilt_300', 'rebuilt_current', 'source_relation_only')
STEPS = (300, 500, 750, 1000)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_design(keys, labels, support_per_group=2):
    groups = defaultdict(list)
    for i, (key, label) in enumerate(zip(keys, labels)):
        groups[(int(key[0]), int(label))].append(i)
    support, heldout = [], []
    for (subject, label), indices in sorted(groups.items()):
        rng = np.random.default_rng(20260921 + subject * 101 + label * 10007)
        indices = np.asarray(indices)[rng.permutation(len(indices))].tolist()
        if len(indices) < support_per_group + 1:
            raise ValueError('Insufficient source trials for disjoint support/holdout')
        support.extend(indices[:support_per_group])
        heldout.append(indices[support_per_group])
    assert not set(support) & set(heldout)
    return support, heldout


def assign_slots(z, labels, memory, initialized):
    z = F.normalize(z.float(), dim=-1)
    memory = F.normalize(memory.float(), dim=-1)
    result = torch.empty(z.shape[:2], dtype=torch.long)
    for s in range(z.shape[1]):
        for c in range(memory.shape[1]):
            take = labels == c
            sim = z[take, s] @ memory[s, c].T
            sim[:, ~initialized[s, c]] = -torch.inf
            if not initialized[s, c].any():
                raise ValueError('Source class has no initialized prototypes')
            result[take, s] = sim.argmax(-1)
    return result


def rebuild(z, labels, assignment, fallback):
    """Identical membership and aggregation for old/current encoders.

    Slots with no support retain the SAME archived step-300 vector in both arms.
    This is an explicit fixed-source reconstruction, not an EMA history replay.
    """
    z = F.normalize(z.float(), dim=-1)
    out, counts = fallback.clone(), torch.zeros(fallback.shape[:3], dtype=torch.long)
    for s in range(out.shape[0]):
        for c in range(out.shape[1]):
            for k in range(out.shape[2]):
                take = (labels == c) & (assignment[:, s] == k)
                counts[s, c, k] = int(take.sum())
                if take.any():
                    out[s, c, k] = F.normalize(z[take, s].mean(0), dim=0)
    return out, counts


def metrics(pred, truth):
    pred, truth = np.asarray(pred, dtype=int), np.asarray(truth, dtype=int)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (truth, pred), 1)
    tp, nc, nt = np.diag(cm), cm.sum(0), cm.sum(1)
    rec = np.divide(tp, nt, out=np.zeros(3, dtype=float), where=nt > 0)
    f1 = np.divide(2 * tp, nc + nt, out=np.zeros(3, dtype=float), where=(nc + nt) > 0)
    return dict(accuracy=float(tp.sum() / len(truth)), balanced_accuracy=float(rec.mean()),
                macro_f1=float(f1.mean()), confusion_matrix=cm.tolist(),
                recall=dict(zip(CLASSES, rec.tolist())),
                precision={c: float(tp[i] / nc[i]) if nc[i] else None for i, c in enumerate(CLASSES)})


def flips(new, old, truth):
    new, old, truth = map(np.asarray, (new, old, truth))
    a, b = (old != truth) & (new == truth), (old == truth) & (new != truth)
    return dict(wrong_to_correct=int(a.sum()), correct_to_wrong=int(b.sum()),
                net_corrected=int(a.sum() - b.sum()), changed=int((old != new).sum()),
                by_true_class={c: dict(wrong_to_correct=int((a & (truth == i)).sum()),
                                      correct_to_wrong=int((b & (truth == i)).sum())) for i, c in enumerate(CLASSES)})


def geometry(z, memory, initialized, truth=None):
    """Same top-two cosine retrieval as R2, then equal scale aggregation.

    These are diagnostic geometric predictions, not the R2 classifier output.
    """
    z, m = F.normalize(z.float(), dim=-1), F.normalize(memory.float(), dim=-1)
    sim = torch.einsum('nsd,sckd->nsck', z, m).masked_fill(~initialized[None], -torch.inf)
    best = sim.topk(min(2, m.shape[2]), dim=-1).values
    valid = torch.isfinite(best)
    safe = best.masked_fill(~valid, -1e4)
    w = (safe / .25).softmax(-1) * valid
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
    scores = (best.masked_fill(~valid, 0) * w).sum(-1).mean(1)
    result = dict(scores=scores, prediction=scores.argmax(-1))
    if truth is not None:
        y = torch.as_tensor(truth, dtype=torch.long)
        other = scores.clone()
        other[torch.arange(len(y)), y] = -torch.inf
        margin = scores[torch.arange(len(y)), y] - other.max(-1).values
        result['true_class_margin'] = {c: float(margin[y == i].mean()) if (y == i).any() else None for i, c in enumerate(CLASSES)}
        result['metrics'] = metrics(result['prediction'], y)
    return result


def slot_audit(memory, initialized, assignments, labels, ids, emotions):
    rows = []
    normalized = F.normalize(memory, dim=-1)
    for s in range(memory.shape[0]):
        for c in range(memory.shape[1]):
            ready = normalized[s, c, initialized[s, c]]
            sim = ready @ ready.T
            pairs = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
            slots = []
            for k in range(memory.shape[2]):
                take = ((labels == c) & (assignments[:, s] == k)).nonzero().flatten().tolist()
                subjects = Counter(int(ids[i][0]) for i in take)
                subclasses = Counter(emotions[i] for i in take)
                slots.append(dict(slot=k, support=len(take), subject_counts=dict(subjects),
                                  subclass_counts=dict(subclasses),
                                  dominant_subject_share=max(subjects.values()) / len(take) if take else None))
            rows.append(dict(scale=s, class_name=CLASSES[c], pairwise_cosine=pairs.tolist(),
                             maximum_cosine=float(pairs.max()) if len(pairs) else None, slots=slots))
    return rows


def package_runtime():
    from MPUS_GA.trial_temporal import train
    from MPUS_GA.trial_temporal import data
    from MPUS_GA.trial_temporal.pcdiag import cpu, move, identities
    return train, data, cpu, move, identities


def build_model(cp, device):
    train, data, _, _, _ = package_runtime()
    args = SimpleNamespace(**cp['args'])
    spec = train.ExperimentSpec(**cp['spec'])
    prepared = data.PreparedMultiSource(tuple(cp['source_domains']), (), cp['source_stats'])
    model = train.build_fold_model(args, spec, prepared, device)
    model.load_state_dict(cp['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, args, spec, prepared


@torch.inference_mode()
def source_embeddings(model, dataset, indices, device):
    train, data, _, _, _ = package_runtime()
    loader = DataLoader(Subset(data.UnlabeledMultiScaleView(dataset), indices), batch_size=4,
                        shuffle=False, collate_fn=data.collate_multiscale)
    values = []
    for batch in loader:
        assert 'y' not in batch
        x, mask = train._batch_to_device(batch, device)
        pooled = [model.temporal[key].pool_sequence(model.encode_window_sequence(x[key], mask[key], key), mask[key])
                  for key in model.scale_keys]
        z = model.scale_output_norm(torch.stack(pooled, 1) + model.scale_embedding[None])
        values.append(z.cpu())
    return torch.cat(values)


@torch.inference_mode()
def infer_conditions(model, args, target, contexts, device):
    train, data, cpu, move, identities = package_runtime()
    collected = {name: defaultdict(list) for name in contexts}
    loader = DataLoader(data.UnlabeledMultiScaleView(target), batch_size=args.target_batch_size,
                        shuffle=False, collate_fn=data.collate_multiscale)
    for batch in loader:
        assert 'y' not in batch
        x, mask = train._batch_to_device(batch, device)
        # The scale encoders are upstream of every intervention. Cache only this
        # label-free prefix, per batch; never cache fusion or classifier outputs.
        original_encode = model.encode_window_sequence
        cached = {}
        def encode(a, b, key):
            if key not in cached:
                cached[key] = original_encode(a, b, key)
            return cached[key]
        model.encode_window_sequence = encode
        try:
            for name, context in contexts.items():
                out = model(x, mask, **move(context, device))
                con = train.independent_scale_consensus(out['calibrated_scale_logits'], args.pseudo_confidence_threshold,
                                                        args.consensus_jsd_threshold, args.consensus_minimum_votes)
                selected = dict(ids=identities(batch), fused_probability=out['probability'],
                                scale_logits=out['scale_logits'], calibrated_scale_logits=out['calibrated_scale_logits'],
                                scale_embeddings=out['scale_embeddings'],
                                pseudo_label=con.pseudo_label, accepted=con.valid_mask,
                                confidence=con.confidence, votes=con.vote_count, jsd=con.js_divergence)
                for key, value in selected.items():
                    collected[name][key].append(cpu(value))
        finally:
            model.encode_window_sequence = original_encode
    return {name: {key: torch.cat(value) for key, value in row.items()} for name, row in collected.items()}


def replay_check(actual, reference):
    mapping = dict(ids='target_ids', fused_probability='fused_probability', scale_embeddings='scale_embeddings',
                   pseudo_label='pseudo_label', accepted='valid_mask', confidence='confidence', votes='vote_count', jsd='js_divergence')
    differences = {}
    if not torch.equal(actual['fused_probability'].argmax(-1), reference['fused_probability'].argmax(-1)):
        raise RuntimeError('Archived replay fused class mismatch')
    for key, old in mapping.items():
        a, b = actual[key].cpu(), reference[old].cpu()
        differences[key] = float((a.float() - b.float()).abs().max())
        if a.dtype in (torch.bool, torch.long):
            if not torch.equal(a, b):
                raise RuntimeError('Archived replay discrete mismatch: ' + key)
        else:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    return differences


def source_only_relation(cp):
    bank = cp['objects']['prototype_bank']
    relation = bank['source_relation'].float()
    mix = bank['relation_uniform_mix']
    relation = (1 - mix) * relation + mix / relation.shape[1]
    relation = relation / relation.sum(1, keepdim=True)
    return (relation / relation.sum((0, 1), keepdim=True)).sum(0)


def target_update_composition(directory, data_dir, dataset, subject, steps):
    from MPUS_GA.trial_temporal.pcdiag_observe import load_truth
    truth = load_truth(data_dir, dataset, subject)
    records = []
    for p in sorted(directory.glob('batches_*.pt')):
        for r in torch.load(p, map_location='cpu', weights_only=False):
            if r['step'] <= 300:
                continue
            records.append(r)
    outputs = {}
    for step in steps:
        rows = [dict(class_name=c, exposure_count=0, correct_exposures=0, effective_mass=0.,
                     correct_effective_mass=0., true_class_counts=Counter(), subclass_counts=Counter()) for c in CLASSES]
        for r in records:
            if r['step'] > step:
                continue
            for i, key in enumerate(r['target_ids'].tolist()):
                if not bool(r['memory_used'][i]) or not bool(r['valid_mask'][i]) or float(r['effective_weight'][i]) <= 0:
                    continue
                y, emotion = truth[tuple(key)]
                c, w = int(r['pseudo_label'][i]), float(r['effective_weight'][i])
                row = rows[c]
                row['exposure_count'] += 1
                row['correct_exposures'] += int(y == c)
                row['effective_mass'] += w
                row['correct_effective_mass'] += w * int(y == c)
                row['true_class_counts'][CLASSES[y]] += 1
                row['subclass_counts'][emotion] += 1
        outputs[step] = rows
    return outputs


def run_fold(cli, direction, subject, device, source_cache):
    train, data, cpu, move, _ = package_runtime()
    from MPUS_GA.trial_temporal.pcdiag_observe import load_truth
    # Match the archived observer, not one global numerical policy. The legacy
    # A observer used cuDNN TF32; the later deterministic B/C/E observer did not.
    # A probe isolated this: restoring legacy TF32 reduced max probability error
    # from 1.13e-5 to 5.96e-8, without relaxing the replay tolerance.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = direction == 'A'
    torch.backends.cudnn.deterministic = direction != 'A'
    torch.use_deterministic_algorithms(direction != 'A')
    root = cli.baseline_root if direction == 'A' else cli.oracle_root / 'R2'
    result = root / direction / f'seed_42_subject_{subject:02d}.json'
    directory = result.with_suffix('.audit')
    outdir = cli.output / direction / f'subject_{subject:02d}'
    outdir.mkdir(parents=True, exist_ok=True)
    composition_file = outdir / 'target_update_composition.json'
    if (all((outdir / f'step_{step:04d}.json').exists() for step in cli.steps)
            and composition_file.exists()
            and all(str(step) in json.loads(composition_file.read_text())['nodes'] for step in cli.steps)):
        print(f'SKIP COMPLETE {direction}/{subject:02d}', flush=True)
        return
    start = torch.load(directory / 'step_0300.pt', map_location='cpu', weights_only=False)
    model, args, spec, prepared = build_model(start, device)
    source_name = start['source_domains'][0]
    if len(start['source_domains']) != 1:
        raise ValueError('This audit is fixed to the single-source six-direction protocol')
    if source_name not in source_cache:
        arrays = {float(s): data.load_scale_arrays(cli.data_dir, source_name, float(s)) for s in spec.scales}
        source_cache.clear()
        source_cache[source_name] = arrays
    source = data.MultiScaleTrialDataset(source_cache[source_name], start['source_stats'], domain_id=0)
    support, heldout = source_design(source.keys, source.labels.tolist())
    indices = support + heldout
    z300_all = source_embeddings(model, source, indices, device)
    z300 = z300_all[:len(support)]
    y = source.labels[support].cpu()
    yh = source.labels[heldout].cpu()
    memory = start['objects']['source_prototype_memory']['memory'].cpu()
    initialized = start['objects']['source_prototype_memory']['initialized'].cpu()
    assignment = assign_slots(z300, y, memory, initialized)
    rebuilt300, counts = rebuild(z300, y, assignment, memory)
    source_truth = {}
    for sub in sorted({source.keys[i][0] for i in support}):
        source_truth.update(load_truth(cli.data_dir, source_name, sub))
    source_ids = [source.keys[i] for i in support]
    source_emotions = [source_truth[key][1] for key in source_ids]
    save_json(outdir / 'source_design.json', dict(source_dataset=source_name, support_ids=source_ids,
              support_labels=y.tolist(), holdout_ids=[source.keys[i] for i in heldout],
              holdout_scope='Not used in prototype reconstruction; seen by original source encoder training',
              fixed_slot_assignment=assignment.tolist(), slot_support=counts.tolist(),
              empty_slot_policy='Keep archived step300 vector in BOTH reconstruction conditions',
              historical_member_ids_available=False,
              slot_audit=slot_audit(memory, initialized, assignment, y, source_ids, source_emotions)))
    target = data.prepare_target(cli.data_dir, spec.target_dataset, subject, prepared, spec.scales)
    for step in cli.steps:
        output_path = outdir / f'step_{step:04d}.json'
        if output_path.exists():
            continue
        begin = time.monotonic()
        cp_path = directory / f'step_{step:04d}.pt'
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        model.load_state_dict(cp['model'], strict=True)
        model.eval()
        model.requires_grad_(False)
        for scale in prepared.stats:
            for a, b in zip(prepared.stats[scale], cp['source_stats'][scale]):
                np.testing.assert_array_equal(a, b)
        if not torch.equal(cp['objects']['source_prototype_memory']['memory'], memory):
            raise RuntimeError('Original source memory was not frozen after step300')
        z_all = z300_all if step == 300 else source_embeddings(model, source, indices, device)
        current, current_counts = rebuild(z_all[:len(support)], y, assignment, memory)
        assert torch.equal(counts, current_counts)
        contexts = {name: cpu(cp['context']) for name in CONDITIONS}
        contexts['memory_off']['source_prototype_memory'] = None
        contexts['memory_off']['source_prototype_initialized'] = None
        contexts['rebuilt_300']['source_prototype_memory'] = rebuilt300
        contexts['rebuilt_current']['source_prototype_memory'] = current
        contexts['source_relation_only']['scale_class_reliability'] = source_only_relation(cp)
        if step == 1000:
            contexts['formal_original'] = torch.load(directory / 'formal_final_context.pt', map_location='cpu', weights_only=False)
        predictions = infer_conditions(model, args, target, contexts, device)
        baseline = predictions['original']
        old = torch.load(directory / 'offline' / f'observation_{step:04d}.pt', map_location='cpu', weights_only=False)
        checks = replay_check(baseline, old)
        if step == 1000:
            formal_old = torch.load(directory / 'offline' / 'formal_final.pt', map_location='cpu', weights_only=False)
            checks['formal_final'] = replay_check(predictions['formal_original'], formal_old)
        for name in CONDITIONS:
            for key in ('ids', 'scale_logits', 'calibrated_scale_logits', 'scale_embeddings', 'pseudo_label', 'accepted'):
                if not torch.equal(predictions[name][key], baseline[key]):
                    raise RuntimeError(f'Unexpected upstream change: {name}/{key}')
        # Target truth is joined ONLY after generating every counterfactual.
        truth_map = load_truth(cli.data_dir, spec.target_dataset, subject)
        truth = np.asarray([truth_map[tuple(key)][0] for key in baseline['ids'].tolist()])
        original_pred = baseline['fused_probability'].argmax(-1).numpy()
        condition_rows = {}
        for name in CONDITIONS:
            p = predictions[name]
            pred = p['fused_probability'].argmax(-1).numpy()
            condition_rows[name] = dict(fused=metrics(pred, truth), changes=flips(pred, original_pred, truth),
                                       maximum_probability_change=float((p['fused_probability'] - baseline['fused_probability']).abs().max()),
                                       pseudo_changes=int((p['pseudo_label'] != baseline['pseudo_label']).sum()),
                                       accepted_changes=int((p['accepted'] != baseline['accepted']).sum()))
        paired = flips(predictions['rebuilt_current']['fused_probability'].argmax(-1).numpy(),
                       predictions['rebuilt_300']['fused_probability'].argmax(-1).numpy(), truth)
        geometries, source_geometries = {}, {}
        raw_geometry = {}
        for name, mem in [('original', memory), ('rebuilt_300', rebuilt300), ('rebuilt_current', current)]:
            g = geometry(baseline['scale_embeddings'], mem, initialized, truth)
            raw_geometry[name] = dict(scores=g.pop('scores'), prediction=g.pop('prediction'))
            geometries[name] = g
            hg = geometry(z_all[len(support):], mem, initialized, yh)
            hg.pop('scores'); hg.pop('prediction')
            source_geometries[name] = hg
        active = counts > 0
        drift = 1 - (F.normalize(rebuilt300, dim=-1) * F.normalize(current, dim=-1)).sum(-1)
        feature_drift = 1 - (F.normalize(z300_all, dim=-1) * F.normalize(z_all, dim=-1)).sum(-1)
        summary = dict(direction=direction, subject=subject, seed=42, step=step,
                       cohort='original_pcdiag' if direction == 'A' else 'deterministic_oracle_R2_control',
                       checkpoint=str(cp_path), checkpoint_sha256=sha(cp_path), replay=checks,
                       condition_results=condition_rows, matched_refresh_changes=paired,
                       consensus=metrics(baseline['pseudo_label'], truth),
                       geometric_predictions=geometries, source_holdout_geometry=source_geometries,
                       prototype_drift=dict(mean=float(drift[active].mean()), maximum=float(drift[active].max()), by_slot=drift.tolist()),
                       source_feature_drift_mean=float(feature_drift.mean()),
                       real_target_prototype_cosine_to_source_references={name: (F.normalize(cp['objects']['prototype_bank']['target'], dim=-1) * F.normalize(mem.mean(2), dim=-1)).sum(-1).tolist() for name, mem in [('old', rebuilt300), ('current', current)]},
                       source_only_reliability=source_only_relation(cp).tolist(),
                       original_reliability=cp['context']['scale_class_reliability'].tolist(),
                       elapsed_seconds=time.monotonic()-begin,
                       scope='Frozen checkpoint sensitivity; not a retraining or causal performance claim')
        summary['numerics'] = dict(matmul_tf32=False, cudnn_tf32=direction == 'A',
                                   strict_determinism=direction != 'A', replay_atol=2e-6, replay_rtol=1e-5)
        if step == 1000:
            formal_metrics = metrics(predictions['formal_original']['fused_probability'].argmax(-1), truth)
            formal_result = json.loads(result.read_text())['evaluation']['fused']
            if formal_metrics['confusion_matrix'] != formal_result['confusion_matrix']:
                raise RuntimeError('Formal score replay mismatch')
            summary['formal_final_metrics'] = formal_metrics
        state_unchanged = all(torch.equal(v.detach().cpu(), cp['model'][k]) for k, v in model.state_dict().items())
        if not state_unchanged:
            raise RuntimeError('Inference mutated model state')
        summary['model_state_unchanged'] = True
        torch.save(dict(predictions=predictions, target_truth=truth, geometric=raw_geometry,
                        support_embeddings=z_all[:len(support)], holdout_embeddings=z_all[len(support):],
                        rebuilt_300=rebuilt300, rebuilt_current=current), outdir / f'step_{step:04d}.pt')
        save_json(output_path, summary)
        print(f'COMPLETE {direction}/{subject:02d} step={step} sec={summary["elapsed_seconds"]:.1f} '
              f'origF1={condition_rows["original"]["fused"]["macro_f1"]:.4f} '
              f'refresh_net={paired["net_corrected"]}', flush=True)
        del cp, predictions
        gc.collect()
    exposures = target_update_composition(directory, cli.data_dir, spec.target_dataset, subject, cli.steps)
    save_json(outdir / 'target_update_composition.json', dict(
        scope='Repeated positive-weight update exposures; not independent trials or exact EMA vector decomposition', nodes=exposures))
    del model, source, target
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--oracle-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--directions', nargs='+', default=list('ABCE'), choices=list('ABCE'))
    p.add_argument('--subjects', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--steps', nargs='+', type=int, default=list(STEPS), choices=STEPS)
    p.add_argument('--device', default='cuda:0')
    cli = p.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    with (cli.output / '.audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        device = torch.device(cli.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(.40, device)
        import MPUS_GA
        package = Path(MPUS_GA.__file__).parent
        manifest = dict(script_sha256=sha(__file__), code_package=str(package),
                        package_sha256={name:sha(package / 'trial_temporal' / name) for name in ('train.py', 'model.py', 'data.py', 'pcdiag.py')},
                        torch=torch.__version__, cuda=torch.version.cuda, steps=list(STEPS),
                        conditions=list(CONDITIONS), source_support_per_subject_class=2,
                        source_holdout_per_subject_class=1, target_truth_used_for_generation=False,
                        cohort_numerics={'A': 'legacy observer: cuDNN TF32, non-strict',
                                         'BCE': 'strict deterministic observer, TF32 disabled'},
                        training=False, optimizer_steps=0, source_slot_assignment='fixed at step300')
        mpath = cli.output / 'manifest.json'
        if mpath.exists() and json.loads(mpath.read_text()) != manifest:
            raise RuntimeError('Audit configuration/code changed: use a fresh output directory')
        save_json(mpath, manifest)
        cache = {}
        try:
            for direction in cli.directions:
                for subject in cli.subjects:
                    save_json(cli.output / 'status.json', dict(status='running', direction=direction, subject=subject,
                              started=datetime.datetime.now().astimezone().isoformat()))
                    print(f'START {direction}/{subject:02d}', flush=True)
                    run_fold(cli, direction, subject, device, cache)
        except Exception as exc:
            save_json(cli.output / 'status.json', dict(status='failed', direction=direction, subject=subject, error=repr(exc)))
            raise
        save_json(cli.output / 'status.json', dict(status='complete', directions=cli.directions, subjects=cli.subjects, steps=cli.steps))
        print('OFFLINE AUDIT COMPLETE; NO TRAINING PERFORMED', flush=True)


if __name__ == '__main__':
    main()
