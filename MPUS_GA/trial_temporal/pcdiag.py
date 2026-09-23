"""Passive R2 observations and explicitly supervised, isolated Oracle interventions.

Baseline capture never reads target labels, samples randomness, or runs a forward.
Full-target observations are performed later by pcdiag_observe in another process.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

NODES = (0, 250, 300, 500, 750, 1000)
STATE_OBJECTS = ('prototype_bank', 'source_prototype_memory', 'target_prior_estimator')
OUTPUTS = ('scale_logits', 'calibrated_scale_logits', 'logits', 'probability',
           'scale_embeddings', 'embedding')


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, torch.device):
        return device
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    return value


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def assert_replay_state(actual, expected, path='state'):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-6, rtol=1e-5, msg='Replay mismatch: ' + path)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError('Replay keys mismatch: ' + path)
        for k in actual:
            assert_replay_state(actual[k], expected[k], path + '.' + str(k))
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise RuntimeError('Replay length mismatch: ' + path)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_replay_state(a, b, path + '.' + str(i))
    elif actual != expected:
        raise RuntimeError('Replay value mismatch: ' + path)


def frozen_profile(direction):
    from . import train
    frozen = json.loads(Path(__file__).with_name('pcdiag_frozen.json').read_text())
    spec = dict(frozen['spec'])
    spec['scales'], spec['source_domains'] = tuple(spec['scales']), tuple(spec['source_domains'])
    transfer = train._FINAL_TRANSFER_DIRECTIONS[direction]
    spec.update(name=direction, transfer_direction=direction, ablation='original_r2_pcdiag',
                **transfer)
    # Transfer dictionaries include only identity, never model or loss settings.
    return train.ExperimentSpec(**spec), frozen['args']


def identities(batch):
    keys = torch.stack([batch[k] for k in ('subject_id', 'session_id', 'trial_id')], dim=1)
    if 'trial_key_by_scale' in batch:
        for value in batch['trial_key_by_scale'].values():
            if not torch.equal(value.cpu(), keys.cpu()):
                raise ValueError('Multiscale trial identities do not agree')
    return keys.cpu().clone()


def override_consumers(probability, valid, labels, weights):
    """Only the Oracle supplies these arguments; never reinterpret them as confidence."""
    n = len(probability)
    if valid is None or weights is None or any(v.shape != (n,) for v in (valid, labels, weights)):
        raise ValueError('Oracle labels/mask/effective weights must all be [batch]')
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError('Invalid Oracle effective weights')
    if (labels < 0).any() or (labels >= probability.shape[1]).any():
        raise ValueError('Invalid Oracle class')
    return labels.detach().to(probability.device), valid.detach().bool().to(probability.device), weights.detach().to(probability)


def oracle_policy(reference, truth, mode, iteration, budget=4):
    """Same reference evidence for every branch; a deterministic round-robin budget."""
    labels = reference['pseudo_label'].clone()
    accepted = reference['valid_mask'].clone()
    weights = reference['effective_weight'].clone()
    candidates = ~accepted & (labels == truth)
    queues = []
    ids = reference['target_ids'].tolist()
    for c in range(3):
        indices = (candidates & (labels == c)).nonzero().flatten().tolist()
        queues.append(sorted(indices, key=lambda i: (-float(reference['confidence'][i]), ids[i], i)))
    added = torch.zeros_like(accepted)
    # Replacement sampling may duplicate a trial; new budget counts unique trials.
    seen = set()
    while int(added.sum()) < budget:
        changed = False
        for offset in range(3):
            c = (iteration + offset) % 3
            while queues[c] and tuple(ids[queues[c][0]]) in seen:
                queues[c].pop(0)
            if queues[c] and int(added.sum()) < budget:
                index = queues[c].pop(0)
                added[index] = True
                seen.add(tuple(ids[index]))
                changed = True
        if not changed:
            break
    if mode in ('P', 'PC'):
        labels[accepted] = truth[accepted]
    if mode in ('C', 'PC'):
        accepted |= added
        weights[added] = .25
    else:
        added.zero_()
    return labels, accepted, weights, added


def inference_context(runtime, step, final_evidence=None):
    """Reproduce the original calibrated target context without updating any state."""
    from . import train
    args, spec = runtime['args'], runtime['spec']
    bank, memory, estimator = (runtime[k] for k in STATE_OBJECTS)
    ramp = train._adaptation_ramp(step, args.adaptation_warmup_iterations, args.adaptation_ramp_end)
    extra = {} if final_evidence is None else {
        'boundary_mean_probability': final_evidence.mean_probability,
        'boundary_hard_frequency': final_evidence.hard_frequency}
    bias = None
    if ramp > 0:
        bias = estimator.common_bias_adjustments(
            source_excess_strength=args.common_bias_strength * ramp,
            source_relative_tolerance=args.common_bias_relative_tolerance,
            boundary_strength=args.boundary_bias_strength * ramp,
            boundary_ratio_tolerance=args.boundary_bias_ratio_tolerance,
            maximum_adjustment=args.common_bias_max_adjustment, **extra)['combined']
    return cpu({'compute_domain': False, 'scale_class_reliability': bank.scale_class_reliability(),
                'multiview_source_anchor': bank.source_relation_weights().mean(0),
                'class_logit_adjustment': bias,
                'pyramid_gate_ramp': ramp if spec.use_pyramid_gate_warmup else 1.,
                'pyramid_bias_risk': (-bias / args.common_bias_max_adjustment).clamp(0, 1) if bias is not None else None,
                'source_prototype_memory': memory.memory,
                'source_prototype_initialized': memory.initialized})


class Audit:
    def __init__(self, directory, mode='baseline', baseline_directory=None, nodes=NODES, resume_step=300):
        self.directory = Path(directory)
        self.mode, self.nodes = mode, tuple(nodes)
        self.resume_step = resume_step
        self.baseline_directory = Path(baseline_directory) if baseline_directory else None
        self.start_iteration = 0 if mode == 'baseline' else resume_step
        self.pending, self.reference_cache = [], {}

    def bind(self, runtime):
        # Store only explicit runtime references: no target dataset/labels in baseline capture.
        self.runtime = {k: runtime[k] for k in ('args', 'spec', 'prepared', 'model', 'optimizer',
                                               'scheduler', 'device', 'seed', 'subject', *STATE_OBJECTS)}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = runtime['device']
        if self.start_iteration:
            self.target = runtime['target']  # Truth is accessible only in the Oracle branch.
            self.target_index = {tuple(k): i for i, k in enumerate(self.target.keys)}
            self.source_indices = [{tuple(k): i for i, k in enumerate(d.keys)} for d in runtime['prepared'].datasets]
            self.resume = torch.load(self.baseline_directory / f'step_{self.start_iteration:04d}.pt', map_location='cpu', weights_only=False)
            runtime['model'].load_state_dict(self.resume['model'])
            for name in STATE_OBJECTS:
                runtime[name].__dict__.update(move(self.resume['objects'][name], self.device))
            runtime['optimizer'].load_state_dict(self.resume['optimizer'])
            runtime['scheduler'].load_state_dict(self.resume['scheduler'])

    def ready(self):
        if self.start_iteration:
            # Restore after model, optimizer and DataLoader iterator construction.
            restore_rng(self.resume['rng'], self.device)
        else:
            self.snapshot(0)

    def snapshot(self, step):
        r = self.runtime
        args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(r['args']).items() if not k.startswith('_')}
        value = {'step': step, 'model': cpu(r['model'].state_dict()), 'args': args,
                 'spec': asdict(r['spec']), 'seed': r['seed'], 'subject': r['subject'],
                 'source_stats': cpu(r['prepared'].stats), 'source_domains': r['prepared'].domain_names,
                 'objects': {k: cpu(vars(r[k])) for k in STATE_OBJECTS},
                 'context': inference_context(r, step), 'teacher_present': False, 'teacher': None,
                 'mode': self.mode, 'observation': 'post_update_before_final_refresh',
                 'rng': rng_state(self.device)}
        selected = getattr(self, 'checkpoint_optimizer', False) or (r['spec'].name in ('B', 'C') and r['seed'] == 42 and r['subject'] <= 5)
        if step == self.start_iteration or (selected and step in (self.resume_step, self.nodes[-1])):
            value['optimizer'], value['scheduler'] = cpu(r['optimizer'].state_dict()), cpu(r['scheduler'].state_dict())
        save_pt(self.directory / f'step_{step:04d}.pt', value)

    def reference(self, iteration):
        end = min(((iteration - 1) // 100 + 1) * 100, self.nodes[-1])
        if self.reference_cache.get('end') != end:
            records = torch.load(self.baseline_directory / f'batches_{end:04d}.pt', weights_only=False)
            self.reference_cache = {'end': end, 'records': {r['step']: r for r in records}}
        return self.reference_cache['records'][iteration]

    def replay_batches(self, iteration):
        from .data import collate_multiscale
        reference = self.reference(iteration)
        datasets = self.runtime['prepared'].datasets
        source = [collate_multiscale([d._item(indices[tuple(k)], True) for k in ids.tolist()])
                  for d, indices, ids in zip(datasets, self.source_indices, reference['source_ids'], strict=True)]
        target = collate_multiscale([self.target._item(self.target_index[tuple(k)], False)
                                    for k in reference['target_ids'].tolist()])
        return source, target

    def capture(self, iteration, source_batches, target_batch, output, consensus, active, source_labels, threshold):
        if 'y' in target_batch:
            raise ValueError('Target labels may not enter the adaptation batch')
        reference = None
        current = cpu(asdict(consensus))
        original = consensus
        ids = identities(target_batch)
        weights = ((consensus.confidence - threshold) / max(1 - threshold, 1e-6)).clamp(0, 1).detach()
        intervention, added = {}, torch.zeros_like(consensus.valid_mask)
        if self.start_iteration:
            from .train import TargetScaleConsensus
            reference = self.reference(iteration)
            if not torch.equal(ids, reference['target_ids']):
                raise ValueError('Replay target identity mismatch')
            values = {k: reference[k].to(self.device) for k in asdict(consensus)}
            if self.mode == 'replay':
                for k, v in current.items():
                    self.compare_evidence(iteration, k, v, reference[k])
            consensus = TargetScaleConsensus(**values)
            weights = reference['effective_weight'].to(self.device)
            if self.mode != 'replay':
                truth = torch.stack([self.target.labels[self.target_index[tuple(k)]] for k in ids.tolist()]).cpu()
                labels, accepted, effective, added_cpu = self.policy(reference, truth, iteration)
                consensus = replace(consensus, pseudo_label=labels.to(self.device), valid_mask=accepted.to(self.device))
                weights, added = effective.to(self.device), added_cpu.to(self.device)
                intervention = {'pseudo_label_override': consensus.pseudo_label, 'effective_weight_override': weights}
        source_present = torch.stack([torch.stack([(y == c).any() for c in range(3)]) for y in source_labels])
        positive = weights > 0
        mass = torch.stack([(weights * consensus.valid_mask * (consensus.pseudo_label == c)).sum() for c in range(3)])
        memory_used = consensus.valid_mask & positive & (mass[consensus.pseudo_label] > 1e-6) & active
        prototype_used = memory_used & source_present.any(0)[consensus.pseudo_label]
        record = {'step': iteration, 'mode': 'train_batch', 'oracle': self.mode not in ('baseline', 'replay'),
                  'target_ids': ids, 'source_ids': [identities(b) for b in source_batches],
                  'window_counts': {k: v.sum(1).cpu() for k, v in target_batch['mask'].items()},
                  **cpu(asdict(consensus)), **{k: cpu(output[k]) for k in OUTPUTS},
                  'raw_scale_probability': cpu(output['scale_logits'].softmax(-1)),
                  'calibrated_scale_probability': cpu(output['calibrated_scale_logits'].softmax(-1)),
                  'fused_probability': cpu(output['probability']),
                  'accepted_rule': cpu(original.valid_mask),
                  'passes_confidence': cpu(original.confidence >= threshold),
                  'passes_votes': cpu(original.vote_count >= self.runtime['args'].consensus_minimum_votes),
                  'passes_jsd': cpu(original.js_divergence <= self.runtime['args'].consensus_jsd_threshold),
                  'adaptation_active': active, 'effective_weight': cpu(weights),
                  'domain_used': cpu(consensus.valid_mask & active), 'prototype_used': cpu(prototype_used),
                  'memory_used': cpu(memory_used), 'source_class_present': cpu(source_present),
                  'added': cpu(added), 'teacher_present': False, 'teacher': None}
        # Keep the actual consensus probability separate from the fused model probability.
        record['probability'] = cpu(consensus.probability)
        if reference is not None:
            record['baseline_accepted'] = reference['valid_mask']
            record['baseline_pseudo_label'] = reference['pseudo_label']
            record['baseline_confidence'] = reference['confidence']
            record['budget_shortfall'] = 4 - int(added.sum()) if self.mode in ('C', 'PC', 'C_truth', 'PC_truth') else 0
        self.pending.append(record)
        return consensus, intervention

    def policy(self, reference, truth, iteration):
        return oracle_policy(reference, truth, self.mode, iteration)

    def compare_evidence(self, iteration, key, actual, expected):
        # Retain PyTorch's numerical diagnostics instead of hiding them with msg=.
        try:
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        except AssertionError as exc:
            raise AssertionError(f'Replay evidence mismatch at {iteration}: {key}\n{exc}') from exc

    def after_step(self, iteration, record):
        self.pending[-1]['losses'] = {k: record[k] for k in ('total', 'classification', 'domain', 'prototype')}
        if self.mode == 'replay':
            ref = self.reference(iteration)['losses']
            for k, v in self.pending[-1]['losses'].items():
                if not np.isclose(v, ref[k], atol=2e-6, rtol=1e-5):
                    raise RuntimeError(f'Replay loss mismatch step {iteration} {k}: {v} vs {ref[k]}')
        if iteration % 100 == 0 or iteration == self.nodes[-1]:
            save_pt(self.directory / f'batches_{iteration:04d}.pt', self.pending)
            self.pending = []
        if iteration in self.nodes:
            self.snapshot(iteration)

    def finish(self, result, final_evidence):
        step = self.nodes[-1]
        save_pt(self.directory / 'formal_final_context.pt', inference_context(self.runtime, step, final_evidence))
        if self.mode == 'replay':
            reference = torch.load(self.baseline_directory / f'step_{step:04d}.pt', map_location='cpu', weights_only=False)
            actual = torch.load(self.directory / f'step_{step:04d}.pt', map_location='cpu', weights_only=False)
            for name in ('model', 'objects', 'rng', 'optimizer', 'scheduler'):
                assert_replay_state(actual[name], reference[name], name)
        result['method'] = 'original_r2_pcdiag' if self.mode == 'baseline' else 'oracle_pcdiag_' + self.mode
        result['oracle'] = self.mode not in ('baseline', 'replay')
        result['diagnostic'] = {'mode': self.mode, 'nodes': self.nodes, 'start_iteration': self.start_iteration,
                                'directory': str(self.directory), 'teacher_present': False,
                                'target_labels_used_for_training': result['oracle'],
                                'reference': str(self.baseline_directory) if self.baseline_directory else None}
        save_json(self.directory / 'complete.json', result['diagnostic'])


def code_identity(package):
    from ..scripts.run_six_direction_final_suite import _code_hashes
    result = _code_hashes(Path(package))
    result['trial_temporal/pcdiag_frozen.json'] = hashlib.sha256(Path(__file__).with_name('pcdiag_frozen.json').read_bytes()).hexdigest()
    return result
