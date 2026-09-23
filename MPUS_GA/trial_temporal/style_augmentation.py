"""Training-only hierarchical target statistics and matched-subgroup variance.

Labels, per-scale validity, prototypes and matches come from R2 co-teaching.
Published statistics contain unique trials, never repeated sampler draws.
No source prototype memory is written here. No target truth is accepted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from .multiscale_coteaching import js_divergence


@dataclass(frozen=True)
class StyleConfig:
    enabled: bool = True
    max_level: int = 2  # 0 global; 1 class; 2 matched subgroup
    coverage: bool = True
    semantic: bool = True
    variance: bool = True
    zero_perturbation: bool = False
    momentum: float = .95  # Per published refresh, not per sampler draw.
    eps: float = 1e-5
    ratio_min: float = .5
    ratio_max: float = 2.
    delta_clip: float = .5
    global_strength: float = .05
    class_strength: float = .12
    subgroup_strength: float = .20
    max_strength: float = .25
    class_min_count: int = 3
    subgroup_min_count: int = 3
    count_reference: int = 32  # Soft confidence scale, not an activation threshold.
    variance_min_count: int = 3
    coverage_min: float = .5
    coverage_max: float = 2.
    loss_cls: float = .5
    loss_sem: float = .1
    loss_var: float = .01

    def validate(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Style parameters must be finite')
        if self.max_level not in (0, 1, 2) or not 0 <= self.momentum < 1:
            raise ValueError('Invalid style level/momentum')
        if min(self.class_min_count, self.subgroup_min_count, self.variance_min_count) < 2:
            raise ValueError('Statistics require at least two distinct trials')
        if self.count_reference < 1 or self.eps <= 0:
            raise ValueError('Invalid count reference/epsilon')
        if not 0 < self.ratio_min <= self.ratio_max or self.delta_clip <= 0:
            raise ValueError('Invalid transfer bounds')
        if not 0 <= min(self.global_strength, self.class_strength, self.subgroup_strength):
            raise ValueError('Style strengths must be nonnegative')
        if not 0 <= self.max_strength <= .25:
            raise ValueError('Style gate must be within [0,.25]')
        if not 0 < self.coverage_min <= 1 <= self.coverage_max:
            raise ValueError('Coverage bounds must enclose one')
        if min(self.loss_cls, self.loss_sem, self.loss_var) < 0:
            raise ValueError('Loss coefficients must be nonnegative')


STYLE_VARIANTS = ('full', 'baseline', 'global', 'class', 'no_coverage',
                  'no_semantic', 'no_variance', 'ce_control')


def style_config(variant='full', **overrides):
    if variant not in STYLE_VARIANTS:
        raise ValueError(f'Unknown R3 variant: {variant}')
    changes = {'baseline': {'enabled': False}, 'global': {'max_level': 0},
               'class': {'max_level': 1}, 'no_coverage': {'coverage': False},
               'no_semantic': {'semantic': False}, 'no_variance': {'variance': False},
               'ce_control': {'zero_perturbation': True}}
    result = replace(StyleConfig(**overrides), **changes.get(variant, {}))
    result.validate()
    return result


@torch.no_grad()
def coverage_weights(counts, config):
    """Inverse coverage, clipped then normalized (bounds are pre-normalization)."""
    counts = counts.detach().float()
    if not config.coverage or not counts.sum():
        return torch.ones_like(counts)
    q = (counts + config.eps) / (counts.sum() + len(counts) * config.eps)
    weights = ((1 / len(counts)) / (q + config.eps)).clamp(config.coverage_min, config.coverage_max)
    return weights / weights.mean()


@torch.no_grad()
def domain_assignments(bank, domain, features, labels, valid):
    """Assign to existing R2 slots using its cosine rule; no new clustering."""
    result = torch.full(features.shape[:2], -1, dtype=torch.long, device=features.device)
    for scale in range(bank.scales):
        for label in range(bank.classes):
            selected = valid[:, scale] & (labels == label) & torch.isfinite(features[:, scale]).all(-1)
            support = bank.support[domain, scale, label] > 0
            if not selected.any() or not support.any():
                continue
            query = F.normalize(features[selected, scale].detach(), dim=-1)
            scores = query @ bank.prototypes[domain, scale, label].T
            scores = scores.masked_fill(~support, -2)
            cosine, slot = scores.max(-1)
            result[selected, scale] = torch.where(cosine >= bank.config.assignment_threshold, slot, -1)
    return result


class TargetStyleBank(nn.Module):
    """EMA global/class moments, snapshot-local subgroup moments, unique support."""
    def __init__(self, scales, classes, slots, dimension, config):
        super().__init__()
        config.validate()
        self.config = config
        self.scales, self.classes, self.slots, self.dimension = scales, classes, slots, dimension
        for prefix in ('target', 'source'):
            for level, shape in (('global', (scales,)), ('class', (scales, classes)),
                                 ('subgroup', (scales, classes, slots))):
                for moment in ('mu', 'second'):
                    self.register_buffer(f'{prefix}_{level}_{moment}', torch.zeros(*shape, dimension))
                self.register_buffer(f'{prefix}_{level}_count', torch.zeros(shape, dtype=torch.long))
        self.register_buffer('class_confidence', torch.zeros(scales, classes))
        self.register_buffer('accepted_counts', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('accepted_confidence', torch.zeros(classes))
        self.register_buffer('pair_reliability', torch.zeros(scales, classes, slots))
        self.register_buffer('target_slot', torch.full((scales, classes, slots), -1, dtype=torch.long))
        self.register_buffer('snapshot_iteration', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _write(self, prefix, level, index, vectors, ema=False):
        mean = getattr(self, f'{prefix}_{level}_mu')
        second = getattr(self, f'{prefix}_{level}_second')
        count = getattr(self, f'{prefix}_{level}_count')
        old_count = int(count[index])
        vectors = vectors[torch.isfinite(vectors).all(-1)]
        count[index] = len(vectors)
        if len(vectors) < 2:
            mean[index].zero_(); second[index].zero_()
            return
        mu, m2 = vectors.mean(0), vectors.square().mean(0)
        if ema and old_count >= 2:
            mean[index].lerp_(mu, 1 - self.config.momentum)
            second[index].lerp_(m2, 1 - self.config.momentum)
        else:
            mean[index].copy_(mu); second[index].copy_(m2)

    def moments(self, prefix, level, index):
        mu = getattr(self, f'{prefix}_{level}_mu')[index].detach()
        second = getattr(self, f'{prefix}_{level}_second')[index].detach()
        return mu, (second - mu.square()).clamp_min(0).add(self.config.eps).sqrt()

    @torch.no_grad()
    def refresh(self, source, target_all, target_selected, alignment, iteration):
        """Rows: key -> (raw teacher feature, coarse label, confidence, scale validity)."""
        reference = self.target_global_mu
        self.pair_reliability.zero_(); self.target_slot.fill_(-1)
        self.accepted_counts.zero_(); self.accepted_confidence.zero_(); self.class_confidence.zero_()
        snapshots = []
        for domain, (prefix, rows) in enumerate((('source', source), ('target', target_selected))):
            keys = sorted(rows)
            features = torch.stack([rows[k][0] for k in keys]).to(reference) if keys else reference.new_empty(0, self.scales, self.dimension)
            labels = torch.tensor([rows[k][1] for k in keys], device=reference.device, dtype=torch.long)
            confidence = torch.tensor([rows[k][2] for k in keys], device=reference.device)
            valid = torch.stack([rows[k][3] for k in keys]).to(reference.device) if keys else torch.empty(0, self.scales, dtype=torch.bool, device=reference.device)
            valid = valid & torch.isfinite(features).all(-1)
            slots = domain_assignments(alignment.bank, domain, features, labels, valid)
            snapshots.append({key: (features[i].detach().cpu(), int(labels[i]), valid[i].cpu(), slots[i].cpu()) for i, key in enumerate(keys)})
            global_features = features
            if domain == 1:
                global_features = torch.stack(list(target_all.values())).to(reference) if target_all else features
            for scale in range(self.scales):
                self._write(prefix, 'global', scale, global_features[:, scale], ema=(domain == 1))
                for label in range(self.classes):
                    selected = (labels == label) & valid[:, scale]
                    self._write(prefix, 'class', (scale, label), features[selected, scale], ema=(domain == 1))
                    if domain == 1 and selected.any():
                        self.class_confidence[scale, label] = confidence[selected].mean()
                    for slot in range(self.slots):
                        members = selected & (slots[:, scale] == slot)
                        # Slot IDs are rebuilt at every R2 refresh; never EMA across IDs.
                        self._write(prefix, 'subgroup', (scale, label, slot), features[members, scale])
            if domain == 1:
                for label in range(self.classes):
                    selected = (labels == label) & valid.any(-1)
                    self.accepted_counts[label] = selected.sum()
                    if selected.any():
                        self.accepted_confidence[label] = confidence[selected].mean()
        for scale in range(self.scales):
            for label in range(self.classes):
                for left, right in alignment.bank.matches[scale, label].nonzero().tolist():
                    if min(int(self.source_subgroup_count[scale, label, left]),
                           int(self.target_subgroup_count[scale, label, right])) < self.config.subgroup_min_count:
                        continue
                    reliability = (alignment.readiness[scale, label]
                                   * alignment.bank.quality[0, scale, label, left]
                                   * alignment.bank.quality[1, scale, label, right]
                                   * alignment.bank.similarity[scale, label, left, right].clamp(0, 1))
                    self.pair_reliability[scale, label, left] = reliability
                    self.target_slot[scale, label, left] = right
        self.snapshot_iteration.fill_(iteration)
        return snapshots


class ReliabilityGatedStyleAugmentor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def transfer(self, z, src_mu, src_std, tgt_mu, tgt_std, gate):
        cfg = self.config
        ratio = (tgt_std.detach() / (src_std.detach() + cfg.eps)).clamp(cfg.ratio_min, cfg.ratio_max)
        delta = (tgt_mu.detach() + ratio * (z.detach() - src_mu.detach()) - z.detach()).clamp(-cfg.delta_clip, cfg.delta_clip)
        finite = torch.isfinite(delta).all(-1, keepdim=True)
        delta = torch.where(finite, delta, torch.zeros_like(delta))
        gate = torch.as_tensor(gate, device=z.device, dtype=z.dtype).clamp(0, cfg.max_strength)
        return z + gate * delta.detach()

    def forward(self, z, labels, source_slots, bank, ramp, confidence_floor):
        cfg = self.config
        styled = z.clone()
        modes = torch.full(z.shape[:2], -1, dtype=torch.long, device=z.device)
        gates = z.new_zeros(z.shape[:2])
        for scale in range(bank.scales):
            for label in range(bank.classes):
                selected = (labels == label).nonzero().flatten()
                for index in selected.tolist():
                    mode, reliability = 0, 1.
                    src_index = tgt_index = scale
                    level = 'global'
                    slot = int(source_slots[index, scale])
                    if min(int(bank.source_global_count[scale]), int(bank.target_global_count[scale])) < 2:
                        continue
                    class_conf = ((float(bank.class_confidence[scale, label]) - confidence_floor) / max(1-confidence_floor, cfg.eps))
                    class_r = min(float(bank.target_class_count[scale, label]) / cfg.count_reference, 1) * max(min(class_conf, 1), 0)
                    if cfg.max_level >= 1 and min(int(bank.source_class_count[scale, label]), int(bank.target_class_count[scale, label])) >= cfg.class_min_count and class_r > 0:
                        mode, level, reliability = 1, 'class', class_r
                        src_index = tgt_index = (scale, label)
                    if cfg.max_level >= 2 and slot >= 0 and bank.pair_reliability[scale, label, slot] > 0:
                        mode, level = 2, 'subgroup'
                        reliability = float(bank.pair_reliability[scale, label, slot])
                        src_index = (scale, label, slot)
                        tgt_index = (scale, label, int(bank.target_slot[scale, label, slot]))
                    strength = (cfg.global_strength, cfg.class_strength, cfg.subgroup_strength)[mode]
                    gate = min(strength * ramp * reliability, cfg.max_strength)
                    if cfg.zero_perturbation:
                        gate = 0.
                    styled[index, scale] = self.transfer(z[index, scale], *bank.moments('source', level, src_index),
                                                          *bank.moments('target', level, tgt_index), gate)
                    modes[index, scale], gates[index, scale] = mode, gate
        return styled, modes, gates


def matched_variance_loss(bank, snapshots, current, config):
    """Unique-trial moments with current student rows replacing cached teacher rows.

Historical rows are detached. A term requires current student participation on
both sides and enough unique support. Gradients therefore cannot vanish merely
because all cached teacher statistics were detached.
"""
    zero = sum(item['features'].sum() * 0 for item in current)
    terms, pair_count, active_rows = [], 0, 0
    lookup = [{key: i for i, key in enumerate(item['keys'])} for item in current]
    for scale, label, left in (bank.pair_reliability > 0).nonzero().tolist():
        right = int(bank.target_slot[scale, label, left])
        moments, participation = [], []
        for domain, slot in enumerate((left, right)):
            vectors, replacements, replacement_indices = [], [], []
            item = current[domain]
            for key, (feature, coarse, valid, slots) in snapshots[domain].items():
                if coarse != label or not bool(valid[scale]) or int(slots[scale]) != slot:
                    continue
                index = lookup[domain].get(key)
                if index is None:
                    vectors.append(feature[scale])
                elif (int(item['labels'][index]) == label and bool(item['valid'][index, scale])
                      and int(item['slots'][index, scale]) == slot
                      and bool(torch.isfinite(item['features'][index, scale]).all())):
                    replacement_indices.append(len(vectors))
                    replacements.append(item['features'][index, scale])
                    vectors.append(feature[scale])
            if len(vectors) < config.variance_min_count or not replacements:
                break
            # One host-to-device transfer per group, not one transfer per cached row.
            values = torch.stack(vectors).to(item['features']).detach()
            positions = torch.tensor(replacement_indices, device=values.device)
            values = values.index_copy(0, positions, torch.stack(replacements))
            moments.append(values.var(0, unbiased=False))
            participation.append(len(replacements))
        if len(moments) == 2:
            terms.append((moments[0] - moments[1]).square().mean() * bank.pair_reliability[scale, label, left].detach())
            pair_count += 1; active_rows += sum(participation)
    return (torch.stack(terms).mean() if terms else zero), pair_count, active_rows


class R3StyleController:
    def __init__(self, student, alignment, config):
        config.validate()
        self.config = config
        self.bank = TargetStyleBank(len(student.scales), student.num_classes, alignment.bank.config.k,
                                    student.classifier.in_features, config).to(next(student.parameters()).device)
        self.augmentor = ReliabilityGatedStyleAugmentor(config)
        self.source_raw = {}
        self.snapshots = [{}, {}]
        self.cumulative = {'style_samples': 0, 'variance_pairs': 0, 'variance_student_rows': 0}

    def loss(self, student, source_outputs, target_output, alignment, iteration):
        z = torch.cat([output['scale_embeddings'] for output in source_outputs])
        zero = z.sum() * 0
        cfg, evidence = self.config, alignment.training_evidence
        ramp = alignment.config.ramp(iteration)
        record = {'enabled': cfg.enabled, 'ramp': ramp, 'snapshot_iteration': int(self.bank.snapshot_iteration),
                  'style_cls': 0., 'semantic': 0., 'variance': 0., 'displacement': 0., 'loss': 0.,
                  'variance_pairs': 0, 'variance_student_rows': 0, 'active_style_samples': 0,
                  'accepted_count_per_class': self.bank.accepted_counts.tolist(),
                  'accepted_ratio_per_class': (self.bank.accepted_counts.float()/self.bank.accepted_counts.sum().clamp_min(1)).tolist(),
                  'accepted_confidence_per_class': self.bank.accepted_confidence.tolist(),
                  'coverage_weights': coverage_weights(self.bank.accepted_counts, cfg).tolist(),
                  'reliable_subgroup_pairs': int((self.bank.pair_reliability > 0).sum()),
                  'style_mode_ratio': [0., 0., 0.], 'style_gate_per_scale': [0.] * self.bank.scales,
                  'style_gate_per_class': [0.] * self.bank.classes, 'fallback_count': 0}
        if not cfg.enabled or ramp <= 0 or evidence is None:
            return zero, record
        labels = evidence['source_labels']
        source_valid = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        source_slots = domain_assignments(alignment.bank, 0, evidence['source_features'], labels, source_valid)
        styled, modes, gates = self.augmentor(z, labels, source_slots, self.bank, ramp, alignment.config.confidence)
        active = modes >= 0
        logits = student.classifier(styled)
        original = torch.cat([output['scale_logits'] for output in source_outputs]).detach().softmax(-1)
        weights = coverage_weights(self.bank.accepted_counts, cfg).to(z)
        ce = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(z.shape[:2]).reshape(-1), reduction='none').reshape(z.shape[:2])
        cls = (ce * weights[labels, None])[active].mean() if active.any() else zero
        sem = js_divergence(original, logits.softmax(-1))[active].mean() if cfg.semantic and active.any() else zero
        displacement = (styled.detach() - z.detach()).square().mean()
        target_slots = domain_assignments(alignment.bank, 1, evidence['target_features'], evidence['target_labels'], evidence['target_valid'])
        current = [{'keys': evidence['source_keys'], 'features': z, 'labels': labels, 'valid': source_valid, 'slots': source_slots},
                   {'keys': evidence['target_keys'], 'features': target_output['scale_embeddings'], 'labels': evidence['target_labels'],
                    'valid': evidence['target_valid'], 'slots': target_slots}]
        var, pairs, rows = matched_variance_loss(self.bank, self.snapshots, current, cfg) if cfg.variance else (zero, 0, 0)
        loss = ramp * (cfg.loss_cls * cls + cfg.loss_sem * sem + cfg.loss_var * var)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite R3 loss')
        record.update({'style_cls': float(cls.detach()), 'semantic': float(sem.detach()), 'variance': float(var.detach()),
                       'displacement': float(displacement), 'loss': float(loss.detach()), 'variance_pairs': pairs,
                       'variance_student_rows': rows, 'active_style_samples': int(active.sum()),
                       'style_mode_ratio': [float((modes == level).float().mean()) for level in range(3)],
                       'style_gate_per_scale': gates.detach().mean(0).tolist(),
                       'style_gate_per_class': [float(gates[labels == c].mean()) if (labels == c).any() else 0. for c in range(self.bank.classes)],
                       'fallback_count': int(((modes >= 0) & (modes < cfg.max_level)).sum())})
        self.cumulative['style_samples'] += int(active.sum())
        self.cumulative['variance_pairs'] += pairs
        self.cumulative['variance_student_rows'] += rows
        return loss, record

    @torch.no_grad()
    def after_step(self, alignment, iteration):
        if not self.config.enabled:
            return
        evidence = alignment.training_evidence
        if evidence is None:
            return
        for key, feature, label in zip(evidence['source_keys'], evidence['source_features'].detach().cpu(),
                                       evidence['source_labels'].detach().cpu(), strict=True):
            self.source_raw[key] = (feature.clone(), int(label), iteration)
        for key in list(self.source_raw):
            if iteration - self.source_raw[key][2] > alignment.config.max_age:
                del self.source_raw[key]
        for label in range(self.bank.classes):
            keys = sorted((k for k in self.source_raw if self.source_raw[k][1] == label), key=lambda k: (self.source_raw[k][2], k))
            for key in keys[:-alignment.config.memory_per_class]:
                del self.source_raw[key]
        if alignment.evidence.last_refresh != iteration:
            return
        source = {k: (row[0], row[1], 1., torch.ones(self.bank.scales, dtype=torch.bool)) for k, row in self.source_raw.items()}
        target_all = {k: row[1] for k, row in alignment.evidence.raw.items()}
        selected = {k: (alignment.evidence.raw[k][1], row['label'], row['confidence'], row['scale_valid'])
                    for k, row in alignment.evidence.published.items()
                    if row['selected'] and iteration >= alignment.config.warmup
                    and iteration - row['last_seen'] <= alignment.config.max_age}
        self.snapshots = self.bank.refresh(source, target_all, selected, alignment, iteration)

    def state(self):
        return {'config': asdict(self.config), 'cumulative': dict(self.cumulative),
                'snapshot_iteration': int(self.bank.snapshot_iteration),
                'target_class_unique_support': self.bank.target_class_count.tolist(),
                'target_subgroup_unique_support': self.bank.target_subgroup_count.tolist(),
                'statistics_policy': 'raw_teacher_embeddings; unique_trials; refresh_after_optimizer; subgroup_slots_reset',
                'displacement_policy': 'diagnostic_only_detached_residual',
                'variance_policy': 'matched_unique_trial_cache_with_current_student_replacement'}

    def state_dict(self):
        return {'config': asdict(self.config), 'bank': deepcopy(self.bank.state_dict()),
                'source_raw': deepcopy(self.source_raw), 'snapshots': deepcopy(self.snapshots),
                'cumulative': dict(self.cumulative)}

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('R3 state configuration mismatch')
        self.bank.load_state_dict(state['bank'])
        self.source_raw = deepcopy(state['source_raw'])
        self.snapshots = deepcopy(state['snapshots'])
        self.cumulative = dict(state['cumulative'])
