"""Read-only R2 evidence -> reliable expansion -> auxiliary target CE only.

No target labels, probability correction, new teacher, clustering, or writes to
R2's evidence/prototype/subgroup banks. All admission decisions are detached.
"""
from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F


CARE_VARIANTS = ('full', 'r2', 'anchor_ce', 'no_geometry', 'fixed_selection')


@dataclass(frozen=True)
class CareConfig:
    enabled: bool = True
    geometry: bool = True
    adaptive: bool = True
    anchor_only: bool = False
    confidence_floor: float = .45
    margin_floor: float = .05
    jsd_ceiling: float = .15
    geometry_similarity: float = .30
    geometry_margin: float = .10
    minimum_votes: int = 2
    stable_refreshes: int = 2
    temporal_jsd: float = .10
    base_fraction: float = .20
    max_fraction: float = .70
    coverage_ema: float = .90
    loss_weight: float = .05

    def validate(self):
        for name in ('confidence_floor', 'margin_floor', 'jsd_ceiling', 'temporal_jsd',
                     'base_fraction', 'max_fraction', 'coverage_ema'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'Invalid CARE {name}')
        if not self.base_fraction <= self.max_fraction or self.coverage_ema >= 1:
            raise ValueError('Invalid CARE selection/EMA range')
        if not -1 <= self.geometry_similarity <= 1 or not 0 <= self.geometry_margin <= 2:
            raise ValueError('Invalid CARE geometry floor')
        if self.minimum_votes not in (2, 3) or self.stable_refreshes < 2 or self.loss_weight < 0:
            raise ValueError('CARE requires multi-scale support and distinct stable refreshes')


def care_config(variant='full', **overrides):
    if variant not in CARE_VARIANTS:
        raise ValueError(f'Unknown CARE variant: {variant}')
    switches = {'r2': {'enabled': False}, 'anchor_ce': {'anchor_only': True},
                'no_geometry': {'geometry': False}, 'fixed_selection': {'adaptive': False}}
    # Variant identity takes precedence over CLI defaults.
    config = replace(CareConfig(), **overrides)
    config = replace(config, **switches.get(variant, {}))
    config.validate()
    return config


@torch.no_grad()
def candidate_evidence(probability, features, source_prototypes, support, config):
    """Source prototypes and target features must be in the SAME teacher space."""
    probability = probability.detach()
    q = probability.mean(1)
    top = q.topk(2, dim=-1)
    labels = top.indices[:, 0]
    confidence = top.values[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]
    jsd = (probability * (probability.clamp_min(1e-8).log()
                         - q[:, None].clamp_min(1e-8).log())).sum(-1).mean(-1)
    votes = probability.argmax(-1) == labels[:, None]
    valid = ((votes.sum(-1) >= config.minimum_votes)
             & (confidence >= config.confidence_floor) & (margin >= config.margin_floor)
             & (jsd <= config.jsd_ceiling))
    score = confidence * (1 - jsd / math.log(probability.shape[-1])).clamp(0, 1)
    geometry_votes = torch.zeros_like(votes)
    if config.geometry:
        similarity = torch.einsum('bsd,sckd->bsck', F.normalize(features.detach(), dim=-1),
                                  F.normalize(source_prototypes.detach(), dim=-1))
        similarity.masked_fill_(~(support > 0)[None], -2.)
        class_similarity = similarity.max(-1).values
        values, classes = class_similarity.topk(2, dim=-1)
        gap = values[..., 0] - values[..., 1]
        # Require all coarse classes to have a supported prototype on a scale;
        # missing competitors must not manufacture a large geometric margin.
        ready = (support > 0).any(-1).all(-1)
        geometry_votes = (votes & ready[None] & (classes[..., 0] == labels[:, None])
                          & (values[..., 0] >= config.geometry_similarity)
                          & (gap >= config.geometry_margin))
        valid &= geometry_votes.sum(-1) >= config.minimum_votes
        quality = ((values[..., 0] + 1) / 2).clamp(0, 1) * (gap / 2).clamp(0, 1).sqrt()
        quality = (quality * geometry_votes).sum(-1) / geometry_votes.sum(-1).clamp_min(1)
        score = (score * quality).clamp_min(0).sqrt()
    return {'q': q, 'labels': labels, 'confidence': confidence, 'margin': margin,
            'jsd': jsd, 'eligible': valid, 'score': score, 'geometry_votes': geometry_votes}


def _jsd(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    m = (a + b) / 2
    return float(.5 * ((a * (a.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum()
                       + (b * (b.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum()))


class CareController:
    def __init__(self, config, r2_config, classes=3):
        config.validate()
        self.config, self.r2_config, self.classes = config, r2_config, classes
        self.cache, self.previous, self.selected = {}, {}, {}
        self.last_refresh = 0
        self.anchor_ema = torch.zeros(classes)
        self.candidate_ema = torch.zeros(classes)
        self.coverage = torch.zeros(classes)
        self.fractions = torch.full((classes,), config.base_fraction)
        self.history = []
        self.latest = {}

    def refresh(self, iteration):
        if iteration % self.r2_config.refresh_interval:
            return
        self.cache = {k: r for k, r in self.cache.items()
                      if iteration - r['seen'] <= self.r2_config.max_age}
        fresh = {k: r for k, r in self.cache.items() if r['seen'] > self.last_refresh}
        previous, stable = {}, {}
        anchors, candidates = torch.zeros(self.classes), torch.zeros(self.classes)
        for key, row in fresh.items():
            label = row['label']
            if row['anchor']:
                anchors[row['anchor_label']] += 1
            old = self.previous.get(key)
            same = (old is not None and old['label'] == label
                    and _jsd(row['q'], old['q']) <= self.config.temporal_jsd)
            streak = (old['streak'] + 1 if same else 1) if row['eligible'] else 0
            previous[key] = {**row, 'streak': streak}
            if streak >= self.config.stable_refreshes and row['eligible']:
                stable[key] = row
                candidates[label] += 1
        self.previous = previous
        decay = self.config.coverage_ema
        self.anchor_ema.mul_(decay).add_(anchors, alpha=1 - decay)
        self.candidate_ema.mul_(decay).add_(candidates, alpha=1 - decay)
        demand = self.anchor_ema + self.candidate_ema
        self.coverage = self.anchor_ema / demand.clamp_min(1e-8)
        gap = torch.where(demand > 0, 1 - self.coverage, torch.zeros_like(demand))
        self.fractions.fill_(self.config.base_fraction)
        if self.config.adaptive:
            self.fractions += (self.config.max_fraction - self.config.base_fraction) * gap
        self.selected = {}
        for label in range(self.classes):
            ranked = sorted((k for k, r in stable.items() if r['label'] == label),
                            key=lambda k: (-stable[k]['score'], k))
            # Do not turn 10 * float32(.2) into three accepted trials.
            fraction = round(float(self.fractions[label]), 6)
            count = min(len(ranked), math.ceil(len(ranked) * fraction - 1e-9))
            threshold = stable[ranked[count - 1]]['score'] if count else 1.
            for key in ranked[:count]:
                self.selected[key] = {**stable[key], 'threshold': threshold}
        self.last_refresh = iteration
        self.latest = {'iteration': iteration, 'anchor_unique': anchors.tolist(),
                       'candidate_unique': candidates.tolist(),
                       'selected_unique': [sum(r['label'] == c for r in self.selected.values())
                                           for c in range(self.classes)],
                       'anchor_ema': self.anchor_ema.tolist(), 'candidate_ema': self.candidate_ema.tolist(),
                       'coverage_proxy': self.coverage.tolist(), 'selection_fraction': self.fractions.tolist()}
        self.history.append(dict(self.latest))

    def loss(self, alignment, target_output, target_batch, iteration):
        logits = target_output['scale_logits']
        zero = logits.sum() * 0.
        if not self.config.enabled:
            return zero, None
        if any(k in target_batch for k in ('y', 'target_y', 'target_labels', 'target_label')):
            raise RuntimeError('CARE must never receive target truth')
        if alignment.pending is None:
            raise RuntimeError('CARE requires the current, read-only R2 teacher evidence')
        _, _, _, _, keys, features, probability, evidence_iteration = alignment.pending
        if evidence_iteration != iteration:
            raise RuntimeError('Stale R2 evidence')
        ids = target_batch.get('trial_key_by_scale')
        if ids is not None:
            expected = torch.stack([target_batch[n] for n in ('subject_id', 'session_id', 'trial_id')], -1)
            if any(not torch.equal(v.cpu(), expected.cpu()) for v in ids.values()):
                raise ValueError('CARE requires matching trial identities across scales')
        current = alignment.training_evidence
        anchor = current['target_valid'].any(-1).detach()
        anchor_labels = current['target_labels'].detach()
        if self.config.anchor_only:
            labels = anchor_labels
            mask = anchor
            weights = current['target_confidence'].detach()
        else:
            evidence = candidate_evidence(probability, features, alignment.bank.prototypes[0],
                                          alignment.bank.support[0], self.config)
            labels, weights = evidence['labels'], evidence['score']
            # Protect every live R2-published selection, including trials whose
            # current scale/peer filters do not produce a training anchor.
            protected = torch.tensor([
                bool((r := alignment.evidence.published.get(key)) is not None
                     and r['selected'] and iteration - r['last_seen'] <= self.r2_config.max_age)
                for key in keys], device=logits.device)
            eligible = evidence['eligible'] & ~anchor & ~protected
            for i, key in enumerate(keys):
                self.cache[key] = {'label': int(labels[i]), 'q': evidence['q'][i].cpu().tolist(),
                                   'score': float(weights[i]), 'eligible': bool(eligible[i]),
                                   'anchor': bool(anchor[i]), 'anchor_label': int(anchor_labels[i]),
                                   'seen': iteration}
            self.refresh(iteration)
            mask = eligible.clone()
            for i, key in enumerate(keys):
                selected = self.selected.get(key)
                mask[i] &= (selected is not None and selected['label'] == int(labels[i])
                            and iteration - selected['seen'] <= self.r2_config.max_age
                            and float(weights[i]) >= selected['threshold'])
        # Repeated sampling of one trial in a batch gives it only one CE term.
        seen = set()
        for i, key in enumerate(keys):
            if key in seen:
                mask[i] = False
            else:
                seen.add(key)
        ce = zero
        if mask.any():
            chosen = logits[mask]
            per_scale = F.cross_entropy(chosen.flatten(0, 1),
                                       labels[mask, None].expand(-1, chosen.shape[1]).reshape(-1),
                                       reduction='none').view(chosen.shape[:2])
            ce = (per_scale.mean(-1) * weights[mask].detach()).mean()
        added = self.config.loss_weight * self.r2_config.ramp(iteration) * ce
        record = {**self.latest, 'iteration': iteration, 'L_ce': float(ce.detach()),
                  'added_loss': float(added.detach()), 'ramp': self.r2_config.ramp(iteration),
                  'accepted_by_class': torch.bincount(labels[mask], minlength=self.classes).tolist(),
                  'anchor_batch': torch.bincount(anchor_labels[anchor], minlength=self.classes).tolist(),
                  'anchor_only': self.config.anchor_only}
        return added, record

    def state(self):
        return {'config': asdict(self.config), 'history': self.history,
                'policy': 'read_only_R2_teacher_source_subgroups; unique_trial_expansion_CE_only',
                'coverage': 'EMA(anchor_unique)/(EMA(anchor_unique)+EMA(stable_candidate_unique)); not target prior',
                'resume': 'completed_folds_skip; interrupted_folds_restart_from_seed'}
