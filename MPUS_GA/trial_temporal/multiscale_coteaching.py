"""Class-conditional peer teaching with periodic evidence and R2 fallback.

No target labels are accepted. Teacher probabilities, trial identities, and source
coarse labels are the only inputs to the evidence policy. Published decisions
remain fixed between refreshes; nothing updates from the current optimization step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .subgroup_alignment import (
    SelectiveSubgroupAlignment, SubgroupConfig, trial_keys,
)


@dataclass(frozen=True)
class CoTeachingConfig(SubgroupConfig):
    confidence: float = 0.60
    keep_fraction: float = 0.50
    stable_refreshes: int = 2
    temporal_jsd: float = 0.10
    source_momentum: float = 0.95
    source_reliability_floor: float = 0.10
    teaching_weight: float = 0.05
    replacement_refreshes: int = 2
    evidence_capacity: int = 512

    def validate(self):
        super().validate()
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0,1]")
        if min(self.stable_refreshes, self.replacement_refreshes) < 2:
            raise ValueError("Stability and replacement require at least two refreshes")
        if self.temporal_jsd < 0 or self.teaching_weight < 0:
            raise ValueError("Temporal JSD and teaching weight must be nonnegative")
        if not 0 <= self.source_momentum < 1 or not 0 < self.source_reliability_floor <= 1:
            raise ValueError("Invalid source reliability settings")
        if self.evidence_capacity < self.memory_per_class:
            raise ValueError("Evidence capacity must cover the per-class memory limit")


def js_divergence(left, right):
    middle = (left + right) * 0.5
    return 0.5 * ((left * (left.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
                  + (right * (right.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1))


@torch.no_grad()
def peer_probabilities(probability, reliability):
    """Scale s is taught by the other scales, never by its own prediction."""
    scales = probability.shape[1]
    if scales != 3:
        raise ValueError("Peer teaching requires the three R2 scales")
    distributions, agree, quality = [], [], []
    for scale in range(scales):
        peers = [s for s in range(scales) if s != scale]
        weights = reliability[peers].clamp_min(1e-8)
        scores = (probability[:, peers] * weights[None]).sum(1) / weights.sum(0)[None]
        scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-8)
        distributions.append(scores)
        votes = probability[:, peers].argmax(-1)
        agree.append((votes[:, 0] == votes[:, 1]) & (votes[:, 0] == scores.argmax(-1)))
        quality.append(weights.mean(0)[scores.argmax(-1)])
    return torch.stack(distributions, 1), torch.stack(agree, 1), torch.stack(quality, 1)


@torch.no_grad()
def voting_consensus(probability, reliability):
    votes = F.one_hot(probability.argmax(-1), probability.shape[-1]).sum(1)
    labels = votes.argmax(-1)
    agree = probability.argmax(-1) == labels[:, None]
    weight = reliability[:, labels].T * agree
    distribution = (probability * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-8)
    return labels, distribution, agree, votes.max(-1).values >= 2


class PeriodicPeerEvidence:
    def __init__(self, scales, classes, config):
        self.config = config
        self.scales, self.classes = scales, classes
        self.source_confusion = torch.zeros(scales, classes, classes)
        self.reliability = torch.ones(scales, classes)
        self.raw = {}
        self.previous = {}
        self.published = {}
        self.last_refresh = 0
        self.history = []

    @torch.no_grad()
    def observe_source(self, logits, labels):
        predictions = logits.detach().cpu().argmax(-1)
        labels = labels.detach().cpu()
        for scale in range(self.scales):
            matrix = torch.bincount(labels * self.classes + predictions[:, scale],
                                    minlength=self.classes ** 2).reshape(self.classes, self.classes).float()
            self.source_confusion[scale].lerp_(matrix, 1 - self.config.source_momentum)

    @torch.no_grad()
    def observe_target(self, keys, probabilities, features, iteration):
        for key, probability, feature in zip(keys, probabilities.detach().cpu(),
                                             features.detach().cpu(), strict=True):
            self.raw[key] = (probability.clone(), feature.clone(), iteration)
        for key in list(self.raw):
            if iteration - self.raw[key][2] > self.config.max_age:
                del self.raw[key]
        ordered = sorted(self.raw, key=lambda key: (self.raw[key][2], key))
        for key in ordered[:-self.config.evidence_capacity]:
            del self.raw[key]

    @torch.no_grad()
    def refresh(self, iteration):
        if iteration % self.config.refresh_interval:
            return False
        for key in list(self.raw):
            if iteration - self.raw[key][2] > self.config.max_age:
                del self.raw[key]
        matrix = self.source_confusion
        precision = matrix.diagonal(dim1=1, dim2=2) / matrix.sum(1).clamp_min(1e-8)
        recall = matrix.diagonal(dim1=1, dim2=2) / matrix.sum(2).clamp_min(1e-8)
        self.reliability = (2 * precision * recall / (precision + recall).clamp_min(1e-8)).clamp(
            self.config.source_reliability_floor, 1)
        self.published = {}
        if not self.raw:
            self.previous = {}
            self.last_refresh = iteration
            return True
        keys = sorted(self.raw)
        probabilities = torch.stack([self.raw[key][0] for key in keys])
        labels, consensus, scale_agree, majority = voting_consensus(probabilities, self.reliability)
        peers, peer_agree, peer_quality = peer_probabilities(probabilities, self.reliability)
        confidence = consensus.gather(1, labels[:, None])[:, 0]
        fresh_previous = {}
        stages = {name: [0] * self.classes for name in ('candidates', 'majority', 'confidence',
                  'stable', 'selected', 'peer_supervised')}
        for index, key in enumerate(keys):
            label = int(labels[index])
            prior = self.previous.get(key)
            # A repeated draw within a batch/refresh interval never increases stability.
            fresh = self.raw[key][2] > self.last_refresh
            same = (prior is not None and prior['label'] == label and fresh
                    and bool(js_divergence(consensus[index], prior['distribution']) <= self.config.temporal_jsd))
            streak = (prior['streak'] + 1 if same else 1) if majority[index] and fresh else 0
            peer_streak = torch.zeros(self.scales, dtype=torch.long)
            for scale in range(self.scales):
                valid_peer = (peer_agree[index, scale] and peers[index, scale].argmax() == label
                              and peers[index, scale, label] >= self.config.confidence and fresh)
                same_peer = (same and prior['peer_labels'][scale] == label
                             and js_divergence(peers[index, scale], prior['peers'][scale]) <= self.config.temporal_jsd)
                if valid_peer:
                    peer_streak[scale] = prior['peer_streak'][scale] + 1 if same_peer else 1
            candidate = bool(majority[index] and confidence[index] >= self.config.confidence
                             and streak >= self.config.stable_refreshes)
            peer_valid = peer_streak >= self.config.stable_refreshes
            row = {'label': label, 'distribution': consensus[index], 'confidence': float(confidence[index]),
                   'streak': streak, 'peer_labels': peers[index].argmax(-1), 'peers': peers[index],
                   'peer_streak': peer_streak, 'peer_valid': peer_valid,
                   'peer_quality': peer_quality[index], 'scale_valid': scale_agree[index],
                   'candidate': candidate, 'selected': False, 'last_seen': self.raw[key][2]}
            fresh_previous[key] = row
            stages['candidates'][label] += 1
            stages['majority'][label] += int(majority[index])
            stages['confidence'][label] += int(majority[index] and confidence[index] >= self.config.confidence)
            stages['stable'][label] += int(candidate)
        for label in range(self.classes):
            ranked = sorted((key for key in keys if fresh_previous[key]['label'] == label
                             and fresh_previous[key]['candidate']),
                            key=lambda key: (-fresh_previous[key]['confidence'], key))
            count = math.ceil(len(ranked) * self.config.keep_fraction)
            for key in ranked[:count]:
                fresh_previous[key]['selected'] = True
                stages['selected'][label] += 1
                stages['peer_supervised'][label] += int(fresh_previous[key]['peer_valid'].sum())
        self.published = fresh_previous
        self.previous = fresh_previous
        self.last_refresh = iteration
        self.history.append({'iteration': iteration, 'source_class_scale_reliability': self.reliability.tolist(),
                             'stages_by_class': stages,
                             'mean_consensus_confidence': float(confidence.mean()),
                             'unique_target_trials': len(keys)})
        return True


class ClassConditionalCoTeaching(SelectiveSubgroupAlignment):
    def __init__(self, student, config: CoTeachingConfig, device):
        super().__init__(student, config, device)
        self.evidence = PeriodicPeerEvidence(self.bank.scales, self.bank.classes, config)
        self.readiness = torch.zeros(self.bank.scales, self.bank.classes, device=device)
        self.match_streak = torch.zeros_like(self.readiness, dtype=torch.long)
        self.previous_matched_members = {}
        self.transition_history = []
        self.teaching_loss = None
        self.cumulative_teaching_pairs = 0
        # Read-only inputs for optional training auxiliaries; decisions stay owned here.
        self.training_evidence = None

    def centroid_strength(self, iteration):
        if self.config.weight == 0:
            return torch.ones_like(self.readiness)
        readiness=self.readiness.detach()
        muse=getattr(self,'muse',None)
        if muse is not None and muse.config.alignment_enabled:
            readiness=readiness*muse.class_gate()[None]
        return 1 - readiness * self.config.ramp(iteration)

    def loss(self, source_outputs, source_batches, target_output, target_batch, iteration):
        if 'y' in target_batch:
            raise RuntimeError('Co-teaching target batch must not contain labels')
        self.teacher.eval()
        sf, sl, sy, sk = [], [], [], []
        for index, batch in enumerate(source_batches):
            feature, logits = self._predict(batch)
            sf.append(feature); sl.append(logits); sy.append(batch['y'].to(feature.device))
            sk.extend(trial_keys(batch, index))
        sf, sl, sy = torch.cat(sf), torch.cat(sl), torch.cat(sy)
        tf, tl = self._predict(target_batch)
        tk = trial_keys(target_batch, 0)
        probability = tl.softmax(-1)
        muse = getattr(self, 'muse', None)
        if muse is not None:
            muse.observe(probability, target_batch, iteration)
        device = tf.device
        labels = torch.zeros(len(tk), dtype=torch.long, device=device)
        confidence = torch.zeros(len(tk), device=device)
        valid = torch.zeros(tf.shape[:2], dtype=torch.bool, device=device)
        peer_targets = torch.zeros_like(tl)
        peer_weight = torch.zeros(tf.shape[:2], device=device)
        current_peers, peer_agree, _ = peer_probabilities(probability, self.evidence.reliability.to(device))
        for index, key in enumerate(tk):
            row = self.evidence.published.get(key)
            if row is None or not row['selected'] or iteration - row['last_seen'] > self.config.max_age:
                continue
            label = row['label']
            labels[index] = label
            confidence[index] = row['confidence']
            current_agree = probability[index].argmax(-1) == label
            if int(current_agree.sum()) >= 2:
                valid[index] = current_agree & row['scale_valid'].to(device)
            peer_targets[index] = row['peers'].to(device)
            peer_ok = (row['peer_valid'].to(device) & peer_agree[index]
                       & (current_peers[index].argmax(-1) == label)
                       & (current_peers[index, :, label] >= self.config.confidence)
                       & (js_divergence(current_peers[index], peer_targets[index]) <= self.config.temporal_jsd))
            peer_weight[index] = (peer_ok * row['peers'][:, label].to(device)
                                  * row['peer_quality'].to(device))
        student_features = [torch.cat([output['scale_embeddings'] for output in source_outputs]),
                            target_output['scale_embeddings']]
        source_valid = torch.ones(sf.shape[:2], dtype=torch.bool, device=device)
        strength=self.readiness
        contrast_valid=valid
        if muse is not None and muse.config.alignment_enabled:
            strength=strength*muse.class_gate()[None]
            contrast_valid=valid & muse.evidence['hard_mask'][:,None]
        contrast, record = self.bank.loss(student_features, [sf, tf], [sy, labels],
                                          [sf.new_ones(len(sf)), confidence], [source_valid, contrast_valid],
                                          term_strength=strength)
        # KL transfers coarse semantics, preserving each scale's feature representation.
        terms = []
        for scale in range(self.bank.scales):
            for label in range(self.bank.classes):
                mask = (labels == label) & (peer_weight[:, scale] > 0)
                if mask.any():
                    divergence = F.kl_div(target_output['scale_logits'][mask, scale].log_softmax(-1),
                                          peer_targets[mask, scale].detach(), reduction='none').sum(-1)
                    terms.append((divergence * peer_weight[mask, scale].detach()).mean())
        self.teaching_loss = torch.stack(terms).mean() if terms else target_output['scale_logits'].sum() * 0
        used = int((peer_weight > 0).sum()) if self.config.ramp(iteration) > 0 and self.config.teaching_weight > 0 else 0
        self.cumulative_teaching_pairs += used
        self.pending = (sk, sf, sl, sy, tk, tf, probability, iteration)
        self.training_evidence = {'source_keys': sk, 'source_features': sf,
                                  'source_labels': sy, 'target_keys': tk,
                                  'target_features': tf, 'target_labels': labels,
                                  'target_confidence': confidence, 'target_valid': valid}
        record.update({'target_accepted_by_class': torch.bincount(labels[valid.any(-1)], minlength=3).tolist(),
                       'teacher_mean_confidence': float(probability.mean(1).max(-1).values.mean()),
                       'ramp': self.config.ramp(iteration), 'loss': float(contrast.detach()),
                       'teaching_loss': float(self.teaching_loss.detach()), 'teaching_pairs': used,
                       'evidence_snapshot_iteration': self.evidence.last_refresh,
                       'replacement_strength': (1 - self.centroid_strength(iteration)).tolist()})
        for key in self.cumulative_pairs:
            self.cumulative_pairs[key] += record[key]
        return contrast, record

    @torch.no_grad()
    def _refresh_readiness(self, iteration):
        for scale in range(self.bank.scales):
            for label in range(self.bank.classes):
                # Both directions need reliable different-class negatives before
                # retiring any R2 attraction for this class/scale.
                other_classes = [c for c in range(self.bank.classes) if c != label]
                negative_ready = all(bool((self.bank.support[d, scale, other_classes] > 0).any()) for d in (0, 1))
                members = set()
                for key, row in self.evidence.published.items():
                    if not row['selected'] or row['label'] != label or not row['scale_valid'][scale]:
                        continue
                    feature = F.normalize(self.evidence.raw[key][1][scale].to(self.bank.prototypes), dim=-1)
                    support = self.bank.support[1, scale, label] > 0
                    if not support.any():
                        continue
                    scores = (feature @ self.bank.prototypes[1, scale, label].T).masked_fill(~support, -2)
                    cosine, assignment = scores.max(0)
                    if cosine >= self.config.assignment_threshold and self.bank.matches[scale, label, :, assignment].any():
                        members.add(key)
                previous = self.previous_matched_members.get((scale, label), set())
                ready = negative_ready and len(members & previous) >= self.config.min_support
                self.match_streak[scale, label] = self.match_streak[scale, label] + 1 if ready else 0
                self.readiness[scale, label] = min(float(self.match_streak[scale, label]) / self.config.replacement_refreshes, 1)
                self.previous_matched_members[scale, label] = members
        if self.config.weight == 0:
            self.readiness.zero_()
        self.transition_history.append({'iteration': iteration, 'readiness': self.readiness.tolist(),
                                        'matched_trial_counts': [[len(self.previous_matched_members[s, c])
                                                                  for c in range(3)] for s in range(3)]})

    @torch.no_grad()
    def after_step(self, student):
        pending = self.pending
        self.pending = None
        super().after_step(student)  # Same EMA update, without the old pseudo-label policy.
        if pending is None:
            return
        sk, sf, sl, sy, tk, tf, probability, iteration = pending
        self.bank.observe(0, sk, sf, sy, sf.new_ones(len(sf)),
                          torch.ones(sf.shape[:2], dtype=torch.bool, device=sf.device), iteration)
        self.evidence.observe_source(sl, sy)
        self.evidence.observe_target(tk, probability, tf, iteration)
        muse=getattr(self,'muse',None)
        if muse is not None and muse.config.alignment_enabled:
            muse.observe_target_subgroups(self.bank,tk,tf,probability,iteration)
        if not self.evidence.refresh(iteration):
            return
        # Only the periodic published decisions enter the target subgroup bank.
        muse_subgroups=muse is not None and muse.config.alignment_enabled
        if not muse_subgroups:self.bank.memory[1].clear()
        selected = [] if muse_subgroups else [(key, row) for key, row in self.evidence.published.items() if row['selected']]
        if selected:
            keys, rows = zip(*selected)
            self.bank.observe(1, keys, torch.stack([self.evidence.raw[key][1] for key in keys]),
                              torch.tensor([row['label'] for row in rows]),
                              torch.tensor([row['confidence'] for row in rows]),
                              torch.stack([row['scale_valid'] for row in rows]), iteration)
        if self.bank.refresh(iteration):
            if muse_subgroups:muse.after_bank_refresh(self.bank)
            self._refresh_readiness(iteration)

    def state(self):
        result = super().state()
        result.update({'config': asdict(self.config), 'policy': 'class_conditional_peer_teaching_with_R2_centroid_fallback',
                       'cumulative_teaching_pairs': self.cumulative_teaching_pairs,
                       'evidence_history': self.evidence.history, 'transition_history': self.transition_history,
                       'final_readiness': self.readiness.tolist(),
                       'teacher_inputs': 'source_coarse_labels_and_unlabeled_target_trials_only'})
        return result
