"""Latent subgroup matching with coarse labels and an unlabeled EMA teacher.

Banks are fold-local, keyed by unique trial identity and refreshed periodically.
Unmatched same-class prototypes never enter the contrastive denominator.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SubgroupConfig:
    k: int = 4
    temperature: float = 0.1
    confidence: float = 0.9
    match_threshold: float = 0.8
    match_margin: float = 0.02
    assignment_threshold: float = 0.5
    weight: float = 0.05
    ema_momentum: float = 0.99
    warmup: int = 300
    ramp_end: int = 600
    refresh_interval: int = 50
    min_support: int = 3
    memory_per_class: int = 256
    max_age: int = 200
    cluster_iterations: int = 10
    jsd_threshold: float = 0.15
    minimum_votes: int = 2

    def validate(self) -> None:
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Subgroup parameters must be finite")
        if min(self.k, self.refresh_interval, self.min_support,
               self.memory_per_class, self.max_age, self.cluster_iterations) < 1:
            raise ValueError("Subgroup sizes and intervals must be positive")
        if self.memory_per_class < self.min_support or self.max_age < self.refresh_interval:
            raise ValueError("Subgroup memory/support/refresh settings are inconsistent")
        if self.temperature <= 0 or self.weight < 0:
            raise ValueError("Subgroup temperature must be positive and weight nonnegative")
        if not 0 <= self.confidence < 1 or not 0 <= self.ema_momentum < 1:
            raise ValueError("Confidence and EMA momentum must be in [0,1)")
        if not -1 <= self.match_threshold < 1 or not -1 <= self.assignment_threshold <= 1:
            raise ValueError("Subgroup cosine thresholds are invalid")
        if not 0 <= self.match_margin <= 2 or self.jsd_threshold < 0:
            raise ValueError("Subgroup matching margin/JSD are invalid")
        if not 0 <= self.warmup < self.ramp_end <= 1000 or self.minimum_votes < 1:
            raise ValueError("Subgroup warmup/ramp/votes are invalid")

    def ramp(self, iteration: int) -> float:
        fraction = min(max((iteration - self.warmup) / (self.ramp_end - self.warmup), 0), 1)
        return 0.5 - 0.5 * math.cos(math.pi * fraction)


@torch.no_grad()
def teacher_evidence(logits: torch.Tensor, config: SubgroupConfig):
    probability = logits.detach().softmax(dim=-1)
    mean = probability.mean(dim=1)
    confidence, labels = mean.max(dim=-1)
    votes = probability.argmax(dim=-1) == labels[:, None]
    jsd = (probability * (probability.clamp_min(1e-8).log()
                         - mean[:, None].clamp_min(1e-8).log())).sum(-1).mean(-1)
    valid = ((confidence >= config.confidence)
             & (votes.sum(-1) >= min(config.minimum_votes, logits.shape[1]))
             & (jsd <= config.jsd_threshold))
    # A disagreeing scale does not receive the consensus class as cluster truth.
    return labels, confidence, valid[:, None] & votes


def trial_keys(batch: dict, domain_index: int) -> list[tuple[int, ...]]:
    required = ("subject_id", "session_id", "trial_id")
    if any(key not in batch for key in required):
        raise ValueError("Subgroup memory requires subject/session/trial IDs")
    columns = [batch[key].detach().cpu().tolist() for key in required]
    return [(domain_index, *map(int, values)) for values in zip(*columns, strict=True)]


@torch.no_grad()
def spherical_clusters(vectors: torch.Tensor, confidence: torch.Tensor, config: SubgroupConfig):
    """Deterministic farthest-point initialization; K is an upper bound."""
    count = len(vectors)
    k = min(config.k, count // config.min_support)
    if not k:
        return vectors[:0], torch.zeros(0, dtype=torch.long), confidence[:0]
    vectors = F.normalize(vectors, dim=-1)
    center = F.normalize(vectors.mean(0), dim=0)
    chosen = [int((vectors @ center).argmax())]
    for _ in range(1, k):
        distances = 1 - (vectors @ vectors[chosen].T).max(dim=1).values
        if float(distances.max()) < 1e-5:
            break
        chosen.append(int(distances.argmax()))
    prototypes = vectors[chosen].clone()
    for _ in range(config.cluster_iterations):
        assignment = (vectors @ prototypes.T).argmax(-1)
        updated = prototypes.clone()
        for index in range(len(prototypes)):
            selected = assignment == index
            if selected.any():
                updated[index] = F.normalize(
                    (vectors[selected] * confidence[selected, None]).sum(0), dim=0
                )
        if torch.allclose(prototypes, updated, atol=1e-5):
            prototypes = updated
            break
        prototypes = updated
    assignment = (vectors @ prototypes.T).argmax(-1)
    support = torch.bincount(assignment, minlength=len(prototypes))
    valid = support >= config.min_support
    quality = torch.stack([
        confidence[assignment == index].mean() if support[index] else confidence.new_zeros(())
        for index in range(len(prototypes))
    ])
    return prototypes[valid], support[valid], quality[valid]


@torch.no_grad()
def mutual_matches(source: torch.Tensor, target: torch.Tensor, config: SubgroupConfig):
    similarity = source @ target.T
    edges = torch.zeros_like(similarity, dtype=torch.bool)
    if not len(source) or not len(target):
        return edges, similarity
    best_target = similarity.argmax(1)
    best_source = similarity.argmax(0)
    for left in range(len(source)):
        right = int(best_target[left])
        if int(best_source[right]) != left or similarity[left, right] < config.match_threshold:
            continue
        source_margin = (similarity[left].topk(2).values.diff().abs()[0]
                         if len(target) > 1 else similarity.new_tensor(2.0))
        target_margin = (similarity[:, right].topk(2).values.diff().abs()[0]
                         if len(source) > 1 else similarity.new_tensor(2.0))
        if min(float(source_margin), float(target_margin)) >= config.match_margin:
            edges[left, right] = True
    return edges, similarity


def selective_prototype_contrast(anchors, positives, negatives, weights, temperature):
    """Only one matched positive and reliable DIFFERENT-class negatives per row."""
    if not len(anchors) or not len(negatives):
        return anchors.sum() * 0.0
    anchors = F.normalize(anchors, dim=-1)
    positive_score = (anchors * positives.detach()).sum(-1, keepdim=True) / temperature
    negative_score = anchors @ negatives.detach().T / temperature
    scores = torch.cat((positive_score, negative_score), dim=1)
    return ((torch.logsumexp(scores, dim=1) - positive_score[:, 0]) * weights.detach()).mean()


class SubgroupBank:
    def __init__(self, scales: int, classes: int, dimension: int,
                 config: SubgroupConfig, device: torch.device):
        config.validate()
        self.config = config
        self.scales, self.classes = scales, classes
        self.memory = [{}, {}]
        self.prototypes = torch.zeros(2, scales, classes, config.k, dimension, device=device)
        self.support = torch.zeros(2, scales, classes, config.k, dtype=torch.long, device=device)
        self.quality = torch.zeros_like(self.support, dtype=torch.float32)
        self.matches = torch.zeros(scales, classes, config.k, config.k, dtype=torch.bool, device=device)
        self.similarity = torch.zeros_like(self.matches, dtype=torch.float32)
        self.last_refresh = 0
        self.history = []

    @torch.no_grad()
    def observe(self, domain, keys, embeddings, labels, confidence, valid, iteration):
        if embeddings.shape[:2] != valid.shape or len(keys) != len(labels):
            raise ValueError("Subgroup observations have inconsistent shapes")
        data = self.memory[domain]
        vectors = F.normalize(embeddings.detach(), dim=-1).cpu()
        for index, key in enumerate(keys):
            # Latest evidence replaces earlier pseudo-labels for this unique trial.
            data.pop(key, None)
            if not valid[index].any():
                continue
            label = int(labels[index])
            if not 0 <= label < self.classes:
                raise ValueError("Subgroup bank accepts coarse labels only")
            data[key] = (vectors[index].clone(), label, float(confidence[index]),
                         valid[index].detach().cpu().clone(), iteration)
        for key in list(data):
            if iteration - data[key][4] > self.config.max_age:
                del data[key]
        for label in range(self.classes):
            ordered = sorted((key for key in data if data[key][1] == label),
                             key=lambda key: (data[key][4], key))
            for key in ordered[:-self.config.memory_per_class]:
                del data[key]

    @torch.no_grad()
    def refresh(self, iteration: int) -> bool:
        if iteration < self.config.warmup or iteration % self.config.refresh_interval:
            return False
        self.prototypes.zero_()
        self.support.zero_()
        self.quality.zero_()
        self.matches.zero_()
        self.similarity.zero_()
        for domain in (0, 1):
            data = self.memory[domain]
            for key in list(data):
                if iteration - data[key][4] > self.config.max_age:
                    del data[key]
            for scale in range(self.scales):
                for label in range(self.classes):
                    rows = [data[key] for key in sorted(data)
                            if data[key][1] == label and data[key][3][scale]]
                    if len(rows) < self.config.min_support:
                        continue
                    vectors = torch.stack([row[0][scale] for row in rows])
                    confidence = torch.tensor([row[2] for row in rows])
                    prototypes, support, quality = spherical_clusters(vectors, confidence, self.config)
                    count = len(prototypes)
                    self.prototypes[domain, scale, label, :count] = prototypes.to(self.prototypes)
                    self.support[domain, scale, label, :count] = support.to(self.support)
                    self.quality[domain, scale, label, :count] = quality.to(self.quality)
        for scale in range(self.scales):
            for label in range(self.classes):
                ns = int((self.support[0, scale, label] > 0).sum())
                nt = int((self.support[1, scale, label] > 0).sum())
                edges, similarity = mutual_matches(
                    self.prototypes[0, scale, label, :ns],
                    self.prototypes[1, scale, label, :nt], self.config,
                )
                self.matches[scale, label, :ns, :nt] = edges
                self.similarity[scale, label, :ns, :nt] = similarity
        self.last_refresh = iteration
        self.history.append(self.state())
        return True

    def loss(self, student_embeddings, teacher_embeddings, labels, confidence, valid,
             term_strength=None):
        zero = sum(tensor.sum() * 0.0 for tensor in student_embeddings)
        totals = {key: 0 for key in ("positive_pairs", "negative_pairs", "ignored_same_class_pairs",
                                    "eligible_anchors", "used_anchors")}
        details, losses = [], []
        for domain in (0, 1):
            other = 1 - domain
            for scale in range(self.scales):
                for label in range(self.classes):
                    strength = 1.0 if term_strength is None else float(term_strength[scale, label])
                    if strength <= 0:
                        continue
                    selected = valid[domain][:, scale] & (labels[domain] == label)
                    eligible = int(selected.sum())
                    totals["eligible_anchors"] += eligible
                    ns = int((self.support[domain, scale, label] > 0).sum())
                    nt = int((self.support[other, scale, label] > 0).sum())
                    if not eligible or not ns:
                        continue
                    queries = F.normalize(teacher_embeddings[domain][selected, scale].detach(), dim=-1)
                    assignment_score = queries @ self.prototypes[domain, scale, label, :ns].T
                    assignment_cosine, assignment = assignment_score.max(-1)
                    edges = self.matches[scale, label]
                    if domain == 1:
                        edges = edges.T
                    matched = edges[assignment]
                    strong = matched.any(-1) & (assignment_cosine >= self.config.assignment_threshold)
                    totals["ignored_same_class_pairs"] += eligible * nt - int(strong.sum())
                    neg_mask = self.support[other, scale] > 0
                    neg_mask = neg_mask.clone()
                    neg_mask[label] = False
                    negatives = self.prototypes[other, scale][neg_mask]
                    if not strong.any() or not len(negatives):
                        continue
                    own = assignment[strong]
                    paired = matched[strong].long().argmax(-1)
                    similarity = self.similarity[scale, label]
                    if domain == 1:
                        similarity = similarity.T
                    pair_quality = similarity[own, paired].clamp(0, 1)
                    weights = (confidence[domain][selected][strong]
                               * self.quality[domain, scale, label, own]
                               * self.quality[other, scale, label, paired] * pair_quality)
                    value = selective_prototype_contrast(
                        student_embeddings[domain][selected, scale][strong],
                        self.prototypes[other, scale, label, paired], negatives,
                        weights, self.config.temperature,
                    )
                    # Equal coarse-class/scale/direction terms, not population weighting.
                    losses.append(value * strength)
                    used = int(strong.sum())
                    totals["used_anchors"] += used
                    totals["positive_pairs"] += used
                    totals["negative_pairs"] += used * len(negatives)
                    details.append({"domain": domain, "scale": scale, "class": label,
                                    "positive_pairs": used, "negative_pairs": used * len(negatives),
                                    "loss": float(value.detach())})
        totals["sample_match_coverage"] = totals["used_anchors"] / max(totals["eligible_anchors"], 1)
        totals["terms"] = details
        totals["snapshot_iteration"] = self.last_refresh
        return (torch.stack(losses).mean() if losses else zero), totals

    @torch.no_grad()
    def state(self):
        active = (self.support > 0).sum(-1)
        matched = torch.stack((self.matches.any(-1).sum(-1), self.matches.any(-2).sum(-1)))
        coverage = matched.float() / active.clamp_min(1)
        return {
            "snapshot_iteration": self.last_refresh,
            "axes": "domain[source,target], scale[1s,2s,4s], coarse_class[positive,neutral,negative]",
            "effective_subgroups": active.cpu().tolist(),
            "unique_trial_support": self.support.cpu().tolist(),
            "prototype_similarity": self.similarity.cpu().tolist(),
            "reliable_matches": self.matches.cpu().tolist(),
            "prototype_match_coverage": coverage.cpu().tolist(),
            "unmatched_subgroup_fraction": torch.where(active > 0, 1 - coverage, 0).cpu().tolist(),
            "no_subgroup_evidence": (active == 0).cpu().tolist(),
        }


class SelectiveSubgroupAlignment:
    """Training-only controller; evaluation always uses the student."""

    def __init__(self, student: nn.Module, config: SubgroupConfig, device: torch.device):
        config.validate()
        self.config = config
        self.teacher = deepcopy(student).eval().requires_grad_(False)
        self.bank = SubgroupBank(len(student.scales), student.num_classes,
                                 student.classifier.in_features, config, device)
        self.teacher_updates = 0
        self.pending = None
        self.cumulative_pairs = {"positive_pairs": 0, "negative_pairs": 0,
                                 "ignored_same_class_pairs": 0}

    @torch.no_grad()
    def _predict(self, batch):
        device = next(self.teacher.parameters()).device
        x = {key: value.to(device) for key, value in batch["x"].items()}
        mask = {key: value.to(device) for key, value in batch["mask"].items()}
        output = self.teacher(x, mask, compute_domain=False)
        return output["scale_embeddings"].detach(), output["scale_logits"].detach()

    def loss(self, source_outputs, source_batches, target_output, target_batch, iteration):
        if "y" in target_batch:
            raise RuntimeError("Subgroup target batch must not contain labels")
        self.teacher.eval()
        source_features, source_labels, source_keys = [], [], []
        for index, batch in enumerate(source_batches):
            features, _ = self._predict(batch)
            source_features.append(features)
            source_labels.append(batch["y"].to(features.device))
            source_keys.extend(trial_keys(batch, index))
        source_features = torch.cat(source_features)
        source_labels = torch.cat(source_labels)
        target_features, target_logits = self._predict(target_batch)
        target_labels, target_confidence, target_valid = teacher_evidence(target_logits, self.config)
        source_confidence = source_features.new_ones(len(source_features))
        source_valid = torch.ones(source_features.shape[:2], dtype=torch.bool, device=source_features.device)
        student_features = [torch.cat([output["scale_embeddings"] for output in source_outputs]),
                            target_output["scale_embeddings"]]
        teacher_features = [source_features, target_features]
        labels = [source_labels, target_labels]
        confidence = [source_confidence, target_confidence]
        valid = [source_valid, target_valid]
        loss, record = self.bank.loss(student_features, teacher_features, labels, confidence, valid)
        keys = [source_keys, trial_keys(target_batch, 0)]
        self.pending = (keys, teacher_features, labels, confidence, valid, iteration)
        record["target_accepted_by_class"] = torch.bincount(
            target_labels[target_valid.any(-1)], minlength=self.bank.classes
        ).cpu().tolist()
        record["teacher_mean_confidence"] = float(target_confidence.mean())
        record["ramp"] = self.config.ramp(iteration)
        record["loss"] = float(loss.detach())
        for key in self.cumulative_pairs:
            self.cumulative_pairs[key] += record[key]
        return loss, record

    @torch.no_grad()
    def after_step(self, student):
        # Bias-correct the first EMA steps; buffers (including non-floats) are copied.
        momentum = min(self.config.ema_momentum, self.teacher_updates / (self.teacher_updates + 1))
        for teacher, current in zip(self.teacher.parameters(), student.parameters(), strict=True):
            teacher.mul_(momentum).add_(current.detach(), alpha=1 - momentum)
        for teacher, current in zip(self.teacher.buffers(), student.buffers(), strict=True):
            teacher.copy_(current)
        self.teacher_updates += 1
        if self.pending is None:
            return
        keys, features, labels, confidence, valid, iteration = self.pending
        for domain in (0, 1):
            self.bank.observe(domain, keys[domain], features[domain], labels[domain],
                              confidence[domain], valid[domain], iteration)
        self.bank.refresh(iteration)
        self.pending = None

    def state(self):
        return {"config": asdict(self.config), "teacher_updates": self.teacher_updates,
                "cumulative_pair_counts": dict(self.cumulative_pairs),
                "final_snapshot": self.bank.state(), "refresh_history": self.bank.history,
                "policy": "coarse_source_labels_and_teacher_target_pseudo_labels_only; "
                          "unmatched_same_class_excluded_from_denominator; student_final_evaluation"}
