"""Reliability and soft-graph objectives for the MPUS-GA MVP."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EPS = 1e-8


def entropy_mi_reliability(
    mc_probabilities: torch.Tensor,
    beta: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return teacher mean probability, normalized entropy, MI and reliability.

    ``mc_probabilities`` is shaped ``[M, B, C]``.
    """

    if mc_probabilities.ndim != 3 or mc_probabilities.shape[0] < 1:
        raise ValueError("mc_probabilities must be shaped [M,B,C]")
    class_count = mc_probabilities.shape[-1]
    if class_count < 2:
        raise ValueError("at least two classes are required")
    probabilities = mc_probabilities.clamp_min(EPS)
    mean_probability = probabilities.mean(dim=0)
    entropy = -(mean_probability * mean_probability.log()).sum(dim=-1)
    expected_entropy = -(
        probabilities * probabilities.log()
    ).sum(dim=-1).mean(dim=0)
    normalizer = math.log(class_count)
    normalized_entropy = (entropy / normalizer).clamp(0.0, 1.0)
    normalized_mi = ((entropy - expected_entropy) / normalizer).clamp_min(0.0)
    reliability = (
        (1.0 - normalized_entropy) * torch.exp(-beta * normalized_mi)
    ).clamp(0.0, 1.0)
    return mean_probability, normalized_entropy, normalized_mi, reliability


def balanced_reliable_gate(
    mean_probability: torch.Tensor,
    reliability: torch.Tensor,
    min_confidence: float,
    min_reliability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select equal top-score target counts per predicted class.

    Target graph supervision is disabled unless every class has a candidate.
    """

    confidence, pseudo_label = mean_probability.max(dim=1)
    candidate = (confidence >= min_confidence) & (reliability >= min_reliability)
    class_count = mean_probability.shape[1]
    raw_counts = torch.bincount(pseudo_label[candidate], minlength=class_count)
    selected = torch.zeros_like(candidate)
    if torch.all(raw_counts > 0):
        quota = int(raw_counts.min().item())
        score = reliability * confidence
        for class_id in range(class_count):
            indices = torch.where(candidate & (pseudo_label == class_id))[0]
            keep = indices[torch.topk(score[indices], k=quota).indices]
            selected[keep] = True
    return selected, raw_counts


def balanced_topk_gate(
    mean_probability: torch.Tensor,
    reliability: torch.Tensor,
    min_confidence: float,
    max_per_class: int,
    fallback_confidence: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a bounded, class-balanced target set by reliability-confidence.

    Unlike the v1 gate, reliability is a ranking signal rather than an
    absolute threshold.  The soft-edge loss normalizes its reliability weights,
    so discarding every sample merely because the absolute values are small is
    unnecessary.  All classes must still be represented to prevent a collapsed
    teacher from defining the target graph.
    """

    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    if fallback_confidence is not None and not (
        0.0 <= fallback_confidence <= min_confidence
    ):
        raise ValueError(
            "fallback_confidence must be within [0, min_confidence]"
        )
    confidence, pseudo_label = mean_probability.max(dim=1)
    candidate = confidence >= float(min_confidence)
    class_count = mean_probability.shape[1]
    if fallback_confidence is not None:
        for class_id in range(class_count):
            class_candidate = candidate & (pseudo_label == class_id)
            if not bool(class_candidate.any()):
                candidate |= (
                    (pseudo_label == class_id)
                    & (confidence >= float(fallback_confidence))
                )
    raw_counts = torch.bincount(pseudo_label[candidate], minlength=class_count)
    selected = torch.zeros_like(candidate)
    if torch.all(raw_counts > 0):
        quota = min(int(raw_counts.min().item()), int(max_per_class))
        score = reliability * confidence
        for class_id in range(class_count):
            indices = torch.where(candidate & (pseudo_label == class_id))[0]
            keep = indices[torch.topk(score[indices], k=quota).indices]
            selected[keep] = True
    return selected, raw_counts


def _weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or weights.sum() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(EPS)


def soft_graph_edge_loss(
    affinity_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_probability: torch.Tensor,
    target_reliability: torch.Tensor,
    selected_target: torch.Tensor,
    target_graph_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute separately normalized SS, ST and TT soft-affinity losses."""

    source_count = source_labels.numel()
    target_count = target_probability.shape[0]
    expected = source_count + target_count
    if affinity_logits.shape != (expected, expected):
        raise ValueError(
            f"affinity_logits must be [{expected},{expected}], got "
            f"{tuple(affinity_logits.shape)}"
        )

    class_count = target_probability.shape[1]
    source_one_hot = F.one_hot(source_labels, num_classes=class_count).to(
        target_probability.dtype
    )
    ss_logits = affinity_logits[:source_count, :source_count]
    ss_targets = source_one_hot @ source_one_hot.T
    ss_mask = ~torch.eye(source_count, dtype=torch.bool, device=ss_logits.device)
    ss_loss = _weighted_bce(
        ss_logits[ss_mask],
        ss_targets[ss_mask],
        torch.ones_like(ss_targets[ss_mask]),
    )

    zero = affinity_logits.sum() * 0.0
    st_loss = zero
    tt_loss = zero
    if bool(selected_target.any()) and target_graph_weight > 0:
        selected_probability = target_probability[selected_target]
        selected_reliability = target_reliability[selected_target]
        target_indices = torch.where(selected_target)[0] + source_count

        st_logits = affinity_logits[:source_count, target_indices]
        st_targets = source_one_hot @ selected_probability.T
        st_weights = selected_reliability.unsqueeze(0).expand_as(st_targets)
        st_loss = _weighted_bce(st_logits, st_targets, st_weights)

        selected_count = len(target_indices)
        if selected_count > 1:
            tt_logits = affinity_logits[target_indices][:, target_indices]
            tt_targets = selected_probability @ selected_probability.T
            tt_weights = (
                selected_reliability[:, None] * selected_reliability[None, :]
            )
            tt_mask = ~torch.eye(
                selected_count, dtype=torch.bool, device=tt_logits.device
            )
            tt_loss = _weighted_bce(
                tt_logits[tt_mask], tt_targets[tt_mask], tt_weights[tt_mask]
            )

    target_weight = float(target_graph_weight)
    total = ss_loss + target_weight * 0.5 * (st_loss + tt_loss)
    return total, {"ss": ss_loss, "st": st_loss, "tt": tt_loss}


def graph_ramp_weight(iteration: int, warmup: int, rampup: int) -> float:
    if iteration <= warmup:
        return 0.0
    if rampup <= 0 or iteration >= warmup + rampup:
        return 1.0
    progress = (iteration - warmup) / rampup
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def scheduled_confidence_threshold(
    iteration: int,
    warmup: int,
    ramp_iterations: int,
    start: float,
    end: float,
) -> float:
    """Linearly tighten the teacher-confidence floor after warm-up."""

    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("confidence thresholds must satisfy 0 <= start <= end <= 1")
    if iteration <= warmup or ramp_iterations <= 0:
        return float(start)
    progress = min(1.0, (iteration - warmup) / ramp_iterations)
    return float(start + progress * (end - start))
