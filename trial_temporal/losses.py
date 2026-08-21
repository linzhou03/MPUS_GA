"""Losses for class-conditional multi-source multiscale adaptation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def class_conditional_prototype_alignment_loss(
    source_embeddings: Sequence[torch.Tensor],
    source_labels: Sequence[torch.Tensor],
    target_embeddings: torch.Tensor,
    target_probability: torch.Tensor,
    joint_weights: torch.Tensor,
    confidence_threshold: float,
) -> tuple[torch.Tensor, float]:
    """Align each source-domain/scale/class centroid to its target centroid.

    Pseudo-label probabilities are detached, while source and target embeddings
    remain in the graph. Missing source classes are skipped; their historical
    prototypes remain available to the separate EMA reliability bank.
    """

    if not source_embeddings:
        raise ValueError("At least one source embedding tensor is required")
    if len(source_embeddings) != len(source_labels):
        raise ValueError("Each source embedding tensor needs labels")
    if target_embeddings.ndim != 3:
        raise ValueError("target_embeddings must be [batch, scales, features]")
    if target_probability.ndim != 2:
        raise ValueError("target_probability must be [batch, classes]")
    domains = len(source_embeddings)
    scales = target_embeddings.shape[1]
    classes = target_probability.shape[1]
    if joint_weights.shape != (domains, scales, classes):
        raise ValueError("joint_weights must be [domains, scales, classes]")
    if any(item.ndim != 3 for item in source_embeddings):
        raise ValueError("source embeddings must be [batch, scales, features]")

    detached_probability = target_probability.detach()
    confidence = detached_probability.max(dim=1).values
    confidence_weight = (
        (confidence - confidence_threshold)
        / max(1.0 - confidence_threshold, 1e-6)
    ).clamp(0.0, 1.0)
    coverage = float((confidence >= confidence_threshold).float().mean())
    losses = []
    weights = []
    for class_index in range(classes):
        target_weight = (
            detached_probability[:, class_index] * confidence_weight
        )
        target_mass = target_weight.sum()
        if float(target_mass) <= 1e-6:
            continue
        target_centroid = (
            target_embeddings * target_weight[:, None, None]
        ).sum(dim=0) / target_mass.clamp_min(1e-8)
        target_centroid = F.normalize(target_centroid, dim=-1)
        for domain_index, (embedding, labels) in enumerate(
            zip(source_embeddings, source_labels, strict=True)
        ):
            selected = labels == class_index
            if not torch.any(selected):
                continue
            source_centroid = F.normalize(
                embedding[selected].mean(dim=0), dim=-1
            )
            distance = 1.0 - (source_centroid * target_centroid).sum(dim=-1)
            for scale_index in range(scales):
                losses.append(distance[scale_index])
                weights.append(
                    joint_weights[domain_index, scale_index, class_index]
                )
    if not losses:
        return target_embeddings.sum() * 0.0, coverage
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).detach().to(loss_tensor)
    return (
        (loss_tensor * weight_tensor).sum()
        / weight_tensor.sum().clamp_min(1e-8),
        coverage,
    )
