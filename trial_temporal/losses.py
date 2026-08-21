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
    target_valid_mask: torch.Tensor | None = None,
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
    if target_valid_mask is not None and target_valid_mask.shape != (
        len(target_embeddings),
    ):
        raise ValueError("target_valid_mask must be [batch]")

    detached_probability = target_probability.detach()
    confidence, pseudo_label = detached_probability.max(dim=1)
    valid = confidence >= confidence_threshold
    if target_valid_mask is not None:
        valid = valid & target_valid_mask.detach().bool().to(valid.device)
    confidence_weight = (
        (confidence - confidence_threshold)
        / max(1.0 - confidence_threshold, 1e-6)
    ).clamp(0.0, 1.0)
    coverage = float(valid.float().mean())
    class_losses = []
    for class_index in range(classes):
        selected_target = valid & (pseudo_label == class_index)
        target_weight = confidence_weight * selected_target.to(confidence_weight)
        target_mass = target_weight.sum()
        if float(target_mass) <= 1e-6:
            continue
        target_centroid = (
            target_embeddings * target_weight[:, None, None]
        ).sum(dim=0) / target_mass.clamp_min(1e-8)
        target_centroid = F.normalize(target_centroid, dim=-1)
        losses = []
        weights = []
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
        if losses:
            loss_tensor = torch.stack(losses)
            weight_tensor = torch.stack(weights).detach().to(loss_tensor)
            class_losses.append(
                (loss_tensor * weight_tensor).sum()
                / weight_tensor.sum().clamp_min(1e-8)
            )
    if not class_losses:
        return target_embeddings.sum() * 0.0, coverage
    return torch.stack(class_losses).mean(), coverage
