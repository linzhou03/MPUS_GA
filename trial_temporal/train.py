"""Class-conditional multiscale multi-source UDA training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm.auto import tqdm

from MPUS_GA.protocols import FIXED_UDA_PROTOCOL

from .data import (
    NUM_CLASSES,
    PreparedMultiSource,
    UnlabeledMultiScaleView,
    collate_multiscale,
    prepare_sources,
    prepare_target,
    scale_key,
)
from .losses import class_conditional_prototype_alignment_loss
from .model import MultiScaleMultiSourceDANN


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_DIR / "data_processed"
DEFAULT_RESULT_ROOT = PACKAGE_DIR / "results_bidirectional_full_ablation"
VARIANT = "class-conditional-boundary-reliable-weighted-pyramid"
CLASS_NAMES = ("positive", "neutral", "negative")
EVALUATION_PROTOCOL_FIXED_FINAL = "fixed_final"
EVALUATION_PROTOCOL_CAGA_TARGET_BEST = "caga_target_best"
EVALUATION_PROTOCOLS = (
    EVALUATION_PROTOCOL_FIXED_FINAL,
    EVALUATION_PROTOCOL_CAGA_TARGET_BEST,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    transfer_direction: str
    ablation: str
    source_domains: tuple[str, ...]
    scales: tuple[float, ...]
    fusion_mode: str
    domain_mode: str
    use_prototypes: bool
    use_feature_pyramid: bool = True
    use_source_excess_suppression: bool = True
    use_boundary_attractor_suppression: bool = True
    target_dataset: str = "seed_v"
    target_subject_count: int = 16
    target_trials: int = 45
    domain_weight: float = 0.2
    prototype_weight: float = 0.1


@dataclass(frozen=True)
class TargetEvidenceSnapshot:
    """Complete unlabeled-target evidence from independent scale heads."""

    mean_probability: torch.Tensor
    hard_frequency: torch.Tensor
    trials: int

    def state(self) -> dict:
        return {
            "mean_probability_by_scale": self.mean_probability.cpu().tolist(),
            "hard_frequency_by_scale": self.hard_frequency.cpu().tolist(),
            "trials": self.trials,
        }


_ABLATION_DEFINITIONS = {
    "0": {
        "description": "1-second single-scale baseline",
        "scales": (1.0,),
        "use_feature_pyramid": False,
        "use_source_excess_suppression": False,
        "use_boundary_attractor_suppression": False,
    },
    "1": {
        "description": "2-second single-scale baseline",
        "scales": (2.0,),
        "use_feature_pyramid": False,
        "use_source_excess_suppression": False,
        "use_boundary_attractor_suppression": False,
    },
    "2": {
        "description": "4-second single-scale baseline",
        "scales": (4.0,),
        "use_feature_pyramid": False,
        "use_source_excess_suppression": False,
        "use_boundary_attractor_suppression": False,
    },
    "3": {
        "description": "uniform 1/2/4-second multiscale fusion",
        "scales": (1.0, 2.0, 4.0),
        "fusion_mode": "uniform",
    },
    "4": {
        "description": "relation-weighted logits without the feature pyramid",
        "scales": (1.0, 2.0, 4.0),
        "use_feature_pyramid": False,
    },
    "5": {
        "description": "source-excess suppression without boundary attraction",
        "scales": (1.0, 2.0, 4.0),
        "use_boundary_attractor_suppression": False,
    },
    "6": {
        "description": "no source-excess or boundary-attractor suppression",
        "scales": (1.0, 2.0, 4.0),
        "use_source_excess_suppression": False,
        "use_boundary_attractor_suppression": False,
    },
    "main": {
        "description": "complete class-conditional boundary-reliable pyramid",
        "scales": (1.0, 2.0, 4.0),
    },
}

_TRANSFER_DIRECTIONS = {
    "A": {
        "description": "SEED-VII to SEED-V",
        "source_domains": ("seed_vii",),
        "target_dataset": "seed_v",
        "target_subject_count": 16,
        "target_trials": 45,
    },
    "B": {
        "description": "SEED-V to SEED-VII",
        "source_domains": ("seed_v",),
        "target_dataset": "seed_vii",
        "target_subject_count": 20,
        "target_trials": 80,
    },
}

EXPERIMENT_ORDER = tuple(
    name
    for ablation in (*map(str, range(7)), "main")
    for name in (
        f"A_{ablation}" if ablation == "main" else f"A{ablation}",
        f"B_{ablation}" if ablation == "main" else f"B{ablation}",
    )
)


def _build_experiments() -> dict[str, ExperimentSpec]:
    experiments = {}
    for ablation in (*map(str, range(7)), "main"):
        definition = _ABLATION_DEFINITIONS[ablation]
        for direction, transfer in _TRANSFER_DIRECTIONS.items():
            name = (
                f"{direction}_main"
                if ablation == "main"
                else f"{direction}{ablation}"
            )
            experiments[name] = ExperimentSpec(
                name=name,
                description=(
                    f"{transfer['description']}; {definition['description']}"
                ),
                transfer_direction=direction,
                ablation=ablation,
                source_domains=transfer["source_domains"],
                scales=definition["scales"],
                fusion_mode=definition.get("fusion_mode", "class_conditional"),
                domain_mode="scale_conditional",
                use_prototypes=True,
                use_feature_pyramid=definition.get(
                    "use_feature_pyramid", True
                ),
                use_source_excess_suppression=definition.get(
                    "use_source_excess_suppression", True
                ),
                use_boundary_attractor_suppression=definition.get(
                    "use_boundary_attractor_suppression", True
                ),
                target_dataset=transfer["target_dataset"],
                target_subject_count=transfer["target_subject_count"],
                target_trials=transfer["target_trials"],
            )
    return experiments


EXPERIMENTS = _build_experiments()


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_integer_set(value: str, minimum: int, maximum: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(minimum, maximum + 1))
    result: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    if not result or any(item < minimum or item > maximum for item in result):
        raise argparse.ArgumentTypeError(
            f"values must be within {minimum}..{maximum}"
        )
    return result


def _target_subjects(value: str, maximum: int = 16) -> list[int]:
    return _parse_integer_set(value, 1, maximum)


def source_sampling_weights(
    labels: torch.Tensor, balance_alpha: float
) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=NUM_CLASSES)
    if torch.any(counts == 0):
        raise ValueError(f"Source domain has an empty class: {counts.tolist()}")
    return counts.float().pow(-balance_alpha)[labels.long()].double()


def _source_loader(
    dataset,
    batch_size: int,
    iterations: int,
    seed: int,
    balance_alpha: float,
    pin_memory: bool,
) -> DataLoader:
    sampler = WeightedRandomSampler(
        source_sampling_weights(dataset.labels, balance_alpha),
        num_samples=batch_size * iterations,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_multiscale,
    )


def _target_loaders(
    target,
    batch_size: int,
    iterations: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    unlabeled = UnlabeledMultiScaleView(target)
    sampler = RandomSampler(
        unlabeled,
        replacement=True,
        num_samples=batch_size * iterations,
        generator=torch.Generator().manual_seed(seed),
    )
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": pin_memory,
        "collate_fn": collate_multiscale,
    }
    return (
        DataLoader(unlabeled, sampler=sampler, drop_last=True, **common),
        DataLoader(unlabeled, shuffle=False, drop_last=False, **common),
        DataLoader(target, shuffle=False, drop_last=False, **common),
    )


def _batch_to_device(batch: dict, device: torch.device) -> tuple[dict, dict]:
    return (
        {
            key: value.to(device, non_blocking=True)
            for key, value in batch["x"].items()
        },
        {
            key: value.to(device, non_blocking=True)
            for key, value in batch["mask"].items()
        },
    )


def _natural_source_class_priors(
    prepared: PreparedMultiSource, device: torch.device
) -> torch.Tensor:
    priors = []
    for dataset in prepared.datasets:
        counts = torch.bincount(dataset.labels.long(), minlength=NUM_CLASSES)
        priors.append(counts.float() / counts.sum())
    return torch.stack(priors).to(device)


def _effective_source_class_priors(
    prepared: PreparedMultiSource,
    balance_alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the class prior induced by softened inverse-frequency sampling."""

    priors = []
    for dataset in prepared.datasets:
        counts = torch.bincount(dataset.labels.long(), minlength=NUM_CLASSES)
        mass = counts.float().pow(1.0 - balance_alpha)
        priors.append(mass / mass.sum())
    return torch.stack(priors).to(device)


def _adaptation_ramp(
    iteration: int, warmup_iterations: int, ramp_end_iteration: int
) -> float:
    if iteration <= warmup_iterations:
        return 0.0
    if ramp_end_iteration <= warmup_iterations:
        return 1.0
    progress = (iteration - warmup_iterations) / (
        ramp_end_iteration - warmup_iterations
    )
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


@dataclass(frozen=True)
class TargetScaleConsensus:
    """Detached pseudo-label evidence derived only from independent scales."""

    probability: torch.Tensor
    pseudo_label: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor
    vote_count: torch.Tensor
    js_divergence: torch.Tensor

    def statistics(self, class_count: int) -> dict:
        valid_labels = self.pseudo_label[self.valid_mask]
        counts = torch.bincount(valid_labels, minlength=class_count)
        return {
            "coverage": float(self.valid_mask.float().mean()),
            "class_counts": counts.cpu().tolist(),
            "mean_confidence": float(self.confidence.mean()),
            "mean_js_divergence": float(self.js_divergence.mean()),
            "mean_vote_count": float(self.vote_count.float().mean()),
        }


@torch.no_grad()
def independent_scale_consensus(
    scale_logits: torch.Tensor,
    confidence_threshold: float,
    js_divergence_threshold: float,
    minimum_votes: int,
) -> TargetScaleConsensus:
    """Build target pseudo-labels before relation-guided fusion (issue 7)."""

    if scale_logits.ndim != 3:
        raise ValueError("scale_logits must be [batch, scales, classes]")
    scale_probability = F.softmax(scale_logits.detach(), dim=-1)
    consensus = scale_probability.mean(dim=1)
    confidence, pseudo_label = consensus.max(dim=1)
    vote_count = (scale_probability.argmax(dim=-1) == pseudo_label[:, None]).sum(
        dim=1
    )
    consensus_safe = consensus.clamp_min(1e-8)
    js_divergence = (
        scale_probability
        * (
            torch.log(scale_probability.clamp_min(1e-8))
            - torch.log(consensus_safe[:, None, :])
        )
    ).sum(dim=-1).mean(dim=1)
    required_votes = min(max(int(minimum_votes), 1), scale_logits.shape[1])
    valid = (
        (confidence >= confidence_threshold)
        & (vote_count >= required_votes)
        & (js_divergence <= js_divergence_threshold)
    )
    return TargetScaleConsensus(
        probability=consensus,
        pseudo_label=pseudo_label,
        confidence=confidence,
        valid_mask=valid,
        vote_count=vote_count,
        js_divergence=js_divergence,
    )


class TargetPriorEstimator:
    """Estimate target label proportions from independent multiscale evidence.

    Each scale maintains a source soft-confusion matrix P(prediction | class).
    The unlabeled target mean prediction is then matched jointly across scales
    by a ridge-regularized label-shift solve. Fused predictions never enter the
    estimator, which prevents a pyramid-error feedback loop.
    """

    def __init__(
        self,
        domain_count: int,
        scale_count: int,
        class_count: int,
        source_prior: torch.Tensor,
        device: torch.device,
        natural_source_prior: torch.Tensor | None = None,
        momentum: float = 0.99,
        ridge: float = 0.10,
        prior_floor: float = 0.03,
    ) -> None:
        if domain_count < 1 or scale_count < 1 or class_count < 2:
            raise ValueError("prior estimator dimensions must be positive")
        if source_prior.shape != (class_count,):
            raise ValueError("source_prior must be [classes]")
        if not 0 <= momentum < 1:
            raise ValueError("prior momentum must be within [0,1)")
        if ridge < 0:
            raise ValueError("prior ridge must be nonnegative")
        if not 0 <= prior_floor < 1.0 / class_count:
            raise ValueError("prior floor must be within [0,1/classes)")
        self.momentum = float(momentum)
        self.ridge = float(ridge)
        self.prior_floor = float(prior_floor)
        anchor = source_prior.detach().float().to(device).clamp_min(1e-8)
        self.source_prior = anchor / anchor.sum()
        natural_anchor = (
            source_prior if natural_source_prior is None else natural_source_prior
        )
        if natural_anchor.shape != (class_count,):
            raise ValueError("natural_source_prior must be [classes]")
        natural_anchor = natural_anchor.detach().float().to(device).clamp_min(1e-8)
        self.natural_source_prior = natural_anchor / natural_anchor.sum()
        identity = torch.eye(class_count, device=device)
        self.source_confusion = identity.view(
            1, 1, class_count, class_count
        ).expand(domain_count, scale_count, -1, -1).clone()
        self.source_hard_confusion = self.source_confusion.clone()
        self.source_initialized = torch.zeros(
            domain_count,
            scale_count,
            class_count,
            dtype=torch.bool,
            device=device,
        )
        self.source_updates = torch.zeros_like(
            self.source_initialized, dtype=torch.long
        )
        self.target_mean_probability = self.source_prior.unsqueeze(0).expand(
            scale_count, -1
        ).clone()
        self.target_hard_frequency = self.target_mean_probability.clone()
        self.target_initialized = False
        self.target_updates = 0
        self.estimated_prior = self.source_prior.clone()

    @torch.no_grad()
    def update_source(
        self,
        domain_index: int,
        scale_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        probability = F.softmax(scale_logits.detach(), dim=-1)
        hard_prediction = probability.argmax(dim=-1)
        if probability.shape[1:] != self.source_confusion.shape[1:3]:
            raise ValueError("source scale logits do not match prior estimator")
        for class_index in range(self.source_prior.numel()):
            selected = labels == class_index
            if not torch.any(selected):
                continue
            observation = probability[selected].mean(dim=0)
            hard_observation = F.one_hot(
                hard_prediction[selected], num_classes=self.source_prior.numel()
            ).float().mean(dim=0)
            for scale_index in range(probability.shape[1]):
                index = (domain_index, scale_index, class_index)
                column = self.source_confusion[
                    domain_index, scale_index, :, class_index
                ]
                if bool(self.source_initialized[index]):
                    column.mul_(self.momentum).add_(
                        observation[scale_index], alpha=1.0 - self.momentum
                    )
                    hard_column = self.source_hard_confusion[
                        domain_index, scale_index, :, class_index
                    ]
                    hard_column.mul_(self.momentum).add_(
                        hard_observation[scale_index],
                        alpha=1.0 - self.momentum,
                    )
                else:
                    column.copy_(observation[scale_index])
                    hard_column = self.source_hard_confusion[
                        domain_index, scale_index, :, class_index
                    ]
                    hard_column.copy_(hard_observation[scale_index])
                    self.source_initialized[index] = True
                column.div_(column.sum().clamp_min(1e-8))
                hard_column.div_(hard_column.sum().clamp_min(1e-8))
                self.source_updates[index] += 1

    @torch.no_grad()
    def update_target(self, scale_logits: torch.Tensor) -> None:
        probability = F.softmax(scale_logits.detach(), dim=-1)
        if probability.shape[1:] != self.target_mean_probability.shape:
            raise ValueError("target scale logits do not match prior estimator")
        observation = probability.mean(dim=0)
        hard_observation = F.one_hot(
            probability.argmax(dim=-1), num_classes=self.source_prior.numel()
        ).float().mean(dim=0)
        if self.target_initialized:
            self.target_mean_probability.mul_(self.momentum).add_(
                observation, alpha=1.0 - self.momentum
            )
            self.target_hard_frequency.mul_(self.momentum).add_(
                hard_observation, alpha=1.0 - self.momentum
            )
        else:
            self.target_mean_probability.copy_(observation)
            self.target_hard_frequency.copy_(hard_observation)
            self.target_initialized = True
        self.target_updates += 1
        self._solve()

    @torch.no_grad()
    def _solve(self) -> None:
        if not self.target_initialized:
            self.estimated_prior.copy_(self.source_prior)
            return
        confusion = self.source_confusion.mean(dim=0)
        design = confusion.reshape(-1, self.source_prior.numel())
        observed = self.target_mean_probability.reshape(-1)
        identity = torch.eye(
            self.source_prior.numel(), device=design.device, dtype=design.dtype
        )
        system = design.T @ design + self.ridge * identity
        rhs = design.T @ observed + self.ridge * self.source_prior
        estimate = torch.linalg.solve(system, rhs)
        estimate = estimate.clamp_min(self.prior_floor)
        self.estimated_prior.copy_(estimate / estimate.sum().clamp_min(1e-8))

    @torch.no_grad()
    def logit_adjustment(self, strength: float) -> torch.Tensor:
        if strength < 0:
            raise ValueError("prior correction strength must be nonnegative")
        ratio = torch.log(self.estimated_prior.clamp_min(1e-8)) - torch.log(
            self.source_prior.clamp_min(1e-8)
        )
        return float(strength) * ratio.clamp(-2.0, 2.0)

    @torch.no_grad()
    def source_prediction_reference(self) -> torch.Tensor:
        """Expected per-scale predictions under the natural source prior."""

        confusion = self.source_confusion.mean(dim=0)
        return torch.einsum(
            "spk,k->sp", confusion, self.natural_source_prior
        ).clamp_min(1e-8)

    @torch.no_grad()
    def source_class_precision(self) -> torch.Tensor:
        """Per-scale/class source precision implied by soft confusion."""

        confusion = self.source_confusion.mean(dim=0)
        true_positive_mass = torch.diagonal(
            confusion, dim1=1, dim2=2
        ) * self.natural_source_prior.unsqueeze(0)
        return (
            true_positive_mass / self.source_prediction_reference()
        ).clamp(0.0, 1.0)

    @torch.no_grad()
    def source_hard_prediction_reference(self) -> torch.Tensor:
        """Expected per-scale hard-vote rates under the natural source prior."""

        confusion = self.source_hard_confusion.mean(dim=0)
        return torch.einsum(
            "spk,k->sp", confusion, self.natural_source_prior
        ).clamp_min(1e-8)

    @torch.no_grad()
    def source_hard_class_precision(self) -> torch.Tensor:
        """Per-scale/class source precision of independent hard decisions."""

        confusion = self.source_hard_confusion.mean(dim=0)
        true_positive_mass = torch.diagonal(
            confusion, dim1=1, dim2=2
        ) * self.natural_source_prior.unsqueeze(0)
        return (
            true_positive_mass / self.source_hard_prediction_reference()
        ).clamp(0.0, 1.0)

    @torch.no_grad()
    def common_bias_adjustment(
        self,
        strength: float,
        relative_tolerance: float,
        maximum_adjustment: float,
    ) -> torch.Tensor:
        """Compatibility wrapper for source-reference excess suppression."""

        return self.common_bias_adjustments(
            source_excess_strength=strength,
            source_relative_tolerance=relative_tolerance,
            boundary_strength=0.0,
            boundary_ratio_tolerance=0.0,
            maximum_adjustment=maximum_adjustment,
        )["combined"]

    @torch.no_grad()
    def common_bias_adjustments(
        self,
        source_excess_strength: float,
        source_relative_tolerance: float,
        boundary_strength: float,
        boundary_ratio_tolerance: float,
        maximum_adjustment: float,
        boundary_mean_probability: torch.Tensor | None = None,
        boundary_hard_frequency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return source-excess, boundary-attractor, and combined penalties.

        Source excess detects a class whose target prediction mass exceeds its
        natural-source reference. Boundary attraction detects a different
        failure: independent scale heads choose the same class much more often
        than the probability mass they assign to it. Both require agreement
        from at least half the scales and are weighted by source false-positive
        risk. Neither fused predictions nor target labels enter either signal.
        """

        if min(source_excess_strength, boundary_strength) < 0:
            raise ValueError("common-bias strengths must be nonnegative")
        if min(source_relative_tolerance, boundary_ratio_tolerance) < 0:
            raise ValueError("common-bias tolerances must be nonnegative")
        if maximum_adjustment < 0:
            raise ValueError("maximum common-bias adjustment must be nonnegative")
        zero = torch.zeros_like(self.source_prior)
        empty = {
            "source_excess": zero.clone(),
            "boundary_attractor": zero.clone(),
            "combined": zero.clone(),
            "false_positive_risk": zero.clone(),
            "source_common_log_excess": zero.clone(),
            "boundary_common_log_ratio": zero.clone(),
        }
        if (
            (source_excess_strength == 0 and boundary_strength == 0)
            or maximum_adjustment == 0
            or not self.target_initialized
            or self.target_mean_probability.shape[0] < 2
            or not bool(self.source_initialized.all())
        ):
            return empty

        boundary_probability = (
            self.target_mean_probability
            if boundary_mean_probability is None
            else boundary_mean_probability.detach().to(
                self.target_mean_probability
            )
        )
        boundary_hard = (
            self.target_hard_frequency
            if boundary_hard_frequency is None
            else boundary_hard_frequency.detach().to(
                self.target_hard_frequency
            )
        )
        if (
            boundary_probability.shape != self.target_mean_probability.shape
            or boundary_hard.shape != self.target_hard_frequency.shape
        ):
            raise ValueError("boundary evidence must be [scales, classes]")

        precision = 0.5 * (
            self.source_class_precision()
            + self.source_hard_class_precision()
        )
        false_positive_risk = 1.0 - precision.mean(dim=0)

        soft_reference = self.source_prediction_reference()
        soft_log_excess = (
            torch.log(self.target_mean_probability.clamp_min(1e-8))
            - torch.log(soft_reference)
            - math.log1p(source_relative_tolerance)
        ).clamp_min(0.0)
        hard_reference = self.source_hard_prediction_reference()
        hard_log_excess = (
            torch.log(self.target_hard_frequency.clamp_min(1e-8))
            - torch.log(hard_reference)
            - math.log1p(source_relative_tolerance)
        ).clamp_min(0.0)
        log_excess = torch.maximum(soft_log_excess, hard_log_excess)
        # For three scales the median is positive only if at least two scales
        # report excess evidence. This rejects a single noisy temporal scale.
        common_excess = log_excess.median(dim=0).values
        source_penalty = (
            float(source_excess_strength)
            * false_positive_risk
            * common_excess
        )

        boundary_log_ratio = (
            torch.log(boundary_hard.clamp_min(1e-8))
            - torch.log(boundary_probability.clamp_min(1e-8))
            - math.log1p(boundary_ratio_tolerance)
        ).clamp_min(0.0)
        boundary_common_ratio = boundary_log_ratio.median(dim=0).values
        boundary_penalty = (
            float(boundary_strength)
            * false_positive_risk
            * boundary_common_ratio
        )
        combined_penalty = (source_penalty + boundary_penalty).clamp(
            max=float(maximum_adjustment)
        )
        return {
            "source_excess": -source_penalty,
            "boundary_attractor": -boundary_penalty,
            "combined": -combined_penalty,
            "false_positive_risk": false_positive_risk,
            "source_common_log_excess": common_excess,
            "boundary_common_log_ratio": boundary_common_ratio,
        }

    @torch.no_grad()
    def state(self) -> dict:
        return {
            "source_prior_anchor": self.source_prior.cpu().tolist(),
            "natural_source_prior_anchor": (
                self.natural_source_prior.cpu().tolist()
            ),
            "estimated_target_prior": self.estimated_prior.cpu().tolist(),
            "target_mean_probability_by_scale": (
                self.target_mean_probability.cpu().tolist()
            ),
            "target_hard_frequency_by_scale": (
                self.target_hard_frequency.cpu().tolist()
            ),
            "source_soft_confusion": self.source_confusion.cpu().tolist(),
            "source_hard_confusion": (
                self.source_hard_confusion.cpu().tolist()
            ),
            "source_prediction_reference_by_scale": (
                self.source_prediction_reference().cpu().tolist()
            ),
            "source_class_precision_by_scale": (
                self.source_class_precision().cpu().tolist()
            ),
            "source_hard_prediction_reference_by_scale": (
                self.source_hard_prediction_reference().cpu().tolist()
            ),
            "source_hard_class_precision_by_scale": (
                self.source_hard_class_precision().cpu().tolist()
            ),
            "source_initialized": self.source_initialized.cpu().tolist(),
            "source_updates": self.source_updates.cpu().tolist(),
            "target_updates": self.target_updates,
        }


class PrototypeBank:
    """EMA prototypes plus a source-anchored scale-class relation graph."""

    def __init__(
        self,
        domain_count: int,
        scale_count: int,
        class_count: int,
        feature_dim: int,
        momentum: float,
        temperature: float,
        uniform_mix: float,
        device: torch.device,
        relation_momentum: float = 0.99,
        relation_temperature: float = 0.25,
        relation_uniform_mix: float = 0.10,
    ) -> None:
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.uniform_mix = float(uniform_mix)
        self.relation_momentum = float(relation_momentum)
        self.relation_temperature = float(relation_temperature)
        self.relation_uniform_mix = float(relation_uniform_mix)
        self.source = torch.zeros(
            domain_count, scale_count, class_count, feature_dim, device=device
        )
        self.target = torch.zeros(
            scale_count, class_count, feature_dim, device=device
        )
        self.source_initialized = torch.zeros(
            domain_count, scale_count, class_count, dtype=torch.bool, device=device
        )
        self.target_initialized = torch.zeros(
            scale_count, class_count, dtype=torch.bool, device=device
        )
        self.source_relation = torch.full(
            (domain_count, scale_count, class_count),
            1.0 / scale_count,
            device=device,
        )
        self.source_relation_initialized = torch.zeros(
            domain_count, class_count, dtype=torch.bool, device=device
        )
        self.source_relation_updates = torch.zeros(
            domain_count, class_count, dtype=torch.long, device=device
        )
        self.target_class_updates = torch.zeros(
            class_count, dtype=torch.long, device=device
        )

    @torch.no_grad()
    def _update_vector(
        self, storage: torch.Tensor, initialized: torch.Tensor, index: tuple, value
    ) -> None:
        value = F.normalize(value.detach(), dim=0)
        if bool(initialized[index]):
            storage[index].mul_(self.momentum).add_(
                value, alpha=1.0 - self.momentum
            )
            storage[index].copy_(F.normalize(storage[index], dim=0))
        else:
            storage[index].copy_(value)
            initialized[index] = True

    @torch.no_grad()
    def update_source(
        self,
        domain_index: int,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        for class_index in range(self.source.shape[2]):
            selected = labels == class_index
            if not torch.any(selected):
                continue
            centroids = embeddings[selected].mean(dim=0)
            for scale_index in range(self.source.shape[1]):
                self._update_vector(
                    self.source,
                    self.source_initialized,
                    (domain_index, scale_index, class_index),
                    centroids[scale_index],
                )

    @torch.no_grad()
    def update_source_relation(
        self,
        domain_index: int,
        scale_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Estimate scale-to-class edges from class-wise source margins."""

        if scale_logits.ndim != 3:
            raise ValueError("scale_logits must be [batch, scales, classes]")
        if scale_logits.shape[1:3] != self.source_relation.shape[1:3]:
            raise ValueError("scale_logits do not match relation graph shape")
        detached = scale_logits.detach()
        for class_index in range(self.source_relation.shape[2]):
            selected = labels == class_index
            if not torch.any(selected):
                continue
            class_logits = detached[selected]
            correct = class_logits[:, :, class_index]
            other_mask = torch.ones(
                class_logits.shape[-1], dtype=torch.bool, device=detached.device
            )
            other_mask[class_index] = False
            strongest_other = class_logits[:, :, other_mask].amax(dim=-1)
            mean_margin = (correct - strongest_other).mean(dim=0)
            relation = F.softmax(
                mean_margin / self.relation_temperature, dim=0
            )
            index = (domain_index, class_index)
            if bool(self.source_relation_initialized[index]):
                self.source_relation[domain_index, :, class_index].mul_(
                    self.relation_momentum
                ).add_(relation, alpha=1.0 - self.relation_momentum)
                self.source_relation[domain_index, :, class_index].div_(
                    self.source_relation[domain_index, :, class_index]
                    .sum()
                    .clamp_min(1e-8)
                )
            else:
                self.source_relation[domain_index, :, class_index].copy_(
                    relation
                )
                self.source_relation_initialized[index] = True
            self.source_relation_updates[index] += 1

    @torch.no_grad()
    def source_relation_weights(self) -> torch.Tensor:
        uniform = torch.full_like(
            self.source_relation, 1.0 / self.source_relation.shape[1]
        )
        mixed = (
            (1.0 - self.relation_uniform_mix) * self.source_relation
            + self.relation_uniform_mix * uniform
        )
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def update_target(
        self,
        embeddings: torch.Tensor,
        probability: torch.Tensor,
        confidence_threshold: float,
        valid_mask: torch.Tensor | None = None,
    ) -> float:
        probability = probability.detach()
        confidence, pseudo_label = probability.max(dim=1)
        valid = confidence >= confidence_threshold
        if valid_mask is not None:
            if valid_mask.shape != valid.shape:
                raise ValueError("valid_mask must be [batch]")
            valid = valid & valid_mask.detach().bool().to(valid.device)
        confidence_weight = (
            (confidence - confidence_threshold)
            / max(1.0 - confidence_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        for class_index in range(self.target.shape[1]):
            selected = valid & (pseudo_label == class_index)
            weight = selected.to(confidence_weight) * confidence_weight
            mass = weight.sum()
            if float(mass) <= 1e-6:
                continue
            centroids = (embeddings * weight[:, None, None]).sum(dim=0)
            centroids = centroids / mass.clamp_min(1e-8)
            for scale_index in range(self.target.shape[0]):
                self._update_vector(
                    self.target,
                    self.target_initialized,
                    (scale_index, class_index),
                    centroids[scale_index],
                )
            self.target_class_updates[class_index] += int(selected.sum())
        return float(valid.float().mean())

    @torch.no_grad()
    def joint_weights(self) -> torch.Tensor:
        domains, scales, classes, _ = self.source.shape
        source_relation = self.source_relation_weights()
        result = source_relation / source_relation.sum(
            dim=(0, 1), keepdim=True
        ).clamp_min(1e-8)
        for class_index in range(classes):
            target_ready = self.target_initialized[:, class_index]
            source_ready = self.source_initialized[:, :, class_index]
            valid = source_ready & target_ready.unsqueeze(0)
            if not torch.any(valid):
                continue
            similarity = torch.einsum(
                "dsf,sf->ds",
                self.source[:, :, class_index],
                self.target[:, class_index],
            )
            logits = (
                torch.log(source_relation[:, :, class_index].clamp_min(1e-8))
                + similarity / self.temperature
            ).masked_fill(~valid, -1e9)
            learned = F.softmax(logits.flatten(), dim=0).reshape(domains, scales)
            uniform = torch.full_like(learned, 1.0 / learned.numel())
            result[:, :, class_index] = (
                (1.0 - self.uniform_mix) * learned
                + self.uniform_mix * uniform
            )
        return result

    @torch.no_grad()
    def scale_class_reliability(self) -> torch.Tensor:
        return self.joint_weights().sum(dim=0)

    @torch.no_grad()
    def source_weights(self, source_priors: torch.Tensor) -> torch.Tensor:
        class_prior = source_priors.mean(dim=0)
        domain_class_weight = self.joint_weights().sum(dim=1)
        weight = (domain_class_weight * class_prior.unsqueeze(0)).sum(dim=1)
        return weight / weight.sum().clamp_min(1e-8)

    @torch.no_grad()
    def state(self) -> dict:
        distances: list[list[list[float | None]]] = []
        for domain_index in range(self.source.shape[0]):
            domain_values = []
            for scale_index in range(self.source.shape[1]):
                scale_values = []
                for class_index in range(self.source.shape[2]):
                    ready = bool(
                        self.source_initialized[
                            domain_index, scale_index, class_index
                        ]
                        and self.target_initialized[scale_index, class_index]
                    )
                    value = None
                    if ready:
                        value = float(
                            1.0
                            - torch.dot(
                                self.source[
                                    domain_index, scale_index, class_index
                                ],
                                self.target[scale_index, class_index],
                            )
                        )
                    scale_values.append(value)
                domain_values.append(scale_values)
            distances.append(domain_values)
        return {
            "source_scale_class_relation": (
                self.source_relation_weights().cpu().tolist()
            ),
            "source_relation_initialized": (
                self.source_relation_initialized.cpu().tolist()
            ),
            "source_relation_updates": self.source_relation_updates.cpu().tolist(),
            "joint_source_scale_class_weights": self.joint_weights().cpu().tolist(),
            "scale_class_reliability": self.scale_class_reliability().cpu().tolist(),
            "source_initialized": self.source_initialized.cpu().tolist(),
            "target_initialized": self.target_initialized.cpu().tolist(),
            "target_class_updates": self.target_class_updates.cpu().tolist(),
            "prototype_cosine_distance": distances,
        }


def _scale_classification_loss(
    scale_logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    if scale_logits.shape[1] == 1:
        return scale_logits.sum() * 0.0
    repeated_labels = labels[:, None].expand(-1, scale_logits.shape[1]).reshape(-1)
    return F.cross_entropy(
        scale_logits.reshape(-1, scale_logits.shape[-1]),
        repeated_labels,
        label_smoothing=label_smoothing,
    )


def _gate_supervision_loss(
    scale_logits: torch.Tensor,
    scale_class_weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Teach the true-class gate which independent scale has lower error."""

    if scale_logits.shape[1] == 1 or not scale_class_weight.requires_grad:
        return scale_logits.sum() * 0.0
    true_logit = scale_logits.gather(
        2,
        labels[:, None, None].expand(-1, scale_logits.shape[1], 1),
    ).squeeze(-1)
    log_normalizer = torch.logsumexp(scale_logits, dim=-1)
    per_scale_nll = -(true_logit - log_normalizer)
    teacher = F.softmax(-per_scale_nll.detach() / temperature, dim=1)
    predicted = scale_class_weight.gather(
        2,
        labels[:, None, None].expand(-1, scale_logits.shape[1], 1),
    ).squeeze(-1)
    predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(
        torch.log(predicted.clamp_min(1e-8)), teacher, reduction="batchmean"
    )


def _class_balanced_domain_ce(
    logits: torch.Tensor,
    domain_index: int,
    class_labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if valid_mask is None:
        valid_mask = torch.ones_like(class_labels, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool().to(class_labels.device)
    domain_labels = torch.full(
        (len(logits),), domain_index, dtype=torch.long, device=logits.device
    )
    sample_loss = F.cross_entropy(logits, domain_labels, reduction="none")
    class_losses = []
    for class_index in range(NUM_CLASSES):
        selected = valid_mask & (class_labels == class_index)
        if torch.any(selected):
            class_losses.append(sample_loss[selected].mean())
    if not class_losses:
        return None
    return torch.stack(class_losses).mean()


def _domain_loss(
    source_outputs: list[dict],
    target_output: dict,
    source_labels: list[torch.Tensor],
    target_consensus: TargetScaleConsensus,
) -> tuple[torch.Tensor, list[float]]:
    if not torch.any(target_consensus.valid_mask):
        if target_output["scale_domain_logits"] is not None:
            logits = target_output["scale_domain_logits"]
            return logits.sum() * 0.0, [0.0] * logits.shape[1]
        logits = target_output["fused_domain_logits"]
        return logits.sum() * 0.0, [0.0]
    all_outputs = source_outputs + [target_output]
    all_class_labels = source_labels + [target_consensus.pseudo_label]
    all_valid_masks = [None] * len(source_outputs) + [target_consensus.valid_mask]
    if target_output["scale_domain_logits"] is not None:
        scale_losses = []
        for scale_index in range(target_output["scale_domain_logits"].shape[1]):
            domain_losses = []
            for domain_index, (output, class_labels, valid_mask) in enumerate(
                zip(
                    all_outputs,
                    all_class_labels,
                    all_valid_masks,
                    strict=True,
                )
            ):
                logits = output["scale_domain_logits"][:, scale_index]
                loss = _class_balanced_domain_ce(
                    logits, domain_index, class_labels, valid_mask
                )
                if loss is not None:
                    domain_losses.append(loss)
            scale_losses.append(torch.stack(domain_losses).mean())
        return torch.stack(scale_losses).mean(), [
            float(item.detach()) for item in scale_losses
        ]
    domain_losses = []
    for domain_index, (output, class_labels, valid_mask) in enumerate(
        zip(all_outputs, all_class_labels, all_valid_masks, strict=True)
    ):
        logits = output["fused_domain_logits"]
        loss = _class_balanced_domain_ce(
            logits, domain_index, class_labels, valid_mask
        )
        if loss is not None:
            domain_losses.append(loss)
    loss = torch.stack(domain_losses).mean()
    return loss, [float(loss.detach())]


def train_step(
    model: MultiScaleMultiSourceDANN,
    source_batches: list[dict],
    target_batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    iteration: int,
    spec: ExperimentSpec,
    source_class_priors: torch.Tensor,
    prototype_bank: PrototypeBank | None,
    label_smoothing: float,
    gradient_clip: float,
    adaptation_warmup_iterations: int,
    adaptation_ramp_end: int,
    pseudo_confidence_threshold: float,
    consensus_jsd_threshold: float = 0.15,
    consensus_minimum_votes: int = 2,
    scale_classification_weight: float = 0.30,
    gate_supervision_weight: float = 0.10,
    gate_teacher_temperature: float = 0.25,
    target_prior_estimator: TargetPriorEstimator | None = None,
    prior_correction_strength: float = 0.0,
    common_bias_strength: float = 2.0,
    common_bias_relative_tolerance: float = 0.10,
    boundary_bias_strength: float = 2.0,
    boundary_bias_ratio_tolerance: float = 0.15,
    common_bias_max_adjustment: float = 0.50,
) -> dict:
    if "y" in target_batch:
        raise RuntimeError("Target adaptation batch unexpectedly contains labels")
    if not spec.use_source_excess_suppression:
        common_bias_strength = 0.0
    if not spec.use_boundary_attractor_suppression:
        boundary_bias_strength = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ramp = _adaptation_ramp(
        iteration, adaptation_warmup_iterations, adaptation_ramp_end
    )
    adaptation_active = iteration > adaptation_warmup_iterations
    reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None
        else None
    )
    prior_adjustment = (
        target_prior_estimator.logit_adjustment(
            prior_correction_strength * ramp
        )
        if target_prior_estimator is not None
        and prior_correction_strength > 0
        and ramp > 0
        else None
    )
    common_bias_components = (
        target_prior_estimator.common_bias_adjustments(
            source_excess_strength=common_bias_strength * ramp,
            source_relative_tolerance=common_bias_relative_tolerance,
            boundary_strength=boundary_bias_strength * ramp,
            boundary_ratio_tolerance=boundary_bias_ratio_tolerance,
            maximum_adjustment=common_bias_max_adjustment,
        )
        if target_prior_estimator is not None
        and (common_bias_strength > 0 or boundary_bias_strength > 0)
        and ramp > 0
        else None
    )
    common_bias_adjustment = (
        common_bias_components["combined"]
        if common_bias_components is not None
        else None
    )
    target_logit_adjustment = None
    if prior_adjustment is not None or common_bias_adjustment is not None:
        target_logit_adjustment = torch.zeros(
            NUM_CLASSES, device=device, dtype=torch.float32
        )
        if prior_adjustment is not None:
            target_logit_adjustment.add_(prior_adjustment)
        if common_bias_adjustment is not None:
            target_logit_adjustment.add_(common_bias_adjustment)
    source_outputs = []
    source_labels = []
    for batch in source_batches:
        x, mask = _batch_to_device(batch, device)
        labels = batch["y"].to(device, non_blocking=True)
        source_labels.append(labels)
        source_outputs.append(
            model(
                x,
                mask,
                grl_alpha=ramp,
                scale_class_reliability=reliability,
            )
        )
    target_x, target_mask = _batch_to_device(target_batch, device)
    target_output = model(
        target_x,
        target_mask,
        grl_alpha=ramp,
        scale_class_reliability=reliability,
        class_logit_adjustment=target_logit_adjustment,
    )

    target_consensus = independent_scale_consensus(
        target_output["calibrated_scale_logits"],
        pseudo_confidence_threshold,
        consensus_jsd_threshold,
        consensus_minimum_votes,
    )

    fused_classification_by_source = torch.stack(
        [
            F.cross_entropy(
                output["logits"], labels, label_smoothing=label_smoothing
            )
            for output, labels in zip(source_outputs, source_labels, strict=True)
        ]
    )
    scale_classification_by_source = torch.stack(
        [
            _scale_classification_loss(
                output["scale_logits"], labels, label_smoothing
            )
            for output, labels in zip(source_outputs, source_labels, strict=True)
        ]
    )
    classification_by_source = (
        fused_classification_by_source
        + scale_classification_weight * scale_classification_by_source
    )
    gate_supervision_by_source = torch.stack(
        [
            _gate_supervision_loss(
                output["scale_logits"],
                output["scale_class_weight"],
                labels,
                gate_teacher_temperature,
            )
            for output, labels in zip(source_outputs, source_labels, strict=True)
        ]
    )
    source_weights = (
        prototype_bank.source_weights(source_class_priors)
        if prototype_bank is not None and adaptation_active
        else torch.full_like(
            classification_by_source, 1.0 / len(classification_by_source)
        )
    )
    classification_loss = (classification_by_source * source_weights).sum()
    gate_supervision_loss = (
        gate_supervision_by_source * source_weights
    ).sum()
    domain_loss, domain_by_scale = _domain_loss(
        source_outputs,
        target_output,
        source_labels,
        target_consensus,
    )
    if prototype_bank is not None and adaptation_active:
        prototype_loss, pseudo_coverage = (
            class_conditional_prototype_alignment_loss(
                [output["scale_embeddings"] for output in source_outputs],
                source_labels,
                target_output["scale_embeddings"],
                target_consensus.probability,
                prototype_bank.joint_weights(),
                pseudo_confidence_threshold,
                target_consensus.valid_mask,
            )
        )
    else:
        prototype_loss = target_output["logits"].sum() * 0.0
        pseudo_coverage = float(
            (
                target_consensus.valid_mask
            )
            .float()
            .mean()
        )
    total_loss = (
        classification_loss
        + gate_supervision_weight * gate_supervision_loss
        + ramp
        * (
            spec.domain_weight * domain_loss
            + spec.prototype_weight * prototype_loss
        )
    )
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip
    )
    optimizer.step()
    scheduler.step()

    if target_prior_estimator is not None:
        for domain_index, (output, labels) in enumerate(
            zip(source_outputs, source_labels, strict=True)
        ):
            target_prior_estimator.update_source(
                domain_index, output["scale_logits"], labels
            )
        # The estimator only sees raw independent target-scale logits. It never
        # consumes prior-corrected or pyramid-fused predictions.
        target_prior_estimator.update_target(target_output["scale_logits"])

    if prototype_bank is not None:
        for domain_index, (output, labels) in enumerate(
            zip(source_outputs, source_labels, strict=True)
        ):
            prototype_bank.update_source(
                domain_index, output["scale_embeddings"], labels
            )
            prototype_bank.update_source_relation(
                domain_index, output["scale_logits"], labels
            )
        if adaptation_active:
            pseudo_coverage = prototype_bank.update_target(
                target_output["scale_embeddings"],
                target_consensus.probability,
                pseudo_confidence_threshold,
                target_consensus.valid_mask,
            )
        else:
            pseudo_coverage = 0.0
        reliability = prototype_bank.scale_class_reliability()
        source_weights = prototype_bank.source_weights(source_class_priors)

    mean_gate = target_output["scale_class_weight"].detach().mean(dim=0)
    mean_pyramid_gate = target_output[
        "pyramid_scale_class_weight"
    ].detach().mean(dim=0)
    consensus_statistics = target_consensus.statistics(NUM_CLASSES)
    if not adaptation_active:
        consensus_statistics["active_class_counts"] = [0] * NUM_CLASSES
        consensus_statistics["active_coverage"] = 0.0
    else:
        consensus_statistics["active_class_counts"] = consensus_statistics[
            "class_counts"
        ]
        consensus_statistics["active_coverage"] = consensus_statistics[
            "coverage"
        ]
    record = {
        "iteration": iteration,
        "adaptation_ramp": ramp,
        "prototype_updates_active": (
            prototype_bank is not None and adaptation_active
        ),
        "total": float(total_loss.detach()),
        "classification": float(classification_loss.detach()),
        "fused_classification": float(
            (fused_classification_by_source * source_weights).sum().detach()
        ),
        "scale_classification": float(
            (scale_classification_by_source * source_weights).sum().detach()
        ),
        "gate_supervision": float(gate_supervision_loss.detach()),
        "classification_by_source": [
            float(value) for value in classification_by_source.detach()
        ],
        "domain": float(domain_loss.detach()),
        "domain_by_scale": domain_by_scale,
        "prototype": float(prototype_loss.detach()),
        "pseudo_label_coverage": pseudo_coverage,
        "target_scale_consensus": consensus_statistics,
        "source_weights": [float(value) for value in source_weights.detach()],
        "mean_target_scale_class_gate": mean_gate.cpu().tolist(),
        "mean_target_pyramid_scale_class_gate": (
            mean_pyramid_gate.cpu().tolist()
        ),
        "pyramid_residual_weight": float(
            target_output["pyramid_residual_weight"].detach()
        ),
        "learned_scale_class_relation_residual": (
            model.scale_class_relation_residual.detach().cpu().tolist()
        ),
        "gradient_norm": float(gradient_norm),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    if target_prior_estimator is not None:
        record["estimated_target_prior"] = (
            target_prior_estimator.estimated_prior.cpu().tolist()
        )
        record["prior_logit_adjustment"] = (
            prior_adjustment.cpu().tolist()
            if prior_adjustment is not None
            else [0.0] * NUM_CLASSES
        )
        record["common_bias_logit_adjustment"] = (
            common_bias_adjustment.cpu().tolist()
            if common_bias_adjustment is not None
            else [0.0] * NUM_CLASSES
        )
        record["source_excess_logit_adjustment"] = (
            common_bias_components["source_excess"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        )
        record["boundary_attractor_logit_adjustment"] = (
            common_bias_components["boundary_attractor"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        )
        record["boundary_common_log_ratio"] = (
            common_bias_components["boundary_common_log_ratio"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        )
        record["target_logit_adjustment"] = (
            target_logit_adjustment.detach().cpu().tolist()
            if target_logit_adjustment is not None
            else [0.0] * NUM_CLASSES
        )
    if prototype_bank is not None:
        record["joint_source_scale_class_weights"] = (
            prototype_bank.joint_weights().cpu().tolist()
        )
        record["source_scale_class_relation"] = (
            prototype_bank.source_relation_weights().cpu().tolist()
        )
        record["scale_class_reliability"] = reliability.cpu().tolist()
    return record


def _classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(labels, prediction, labels=list(range(NUM_CLASSES)))
    support = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=support > 0,
    )
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(
            f1_score(labels, prediction, average="macro", zero_division=0)
        ),
        "worst_class_recall": float(recall.min()),
        "recall_gap": float(recall.max() - recall.min()),
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, recall, strict=True)
        },
        "prediction_counts": {
            name: int((prediction == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "true_counts": {
            name: int((labels == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "trials": int(len(labels)),
    }


@torch.no_grad()
def collect_unlabeled_target_evidence(
    model: MultiScaleMultiSourceDANN,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> TargetEvidenceSnapshot:
    """Refresh raw scale evidence on the complete target set without labels."""

    model.eval()
    probability_sum = None
    hard_sum = None
    trials = 0
    for batch in tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        if "y" in batch:
            raise RuntimeError(
                "Target evidence refresh must use an unlabeled target view"
            )
        x, mask = _batch_to_device(batch, device)
        output = model(x, mask, compute_domain=False)
        probability = F.softmax(output["scale_logits"], dim=-1)
        hard = F.one_hot(
            probability.argmax(dim=-1), num_classes=NUM_CLASSES
        ).float()
        batch_probability_sum = probability.sum(dim=0)
        batch_hard_sum = hard.sum(dim=0)
        if probability_sum is None:
            probability_sum = batch_probability_sum
            hard_sum = batch_hard_sum
        else:
            probability_sum.add_(batch_probability_sum)
            hard_sum.add_(batch_hard_sum)
        trials += probability.shape[0]
    if probability_sum is None or hard_sum is None or trials == 0:
        raise RuntimeError("Target evidence refresh received no target trials")
    return TargetEvidenceSnapshot(
        mean_probability=probability_sum / trials,
        hard_frequency=hard_sum / trials,
        trials=trials,
    )


def evaluate_trials(
    model: MultiScaleMultiSourceDANN,
    loader: DataLoader,
    device: torch.device,
    description: str,
    prototype_bank: PrototypeBank | None,
    target_prior_estimator: TargetPriorEstimator | None = None,
    prior_correction_strength: float = 0.0,
    common_bias_strength: float = 2.0,
    common_bias_relative_tolerance: float = 0.10,
    boundary_bias_strength: float = 2.0,
    boundary_bias_ratio_tolerance: float = 0.15,
    common_bias_max_adjustment: float = 0.50,
    final_target_evidence: TargetEvidenceSnapshot | None = None,
) -> dict:
    model.eval()
    probabilities = []
    scale_probabilities = []
    corrected_scale_probabilities = []
    scale_gates = []
    pyramid_scale_gates = []
    labels = []
    reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None
        else None
    )
    prior_adjustment = (
        target_prior_estimator.logit_adjustment(prior_correction_strength)
        if target_prior_estimator is not None
        and prior_correction_strength > 0
        else None
    )
    common_bias_components = (
        target_prior_estimator.common_bias_adjustments(
            source_excess_strength=common_bias_strength,
            source_relative_tolerance=common_bias_relative_tolerance,
            boundary_strength=boundary_bias_strength,
            boundary_ratio_tolerance=boundary_bias_ratio_tolerance,
            maximum_adjustment=common_bias_max_adjustment,
            boundary_mean_probability=(
                final_target_evidence.mean_probability
                if final_target_evidence is not None
                else None
            ),
            boundary_hard_frequency=(
                final_target_evidence.hard_frequency
                if final_target_evidence is not None
                else None
            ),
        )
        if target_prior_estimator is not None
        and (common_bias_strength > 0 or boundary_bias_strength > 0)
        else None
    )
    common_bias_adjustment = (
        common_bias_components["combined"]
        if common_bias_components is not None
        else None
    )
    target_logit_adjustment = None
    if prior_adjustment is not None or common_bias_adjustment is not None:
        target_logit_adjustment = torch.zeros(
            NUM_CLASSES, device=device, dtype=torch.float32
        )
        if prior_adjustment is not None:
            target_logit_adjustment.add_(prior_adjustment)
        if common_bias_adjustment is not None:
            target_logit_adjustment.add_(common_bias_adjustment)
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=description,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            if "y" not in batch:
                raise RuntimeError("Evaluation requires labeled trials")
            x, mask = _batch_to_device(batch, device)
            output = model(
                x,
                mask,
                compute_domain=False,
                scale_class_reliability=reliability,
                class_logit_adjustment=target_logit_adjustment,
            )
            probabilities.append(output["probability"].cpu().numpy())
            scale_probabilities.append(
                F.softmax(output["scale_logits"], dim=-1).cpu().numpy()
            )
            corrected_scale_probabilities.append(
                F.softmax(
                    output["calibrated_scale_logits"], dim=-1
                ).cpu().numpy()
            )
            scale_gates.append(output["scale_class_weight"].cpu().numpy())
            pyramid_scale_gates.append(
                output["pyramid_scale_class_weight"].cpu().numpy()
            )
            labels.append(batch["y"].numpy())
    labels_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    scale_probability_array = np.concatenate(scale_probabilities)
    corrected_scale_probability_array = np.concatenate(
        corrected_scale_probabilities
    )
    gate_array = np.concatenate(scale_gates)
    pyramid_gate_array = np.concatenate(pyramid_scale_gates)
    return {
        "fused": _classification_metrics(labels_array, probability_array),
        "by_scale": {
            scale_key(scale): _classification_metrics(
                labels_array, scale_probability_array[:, index]
            )
            for index, scale in enumerate(model.scales)
        },
        "by_scale_adjusted": {
            scale_key(scale): _classification_metrics(
                labels_array, corrected_scale_probability_array[:, index]
            )
            for index, scale in enumerate(model.scales)
        },
        "mean_scale_class_gate": gate_array.mean(axis=0).tolist(),
        "mean_pyramid_scale_class_gate": (
            pyramid_gate_array.mean(axis=0).tolist()
        ),
        "pyramid_residual_weight": float(
            torch.sigmoid(model.pyramid_residual_logit.detach()).cpu()
        ),
        "estimated_target_prior": (
            target_prior_estimator.estimated_prior.cpu().tolist()
            if target_prior_estimator is not None
            else None
        ),
        "prior_logit_adjustment": (
            prior_adjustment.cpu().tolist()
            if prior_adjustment is not None
            else [0.0] * NUM_CLASSES
        ),
        "common_bias_logit_adjustment": (
            common_bias_adjustment.cpu().tolist()
            if common_bias_adjustment is not None
            else [0.0] * NUM_CLASSES
        ),
        "source_excess_logit_adjustment": (
            common_bias_components["source_excess"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        ),
        "boundary_attractor_logit_adjustment": (
            common_bias_components["boundary_attractor"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        ),
        "boundary_common_log_ratio": (
            common_bias_components["boundary_common_log_ratio"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        ),
        "source_false_positive_risk": (
            common_bias_components["false_positive_risk"].cpu().tolist()
            if common_bias_components is not None
            else [0.0] * NUM_CLASSES
        ),
        "final_unlabeled_target_evidence": (
            final_target_evidence.state()
            if final_target_evidence is not None
            else None
        ),
        "target_logit_adjustment": (
            target_logit_adjustment.cpu().tolist()
            if target_logit_adjustment is not None
            else [0.0] * NUM_CLASSES
        ),
        "source_prediction_reference_by_scale": (
            target_prior_estimator.source_prediction_reference().cpu().tolist()
            if target_prior_estimator is not None
            else None
        ),
        "source_class_precision_by_scale": (
            target_prior_estimator.source_class_precision().cpu().tolist()
            if target_prior_estimator is not None
            else None
        ),
        "source_hard_prediction_reference_by_scale": (
            target_prior_estimator.source_hard_prediction_reference()
            .cpu()
            .tolist()
            if target_prior_estimator is not None
            else None
        ),
        "source_hard_class_precision_by_scale": (
            target_prior_estimator.source_hard_class_precision().cpu().tolist()
            if target_prior_estimator is not None
            else None
        ),
    }


def _should_evaluate_target(
    protocol: str,
    iteration: int,
    total_iterations: int,
    target_eval_interval: int,
) -> bool:
    """Return whether this model state is a target-evaluation candidate."""

    if iteration == total_iterations:
        return True
    return (
        protocol == EVALUATION_PROTOCOL_CAGA_TARGET_BEST
        and iteration % target_eval_interval == 0
    )


def _target_evaluation_is_better(candidate: dict, incumbent: dict | None) -> bool:
    """Match public CAGA-SGA: select strictly higher target Accuracy."""

    return incumbent is None or (
        candidate["fused"]["accuracy"] > incumbent["fused"]["accuracy"]
    )


def _write_summary(result_dir: Path) -> None:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("seed_*_subject_*.json"))
    ]
    if not results:
        return
    rows = []
    metric_paths = {
        "trial_accuracy": ("evaluation", "fused", "accuracy"),
        "trial_balanced_accuracy": (
            "evaluation",
            "fused",
            "balanced_accuracy",
        ),
        "trial_macro_f1": ("evaluation", "fused", "macro_f1"),
        "worst_class_recall": (
            "evaluation",
            "fused",
            "worst_class_recall",
        ),
        "recall_gap": ("evaluation", "fused", "recall_gap"),
    }
    for class_name in CLASS_NAMES:
        metric_paths[f"recall_{class_name}"] = (
            "evaluation",
            "fused",
            "per_class_recall",
            class_name,
        )
    for metric, path in metric_paths.items():
        values = []
        for result in results:
            value = result
            for key in path:
                value = value[key]
            values.append(float(value))
        array = np.asarray(values)
        rows.append(
            {
                "metric": metric,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "folds": len(array),
            }
        )
    with (result_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std", "folds"))
        writer.writeheader()
        writer.writerows(rows)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    iterations: int,
    warmup_iterations: int,
):
    def multiplier(step: int) -> float:
        if step < warmup_iterations:
            return (step + 1) / max(warmup_iterations, 1)
        progress = (step - warmup_iterations) / max(
            iterations - warmup_iterations, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_fold(
    args,
    spec: ExperimentSpec,
    prepared: PreparedMultiSource,
    seed: int,
    subject: int,
    device: torch.device,
) -> None:
    iterations = FIXED_UDA_PROTOCOL.training_iterations
    experiment_dir = args.result_root / spec.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    result_path = experiment_dir / f"seed_{seed}_subject_{subject:02d}.json"
    if result_path.exists() and not args.overwrite:
        tqdm.write(f"Skip existing {result_path}")
        return

    target = prepare_target(
        args.data_dir,
        spec.target_dataset,
        subject,
        prepared,
        spec.scales,
    )
    FIXED_UDA_PROTOCOL.validate_fold(
        target_trials=len(target),
        expected_target_trials=spec.target_trials,
        training_iterations=iterations,
    )
    set_seed(seed, device)
    if args.source_batch_size % len(prepared.datasets):
        raise ValueError(
            "source-batch-size must be divisible by the source-domain count"
        )
    per_source_batch = args.source_batch_size // len(prepared.datasets)
    if per_source_batch < 2:
        raise ValueError("Each source domain needs at least two trials per batch")
    pin_memory = device.type == "cuda"
    source_loaders = [
        _source_loader(
            dataset,
            per_source_batch,
            iterations,
            seed + domain_index * 101,
            args.source_balance_alpha,
            pin_memory,
        )
        for domain_index, dataset in enumerate(prepared.datasets)
    ]
    target_loader, target_evidence_loader, test_loader = _target_loaders(
        target,
        args.target_batch_size,
        iterations,
        seed + 1001,
        pin_memory,
    )
    model = MultiScaleMultiSourceDANN(
        scales=spec.scales,
        num_domains=len(prepared.domain_names) + 1,
        d_model=args.d_model,
        num_heads=args.num_heads,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        fusion_layers=args.fusion_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        spatial_topk=args.spatial_topk,
        use_channel_attention=True,
        channel_attention_reduction=args.channel_attention_reduction,
        fusion_mode=spec.fusion_mode,
        domain_mode=spec.domain_mode,
        detach_domain_probability=True,
        relation_strength=args.relation_strength,
        pyramid_weight_floor=args.pyramid_weight_floor,
        pyramid_residual_initial=args.pyramid_residual_initial,
        use_feature_pyramid=spec.use_feature_pyramid,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = _scheduler(optimizer, iterations, args.warmup_iterations)
    source_class_priors = _natural_source_class_priors(prepared, device)
    effective_source_class_priors = _effective_source_class_priors(
        prepared, args.source_balance_alpha, device
    )
    target_prior_estimator = TargetPriorEstimator(
        len(prepared.datasets),
        len(spec.scales),
        NUM_CLASSES,
        effective_source_class_priors.mean(dim=0),
        device,
        natural_source_prior=source_class_priors.mean(dim=0),
        momentum=args.prior_momentum,
        ridge=args.prior_ridge,
        prior_floor=args.prior_floor,
    )
    common_bias_strength = (
        args.common_bias_strength
        if spec.use_source_excess_suppression
        else 0.0
    )
    boundary_bias_strength = (
        args.boundary_bias_strength
        if spec.use_boundary_attractor_suppression
        else 0.0
    )
    prototype_bank = None
    if spec.use_prototypes:
        prototype_bank = PrototypeBank(
            len(prepared.datasets),
            len(spec.scales),
            NUM_CLASSES,
            args.d_model,
            args.prototype_momentum,
            args.reliability_temperature,
            args.reliability_uniform_mix,
            device,
            relation_momentum=args.relation_momentum,
            relation_temperature=args.relation_temperature,
            relation_uniform_mix=args.relation_uniform_mix,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    source_iterators = [iter(loader) for loader in source_loaders]
    target_iterator = iter(target_loader)
    training_trace = []
    target_evaluation_trace = []
    selected_evaluation = None
    selected_iteration = None
    progress = tqdm(
        range(1, iterations + 1),
        desc=f"{spec.name} | seed={seed} | target={subject:02d}",
        unit="iter",
        leave=False,
        dynamic_ncols=True,
    )
    for iteration in progress:
        source_batches = []
        for index, loader in enumerate(source_loaders):
            try:
                batch = next(source_iterators[index])
            except StopIteration:
                source_iterators[index] = iter(loader)
                batch = next(source_iterators[index])
            source_batches.append(batch)
        try:
            target_batch = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_batch = next(target_iterator)
        record = train_step(
            model,
            source_batches,
            target_batch,
            optimizer,
            scheduler,
            device,
            iteration,
            spec,
            source_class_priors,
            prototype_bank,
            args.label_smoothing,
            args.gradient_clip,
            args.adaptation_warmup_iterations,
            args.adaptation_ramp_end,
            args.pseudo_confidence_threshold,
            args.consensus_jsd_threshold,
            args.consensus_minimum_votes,
            args.scale_classification_weight,
            args.gate_supervision_weight,
            args.gate_teacher_temperature,
            target_prior_estimator,
            args.prior_correction_strength,
            common_bias_strength,
            args.common_bias_relative_tolerance,
            boundary_bias_strength,
            args.boundary_bias_ratio_tolerance,
            args.common_bias_max_adjustment,
        )
        if iteration == 1 or iteration % args.log_interval == 0 or iteration == iterations:
            training_trace.append(record)
            progress.set_postfix(
                total=f"{record['total']:.3f}",
                cls=f"{record['classification']:.3f}",
                dom=f"{record['domain']:.3f}",
                proto=f"{record['prototype']:.3f}",
            )
        if _should_evaluate_target(
            args.evaluation_protocol,
            iteration,
            iterations,
            args.target_eval_interval,
        ):
            final_target_evidence = collect_unlabeled_target_evidence(
                model,
                target_evidence_loader,
                device,
                (
                    f"Unlabeled evidence I{iteration:04d} "
                    f"{spec.name}/seed{seed}/S{subject:02d}"
                ),
            )
            candidate_evaluation = evaluate_trials(
                model,
                test_loader,
                device,
                (
                    f"Target eval I{iteration:04d} "
                    f"{spec.name}/seed{seed}/S{subject:02d}"
                ),
                prototype_bank,
                target_prior_estimator,
                args.prior_correction_strength,
                common_bias_strength,
                args.common_bias_relative_tolerance,
                boundary_bias_strength,
                args.boundary_bias_ratio_tolerance,
                args.common_bias_max_adjustment,
                final_target_evidence,
            )
            target_evaluation_trace.append(
                {"iteration": iteration, "evaluation": candidate_evaluation}
            )
            if _target_evaluation_is_better(
                candidate_evaluation, selected_evaluation
            ):
                selected_evaluation = candidate_evaluation
                selected_iteration = iteration
            tqdm.write(
                f"Target eval {spec.name}/seed{seed}/S{subject:02d} "
                f"I{iteration:04d}: "
                f"acc={candidate_evaluation['fused']['accuracy']:.4f}, "
                f"best_acc={selected_evaluation['fused']['accuracy']:.4f} "
                f"at I{selected_iteration:04d}"
            )

    if selected_evaluation is None or selected_iteration is None:
        raise RuntimeError("No target evaluation was produced")
    evaluation = selected_evaluation
    is_target_selected = (
        args.evaluation_protocol == EVALUATION_PROTOCOL_CAGA_TARGET_BEST
    )
    result = {
        "variant": f"{VARIANT}-{args.evaluation_protocol}",
        "experiment": spec.name,
        "experiment_spec": asdict(spec),
        "protocol": {
            "name": (
                f"{'_'.join(prepared.domain_names)}_to_"
                f"{spec.target_dataset}_transductive_"
                f"{args.evaluation_protocol}"
            ),
            "source_domains": list(prepared.domain_names),
            "target_dataset": spec.target_dataset,
            "target_subject_count": spec.target_subject_count,
            "target_trials": spec.target_trials,
            "training_iterations": FIXED_UDA_PROTOCOL.training_iterations,
            "checkpoint_selection": (
                "highest_target_accuracy"
                if is_target_selected
                else FIXED_UDA_PROTOCOL.checkpoint_selection
            ),
            "target_evaluations": len(target_evaluation_trace),
            "target_eval_interval": (
                args.target_eval_interval if is_target_selected else None
            ),
            "selected_iteration": selected_iteration,
            "target_probability_correction": (
                args.prior_correction_strength > 0
                or common_bias_strength > 0
                or boundary_bias_strength > 0
            ),
            "target_prior_correction": args.prior_correction_strength > 0,
            "cross_scale_common_bias_suppression": (
                common_bias_strength > 0
                or boundary_bias_strength > 0
            ),
            "source_excess_suppression": common_bias_strength > 0,
            "boundary_attractor_suppression": (
                boundary_bias_strength > 0
            ),
            "final_target_evidence": (
                "complete_unlabeled_target_refresh_from_independent_scale_logits"
            ),
        },
        "source_trials": {
            name: len(dataset)
            for name, dataset in zip(
                prepared.domain_names, prepared.datasets, strict=True
            )
        },
        "source_natural_class_priors": {
            name: prior.cpu().tolist()
            for name, prior in zip(
                prepared.domain_names, source_class_priors, strict=True
            )
        },
        "source_effective_sampling_priors": {
            name: prior.cpu().tolist()
            for name, prior in zip(
                prepared.domain_names,
                effective_source_class_priors,
                strict=True,
            )
        },
        "target_subject": subject,
        "random_seed": seed,
        "model_parameters": parameter_count,
        "model": {
            "scales": list(spec.scales),
            "fusion_mode": spec.fusion_mode,
            "domain_mode": spec.domain_mode,
            "use_feature_pyramid": spec.use_feature_pyramid,
            "channel_attention": True,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "spatial_layers": args.spatial_layers,
            "temporal_layers": args.temporal_layers,
            "fusion_layers": args.fusion_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "spatial_topk": args.spatial_topk,
            "relation_strength": args.relation_strength,
            "pyramid_weight_floor": args.pyramid_weight_floor,
            "pyramid_residual_initial": args.pyramid_residual_initial,
            "fusion_output": (
                "one class-specific feature per emotion, scored by the "
                "matching classifier row"
            ),
            "issue_7_independence_boundary": (
                "scale logits precede all cross-scale context and relation fusion"
            ),
        },
        "optimization": {
            "source_batch_size": args.source_batch_size,
            "target_batch_size": args.target_batch_size,
            "source_balance_alpha": args.source_balance_alpha,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_iterations": args.warmup_iterations,
            "adaptation_warmup_iterations": args.adaptation_warmup_iterations,
            "adaptation_ramp_end": args.adaptation_ramp_end,
            "label_smoothing": args.label_smoothing,
            "domain_weight": spec.domain_weight,
            "prototype_weight": spec.prototype_weight,
            "pseudo_confidence_threshold": args.pseudo_confidence_threshold,
            "prototype_momentum": args.prototype_momentum,
            "reliability_temperature": args.reliability_temperature,
            "reliability_uniform_mix": args.reliability_uniform_mix,
            "relation_momentum": args.relation_momentum,
            "relation_temperature": args.relation_temperature,
            "relation_uniform_mix": args.relation_uniform_mix,
            "relation_strength": args.relation_strength,
            "pyramid_weight_floor": args.pyramid_weight_floor,
            "pyramid_residual_initial": args.pyramid_residual_initial,
            "prior_correction_strength": args.prior_correction_strength,
            "common_bias_strength": args.common_bias_strength,
            "effective_common_bias_strength": common_bias_strength,
            "common_bias_relative_tolerance": (
                args.common_bias_relative_tolerance
            ),
            "boundary_bias_strength": args.boundary_bias_strength,
            "effective_boundary_bias_strength": boundary_bias_strength,
            "boundary_bias_ratio_tolerance": (
                args.boundary_bias_ratio_tolerance
            ),
            "common_bias_max_adjustment": (
                args.common_bias_max_adjustment
            ),
            "prior_momentum": args.prior_momentum,
            "prior_ridge": args.prior_ridge,
            "prior_floor": args.prior_floor,
            "scale_classification_weight": args.scale_classification_weight,
            "gate_supervision_weight": args.gate_supervision_weight,
            "gate_teacher_temperature": args.gate_teacher_temperature,
            "consensus_jsd_threshold": args.consensus_jsd_threshold,
            "consensus_minimum_votes": args.consensus_minimum_votes,
            "removed_losses": [
                "supervised_contrastive",
                "information_maximization",
                "one_second_teacher_consistency",
            ],
        },
        "final_prototype_bank": (
            prototype_bank.state() if prototype_bank is not None else None
        ),
        "final_target_prior_estimator": target_prior_estimator.state(),
        "final_learned_scale_class_relation_residual": (
            model.scale_class_relation_residual.detach().cpu().tolist()
        ),
        "final_pyramid_residual_weight": float(
            torch.sigmoid(model.pyramid_residual_logit.detach()).cpu()
        ),
        "training_trace": training_trace,
        "target_evaluation_trace": target_evaluation_trace,
        "evaluation": evaluation,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary(experiment_dir)
    fused = evaluation["fused"]
    result_label = "Selected" if is_target_selected else "Final"
    tqdm.write(
        f"{result_label} {result_path} at iteration {selected_iteration}: "
        f"acc={fused['accuracy']:.4f}, "
        f"bal={fused['balanced_accuracy']:.4f}, f1={fused['macro_f1']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-conditional multiscale fixed-1000 UDA"
    )
    parser.add_argument("--experiment", choices=EXPERIMENT_ORDER, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--source-batch-size", type=int, default=24)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--source-balance-alpha", type=float, default=0.4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--adaptation-warmup-iterations", type=int, default=300)
    parser.add_argument("--adaptation-ramp-end", type=int, default=600)
    parser.add_argument("--pseudo-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--prototype-momentum", type=float, default=0.90)
    parser.add_argument("--reliability-temperature", type=float, default=0.15)
    parser.add_argument("--reliability-uniform-mix", type=float, default=0.10)
    parser.add_argument("--relation-momentum", type=float, default=0.99)
    parser.add_argument("--relation-temperature", type=float, default=0.25)
    parser.add_argument("--relation-uniform-mix", type=float, default=0.10)
    parser.add_argument("--relation-strength", type=float, default=1.0)
    parser.add_argument("--pyramid-weight-floor", type=float, default=0.10)
    parser.add_argument("--pyramid-residual-initial", type=float, default=0.05)
    parser.add_argument("--prior-correction-strength", type=float, default=0.0)
    parser.add_argument("--common-bias-strength", type=float, default=2.0)
    parser.add_argument(
        "--common-bias-relative-tolerance", type=float, default=0.10
    )
    parser.add_argument("--boundary-bias-strength", type=float, default=2.0)
    parser.add_argument(
        "--boundary-bias-ratio-tolerance", type=float, default=0.15
    )
    parser.add_argument(
        "--common-bias-max-adjustment", type=float, default=0.50
    )
    parser.add_argument("--prior-momentum", type=float, default=0.99)
    parser.add_argument("--prior-ridge", type=float, default=0.10)
    parser.add_argument("--prior-floor", type=float, default=0.03)
    parser.add_argument("--scale-classification-weight", type=float, default=0.30)
    parser.add_argument("--gate-supervision-weight", type=float, default=0.10)
    parser.add_argument("--gate-teacher-temperature", type=float, default=0.25)
    parser.add_argument("--consensus-jsd-threshold", type=float, default=0.15)
    parser.add_argument("--consensus-minimum-votes", type=int, default=2)
    parser.add_argument("--channel-attention-reduction", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--spatial-topk", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default=EVALUATION_PROTOCOL_FIXED_FINAL,
        help=(
            "fixed_final evaluates only iteration 1000; caga_target_best "
            "evaluates the labeled target at intervals and reports the "
            "highest-Accuracy iteration"
        ),
    )
    parser.add_argument(
        "--target-eval-interval",
        type=int,
        default=50,
        help="Target evaluation interval for caga_target_best",
    )
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args, spec: ExperimentSpec) -> None:
    if isinstance(args.target_subjects, str):
        args.target_subjects = _target_subjects(
            args.target_subjects, spec.target_subject_count
        )
    elif any(
        subject < 1 or subject > spec.target_subject_count
        for subject in args.target_subjects
    ):
        raise ValueError(
            f"target subjects must be within 1..{spec.target_subject_count}"
        )
    if args.source_batch_size < 2 or args.target_batch_size < 2:
        raise ValueError("Source and target batch sizes must be at least two")
    if args.d_model % args.num_heads:
        raise ValueError("d-model must be divisible by num-heads")
    if min(args.spatial_layers, args.temporal_layers, args.fusion_layers) < 1:
        raise ValueError("All model layer counts must be positive")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be within [0,1)")
    if not 0 <= args.source_balance_alpha <= 1:
        raise ValueError("source-balance-alpha must be within [0,1]")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be nonnegative")
    if (
        args.log_interval < 1
        or args.target_eval_interval < 1
        or args.gradient_clip <= 0
    ):
        raise ValueError(
            "log interval, target evaluation interval, and gradient clip "
            "must be positive"
        )
    if not (
        0
        <= args.adaptation_warmup_iterations
        <= args.adaptation_ramp_end
        <= FIXED_UDA_PROTOCOL.training_iterations
    ):
        raise ValueError(
            "adaptation warmup/ramp must satisfy 0 <= warmup <= ramp <= 1000"
        )
    if not 0 <= args.pseudo_confidence_threshold < 1:
        raise ValueError("pseudo-confidence-threshold must be within [0,1)")
    if not 0 <= args.prototype_momentum < 1:
        raise ValueError("prototype-momentum must be within [0,1)")
    if args.reliability_temperature <= 0:
        raise ValueError("reliability-temperature must be positive")
    if not 0 <= args.reliability_uniform_mix <= 1:
        raise ValueError("reliability-uniform-mix must be within [0,1]")
    if not 0 <= args.relation_momentum < 1:
        raise ValueError("relation-momentum must be within [0,1)")
    if args.relation_temperature <= 0 or args.gate_teacher_temperature <= 0:
        raise ValueError("relation and gate temperatures must be positive")
    if not 0 <= args.relation_uniform_mix <= 1:
        raise ValueError("relation-uniform-mix must be within [0,1]")
    if not 0 <= args.pyramid_weight_floor < 1:
        raise ValueError("pyramid-weight-floor must be within [0,1)")
    if not 0 < args.pyramid_residual_initial < 1:
        raise ValueError("pyramid-residual-initial must be within (0,1)")
    if args.prior_correction_strength < 0:
        raise ValueError("prior-correction-strength must be nonnegative")
    if min(
        args.common_bias_strength,
        args.common_bias_relative_tolerance,
        args.boundary_bias_strength,
        args.boundary_bias_ratio_tolerance,
        args.common_bias_max_adjustment,
    ) < 0:
        raise ValueError("common-bias parameters must be nonnegative")
    if not 0 <= args.prior_momentum < 1:
        raise ValueError("prior-momentum must be within [0,1)")
    if args.prior_ridge < 0:
        raise ValueError("prior-ridge must be nonnegative")
    if not 0 <= args.prior_floor < 1.0 / NUM_CLASSES:
        raise ValueError("prior-floor must be within [0,1/classes)")
    if min(
        args.relation_strength,
        args.scale_classification_weight,
        args.gate_supervision_weight,
        args.consensus_jsd_threshold,
    ) < 0:
        raise ValueError("relation/loss weights and JSD threshold are nonnegative")
    if args.consensus_minimum_votes < 1:
        raise ValueError("consensus-minimum-votes must be positive")
    if args.channel_attention_reduction < 1:
        raise ValueError("channel-attention-reduction must be positive")


def main() -> None:
    args = build_parser().parse_args()
    spec = EXPERIMENTS[args.experiment]
    validate_args(args, spec)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tqdm.write(
        f"Experiment={spec.name}: {spec.description}; "
        f"sources={spec.source_domains}; scales={spec.scales}; device={device}"
    )
    tqdm.write(
        "Method: relation-weighted logit anchor plus class-conditional "
        "weighted feature pyramid, source-calibrated common-bias suppression, "
        "and hard/soft boundary-attractor suppression; independent per-scale "
        "logits precede all bias estimation, cross-scale context, and fusion"
    )
    if args.evaluation_protocol == EVALUATION_PROTOCOL_CAGA_TARGET_BEST:
        tqdm.write(
            "Protocol: CAGA-SGA-style target selection, fixed 1000 training "
            f"iterations, target evaluation every {args.target_eval_interval} "
            "iterations, highest target Accuracy reported"
        )
    else:
        tqdm.write(
            "Protocol: fixed 1000 iterations, source-calibrated cross-scale "
            "common-bias suppression with a final complete unlabeled-target "
            "evidence refresh, diagnostic-only target-prior estimation, no "
            "checkpoint selection, one final labeled target evaluation"
        )
    prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
    for seed in args.random_seeds:
        for subject in args.target_subjects:
            run_fold(args, spec, prepared, seed, subject, device)


if __name__ == "__main__":
    main()
