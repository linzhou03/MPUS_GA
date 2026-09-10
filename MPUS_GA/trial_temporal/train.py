"""Class-conditional multiscale multi-source UDA training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, replace
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

from .anatomical_prior import physiology_metadata
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
from .subgroup_alignment import SelectiveSubgroupAlignment, SubgroupConfig
from .multiscale_coteaching import ClassConditionalCoTeaching, CoTeachingConfig


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
    pyramid_gate_mode: str = "global"
    use_pyramid_bias_guard: bool = False
    use_pyramid_gate_warmup: bool = False
    multiview_fusion_mode: str = "none"
    use_multiview_uncertainty: bool = False
    use_sign_aware_pyramid_guard: bool = False
    use_source_multiview_anchor: bool = False
    use_stable_pyramid_gate: bool = False
    use_temporal_msad: bool = False
    use_source_prototype_memory: bool = False
    use_relative_degradation_fusion: bool = False
    use_anatomical_regions: bool = False
    use_physiology_prior: bool = False
    use_physiology_reliability: bool = False
    use_balanced_multiscale_mixup: bool = False
    use_class_scale_adaptive_augmentation: bool = False
    use_class_scale_reliability_fusion: bool = False
    use_guarded_class_scale_reliability_fusion: bool = False
    use_domain_gap_scale_calibration: bool = False
    use_subgroup_alignment: bool = False
    use_multiscale_coteaching: bool = False
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

_FINAL_TRANSFER_DIRECTIONS = {
    "A": _TRANSFER_DIRECTIONS["A"],
    "B": _TRANSFER_DIRECTIONS["B"],
    "C": {
        "description": "SEED-IV to SEED-V",
        "source_domains": ("seed_iv",),
        "target_dataset": "seed_v",
        "target_subject_count": 16,
        "target_trials": 45,
    },
    "D": {
        "description": "SEED-V to SEED-IV",
        "source_domains": ("seed_v",),
        "target_dataset": "seed_iv",
        "target_subject_count": 15,
        "target_trials": 72,
    },
    "E": {
        "description": "SEED-IV to SEED-VII",
        "source_domains": ("seed_iv",),
        "target_dataset": "seed_vii",
        "target_subject_count": 20,
        "target_trials": 80,
    },
    "F": {
        "description": "SEED-VII to SEED-IV",
        "source_domains": ("seed_vii",),
        "target_dataset": "seed_iv",
        "target_subject_count": 15,
        "target_trials": 72,
    },
}

LEGACY_EXPERIMENT_ORDER = tuple(
    name
    for ablation in (*map(str, range(7)), "main")
    for name in (
        f"A_{ablation}" if ablation == "main" else f"A{ablation}",
        f"B_{ablation}" if ablation == "main" else f"B{ablation}",
    )
)

_G_DEFINITIONS = {
    "G0": {
        "description": "relation-logit anchor without feature pyramid",
        "use_feature_pyramid": False,
    },
    "G1": {
        "description": "class-static safe residual pyramid",
        "pyramid_gate_mode": "class_static",
        "use_pyramid_gate_warmup": True,
    },
    "G2": {
        "description": "sample-class adaptive safe residual pyramid",
        "pyramid_gate_mode": "sample_class",
        "use_pyramid_gate_warmup": True,
    },
    "G3": {
        "description": "bias-guarded sample-class safe residual pyramid",
        "pyramid_gate_mode": "sample_class",
        "use_pyramid_bias_guard": True,
        "use_pyramid_gate_warmup": True,
    },
}

G_EXPERIMENT_ORDER = tuple(
    f"{direction}_{variant}"
    for variant in _G_DEFINITIONS
    for direction in "AB"
)

_H_DEFINITIONS = {
    "H0": {
        "description": "relation-logit anchor without feature fusion",
        "use_feature_pyramid": False,
    },
    "H1": {
        "description": "class-query cross-scale feature fusion",
        "multiview_fusion_mode": "class_query",
    },
    "H2": {
        "description": "class-query fusion with low-rank scale interactions",
        "multiview_fusion_mode": "class_query_low_rank",
    },
    "H3": {
        "description": "uncertainty-aware low-rank multiview fusion",
        "multiview_fusion_mode": "class_query_low_rank",
        "use_multiview_uncertainty": True,
    },
    "H4": {
        "description": "sign-safe uncertainty-aware low-rank multiview fusion",
        "multiview_fusion_mode": "class_query_low_rank",
        "use_multiview_uncertainty": True,
        "use_sign_aware_pyramid_guard": True,
    },
    "H5": {
        "description": (
            "source-anchored stable sign-safe low-rank multiview fusion"
        ),
        "multiview_fusion_mode": "class_query_low_rank",
        "use_multiview_uncertainty": True,
        "use_sign_aware_pyramid_guard": True,
        "use_source_multiview_anchor": True,
        "use_stable_pyramid_gate": True,
    },
}

H_EXPERIMENT_ORDER = tuple(
    f"{direction}_{variant}"
    for variant in _H_DEFINITIONS
    for direction in "AB"
)

_R_DEFINITIONS = {
    "R0": {
        "description": "H5 stable source-anchored multiview baseline",
    },
    "R1": {
        "description": "R0 with anti-aliased temporal MSAD",
        "use_temporal_msad": True,
    },
    "R2": {
        "description": "R1 with class-balanced source multi-prototype memory",
        "use_temporal_msad": True,
        "use_source_prototype_memory": True,
    },
    "R3": {
        "description": (
            "R2 with relative-degradation-aware class-conditional fusion"
        ),
        "use_temporal_msad": True,
        "use_source_prototype_memory": True,
        "use_relative_degradation_fusion": True,
    },
    "R4": {
        "description": (
            "H5 with source multi-prototype memory only; no MSAD or "
            "relative degradation"
        ),
        "use_source_prototype_memory": True,
    },
}

R_EXPERIMENT_ORDER = tuple(
    f"{direction}_{variant}"
    for variant in _R_DEFINITIONS
    for direction in "AB"
)
N_EXPERIMENT_ORDER = tuple(
    f"{direction}_{variant}"
    for variant in ("N1", "N2", "N3")
    for direction in "AB"
)
D_EXPERIMENT_ORDER = tuple(
    f"{direction}_D{variant}"
    for variant in range(4)
    for direction in "AB"
)
FINAL_EXPERIMENT_ORDER = tuple(_FINAL_TRANSFER_DIRECTIONS)
EXPERIMENT_ORDER = (
    LEGACY_EXPERIMENT_ORDER
    + G_EXPERIMENT_ORDER
    + H_EXPERIMENT_ORDER
    + R_EXPERIMENT_ORDER
    + N_EXPERIMENT_ORDER
    + D_EXPERIMENT_ORDER
    + FINAL_EXPERIMENT_ORDER
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
    for variant, definition in _G_DEFINITIONS.items():
        for direction, transfer in _TRANSFER_DIRECTIONS.items():
            name = f"{direction}_{variant}"
            experiments[name] = ExperimentSpec(
                name=name,
                description=(
                    f"{transfer['description']}; {definition['description']}"
                ),
                transfer_direction=direction,
                ablation=variant,
                source_domains=transfer["source_domains"],
                scales=(1.0, 2.0, 4.0),
                fusion_mode="class_conditional",
                domain_mode="scale_conditional",
                use_prototypes=True,
                use_feature_pyramid=definition.get(
                    "use_feature_pyramid", True
                ),
                pyramid_gate_mode=definition.get(
                    "pyramid_gate_mode", "class_static"
                ),
                use_pyramid_bias_guard=definition.get(
                    "use_pyramid_bias_guard", False
                ),
                use_pyramid_gate_warmup=definition.get(
                    "use_pyramid_gate_warmup", False
                ),
                use_source_excess_suppression=True,
                use_boundary_attractor_suppression=True,
                target_dataset=transfer["target_dataset"],
                target_subject_count=transfer["target_subject_count"],
                target_trials=transfer["target_trials"],
            )
    for variant, definition in _H_DEFINITIONS.items():
        for direction, transfer in _TRANSFER_DIRECTIONS.items():
            name = f"{direction}_{variant}"
            experiments[name] = ExperimentSpec(
                name=name,
                description=(
                    f"{transfer['description']}; {definition['description']}"
                ),
                transfer_direction=direction,
                ablation=variant,
                source_domains=transfer["source_domains"],
                scales=(1.0, 2.0, 4.0),
                fusion_mode="class_conditional",
                domain_mode="scale_conditional",
                use_prototypes=True,
                use_feature_pyramid=definition.get(
                    "use_feature_pyramid", True
                ),
                pyramid_gate_mode=(
                    "class_static"
                    if not definition.get("use_feature_pyramid", True)
                    else "sample_class"
                ),
                use_pyramid_gate_warmup=definition.get(
                    "use_feature_pyramid", True
                ),
                multiview_fusion_mode=definition.get(
                    "multiview_fusion_mode", "none"
                ),
                use_multiview_uncertainty=definition.get(
                    "use_multiview_uncertainty", False
                ),
                use_sign_aware_pyramid_guard=definition.get(
                    "use_sign_aware_pyramid_guard", False
                ),
                use_source_multiview_anchor=definition.get(
                    "use_source_multiview_anchor", False
                ),
                use_stable_pyramid_gate=definition.get(
                    "use_stable_pyramid_gate", False
                ),
                use_source_excess_suppression=True,
                use_boundary_attractor_suppression=True,
                target_dataset=transfer["target_dataset"],
                target_subject_count=transfer["target_subject_count"],
                target_trials=transfer["target_trials"],
            )
    for variant, definition in _R_DEFINITIONS.items():
        for direction, transfer in _TRANSFER_DIRECTIONS.items():
            name = f"{direction}_{variant}"
            experiments[name] = ExperimentSpec(
                name=name,
                description=(
                    f"{transfer['description']}; {definition['description']}"
                ),
                transfer_direction=direction,
                ablation=variant,
                source_domains=transfer["source_domains"],
                scales=(1.0, 2.0, 4.0),
                fusion_mode="class_conditional",
                domain_mode="scale_conditional",
                use_prototypes=True,
                use_feature_pyramid=True,
                pyramid_gate_mode="sample_class",
                use_pyramid_gate_warmup=True,
                multiview_fusion_mode="class_query_low_rank",
                use_multiview_uncertainty=True,
                use_sign_aware_pyramid_guard=True,
                use_source_multiview_anchor=True,
                use_stable_pyramid_gate=True,
                use_temporal_msad=definition.get(
                    "use_temporal_msad", False
                ),
                use_source_prototype_memory=definition.get(
                    "use_source_prototype_memory", False
                ),
                use_relative_degradation_fusion=definition.get(
                    "use_relative_degradation_fusion", False
                ),
                use_source_excess_suppression=True,
                use_boundary_attractor_suppression=True,
                target_dataset=transfer["target_dataset"],
                target_subject_count=transfer["target_subject_count"],
                target_trials=transfer["target_trials"],
            )
    for direction in "AB":
        baseline = experiments[f"{direction}_R2"]
        experiments[f"{direction}_N1"] = replace(
            baseline,
            name=f"{direction}_N1",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; R2 with "
                "fixed exhaustive anatomical region-balanced aggregation"
            ),
            ablation="N1",
            use_anatomical_regions=True,
        )
        experiments[f"{direction}_N2"] = replace(
            experiments[f"{direction}_N1"],
            name=f"{direction}_N2",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; N1 with "
                "source-standardised continuous raw-DE physiology descriptors"
            ),
            ablation="N2",
            use_physiology_prior=True,
        )
        experiments[f"{direction}_N3"] = replace(
            baseline,
            name=f"{direction}_N3",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; R2 with "
                "subject-robust compact physiology pseudo-label reliability"
            ),
            ablation="N3",
            use_physiology_reliability=True,
        )
        experiments[f"{direction}_D0"] = replace(
            baseline,
            name=f"{direction}_D0",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; exact R2 "
                "control for class-balanced multiscale augmentation"
            ),
            ablation="D0",
        )
        experiments[f"{direction}_D1"] = replace(
            experiments[f"{direction}_D0"],
            name=f"{direction}_D1",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; R2 with "
                "class-balanced cross-subject scale-synchronous MixUp"
            ),
            ablation="D1",
            use_balanced_multiscale_mixup=True,
        )
        experiments[f"{direction}_D2"] = replace(
            experiments[f"{direction}_D1"],
            name=f"{direction}_D2",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; D1 with "
                "source-real class-by-scale competence weighting"
            ),
            ablation="D2",
            use_class_scale_adaptive_augmentation=True,
        )
        experiments[f"{direction}_D3"] = replace(
            experiments[f"{direction}_D2"],
            name=f"{direction}_D3",
            description=(
                f"{_TRANSFER_DIRECTIONS[direction]['description']}; D2 with "
                "target uncertainty-aware class-scale reliability fusion"
            ),
            ablation="D3",
            use_class_scale_reliability_fusion=True,
        )
    for direction, transfer in _FINAL_TRANSFER_DIRECTIONS.items():
        experiments[direction] = ExperimentSpec(
            name=direction,
            description=(
                f"{transfer['description']}; final domain-gap-calibrated "
                "guarded class-balanced multiscale adaptation"
            ),
            transfer_direction=direction,
            ablation="final",
            source_domains=transfer["source_domains"],
            scales=(1.0, 2.0, 4.0),
            fusion_mode="class_conditional",
            domain_mode="scale_conditional",
            use_prototypes=True,
            use_feature_pyramid=True,
            pyramid_gate_mode="sample_class",
            use_pyramid_gate_warmup=True,
            multiview_fusion_mode="class_query_low_rank",
            use_multiview_uncertainty=True,
            use_sign_aware_pyramid_guard=True,
            use_source_multiview_anchor=True,
            use_stable_pyramid_gate=True,
            use_temporal_msad=True,
            use_source_prototype_memory=True,
            use_balanced_multiscale_mixup=True,
            use_class_scale_adaptive_augmentation=True,
            use_guarded_class_scale_reliability_fusion=True,
            use_domain_gap_scale_calibration=True,
            use_source_excess_suppression=True,
            use_boundary_attractor_suppression=True,
            target_dataset=transfer["target_dataset"],
            target_subject_count=transfer["target_subject_count"],
            target_trials=transfer["target_trials"],
        )
    return experiments


EXPERIMENTS = _build_experiments()


def subgroup_experiment(direction: str) -> ExperimentSpec:
    """R2 backbone with selective subgroup contrast replacing centroid alignment."""
    if direction not in _FINAL_TRANSFER_DIRECTIONS:
        raise ValueError("R2 subgroup experiments use directions A through F")
    transfer = _FINAL_TRANSFER_DIRECTIONS[direction]
    return replace(
        EXPERIMENTS["A_R2"],
        name=direction,
        description=f"{transfer['description']}; R2 selective latent subgroup contrast",
        transfer_direction=direction,
        ablation="r2_subgroup",
        source_domains=transfer["source_domains"],
        target_dataset=transfer["target_dataset"],
        target_subject_count=transfer["target_subject_count"],
        target_trials=transfer["target_trials"],
        use_subgroup_alignment=True,
        prototype_weight=0.0,
    )


def coteaching_experiment(direction: str) -> ExperimentSpec:
    return replace(subgroup_experiment(direction), ablation="r2_coteaching",
                   description=f"{direction}; R2 class-conditional multiscale peer teaching and selective alignment",
                   use_multiscale_coteaching=True,
                   prototype_weight=EXPERIMENTS["A_R2"].prototype_weight)


def subgroup_config(args) -> SubgroupConfig:
    cls = CoTeachingConfig if getattr(args, "method", None) == "r2_coteaching" else SubgroupConfig
    return cls(**{name: (value if (value := getattr(args, f"subgroup_{name}", None)) is not None else default)
                  for name, default in asdict(cls()).items()})


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


def _batch_physiology_to_device(
    batch: dict, device: torch.device
) -> dict[str, torch.Tensor] | None:
    if "physiology" not in batch:
        return None
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch["physiology"].items()
    }


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


def physiology_pseudo_label_reliability(
    eeg_probability: torch.Tensor,
    physiology_probability: torch.Tensor,
    temperature: float = 0.15,
    floor: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded agreement weight; physiology cannot create or relabel a target."""

    if eeg_probability.ndim != 2 or physiology_probability.shape != (
        eeg_probability.shape
    ):
        raise ValueError("EEG and physiology probabilities must be [batch,classes]")
    if temperature <= 0:
        raise ValueError("physiology reliability temperature must be positive")
    if not 0 <= floor <= 1:
        raise ValueError("physiology reliability floor must be within [0,1]")
    eeg = eeg_probability.detach().clamp_min(1e-8)
    physiology = physiology_probability.detach().clamp_min(1e-8)
    midpoint = 0.5 * (eeg + physiology)
    js_divergence = 0.5 * (
        (eeg * (torch.log(eeg) - torch.log(midpoint))).sum(dim=-1)
        + (
            physiology
            * (torch.log(physiology) - torch.log(midpoint))
        ).sum(dim=-1)
    )
    agreement = torch.exp(-js_divergence / temperature)
    return floor + (1.0 - floor) * agreement, js_divergence


@dataclass(frozen=True)
class BalancedMultiscaleMixup:
    """One label-preserving augmented trial triplet for every source row."""

    features: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    labels: torch.Tensor
    sample_weight: torch.Tensor
    partner_index: torch.Tensor
    mixing_coefficient: torch.Tensor


@torch.no_grad()
def _same_class_cross_subject_partners(
    labels: torch.Tensor,
    subjects: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose same-class partners, preferring another source subject."""

    if labels.ndim != 1 or subjects.shape != labels.shape:
        raise ValueError("MixUp labels and subjects must be one-dimensional")
    partner = torch.arange(len(labels), device=labels.device)
    valid = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    for class_index in range(NUM_CLASSES):
        rows = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        for row in rows:
            candidates = rows[subjects[rows] != subjects[row]]
            if not len(candidates):
                candidates = rows[rows != row]
            if len(candidates):
                choice = torch.randint(
                    len(candidates), (1,), device=labels.device
                )
                partner[row] = candidates[choice]
                valid[row] = True
    return partner, valid


@torch.no_grad()
def balanced_multiscale_mixup(
    batch: dict,
    device: torch.device,
    natural_class_prior: torch.Tensor,
    beta_alpha: float,
    rarity_power: float,
) -> BalancedMultiscaleMixup:
    """Generate a synchronized 1/2/4-second minority-class source view.

    A row uses one same-class partner and one mixing coefficient at every
    temporal scale. Partner sequences are linearly resampled to the anchor
    trial length before mixing, so masks and physical trial coverage remain
    aligned even when clips contain different numbers of windows.
    """

    if beta_alpha <= 0 or rarity_power <= 0:
        raise ValueError("MixUp beta alpha and rarity power must be positive")
    features, masks = _batch_to_device(batch, device)
    labels = batch["y"].to(device, non_blocking=True)
    subjects = batch["subject_id"].to(device, non_blocking=True)
    prior = natural_class_prior.to(device=device, dtype=torch.float32)
    if prior.shape != (NUM_CLASSES,) or torch.any(prior <= 0):
        raise ValueError("Natural source class prior must be positive [classes]")

    partner, has_partner = _same_class_cross_subject_partners(labels, subjects)
    rarity = (prior.max() / prior).pow(rarity_power) - 1.0
    rarity = rarity / rarity.max().clamp_min(1e-8)
    sample_weight = rarity[labels] * has_partner.to(rarity.dtype)

    concentration = torch.tensor(beta_alpha, device=device)
    coefficient = torch.distributions.Beta(
        concentration, concentration
    ).sample((len(labels),))
    coefficient = torch.maximum(coefficient, 1.0 - coefficient)
    coefficient = torch.where(
        sample_weight > 0,
        coefficient,
        torch.ones_like(coefficient),
    )

    augmented = {key: value.clone() for key, value in features.items()}
    for key, value in features.items():
        mask = masks[key]
        for row in torch.nonzero(
            sample_weight > 0, as_tuple=False
        ).flatten().tolist():
            other = int(partner[row])
            anchor_length = int(mask[row].sum())
            partner_length = int(mask[other].sum())
            partner_sequence = value[other, :partner_length]
            if partner_length != anchor_length:
                flattened = partner_sequence.reshape(partner_length, -1)
                partner_sequence = F.interpolate(
                    flattened.transpose(0, 1).unsqueeze(0),
                    size=anchor_length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1).reshape(
                    anchor_length, value.shape[2], value.shape[3]
                )
            mix = coefficient[row].to(value.dtype)
            augmented[key][row, :anchor_length] = (
                mix * value[row, :anchor_length]
                + (1.0 - mix) * partner_sequence
            )
    return BalancedMultiscaleMixup(
        features=augmented,
        masks=masks,
        labels=labels,
        sample_weight=sample_weight,
        partner_index=partner,
        mixing_coefficient=coefficient,
    )


class ClassScaleCompetenceEMA:
    """Source-real-only class-by-scale competence for augmentation and fusion."""

    def __init__(
        self,
        scale_count: int,
        class_count: int,
        momentum: float,
        uniform_mix: float,
        device: torch.device,
    ) -> None:
        if scale_count < 1 or class_count < 2:
            raise ValueError("Competence matrix dimensions are invalid")
        if not 0 <= momentum < 1 or not 0 <= uniform_mix <= 1:
            raise ValueError("Competence EMA parameters are invalid")
        self.scale_count = int(scale_count)
        self.class_count = int(class_count)
        self.momentum = float(momentum)
        self.uniform_mix = float(uniform_mix)
        self.value = torch.full(
            (scale_count, class_count),
            1.0 / class_count,
            device=device,
        )
        self.initialized = torch.zeros(
            class_count, dtype=torch.bool, device=device
        )
        self.updates = torch.zeros(
            class_count, dtype=torch.long, device=device
        )

    @torch.no_grad()
    def update(self, scale_logits: torch.Tensor, labels: torch.Tensor) -> None:
        if scale_logits.ndim != 3 or scale_logits.shape[1:] != (
            self.scale_count,
            self.class_count,
        ):
            raise ValueError("Scale logits do not match competence matrix")
        probability = F.softmax(scale_logits.detach(), dim=-1)
        for class_index in range(self.class_count):
            selected = labels == class_index
            if not torch.any(selected):
                continue
            observed = probability[selected, :, class_index].mean(dim=0)
            if self.initialized[class_index]:
                self.value[:, class_index].mul_(self.momentum).add_(
                    observed, alpha=1.0 - self.momentum
                )
            else:
                self.value[:, class_index].copy_(observed)
                self.initialized[class_index] = True
            self.updates[class_index] += 1

    @torch.no_grad()
    def reliability(self) -> torch.Tensor:
        normalized = self.value.clamp_min(1e-6)
        normalized = normalized / normalized.sum(dim=0, keepdim=True)
        uniform = torch.full_like(normalized, 1.0 / self.scale_count)
        return (
            (1.0 - self.uniform_mix) * normalized
            + self.uniform_mix * uniform
        )

    @torch.no_grad()
    def deficit_weight(self, floor: float, power: float) -> torch.Tensor:
        if floor <= 0 or power <= 0:
            raise ValueError("Competence deficit parameters must be positive")
        difficulty = (1.0 - self.value.clamp(0.0, 1.0) + floor).pow(power)
        weight = difficulty / difficulty.mean(dim=0, keepdim=True).clamp_min(
            1e-8
        )
        return weight.clamp(0.5, 2.0)

    def state(self) -> dict:
        return {
            "true_class_probability_by_scale_class": self.value.cpu().tolist(),
            "normalized_reliability_by_scale_class": (
                self.reliability().cpu().tolist()
            ),
            "initialized_by_class": self.initialized.cpu().tolist(),
            "updates_by_class": self.updates.cpu().tolist(),
            "momentum": self.momentum,
            "uniform_mix": self.uniform_mix,
            "statistics_source": "real_labeled_source_batches_only",
        }


class DomainGapScaleCalibrator:
    """Bounded scale transferability from source and unlabeled target evidence."""

    def __init__(
        self,
        scale_count: int,
        class_count: int,
        momentum: float,
        agreement_momentum: float,
        spread_weight: float,
        gap_strength: float,
        uniform_mix: float,
        max_log_deviation: float,
        minimum_updates: int,
        device: torch.device,
    ) -> None:
        if scale_count < 2 or class_count < 2:
            raise ValueError("Domain-gap calibration requires multiple scales")
        if not 0 <= momentum < 1 or not 0 <= agreement_momentum < 1:
            raise ValueError("Domain-gap EMA momenta must be within [0,1)")
        if spread_weight < 0 or gap_strength < 0:
            raise ValueError("Domain-gap weights must be nonnegative")
        if not 0 <= uniform_mix <= 1 or max_log_deviation <= 0:
            raise ValueError("Domain-gap calibration bounds are invalid")
        if minimum_updates < 1:
            raise ValueError("Domain-gap minimum updates must be positive")
        self.scale_count = int(scale_count)
        self.class_count = int(class_count)
        self.momentum = float(momentum)
        self.agreement_momentum = float(agreement_momentum)
        self.spread_weight = float(spread_weight)
        self.gap_strength = float(gap_strength)
        self.uniform_mix = float(uniform_mix)
        self.max_log_deviation = float(max_log_deviation)
        self.minimum_updates = int(minimum_updates)
        self.domain_gap = torch.zeros(scale_count, device=device)
        self.domain_initialized = torch.tensor(False, device=device)
        self.domain_updates = torch.zeros((), dtype=torch.long, device=device)
        self.target_agreement = torch.full(
            (scale_count, class_count), 1.0 / class_count, device=device
        )
        self.agreement_initialized = torch.zeros(
            (scale_count, class_count), dtype=torch.bool, device=device
        )
        self.agreement_updates = torch.zeros(
            (scale_count, class_count), dtype=torch.long, device=device
        )

    @torch.no_grad()
    def update(
        self,
        source_scale_embeddings: list[torch.Tensor],
        target_scale_embeddings: torch.Tensor,
        target_scale_logits: torch.Tensor,
        confidence_threshold: float,
    ) -> None:
        """Update without target labels or fused target predictions."""

        if not source_scale_embeddings:
            raise ValueError("At least one source embedding batch is required")
        source = torch.cat(
            [embedding.detach() for embedding in source_scale_embeddings], dim=0
        )
        target = target_scale_embeddings.detach()
        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("Scale embeddings must be [batch,scales,features]")
        if source.shape[1:] != target.shape[1:]:
            raise ValueError("Source and target scale embeddings must match")
        if source.shape[1] != self.scale_count:
            raise ValueError("Scale embedding count does not match calibrator")
        if target_scale_logits.shape[:2] != target.shape[:2] or (
            target_scale_logits.ndim != 3
            or target_scale_logits.shape[2] != self.class_count
        ):
            raise ValueError("Target scale logits do not match calibrator")
        if not 0 <= confidence_threshold < 1:
            raise ValueError("Confidence threshold must be within [0,1)")

        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        source_variance = source.var(dim=0, unbiased=False)
        target_variance = target.var(dim=0, unbiased=False)
        pooled_variance = 0.5 * (source_variance + target_variance)
        mean_gap = (
            (source_mean - target_mean).square()
            / pooled_variance.clamp_min(1e-6)
        ).mean(dim=-1)
        spread_gap = (
            torch.log(source_variance.clamp_min(1e-6))
            - torch.log(target_variance.clamp_min(1e-6))
        ).square().mean(dim=-1)
        observed_gap = mean_gap + self.spread_weight * spread_gap
        if bool(self.domain_initialized):
            self.domain_gap.mul_(self.momentum).add_(
                observed_gap, alpha=1.0 - self.momentum
            )
        else:
            self.domain_gap.copy_(observed_gap)
            self.domain_initialized.fill_(True)
        self.domain_updates += 1

        probability = F.softmax(target_scale_logits.detach(), dim=-1)
        probability_sum = probability.sum(dim=1)
        for scale_index in range(self.scale_count):
            peer_probability = (
                probability_sum - probability[:, scale_index]
            ) / (self.scale_count - 1)
            peer_confidence, peer_label = peer_probability.max(dim=-1)
            confident = peer_confidence >= confidence_threshold
            for class_index in range(self.class_count):
                selected = confident & (peer_label == class_index)
                if not torch.any(selected):
                    continue
                weight = peer_confidence[selected]
                observed = (
                    probability[selected, scale_index, class_index] * weight
                ).sum() / weight.sum().clamp_min(1e-8)
                if self.agreement_initialized[scale_index, class_index]:
                    self.target_agreement[scale_index, class_index].mul_(
                        self.agreement_momentum
                    ).add_(observed, alpha=1.0 - self.agreement_momentum)
                else:
                    self.target_agreement[scale_index, class_index].copy_(
                        observed
                    )
                    self.agreement_initialized[scale_index, class_index] = True
                self.agreement_updates[scale_index, class_index] += 1

    @torch.no_grad()
    def reliability(
        self, source_competence: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return normalized, uniform-shrunk class-by-scale reliability."""

        if int(self.domain_updates) < self.minimum_updates:
            return None
        uniform = torch.full_like(
            self.target_agreement, 1.0 / self.scale_count
        )
        if source_competence is None:
            source_reliability = uniform
        else:
            if source_competence.shape != self.target_agreement.shape:
                raise ValueError("Source competence does not match calibrator")
            source_reliability = source_competence.clamp_min(1e-8)
            source_reliability = source_reliability / source_reliability.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-8)

        relative_gap = self.domain_gap / self.domain_gap.mean().clamp_min(1e-8)
        domain_reliability = F.softmax(
            -self.gap_strength * relative_gap, dim=0
        ).unsqueeze(-1).expand_as(self.target_agreement)
        target_reliability = self.target_agreement.clamp_min(1e-8)
        target_reliability = target_reliability / target_reliability.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-8)
        log_score = (
            torch.log(source_reliability)
            + torch.log(domain_reliability)
            + torch.log(target_reliability)
        ) / 3.0
        raw = F.softmax(log_score, dim=0)
        bounded_log_relative = torch.log(
            (raw * self.scale_count).clamp_min(1e-8)
        ).clamp(-self.max_log_deviation, self.max_log_deviation)
        bounded = F.softmax(bounded_log_relative, dim=0)
        return (1.0 - self.uniform_mix) * bounded + self.uniform_mix * uniform

    @torch.no_grad()
    def domain_reliability(self) -> torch.Tensor | None:
        if int(self.domain_updates) < self.minimum_updates:
            return None
        relative_gap = self.domain_gap / self.domain_gap.mean().clamp_min(1e-8)
        return F.softmax(-self.gap_strength * relative_gap, dim=0)

    def state(self, source_competence: torch.Tensor | None = None) -> dict:
        calibrated = self.reliability(source_competence)
        domain_reliability = self.domain_reliability()
        return {
            "domain_gap_by_scale": self.domain_gap.cpu().tolist(),
            "domain_reliability_by_scale": (
                domain_reliability.cpu().tolist()
                if domain_reliability is not None
                else None
            ),
            "target_leave_one_out_agreement_by_scale_class": (
                self.target_agreement.cpu().tolist()
            ),
            "calibrated_reliability_by_scale_class": (
                calibrated.cpu().tolist() if calibrated is not None else None
            ),
            "domain_initialized": bool(self.domain_initialized),
            "domain_updates": int(self.domain_updates),
            "agreement_initialized_by_scale_class": (
                self.agreement_initialized.cpu().tolist()
            ),
            "agreement_updates_by_scale_class": (
                self.agreement_updates.cpu().tolist()
            ),
            "momentum": self.momentum,
            "agreement_momentum": self.agreement_momentum,
            "spread_weight": self.spread_weight,
            "gap_strength": self.gap_strength,
            "uniform_mix": self.uniform_mix,
            "max_log_deviation": self.max_log_deviation,
            "minimum_updates": self.minimum_updates,
            "statistics_source": (
                "source_and_unlabeled_target_embeddings_plus_independent_"
                "target_scale_logits; no_target_labels_or_fused_predictions"
            ),
        }


@torch.no_grad()
def _combine_scale_class_reliability(
    first: torch.Tensor | None,
    second: torch.Tensor | None,
) -> torch.Tensor | None:
    """Geometric-mean two source-only reliability matrices."""

    if first is None:
        return second
    if second is None:
        return first
    if first.shape != second.shape:
        raise ValueError("Scale-class reliability matrices must have equal shape")
    combined = torch.sqrt(first.clamp_min(1e-8) * second.clamp_min(1e-8))
    return combined / combined.sum(dim=0, keepdim=True).clamp_min(1e-8)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values)
    if values.shape != weights.shape:
        raise ValueError("Weighted mean values and weights must have equal shape")
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


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
        sample_weight: torch.Tensor | None = None,
    ) -> float:
        probability = probability.detach()
        confidence, pseudo_label = probability.max(dim=1)
        valid = confidence >= confidence_threshold
        if valid_mask is not None:
            if valid_mask.shape != valid.shape:
                raise ValueError("valid_mask must be [batch]")
            valid = valid & valid_mask.detach().bool().to(valid.device)
        if sample_weight is not None and sample_weight.shape != valid.shape:
            raise ValueError("sample_weight must be [batch]")
        confidence_weight = (
            (confidence - confidence_threshold)
            / max(1.0 - confidence_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        if sample_weight is not None:
            confidence_weight = confidence_weight * sample_weight.detach().to(
                confidence_weight
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


class SourceMultiPrototypeMemory:
    """Equal-capacity source-only memory for every scale and emotion class."""

    def __init__(
        self,
        scale_count: int,
        class_count: int,
        slots: int,
        feature_dim: int,
        momentum: float,
        device: torch.device,
    ) -> None:
        if min(scale_count, class_count, slots, feature_dim) < 1:
            raise ValueError("source memory dimensions must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("source memory momentum must be within [0,1)")
        self.momentum = float(momentum)
        self.memory = torch.zeros(
            scale_count,
            class_count,
            slots,
            feature_dim,
            device=device,
        )
        self.initialized = torch.zeros(
            scale_count,
            class_count,
            slots,
            dtype=torch.bool,
            device=device,
        )
        self.updates = torch.zeros(
            scale_count,
            class_count,
            slots,
            dtype=torch.long,
            device=device,
        )
        self.update_calls = 0

    @torch.no_grad()
    def update_source(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Update only from source embeddings paired with source true labels."""

        if embeddings.ndim != 3 or embeddings.shape[1] != self.memory.shape[0]:
            raise ValueError("source memory embeddings must be [batch,scales,d]")
        if embeddings.shape[2] != self.memory.shape[-1]:
            raise ValueError("source memory feature dimension mismatch")
        if labels.shape != (embeddings.shape[0],):
            raise ValueError("source memory labels must be [batch]")
        detached = F.normalize(embeddings.detach(), dim=-1)
        for scale_index in range(self.memory.shape[0]):
            for class_index in range(self.memory.shape[1]):
                vectors = detached[labels == class_index, scale_index]
                if vectors.numel() == 0:
                    continue
                available = torch.where(
                    ~self.initialized[scale_index, class_index]
                )[0]
                fill_count = min(len(available), len(vectors))
                if fill_count:
                    slots = available[:fill_count]
                    self.memory[scale_index, class_index, slots] = vectors[
                        :fill_count
                    ]
                    self.initialized[scale_index, class_index, slots] = True
                    self.updates[scale_index, class_index, slots] += 1
                    vectors = vectors[fill_count:]
                if vectors.numel() == 0:
                    continue
                ready = self.initialized[scale_index, class_index]
                ready_indices = torch.where(ready)[0]
                prototypes = self.memory[
                    scale_index, class_index, ready_indices
                ]
                assignment = (vectors @ prototypes.transpose(0, 1)).argmax(
                    dim=1
                )
                for local_index, slot_index in enumerate(ready_indices):
                    selected = assignment == local_index
                    if not torch.any(selected):
                        continue
                    observation = F.normalize(
                        vectors[selected].mean(dim=0), dim=0
                    )
                    stored = self.memory[
                        scale_index, class_index, slot_index
                    ]
                    stored.mul_(self.momentum).add_(
                        observation, alpha=1.0 - self.momentum
                    )
                    stored.copy_(F.normalize(stored, dim=0))
                    self.updates[scale_index, class_index, slot_index] += int(
                        selected.sum()
                    )
        self.update_calls += 1

    @torch.no_grad()
    def state(self, freeze_iteration: int) -> dict:
        return {
            "policy": (
                "source_true_labels_only; target_query_only; "
                f"frozen_after_iteration_{freeze_iteration}"
            ),
            "shape": list(self.memory.shape),
            "equal_slots_per_scale_class": self.memory.shape[2],
            "initialized": self.initialized.cpu().tolist(),
            "updates": self.updates.cpu().tolist(),
            "update_calls": self.update_calls,
            "prototype_norms": self.memory.norm(dim=-1).cpu().tolist(),
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


def _pyramid_gate_supervision_loss(
    relation_logits: torch.Tensor,
    pyramid_logits: torch.Tensor,
    raw_gate: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    """Open a class gate only when its source-label logit change is useful.

    For the true class, a positive pyramid-minus-anchor change is useful. For
    every false class the sign is reversed, because lowering that logit
    improves the true-class margin. The detached soft teacher supervises only
    the gate; it cannot feed labels into the pyramid logits or scale heads.
    """

    if not raw_gate.requires_grad:
        return relation_logits.sum() * 0.0
    if temperature <= 0 or margin < 0:
        raise ValueError("pyramid gate teacher parameters are invalid")
    if raw_gate.shape != relation_logits.shape or pyramid_logits.shape != (
        relation_logits.shape
    ):
        raise ValueError("pyramid gate tensors must all be [batch,classes]")
    true_class = F.one_hot(labels, num_classes=relation_logits.shape[1]).bool()
    delta = (pyramid_logits - relation_logits).detach()
    useful_delta = torch.where(true_class, delta, -delta)
    teacher = torch.sigmoid((useful_delta - margin) / temperature)
    return F.binary_cross_entropy(
        raw_gate.clamp(1e-6, 1.0 - 1e-6), teacher
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
    pyramid_gate_supervision_weight: float = 0.05,
    pyramid_gate_sparsity_weight: float = 0.005,
    pyramid_gate_teacher_temperature: float = 0.10,
    pyramid_gate_teacher_margin: float = 0.05,
    source_prototype_memory: SourceMultiPrototypeMemory | None = None,
    physiology_classification_weight: float = 0.05,
    physiology_reliability_temperature: float = 0.15,
    physiology_reliability_floor: float = 0.50,
    class_scale_competence: ClassScaleCompetenceEMA | None = None,
    domain_gap_scale_calibrator: DomainGapScaleCalibrator | None = None,
    mixup_warmup_iterations: int = 50,
    mixup_beta_alpha: float = 0.40,
    mixup_rarity_power: float = 0.50,
    mixup_loss_weight: float = 0.25,
    competence_deficit_floor: float = 0.25,
    competence_deficit_power: float = 1.0,
    subgroup_alignment: SelectiveSubgroupAlignment | None = None,
) -> dict:
    if "y" in target_batch:
        raise RuntimeError("Target adaptation batch unexpectedly contains labels")
    if spec.use_subgroup_alignment != (subgroup_alignment is not None):
        raise RuntimeError("Subgroup experiment and training controller must agree")
    if spec.use_multiscale_coteaching != isinstance(subgroup_alignment, ClassConditionalCoTeaching):
        raise RuntimeError("Co-teaching experiment and training controller must agree")
    if not spec.use_source_excess_suppression:
        common_bias_strength = 0.0
    if not spec.use_boundary_attractor_suppression:
        boundary_bias_strength = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ramp = _adaptation_ramp(
        iteration, adaptation_warmup_iterations, adaptation_ramp_end
    )
    pyramid_gate_ramp = ramp if spec.use_pyramid_gate_warmup else 1.0
    adaptation_active = iteration > adaptation_warmup_iterations
    prototype_reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None
        else None
    )
    competence_reliability = (
        class_scale_competence.reliability()
        if class_scale_competence is not None
        else None
    )
    calibrated_reliability = (
        domain_gap_scale_calibrator.reliability(competence_reliability)
        if domain_gap_scale_calibrator is not None
        else None
    )
    reliability = prototype_reliability
    if spec.use_class_scale_reliability_fusion:
        reliability = _combine_scale_class_reliability(
            prototype_reliability, competence_reliability
        )
    elif spec.use_domain_gap_scale_calibration:
        reliability = _combine_scale_class_reliability(
            prototype_reliability, calibrated_reliability
        )
    guarded_reliability = (
        calibrated_reliability
        if calibrated_reliability is not None
        else competence_reliability
    )
    if not spec.use_guarded_class_scale_reliability_fusion:
        guarded_reliability = None
    base_source_multiview_anchor = (
        prototype_bank.source_relation_weights().mean(dim=0)
        if prototype_bank is not None
        and spec.use_source_multiview_anchor
        else None
    )
    source_multiview_anchor = (
        _combine_scale_class_reliability(
            base_source_multiview_anchor, competence_reliability
        )
        if spec.use_class_scale_reliability_fusion
        else base_source_multiview_anchor
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
    pyramid_bias_risk = None
    if (
        spec.use_pyramid_bias_guard
        or spec.use_sign_aware_pyramid_guard
    ) and common_bias_components is not None:
        if common_bias_max_adjustment > 0:
            pyramid_bias_risk = (
                -common_bias_components["combined"]
                / common_bias_max_adjustment
            ).clamp(0.0, 1.0)
        else:
            pyramid_bias_risk = torch.zeros(
                NUM_CLASSES, device=device, dtype=torch.float32
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
        physiology = _batch_physiology_to_device(batch, device)
        labels = batch["y"].to(device, non_blocking=True)
        source_labels.append(labels)
        source_outputs.append(
            model(
                x,
                mask,
                grl_alpha=ramp,
                scale_class_reliability=reliability,
                multiview_source_anchor=source_multiview_anchor,
                pyramid_gate_ramp=pyramid_gate_ramp,
                source_prototype_memory=(
                    source_prototype_memory.memory
                    if source_prototype_memory is not None
                    else None
                ),
                source_prototype_initialized=(
                    source_prototype_memory.initialized
                    if source_prototype_memory is not None
                    else None
                ),
                physiology_by_scale=physiology,
                class_scale_reliability_ramp=0.0,
                guarded_scale_class_reliability=guarded_reliability,
            )
        )
    target_x, target_mask = _batch_to_device(target_batch, device)
    target_physiology = _batch_physiology_to_device(target_batch, device)
    target_output = model(
        target_x,
        target_mask,
        grl_alpha=ramp,
        scale_class_reliability=reliability,
        multiview_source_anchor=source_multiview_anchor,
        class_logit_adjustment=target_logit_adjustment,
        pyramid_gate_ramp=pyramid_gate_ramp,
        pyramid_bias_risk=pyramid_bias_risk,
        source_prototype_memory=(
            source_prototype_memory.memory
            if source_prototype_memory is not None
            else None
        ),
        source_prototype_initialized=(
            source_prototype_memory.initialized
            if source_prototype_memory is not None
            else None
        ),
        physiology_by_scale=target_physiology,
        class_scale_reliability_ramp=(
            ramp
            if (
                spec.use_class_scale_reliability_fusion
                or spec.use_guarded_class_scale_reliability_fusion
            )
            else 0.0
        ),
        guarded_scale_class_reliability=guarded_reliability,
    )

    mixup_active = (
        spec.use_balanced_multiscale_mixup
        and iteration > mixup_warmup_iterations
        and mixup_loss_weight > 0
    )
    augmented_views: list[BalancedMultiscaleMixup | None] = []
    augmented_outputs: list[dict | None] = []
    for domain_index, batch in enumerate(source_batches):
        if not mixup_active:
            augmented_views.append(None)
            augmented_outputs.append(None)
            continue
        augmented = balanced_multiscale_mixup(
            batch,
            device,
            source_class_priors[domain_index],
            mixup_beta_alpha,
            mixup_rarity_power,
        )
        augmented_views.append(augmented)
        if not torch.any(augmented.sample_weight > 0):
            augmented_outputs.append(None)
            continue
        augmented_outputs.append(
            model(
                augmented.features,
                augmented.masks,
                grl_alpha=0.0,
                compute_domain=False,
                scale_class_reliability=reliability,
                multiview_source_anchor=source_multiview_anchor,
                pyramid_gate_ramp=pyramid_gate_ramp,
                source_prototype_memory=(
                    source_prototype_memory.memory
                    if source_prototype_memory is not None
                    else None
                ),
                source_prototype_initialized=(
                    source_prototype_memory.initialized
                    if source_prototype_memory is not None
                    else None
                ),
                class_scale_reliability_ramp=0.0,
                guarded_scale_class_reliability=guarded_reliability,
            )
        )

    target_consensus = independent_scale_consensus(
        target_output["calibrated_scale_logits"],
        pseudo_confidence_threshold,
        consensus_jsd_threshold,
        consensus_minimum_votes,
    )
    if spec.use_physiology_reliability:
        if target_output["physiology_probability"] is None:
            raise RuntimeError("Physiology reliability output is unavailable")
        target_physiology_weight, target_physiology_jsd = (
            physiology_pseudo_label_reliability(
                target_consensus.probability,
                target_output["physiology_probability"],
                physiology_reliability_temperature,
                physiology_reliability_floor,
            )
        )
    else:
        target_physiology_weight = target_consensus.probability.new_ones(
            len(target_consensus.probability)
        )
        target_physiology_jsd = target_consensus.probability.new_zeros(
            len(target_consensus.probability)
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
    if spec.use_physiology_reliability:
        physiology_classification_by_source = torch.stack(
            [
                F.cross_entropy(
                    output["physiology_logits"].reshape(-1, NUM_CLASSES),
                    labels[:, None]
                    .expand(-1, output["physiology_logits"].shape[1])
                    .reshape(-1),
                    label_smoothing=label_smoothing,
                )
                for output, labels in zip(
                    source_outputs, source_labels, strict=True
                )
            ]
        )
    else:
        physiology_classification_by_source = torch.zeros_like(
            fused_classification_by_source
        ) + target_output["logits"].sum() * 0.0
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
    if spec.use_feature_pyramid and spec.pyramid_gate_mode != "global":
        pyramid_gate_supervision_by_source = torch.stack(
            [
                _pyramid_gate_supervision_loss(
                    output["relation_weighted_logits"],
                    output["pyramid_logits"],
                    output["pyramid_raw_gate"],
                    labels,
                    pyramid_gate_teacher_temperature,
                    pyramid_gate_teacher_margin,
                )
                for output, labels in zip(
                    source_outputs, source_labels, strict=True
                )
            ]
        )
        pyramid_gate_sparsity_loss = torch.stack(
            [output["pyramid_raw_gate"].mean() for output in source_outputs]
            + [target_output["pyramid_raw_gate"].mean()]
        ).mean()
    else:
        zero = target_output["logits"].sum() * 0.0
        pyramid_gate_supervision_by_source = torch.zeros_like(
            fused_classification_by_source
        ) + zero
        pyramid_gate_sparsity_loss = zero
    source_weights = (
        prototype_bank.source_weights(source_class_priors)
        if prototype_bank is not None and adaptation_active
        else torch.full_like(
            classification_by_source, 1.0 / len(classification_by_source)
        )
    )
    classification_loss = (classification_by_source * source_weights).sum()
    physiology_classification_loss = (
        physiology_classification_by_source * source_weights
    ).sum()
    gate_supervision_loss = (
        gate_supervision_by_source * source_weights
    ).sum()
    pyramid_gate_supervision_loss = (
        pyramid_gate_supervision_by_source * source_weights
    ).sum()
    augmentation_fused_by_source = []
    augmentation_scale_by_source = []
    augmentation_by_source = []
    competence_deficit = (
        class_scale_competence.deficit_weight(
            competence_deficit_floor, competence_deficit_power
        )
        if class_scale_competence is not None
        and spec.use_class_scale_adaptive_augmentation
        else None
    )
    zero_augmentation = target_output["logits"].sum() * 0.0
    for augmented, output in zip(
        augmented_views, augmented_outputs, strict=True
    ):
        if augmented is None or output is None:
            augmentation_fused_by_source.append(zero_augmentation)
            augmentation_scale_by_source.append(zero_augmentation)
            augmentation_by_source.append(zero_augmentation)
            continue
        fused_loss = F.cross_entropy(
            output["logits"],
            augmented.labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        fused_loss = _weighted_mean(fused_loss, augmented.sample_weight)
        scale_logits = output["scale_logits"]
        expanded_labels = augmented.labels[:, None].expand(
            -1, scale_logits.shape[1]
        )
        scale_loss = F.cross_entropy(
            scale_logits.reshape(-1, NUM_CLASSES),
            expanded_labels.reshape(-1),
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(scale_logits.shape[:2])
        scale_weight = augmented.sample_weight[:, None].expand_as(scale_loss)
        if competence_deficit is not None:
            adaptive_weight = competence_deficit[:, augmented.labels].transpose(
                0, 1
            )
            scale_weight = scale_weight * adaptive_weight
        scale_loss = _weighted_mean(scale_loss, scale_weight)
        augmentation_fused_by_source.append(fused_loss)
        augmentation_scale_by_source.append(scale_loss)
        augmentation_by_source.append(
            fused_loss + scale_classification_weight * scale_loss
        )
    augmentation_fused_by_source = torch.stack(
        augmentation_fused_by_source
    )
    augmentation_scale_by_source = torch.stack(
        augmentation_scale_by_source
    )
    augmentation_by_source = torch.stack(augmentation_by_source)
    augmentation_loss = (augmentation_by_source * source_weights).sum()
    domain_loss, domain_by_scale = _domain_loss(
        source_outputs,
        target_output,
        source_labels,
        target_consensus,
    )
    subgroup_loss = target_output["logits"].sum() * 0.0
    teaching_loss = subgroup_loss
    subgroup_record = None
    subgroup_coefficient = 0.0
    teaching_coefficient = 0.0
    centroid_strength = None
    if subgroup_alignment is not None:
        subgroup_loss, subgroup_record = subgroup_alignment.loss(
            source_outputs, source_batches, target_output, target_batch, iteration
        )
        subgroup_coefficient = subgroup_alignment.config.weight * subgroup_alignment.config.ramp(iteration)
        if spec.use_multiscale_coteaching:
            centroid_strength = subgroup_alignment.centroid_strength(iteration)
            teaching_loss = subgroup_alignment.teaching_loss
            teaching_coefficient = subgroup_alignment.config.teaching_weight * subgroup_alignment.config.ramp(iteration)
    if prototype_bank is not None and adaptation_active and (
        not spec.use_subgroup_alignment or spec.use_multiscale_coteaching
    ):
        prototype_loss, pseudo_coverage = (
            class_conditional_prototype_alignment_loss(
                [output["scale_embeddings"] for output in source_outputs],
                source_labels,
                target_output["scale_embeddings"],
                target_consensus.probability,
                prototype_bank.joint_weights(),
                pseudo_confidence_threshold,
                target_consensus.valid_mask,
                (
                    target_physiology_weight
                    if spec.use_physiology_reliability
                    else None
                ),
                scale_class_strength=centroid_strength,
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
        + subgroup_coefficient * subgroup_loss
        + teaching_coefficient * teaching_loss
        + mixup_loss_weight * augmentation_loss
        + physiology_classification_weight
        * physiology_classification_loss
        + gate_supervision_weight * gate_supervision_loss
        + ramp
        * (
            pyramid_gate_supervision_weight
            * pyramid_gate_supervision_loss
            + pyramid_gate_sparsity_weight * pyramid_gate_sparsity_loss
        )
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
    if subgroup_alignment is not None:
        subgroup_alignment.after_step(model)

    if class_scale_competence is not None:
        for output, labels in zip(
            source_outputs, source_labels, strict=True
        ):
            class_scale_competence.update(output["scale_logits"], labels)

    if domain_gap_scale_calibrator is not None and adaptation_active:
        domain_gap_scale_calibrator.update(
            [output["scale_embeddings"] for output in source_outputs],
            target_output["scale_embeddings"],
            target_output["scale_logits"],
            pseudo_confidence_threshold,
        )

    # R memory has a strict information boundary: source true labels only.
    # It is updated during the source warmup, then frozen for all subsequent
    # target querying and final evaluation. Target labels/pseudo-labels never
    # select a memory class or modify a slot.
    if (
        source_prototype_memory is not None
        and iteration <= adaptation_warmup_iterations
    ):
        for output, labels in zip(
            source_outputs, source_labels, strict=True
        ):
            source_prototype_memory.update_source(
                output["scale_embeddings"], labels
            )

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
                sample_weight=(
                    target_physiology_weight
                    if spec.use_physiology_reliability
                    else None
                ),
            )
        else:
            pseudo_coverage = 0.0
        prototype_reliability = prototype_bank.scale_class_reliability()
        competence_reliability = (
            class_scale_competence.reliability()
            if class_scale_competence is not None
            else None
        )
        calibrated_reliability = (
            domain_gap_scale_calibrator.reliability(competence_reliability)
            if domain_gap_scale_calibrator is not None
            else None
        )
        reliability = prototype_reliability
        if spec.use_class_scale_reliability_fusion:
            reliability = _combine_scale_class_reliability(
                prototype_reliability, competence_reliability
            )
        elif spec.use_domain_gap_scale_calibration:
            reliability = _combine_scale_class_reliability(
                prototype_reliability, calibrated_reliability
            )
        source_weights = prototype_bank.source_weights(source_class_priors)

    mean_gate = target_output["scale_class_weight"].detach().mean(dim=0)
    mean_pyramid_gate = target_output[
        "pyramid_scale_class_weight"
    ].detach().mean(dim=0)
    mean_pyramid_residual_gate = target_output[
        "pyramid_residual_gate"
    ].detach().mean(dim=0)
    mean_pyramid_raw_gate = target_output[
        "pyramid_raw_gate"
    ].detach().mean(dim=0)
    mean_pyramid_guard = target_output[
        "pyramid_guard_factor"
    ].detach().mean(dim=0)
    mean_pyramid_unsupported_excess = target_output[
        "pyramid_unsupported_excess"
    ].detach().mean(dim=0)
    mean_multiview_attention = target_output[
        "multiview_token_attention"
    ].detach().mean(dim=0)
    mean_multiview_dynamic_attention = target_output[
        "multiview_dynamic_attention"
    ].detach().mean(dim=0)
    multiview_source_anchor = target_output[
        "multiview_source_anchor"
    ].detach()
    mean_multiview_uncertainty = target_output[
        "multiview_token_uncertainty"
    ].detach().mean(dim=0)
    mean_multiview_conflict = target_output[
        "multiview_token_conflict"
    ].detach().mean(dim=0)
    mean_rda_msad_filter = target_output[
        "rda_msad_filter_weight"
    ].detach().mean(dim=0)
    rda_msad_gate = target_output["rda_msad_gate"].detach()
    mean_rda_memory_distance = target_output[
        "rda_memory_distance"
    ].detach().mean(dim=0)
    mean_rda_memory_scale_weight = target_output[
        "rda_memory_scale_weight"
    ].detach().mean(dim=0)
    mean_rda_memory_gate = target_output[
        "rda_memory_gate"
    ].detach().mean(dim=0)
    mean_sample_scale_class_reliability = target_output[
        "sample_scale_class_reliability"
    ].detach().mean(dim=0)
    mean_guarded_reliability_activation = target_output[
        "guarded_reliability_activation"
    ].detach().mean(dim=0)
    mean_guarded_log_adjustment = target_output[
        "guarded_scale_class_log_adjustment"
    ].detach().abs().mean(dim=0)
    active_mixup_rows = sum(
        int((view.sample_weight > 0).sum())
        for view in augmented_views
        if view is not None
    )
    total_mixup_rows = sum(
        len(view.sample_weight)
        for view in augmented_views
        if view is not None
    )
    active_coefficients = torch.cat(
        [
            view.mixing_coefficient[view.sample_weight > 0]
            for view in augmented_views
            if view is not None and torch.any(view.sample_weight > 0)
        ]
    ) if active_mixup_rows else target_output["logits"].new_ones(1)
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
        "multiscale_mixup": float(augmentation_loss.detach()),
        "multiscale_mixup_fused": float(
            (augmentation_fused_by_source * source_weights).sum().detach()
        ),
        "multiscale_mixup_scale": float(
            (augmentation_scale_by_source * source_weights).sum().detach()
        ),
        "multiscale_mixup_active": mixup_active,
        "multiscale_mixup_active_fraction": (
            active_mixup_rows / total_mixup_rows if total_mixup_rows else 0.0
        ),
        "multiscale_mixup_mean_anchor_coefficient": float(
            active_coefficients.mean().detach()
        ),
        "physiology_classification": float(
            physiology_classification_loss.detach()
        ),
        "fused_classification": float(
            (fused_classification_by_source * source_weights).sum().detach()
        ),
        "scale_classification": float(
            (scale_classification_by_source * source_weights).sum().detach()
        ),
        "gate_supervision": float(gate_supervision_loss.detach()),
        "pyramid_gate_supervision": float(
            pyramid_gate_supervision_loss.detach()
        ),
        "pyramid_gate_sparsity": float(pyramid_gate_sparsity_loss.detach()),
        "classification_by_source": [
            float(value) for value in classification_by_source.detach()
        ],
        "domain": float(domain_loss.detach()),
        "domain_by_scale": domain_by_scale,
        "prototype": float(prototype_loss.detach()),
        "subgroup_contrast": float(subgroup_loss.detach()),
        "subgroup_contrast_coefficient": subgroup_coefficient,
        "peer_teaching": float(teaching_loss.detach()),
        "peer_teaching_coefficient": teaching_coefficient,
        "subgroup_alignment": subgroup_record,
        "pseudo_label_coverage": pseudo_coverage,
        "physiology_reliability_active": (
            spec.use_physiology_reliability and adaptation_active
        ),
        "mean_target_physiology_reliability": float(
            target_physiology_weight.mean().detach()
        ),
        "min_target_physiology_reliability": float(
            target_physiology_weight.min().detach()
        ),
        "mean_target_physiology_js_divergence": float(
            target_physiology_jsd.mean().detach()
        ),
        "target_scale_consensus": consensus_statistics,
        "source_weights": [float(value) for value in source_weights.detach()],
        "mean_target_scale_class_gate": mean_gate.cpu().tolist(),
        "mean_target_sample_scale_class_reliability": (
            mean_sample_scale_class_reliability.cpu().tolist()
        ),
        "mean_target_guarded_reliability_activation": (
            mean_guarded_reliability_activation.cpu().tolist()
        ),
        "mean_target_guarded_abs_log_adjustment": (
            mean_guarded_log_adjustment.cpu().tolist()
        ),
        "source_real_class_scale_competence": (
            class_scale_competence.value.cpu().tolist()
            if class_scale_competence is not None
            else None
        ),
        "source_real_class_scale_reliability": (
            class_scale_competence.reliability().cpu().tolist()
            if class_scale_competence is not None
            else None
        ),
        "domain_gap_scale_calibration": (
            domain_gap_scale_calibrator.state(competence_reliability)
            if domain_gap_scale_calibrator is not None
            else None
        ),
        "augmentation_class_scale_deficit_weight": (
            competence_deficit.cpu().tolist()
            if competence_deficit is not None
            else None
        ),
        "mean_target_pyramid_scale_class_gate": (
            mean_pyramid_gate.cpu().tolist()
        ),
        "mean_target_pyramid_residual_gate_by_class": (
            mean_pyramid_residual_gate.cpu().tolist()
        ),
        "mean_target_pyramid_raw_gate_by_class": (
            mean_pyramid_raw_gate.cpu().tolist()
        ),
        "mean_target_pyramid_guard_by_class": (
            mean_pyramid_guard.cpu().tolist()
        ),
        "mean_target_pyramid_unsupported_excess_by_class": (
            mean_pyramid_unsupported_excess.cpu().tolist()
        ),
        "mean_target_multiview_token_attention": (
            mean_multiview_attention.cpu().tolist()
        ),
        "mean_target_multiview_dynamic_attention": (
            mean_multiview_dynamic_attention.cpu().tolist()
        ),
        "source_multiview_token_anchor": (
            multiview_source_anchor.cpu().tolist()
        ),
        "mean_target_multiview_token_uncertainty": (
            mean_multiview_uncertainty.cpu().tolist()
        ),
        "mean_target_multiview_token_conflict": (
            mean_multiview_conflict.cpu().tolist()
        ),
        "mean_rda_msad_filter_weight": mean_rda_msad_filter.cpu().tolist(),
        "rda_msad_gate": rda_msad_gate.cpu().tolist(),
        "mean_target_rda_memory_distance": (
            mean_rda_memory_distance.cpu().tolist()
        ),
        "mean_target_rda_memory_scale_weight": (
            mean_rda_memory_scale_weight.cpu().tolist()
        ),
        "mean_target_rda_memory_gate": mean_rda_memory_gate.cpu().tolist(),
        "source_prototype_memory_updates_active": (
            source_prototype_memory is not None
            and iteration <= adaptation_warmup_iterations
        ),
        "pyramid_residual_weight": float(
            target_output["pyramid_residual_weight"].detach()
        ),
        "learned_scale_class_relation_residual": (
            model.scale_class_relation_residual.detach().cpu().tolist()
        ),
        "pyramid_bias_risk": (
            pyramid_bias_risk.detach().cpu().tolist()
            if pyramid_bias_risk is not None
            else [0.0] * NUM_CLASSES
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
        physiology = _batch_physiology_to_device(batch, device)
        physiology_arguments = (
            {"physiology_by_scale": physiology}
            if physiology is not None
            else {}
        )
        output = model(
            x,
            mask,
            compute_domain=False,
            **physiology_arguments,
        )
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
    source_prototype_memory: SourceMultiPrototypeMemory | None = None,
    class_scale_competence: ClassScaleCompetenceEMA | None = None,
    domain_gap_scale_calibrator: DomainGapScaleCalibrator | None = None,
) -> dict:
    model.eval()
    probabilities = []
    scale_probabilities = []
    corrected_scale_probabilities = []
    scale_gates = []
    pyramid_scale_gates = []
    pyramid_residual_gates = []
    pyramid_raw_gates = []
    pyramid_guard_factors = []
    pyramid_unsupported_excesses = []
    multiview_token_attentions = []
    multiview_dynamic_attentions = []
    multiview_source_anchors = []
    multiview_token_uncertainties = []
    multiview_token_conflicts = []
    rda_msad_filter_weights = []
    rda_msad_gates = []
    rda_memory_distances = []
    rda_memory_valid_masks = []
    rda_memory_scale_weights = []
    rda_memory_gates = []
    sample_scale_class_reliabilities = []
    guarded_reliability_activations = []
    guarded_log_adjustments = []
    labels = []
    prototype_reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None
        else None
    )
    competence_reliability = (
        class_scale_competence.reliability()
        if class_scale_competence is not None
        else None
    )
    calibrated_reliability = (
        domain_gap_scale_calibrator.reliability(competence_reliability)
        if domain_gap_scale_calibrator is not None
        else None
    )
    reliability = prototype_reliability
    if model.use_class_scale_reliability_fusion:
        reliability = _combine_scale_class_reliability(
            prototype_reliability, competence_reliability
        )
    elif domain_gap_scale_calibrator is not None:
        reliability = _combine_scale_class_reliability(
            prototype_reliability, calibrated_reliability
        )
    guarded_reliability = (
        (
            calibrated_reliability
            if calibrated_reliability is not None
            else competence_reliability
        )
        if model.use_guarded_class_scale_reliability_fusion
        else None
    )
    base_source_multiview_anchor = (
        prototype_bank.source_relation_weights().mean(dim=0)
        if prototype_bank is not None
        and model.multiview_source_anchor_mix > 0
        else None
    )
    source_multiview_anchor = (
        _combine_scale_class_reliability(
            base_source_multiview_anchor, competence_reliability
        )
        if model.use_class_scale_reliability_fusion
        else base_source_multiview_anchor
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
    pyramid_bias_risk = None
    if (
        model.use_pyramid_bias_guard
        or model.use_sign_aware_pyramid_guard
    ) and common_bias_components is not None:
        if common_bias_max_adjustment > 0:
            pyramid_bias_risk = (
                -common_bias_components["combined"]
                / common_bias_max_adjustment
            ).clamp(0.0, 1.0)
        else:
            pyramid_bias_risk = torch.zeros(
                NUM_CLASSES, device=device, dtype=torch.float32
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
            physiology = _batch_physiology_to_device(batch, device)
            physiology_arguments = (
                {"physiology_by_scale": physiology}
                if physiology is not None
                else {}
            )
            output = model(
                x,
                mask,
                compute_domain=False,
                scale_class_reliability=reliability,
                multiview_source_anchor=source_multiview_anchor,
                class_logit_adjustment=target_logit_adjustment,
                pyramid_gate_ramp=1.0,
                pyramid_bias_risk=pyramid_bias_risk,
                source_prototype_memory=(
                    source_prototype_memory.memory
                    if source_prototype_memory is not None
                    else None
                ),
                source_prototype_initialized=(
                    source_prototype_memory.initialized
                    if source_prototype_memory is not None
                    else None
                ),
                class_scale_reliability_ramp=(
                    1.0
                    if (
                        model.use_class_scale_reliability_fusion
                        or model.use_guarded_class_scale_reliability_fusion
                    )
                    else 0.0
                ),
                guarded_scale_class_reliability=guarded_reliability,
                **physiology_arguments,
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
            pyramid_residual_gates.append(
                output["pyramid_residual_gate"].cpu().numpy()
            )
            pyramid_raw_gates.append(output["pyramid_raw_gate"].cpu().numpy())
            pyramid_guard_factors.append(
                output["pyramid_guard_factor"].cpu().numpy()
            )
            pyramid_unsupported_excesses.append(
                output["pyramid_unsupported_excess"].cpu().numpy()
            )
            multiview_token_attentions.append(
                output["multiview_token_attention"].cpu().numpy()
            )
            multiview_dynamic_attentions.append(
                output["multiview_dynamic_attention"].cpu().numpy()
            )
            multiview_source_anchors.append(
                output["multiview_source_anchor"].cpu().numpy()
            )
            multiview_token_uncertainties.append(
                output["multiview_token_uncertainty"].cpu().numpy()
            )
            multiview_token_conflicts.append(
                output["multiview_token_conflict"].cpu().numpy()
            )
            rda_msad_filter_weights.append(
                output["rda_msad_filter_weight"].cpu().numpy()
            )
            rda_msad_gates.append(output["rda_msad_gate"].cpu().numpy())
            rda_memory_distances.append(
                output["rda_memory_distance"].cpu().numpy()
            )
            rda_memory_valid_masks.append(
                output["rda_memory_valid"].cpu().numpy()
            )
            rda_memory_scale_weights.append(
                output["rda_memory_scale_weight"].cpu().numpy()
            )
            rda_memory_gates.append(
                output["rda_memory_gate"].cpu().numpy()
            )
            sample_scale_class_reliabilities.append(
                output["sample_scale_class_reliability"].cpu().numpy()
            )
            guarded_reliability_activations.append(
                output["guarded_reliability_activation"].cpu().numpy()
            )
            guarded_log_adjustments.append(
                output["guarded_scale_class_log_adjustment"].cpu().numpy()
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
    pyramid_residual_gate_array = np.concatenate(pyramid_residual_gates)
    pyramid_raw_gate_array = np.concatenate(pyramid_raw_gates)
    pyramid_guard_array = np.concatenate(pyramid_guard_factors)
    pyramid_unsupported_excess_array = np.concatenate(
        pyramid_unsupported_excesses
    )
    multiview_attention_array = np.concatenate(multiview_token_attentions)
    multiview_dynamic_attention_array = np.concatenate(
        multiview_dynamic_attentions
    )
    multiview_source_anchor_array = np.stack(multiview_source_anchors)
    multiview_uncertainty_array = np.concatenate(
        multiview_token_uncertainties
    )
    multiview_conflict_array = np.concatenate(multiview_token_conflicts)
    rda_msad_filter_array = np.concatenate(rda_msad_filter_weights)
    rda_msad_gate_array = np.stack(rda_msad_gates)
    rda_memory_distance_array = np.concatenate(rda_memory_distances)
    rda_memory_valid_array = np.concatenate(rda_memory_valid_masks)
    rda_memory_scale_weight_array = np.concatenate(
        rda_memory_scale_weights
    )
    rda_memory_gate_array = np.concatenate(rda_memory_gates)
    sample_scale_class_reliability_array = np.concatenate(
        sample_scale_class_reliabilities
    )
    guarded_reliability_activation_array = np.concatenate(
        guarded_reliability_activations
    )
    guarded_log_adjustment_array = np.concatenate(
        guarded_log_adjustments
    )
    multiview_token_names = list(model.scale_keys)
    if model.multiview_fusion_mode == "class_query_low_rank":
        multiview_token_names.extend(
            f"{model.scale_keys[left]}x{model.scale_keys[right]}"
            for left, right in model.multiview_pairs
        )
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
            pyramid_residual_gate_array.mean()
        ),
        "mean_pyramid_residual_gate_by_class": (
            pyramid_residual_gate_array.mean(axis=0).tolist()
        ),
        "mean_pyramid_raw_gate_by_class": (
            pyramid_raw_gate_array.mean(axis=0).tolist()
        ),
        "mean_pyramid_guard_by_class": (
            pyramid_guard_array.mean(axis=0).tolist()
        ),
        "mean_pyramid_unsupported_excess_by_class": (
            pyramid_unsupported_excess_array.mean(axis=0).tolist()
        ),
        "multiview_token_names": multiview_token_names,
        "mean_multiview_token_attention": (
            multiview_attention_array.mean(axis=0).tolist()
        ),
        "mean_multiview_dynamic_attention": (
            multiview_dynamic_attention_array.mean(axis=0).tolist()
        ),
        "source_multiview_token_anchor": (
            multiview_source_anchor_array.mean(axis=0).tolist()
        ),
        "mean_multiview_token_uncertainty": (
            multiview_uncertainty_array.mean(axis=0).tolist()
        ),
        "mean_multiview_token_conflict": (
            multiview_conflict_array.mean(axis=0).tolist()
        ),
        "mean_rda_msad_filter_weight": (
            rda_msad_filter_array.mean(axis=0).tolist()
        ),
        "mean_rda_msad_gate": rda_msad_gate_array.mean(axis=0).tolist(),
        "mean_rda_memory_distance": (
            rda_memory_distance_array.mean(axis=0).tolist()
        ),
        "rda_memory_valid_fraction": (
            rda_memory_valid_array.mean(axis=0).tolist()
        ),
        "mean_rda_memory_scale_weight": (
            rda_memory_scale_weight_array.mean(axis=0).tolist()
        ),
        "mean_rda_memory_gate": rda_memory_gate_array.mean(axis=0).tolist(),
        "mean_sample_scale_class_reliability": (
            sample_scale_class_reliability_array.mean(axis=0).tolist()
        ),
        "mean_guarded_reliability_activation": (
            guarded_reliability_activation_array.mean(axis=0).tolist()
        ),
        "mean_guarded_abs_log_adjustment": (
            np.abs(guarded_log_adjustment_array).mean(axis=0).tolist()
        ),
        "source_real_class_scale_competence": (
            class_scale_competence.value.cpu().tolist()
            if class_scale_competence is not None
            else None
        ),
        "source_real_class_scale_reliability": (
            class_scale_competence.reliability().cpu().tolist()
            if class_scale_competence is not None
            else None
        ),
        "domain_gap_scale_calibration": (
            domain_gap_scale_calibrator.state(competence_reliability)
            if domain_gap_scale_calibrator is not None
            else None
        ),
        "pyramid_bias_risk": (
            pyramid_bias_risk.cpu().tolist()
            if pyramid_bias_risk is not None
            else [0.0] * NUM_CLASSES
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
        if spec.use_subgroup_alignment:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("experiment_spec") != asdict(spec) or (
                saved.get("subgroup_alignment") or {}
            ).get("config") != asdict(subgroup_config(args)):
                raise ValueError("Existing result uses another method/config; use a new result root")
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
        pyramid_gate_mode=spec.pyramid_gate_mode,
        use_pyramid_bias_guard=spec.use_pyramid_bias_guard,
        pyramid_residual_clip=args.pyramid_residual_clip,
        pyramid_guard_strength=args.pyramid_guard_strength,
        pyramid_guard_tolerance=args.pyramid_guard_tolerance,
        multiview_fusion_mode=spec.multiview_fusion_mode,
        use_multiview_uncertainty=spec.use_multiview_uncertainty,
        use_sign_aware_pyramid_guard=(
            spec.use_sign_aware_pyramid_guard
        ),
        multiview_low_rank=args.multiview_low_rank,
        multiview_uncertainty_strength=(
            args.multiview_uncertainty_strength
        ),
        multiview_conflict_strength=args.multiview_conflict_strength,
        multiview_source_anchor_mix=(
            args.multiview_source_anchor_mix
            if spec.use_source_multiview_anchor
            else 0.0
        ),
        pyramid_gate_shrinkage=(
            args.pyramid_gate_shrinkage
            if spec.use_stable_pyramid_gate
            else 0.0
        ),
        pyramid_gate_ceiling=(
            args.pyramid_gate_ceiling
            if spec.use_stable_pyramid_gate
            else 1.0
        ),
        use_temporal_msad=spec.use_temporal_msad,
        use_source_prototype_memory=spec.use_source_prototype_memory,
        use_relative_degradation_fusion=(
            spec.use_relative_degradation_fusion
        ),
        rda_memory_topk=args.rda_memory_topk,
        rda_memory_temperature=args.rda_memory_temperature,
        rda_memory_strength=args.rda_memory_strength,
        rda_degradation_strength=args.rda_degradation_strength,
        rda_msad_strength=args.rda_msad_strength,
        use_anatomical_regions=spec.use_anatomical_regions,
        anatomical_region_max_strength=args.anatomical_region_max_strength,
        use_physiology_prior=spec.use_physiology_prior,
        physiology_max_strength=args.physiology_max_strength,
        use_physiology_reliability=spec.use_physiology_reliability,
        physiology_reliability_hidden=args.physiology_reliability_hidden,
        use_class_scale_reliability_fusion=(
            spec.use_class_scale_reliability_fusion
        ),
        use_guarded_class_scale_reliability_fusion=(
            spec.use_guarded_class_scale_reliability_fusion
        ),
        class_scale_fusion_strength=args.class_scale_fusion_strength,
        class_scale_entropy_strength=args.class_scale_entropy_strength,
        class_scale_conflict_strength=args.class_scale_conflict_strength,
        class_scale_reliability_floor=args.class_scale_reliability_floor,
        class_scale_guard_max_log_adjustment=(
            args.class_scale_guard_max_log_adjustment
        ),
        class_scale_guard_consensus_floor=(
            args.class_scale_guard_consensus_floor
        ),
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
    source_prototype_memory = None
    if spec.use_source_prototype_memory:
        source_prototype_memory = SourceMultiPrototypeMemory(
            len(spec.scales),
            NUM_CLASSES,
            args.rda_memory_slots,
            args.d_model,
            args.rda_memory_momentum,
            device,
        )
    class_scale_competence = None
    if (
        spec.use_class_scale_adaptive_augmentation
        or spec.use_class_scale_reliability_fusion
        or spec.use_guarded_class_scale_reliability_fusion
        or spec.use_domain_gap_scale_calibration
    ):
        class_scale_competence = ClassScaleCompetenceEMA(
            len(spec.scales),
            NUM_CLASSES,
            args.class_scale_competence_momentum,
            args.class_scale_competence_uniform_mix,
            device,
        )
    domain_gap_scale_calibrator = None
    if spec.use_domain_gap_scale_calibration:
        domain_gap_scale_calibrator = DomainGapScaleCalibrator(
            len(spec.scales),
            NUM_CLASSES,
            args.domain_gap_momentum,
            args.target_scale_agreement_momentum,
            args.domain_gap_spread_weight,
            args.domain_gap_strength,
            args.domain_gap_uniform_mix,
            args.domain_gap_max_log_deviation,
            args.domain_gap_min_updates,
            device,
        )
    subgroup_alignment = (
        (ClassConditionalCoTeaching if spec.use_multiscale_coteaching else SelectiveSubgroupAlignment)(
            model, subgroup_config(args), device)
        if spec.use_subgroup_alignment else None
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
            args.pyramid_gate_supervision_weight,
            args.pyramid_gate_sparsity_weight,
            args.pyramid_gate_teacher_temperature,
            args.pyramid_gate_teacher_margin,
            source_prototype_memory=source_prototype_memory,
            physiology_classification_weight=(
                args.physiology_classification_weight
            ),
            physiology_reliability_temperature=(
                args.physiology_reliability_temperature
            ),
            physiology_reliability_floor=args.physiology_reliability_floor,
            class_scale_competence=class_scale_competence,
            domain_gap_scale_calibrator=domain_gap_scale_calibrator,
            mixup_warmup_iterations=args.mixup_warmup_iterations,
            mixup_beta_alpha=args.mixup_beta_alpha,
            mixup_rarity_power=args.mixup_rarity_power,
            mixup_loss_weight=args.mixup_loss_weight,
            competence_deficit_floor=args.competence_deficit_floor,
            competence_deficit_power=args.competence_deficit_power,
            subgroup_alignment=subgroup_alignment,
        )
        if iteration == 1 or iteration % args.log_interval == 0 or iteration == iterations:
            training_trace.append(record)
            postfix = {
                "total": f"{record['total']:.3f}",
                "cls": f"{record['classification']:.3f}",
                "dom": f"{record['domain']:.3f}",
                "proto": f"{record['prototype']:.3f}",
            }
            if subgroup_alignment is not None:
                postfix["sub"] = f"{record['subgroup_contrast']:.3f}"
                postfix["match"] = f"{record['subgroup_alignment']['sample_match_coverage']:.2f}"
            if spec.use_multiscale_coteaching:
                postfix["peer"] = f"{record['peer_teaching']:.3f}"
            progress.set_postfix(postfix)
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
                source_prototype_memory=source_prototype_memory,
                class_scale_competence=class_scale_competence,
                domain_gap_scale_calibrator=domain_gap_scale_calibrator,
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
        "method": args.method,
        "subgroup_alignment": subgroup_alignment.state() if subgroup_alignment is not None else None,
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
            "pyramid_gate_mode": spec.pyramid_gate_mode,
            "use_pyramid_bias_guard": spec.use_pyramid_bias_guard,
            "use_pyramid_gate_warmup": spec.use_pyramid_gate_warmup,
            "multiview_fusion_mode": spec.multiview_fusion_mode,
            "use_multiview_uncertainty": spec.use_multiview_uncertainty,
            "use_sign_aware_pyramid_guard": (
                spec.use_sign_aware_pyramid_guard
            ),
            "use_source_multiview_anchor": (
                spec.use_source_multiview_anchor
            ),
            "use_stable_pyramid_gate": spec.use_stable_pyramid_gate,
            "use_temporal_msad": spec.use_temporal_msad,
            "use_source_prototype_memory": (
                spec.use_source_prototype_memory
            ),
            "use_relative_degradation_fusion": (
                spec.use_relative_degradation_fusion
            ),
            "use_anatomical_regions": spec.use_anatomical_regions,
            "anatomical_region_max_strength": (
                args.anatomical_region_max_strength
            ),
            "use_physiology_prior": spec.use_physiology_prior,
            "physiology_max_strength": args.physiology_max_strength,
            "use_physiology_reliability": (
                spec.use_physiology_reliability
            ),
            "physiology_reliability_hidden": (
                args.physiology_reliability_hidden
            ),
            "use_balanced_multiscale_mixup": (
                spec.use_balanced_multiscale_mixup
            ),
            "use_class_scale_adaptive_augmentation": (
                spec.use_class_scale_adaptive_augmentation
            ),
            "use_class_scale_reliability_fusion": (
                spec.use_class_scale_reliability_fusion
            ),
            "use_guarded_class_scale_reliability_fusion": (
                spec.use_guarded_class_scale_reliability_fusion
            ),
            "use_domain_gap_scale_calibration": (
                spec.use_domain_gap_scale_calibration
            ),
            "class_scale_fusion_strength": args.class_scale_fusion_strength,
            "class_scale_entropy_strength": (
                args.class_scale_entropy_strength
            ),
            "class_scale_conflict_strength": (
                args.class_scale_conflict_strength
            ),
            "class_scale_reliability_floor": (
                args.class_scale_reliability_floor
            ),
            "class_scale_guard_max_log_adjustment": (
                args.class_scale_guard_max_log_adjustment
            ),
            "class_scale_guard_consensus_floor": (
                args.class_scale_guard_consensus_floor
            ),
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
            "pyramid_residual_clip": args.pyramid_residual_clip,
            "pyramid_guard_strength": args.pyramid_guard_strength,
            "pyramid_guard_tolerance": args.pyramid_guard_tolerance,
            "multiview_low_rank": args.multiview_low_rank,
            "multiview_uncertainty_strength": (
                args.multiview_uncertainty_strength
            ),
            "multiview_conflict_strength": (
                args.multiview_conflict_strength
            ),
            "multiview_source_anchor_mix": (
                args.multiview_source_anchor_mix
                if spec.use_source_multiview_anchor
                else 0.0
            ),
            "pyramid_gate_shrinkage": (
                args.pyramid_gate_shrinkage
                if spec.use_stable_pyramid_gate
                else 0.0
            ),
            "pyramid_gate_ceiling": (
                args.pyramid_gate_ceiling
                if spec.use_stable_pyramid_gate
                else 1.0
            ),
            "rda_memory_slots_per_scale_class": args.rda_memory_slots,
            "rda_memory_topk": args.rda_memory_topk,
            "rda_memory_temperature": args.rda_memory_temperature,
            "rda_memory_strength": args.rda_memory_strength,
            "rda_degradation_strength": args.rda_degradation_strength,
            "rda_msad_strength": args.rda_msad_strength,
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
            "pyramid_residual_clip": args.pyramid_residual_clip,
            "pyramid_guard_strength": args.pyramid_guard_strength,
            "pyramid_guard_tolerance": args.pyramid_guard_tolerance,
            "multiview_low_rank": args.multiview_low_rank,
            "multiview_uncertainty_strength": (
                args.multiview_uncertainty_strength
            ),
            "multiview_conflict_strength": (
                args.multiview_conflict_strength
            ),
            "multiview_source_anchor_mix": (
                args.multiview_source_anchor_mix
                if spec.use_source_multiview_anchor
                else 0.0
            ),
            "pyramid_gate_shrinkage": (
                args.pyramid_gate_shrinkage
                if spec.use_stable_pyramid_gate
                else 0.0
            ),
            "pyramid_gate_ceiling": (
                args.pyramid_gate_ceiling
                if spec.use_stable_pyramid_gate
                else 1.0
            ),
            "rda_memory_slots": args.rda_memory_slots,
            "rda_memory_momentum": args.rda_memory_momentum,
            "rda_memory_topk": args.rda_memory_topk,
            "rda_memory_temperature": args.rda_memory_temperature,
            "rda_memory_strength": args.rda_memory_strength,
            "rda_degradation_strength": args.rda_degradation_strength,
            "rda_msad_strength": args.rda_msad_strength,
            "pyramid_gate_supervision_weight": (
                args.pyramid_gate_supervision_weight
            ),
            "pyramid_gate_sparsity_weight": (
                args.pyramid_gate_sparsity_weight
            ),
            "pyramid_gate_teacher_temperature": (
                args.pyramid_gate_teacher_temperature
            ),
            "pyramid_gate_teacher_margin": (
                args.pyramid_gate_teacher_margin
            ),
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
            "physiology_classification_weight": (
                args.physiology_classification_weight
            ),
            "physiology_reliability_temperature": (
                args.physiology_reliability_temperature
            ),
            "physiology_reliability_floor": (
                args.physiology_reliability_floor
            ),
            "mixup_warmup_iterations": args.mixup_warmup_iterations,
            "mixup_beta_alpha": args.mixup_beta_alpha,
            "mixup_rarity_power": args.mixup_rarity_power,
            "mixup_loss_weight": args.mixup_loss_weight,
            "class_scale_competence_momentum": (
                args.class_scale_competence_momentum
            ),
            "class_scale_competence_uniform_mix": (
                args.class_scale_competence_uniform_mix
            ),
            "competence_deficit_floor": args.competence_deficit_floor,
            "competence_deficit_power": args.competence_deficit_power,
            "domain_gap_momentum": args.domain_gap_momentum,
            "target_scale_agreement_momentum": (
                args.target_scale_agreement_momentum
            ),
            "domain_gap_spread_weight": args.domain_gap_spread_weight,
            "domain_gap_strength": args.domain_gap_strength,
            "domain_gap_uniform_mix": args.domain_gap_uniform_mix,
            "domain_gap_max_log_deviation": (
                args.domain_gap_max_log_deviation
            ),
            "domain_gap_min_updates": args.domain_gap_min_updates,
            "removed_losses": [
                "supervised_contrastive",
                "information_maximization",
                "one_second_teacher_consistency",
            ],
        },
        "final_prototype_bank": (
            prototype_bank.state() if prototype_bank is not None else None
        ),
        "final_source_prototype_memory": (
            source_prototype_memory.state(
                args.adaptation_warmup_iterations
            )
            if source_prototype_memory is not None
            else None
        ),
        "final_class_scale_competence": (
            class_scale_competence.state()
            if class_scale_competence is not None
            else None
        ),
        "final_domain_gap_scale_calibration": (
            domain_gap_scale_calibrator.state(
                class_scale_competence.reliability()
                if class_scale_competence is not None
                else None
            )
            if domain_gap_scale_calibrator is not None
            else None
        ),
        "final_target_prior_estimator": target_prior_estimator.state(),
        "final_learned_scale_class_relation_residual": (
            model.scale_class_relation_residual.detach().cpu().tolist()
        ),
        "anatomical_physiology_prior": (
            {
                **physiology_metadata(),
                "source_only_standardisation_by_scale": (
                    {
                        scale_key(scale): {
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                        }
                        for scale, (mean, std) in (
                            prepared.physiology_stats or {}
                        ).items()
                    }
                    if spec.use_physiology_prior
                    else None
                ),
                "normalisation": (
                    "within_subject_median_mad_using_unlabeled_trials"
                    if spec.use_physiology_reliability
                    else (
                        "source_global_mean_std"
                        if spec.use_physiology_prior
                        else None
                    )
                ),
                "descriptor_mode": (
                    "compact_four_indicators"
                    if spec.use_physiology_reliability
                    else (
                        "full_44_indicators"
                        if spec.use_physiology_prior
                        else None
                    )
                ),
                "role": (
                    "bounded_pseudo_label_prototype_weight_only"
                    if spec.use_physiology_reliability
                    else (
                        "direct_scale_token_residual"
                        if spec.use_physiology_prior
                        else "anatomical_region_balanced_residual"
                    )
                ),
                "final_anatomical_region_gate": (
                    float(
                        args.anatomical_region_max_strength
                        * torch.sigmoid(
                            model.spatial.anatomical_region_gate_logit.detach()
                        )
                    )
                    if model.spatial.anatomical_region_gate_logit is not None
                    else None
                ),
                "final_physiology_gate_by_scale": (
                    (
                        args.physiology_max_strength
                        * torch.sigmoid(model.physiology_gate_logit.detach())
                    ).cpu().tolist()
                    if model.physiology_gate_logit is not None
                    else None
                ),
            }
            if (
                spec.use_anatomical_regions
                or spec.use_physiology_prior
                or spec.use_physiology_reliability
            )
            else None
        ),
        "final_pyramid_residual_weight": float(
            evaluation["pyramid_residual_weight"]
        ),
        "final_pyramid_class_gate_prior": (
            torch.sigmoid(model.pyramid_class_gate_logit.detach()).cpu().tolist()
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
    parser.add_argument("--method", choices=("experiment", "r2_subgroup", "r2_coteaching"), default="experiment",
                        help="R2 subgroup/peer-teaching A-F profiles without later D/N/domain-gap additions")
    for name, default in asdict(CoTeachingConfig()).items():
        parser.add_argument("--subgroup-" + name.replace("_", "-"), type=type(default), default=None)
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
    parser.add_argument("--pyramid-residual-clip", type=float, default=0.50)
    parser.add_argument("--pyramid-guard-strength", type=float, default=4.0)
    parser.add_argument("--pyramid-guard-tolerance", type=float, default=0.05)
    parser.add_argument("--multiview-low-rank", type=int, default=16)
    parser.add_argument(
        "--multiview-uncertainty-strength", type=float, default=1.0
    )
    parser.add_argument(
        "--multiview-conflict-strength", type=float, default=1.0
    )
    parser.add_argument(
        "--multiview-source-anchor-mix", type=float, default=0.50
    )
    parser.add_argument(
        "--pyramid-gate-shrinkage", type=float, default=0.50
    )
    parser.add_argument(
        "--pyramid-gate-ceiling", type=float, default=0.25
    )
    parser.add_argument("--rda-memory-slots", type=int, default=4)
    parser.add_argument("--rda-memory-momentum", type=float, default=0.90)
    parser.add_argument("--rda-memory-topk", type=int, default=2)
    parser.add_argument(
        "--rda-memory-temperature", type=float, default=0.25
    )
    parser.add_argument("--rda-memory-strength", type=float, default=0.25)
    parser.add_argument(
        "--rda-degradation-strength", type=float, default=1.0
    )
    parser.add_argument("--rda-msad-strength", type=float, default=0.25)
    parser.add_argument(
        "--anatomical-region-max-strength", type=float, default=0.25
    )
    parser.add_argument("--physiology-max-strength", type=float, default=0.25)
    parser.add_argument("--physiology-reliability-hidden", type=int, default=16)
    parser.add_argument(
        "--physiology-classification-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--physiology-reliability-temperature", type=float, default=0.15
    )
    parser.add_argument(
        "--physiology-reliability-floor", type=float, default=0.50
    )
    parser.add_argument("--mixup-warmup-iterations", type=int, default=50)
    parser.add_argument("--mixup-beta-alpha", type=float, default=0.40)
    parser.add_argument("--mixup-rarity-power", type=float, default=0.50)
    parser.add_argument("--mixup-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--class-scale-competence-momentum", type=float, default=0.95
    )
    parser.add_argument(
        "--class-scale-competence-uniform-mix", type=float, default=0.10
    )
    parser.add_argument("--domain-gap-momentum", type=float, default=0.95)
    parser.add_argument(
        "--target-scale-agreement-momentum", type=float, default=0.95
    )
    parser.add_argument(
        "--domain-gap-spread-weight", type=float, default=0.25
    )
    parser.add_argument("--domain-gap-strength", type=float, default=0.50)
    parser.add_argument(
        "--domain-gap-uniform-mix", type=float, default=0.50
    )
    parser.add_argument(
        "--domain-gap-max-log-deviation", type=float, default=0.35
    )
    parser.add_argument("--domain-gap-min-updates", type=int, default=20)
    parser.add_argument(
        "--competence-deficit-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--competence-deficit-power", type=float, default=1.0
    )
    parser.add_argument(
        "--class-scale-fusion-strength", type=float, default=0.50
    )
    parser.add_argument(
        "--class-scale-entropy-strength", type=float, default=0.50
    )
    parser.add_argument(
        "--class-scale-conflict-strength", type=float, default=1.0
    )
    parser.add_argument(
        "--class-scale-reliability-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--class-scale-guard-max-log-adjustment", type=float, default=0.25
    )
    parser.add_argument(
        "--class-scale-guard-consensus-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--pyramid-gate-supervision-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--pyramid-gate-sparsity-weight", type=float, default=0.005
    )
    parser.add_argument(
        "--pyramid-gate-teacher-temperature", type=float, default=0.10
    )
    parser.add_argument(
        "--pyramid-gate-teacher-margin", type=float, default=0.05
    )
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
    if spec.use_subgroup_alignment:
        subgroup_config(args).validate()
        if args.evaluation_protocol != EVALUATION_PROTOCOL_FIXED_FINAL:
            raise ValueError("R2 subgroup alignment requires fixed-final evaluation")
        if args.result_root.expanduser().resolve() == DEFAULT_RESULT_ROOT.resolve():
            raise ValueError("R2 subgroup alignment requires an independent --result-root")
        if spec.prototype_weight != 0 and not spec.use_multiscale_coteaching:
            raise ValueError("Subgroup contrast replaces coarse centroid alignment")
        if spec.use_multiscale_coteaching and spec.prototype_weight != EXPERIMENTS["A_R2"].prototype_weight:
            raise ValueError("Co-teaching requires the original R2 centroid weight for fallback")
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
    if args.pyramid_residual_clip <= 0:
        raise ValueError("pyramid-residual-clip must be positive")
    if args.multiview_low_rank < 1:
        raise ValueError("multiview-low-rank must be positive")
    if min(
        args.pyramid_guard_strength,
        args.pyramid_guard_tolerance,
        args.pyramid_gate_supervision_weight,
        args.pyramid_gate_sparsity_weight,
        args.pyramid_gate_teacher_margin,
        args.multiview_uncertainty_strength,
        args.multiview_conflict_strength,
    ) < 0:
        raise ValueError("pyramid gate/guard parameters must be nonnegative")
    if not 0 <= args.multiview_source_anchor_mix <= 1:
        raise ValueError("multiview-source-anchor-mix must be within [0,1]")
    if not 0 <= args.pyramid_gate_shrinkage <= 1:
        raise ValueError("pyramid-gate-shrinkage must be within [0,1]")
    if not 0 < args.pyramid_gate_ceiling <= 1:
        raise ValueError("pyramid-gate-ceiling must be within (0,1]")
    if args.rda_memory_slots < 1 or args.rda_memory_topk < 1:
        raise ValueError("RDA memory slots/top-k must be positive")
    if args.rda_memory_topk > args.rda_memory_slots:
        raise ValueError("RDA memory top-k cannot exceed memory slots")
    if not 0 <= args.rda_memory_momentum < 1:
        raise ValueError("RDA memory momentum must be within [0,1)")
    if args.rda_memory_temperature <= 0:
        raise ValueError("RDA memory temperature must be positive")
    if min(
        args.rda_memory_strength,
        args.rda_degradation_strength,
        args.rda_msad_strength,
        args.anatomical_region_max_strength,
        args.physiology_max_strength,
        args.physiology_classification_weight,
    ) < 0:
        raise ValueError("RDA/anatomical/physiology strengths must be nonnegative")
    if args.physiology_reliability_hidden < 1:
        raise ValueError("physiology-reliability-hidden must be positive")
    if args.physiology_reliability_temperature <= 0:
        raise ValueError("physiology-reliability-temperature must be positive")
    if not 0 <= args.physiology_reliability_floor <= 1:
        raise ValueError("physiology-reliability-floor must be within [0,1]")
    if not 0 <= args.mixup_warmup_iterations < FIXED_UDA_PROTOCOL.training_iterations:
        raise ValueError("mixup-warmup-iterations must be within [0,1000)")
    if min(
        args.mixup_beta_alpha,
        args.mixup_rarity_power,
        args.competence_deficit_floor,
        args.competence_deficit_power,
    ) <= 0:
        raise ValueError("MixUp and competence deficit parameters must be positive")
    if args.mixup_loss_weight < 0:
        raise ValueError("mixup-loss-weight must be nonnegative")
    if not 0 <= args.class_scale_competence_momentum < 1:
        raise ValueError("class-scale-competence-momentum must be within [0,1)")
    if not 0 <= args.class_scale_competence_uniform_mix <= 1:
        raise ValueError("class-scale-competence-uniform-mix must be within [0,1]")
    if not 0 <= args.domain_gap_momentum < 1:
        raise ValueError("domain-gap-momentum must be within [0,1)")
    if not 0 <= args.target_scale_agreement_momentum < 1:
        raise ValueError("target-scale-agreement-momentum must be within [0,1)")
    if min(args.domain_gap_spread_weight, args.domain_gap_strength) < 0:
        raise ValueError("domain-gap weights must be nonnegative")
    if not 0 <= args.domain_gap_uniform_mix <= 1:
        raise ValueError("domain-gap-uniform-mix must be within [0,1]")
    if args.domain_gap_max_log_deviation <= 0:
        raise ValueError("domain-gap-max-log-deviation must be positive")
    if args.domain_gap_min_updates < 1:
        raise ValueError("domain-gap-min-updates must be positive")
    if min(
        args.class_scale_fusion_strength,
        args.class_scale_entropy_strength,
        args.class_scale_conflict_strength,
    ) < 0:
        raise ValueError("class-scale fusion strengths must be nonnegative")
    if not 0 < args.class_scale_reliability_floor <= 1:
        raise ValueError("class-scale-reliability-floor must be within (0,1]")
    if args.class_scale_guard_max_log_adjustment <= 0:
        raise ValueError("class-scale-guard-max-log-adjustment must be positive")
    if not 0 <= args.class_scale_guard_consensus_floor < 1:
        raise ValueError(
            "class-scale-guard-consensus-floor must be within [0,1)"
        )
    if (
        spec.use_class_scale_adaptive_augmentation
        and not spec.use_balanced_multiscale_mixup
    ):
        raise ValueError("Class-scale adaptive augmentation requires MixUp")
    if (
        spec.use_class_scale_reliability_fusion
        and not spec.use_class_scale_adaptive_augmentation
    ):
        raise ValueError("Class-scale reliability fusion requires D2 augmentation")
    if (
        spec.use_guarded_class_scale_reliability_fusion
        and not spec.use_class_scale_adaptive_augmentation
    ):
        raise ValueError("Guarded reliability fusion requires adaptive augmentation")
    if (
        spec.use_domain_gap_scale_calibration
        and not spec.use_guarded_class_scale_reliability_fusion
    ):
        raise ValueError("Domain-gap calibration requires guarded fusion")
    if args.pyramid_gate_teacher_temperature <= 0:
        raise ValueError("pyramid gate teacher temperature must be positive")
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
    spec = (coteaching_experiment(args.experiment) if args.method == "r2_coteaching"
            else subgroup_experiment(args.experiment) if args.method == "r2_subgroup"
            else EXPERIMENTS[args.experiment])
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
    prepared = prepare_sources(
        args.data_dir,
        spec.source_domains,
        spec.scales,
        use_physiology=(
            spec.use_physiology_prior or spec.use_physiology_reliability
        ),
        compact_physiology=spec.use_physiology_reliability,
        subject_robust_physiology=spec.use_physiology_reliability,
    )
    for seed in args.random_seeds:
        for subject in args.target_subjects:
            run_fold(args, spec, prepared, seed, subject, device)


if __name__ == "__main__":
    main()
