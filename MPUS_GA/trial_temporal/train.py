"""Fixed-protocol class-conditional multiscale multi-source UDA training."""

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
DEFAULT_RESULT_ROOT = PACKAGE_DIR / "results_class_conditional_multiscale"
VARIANT = "class-conditional-multisource-multiscale-fixed1000"
CLASS_NAMES = ("positive", "neutral", "negative")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    source_domains: tuple[str, ...]
    scales: tuple[float, ...]
    fusion_mode: str
    domain_mode: str
    use_prototypes: bool
    target_dataset: str = "seed_v"
    target_subject_count: int = 16
    target_trials: int = 45
    domain_weight: float = 0.2
    prototype_weight: float = 0.1


EXPERIMENT_ORDER = (
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "main",
    "B6",
)
EXPERIMENTS = {
    "A0": ExperimentSpec(
        "A0",
        "1-second only with class-conditional multi-source alignment",
        ("seed_vii", "seed_iv"),
        (1.0,),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "A1": ExperimentSpec(
        "A1",
        "2-second only with class-conditional multi-source alignment",
        ("seed_vii", "seed_iv"),
        (2.0,),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "A2": ExperimentSpec(
        "A2",
        "4-second only with class-conditional multi-source alignment",
        ("seed_vii", "seed_iv"),
        (4.0,),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "A3": ExperimentSpec(
        "A3",
        "naive multiscale attention with fused-only conditional alignment",
        ("seed_vii", "seed_iv"),
        (1.0, 2.0, 4.0),
        "attention",
        "fused_conditional",
        False,
        prototype_weight=0.0,
    ),
    "A4": ExperimentSpec(
        "A4",
        "multiscale attention with non-class-conditional per-scale alignment",
        ("seed_vii", "seed_iv"),
        (1.0, 2.0, 4.0),
        "attention",
        "scale_global",
        False,
        prototype_weight=0.0,
    ),
    "A5": ExperimentSpec(
        "A5",
        "class-conditional alignment with uniform multiscale fusion",
        ("seed_vii", "seed_iv"),
        (1.0, 2.0, 4.0),
        "uniform",
        "scale_conditional",
        True,
    ),
    "A6": ExperimentSpec(
        "A6",
        "full method with SEED-VII as the only source",
        ("seed_vii",),
        (1.0, 2.0, 4.0),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "A7": ExperimentSpec(
        "A7",
        "full method with SEED-IV as the only source",
        ("seed_iv",),
        (1.0, 2.0, 4.0),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "main": ExperimentSpec(
        "main",
        "class-conditional source-scale reliability and dynamic class fusion",
        ("seed_vii", "seed_iv"),
        (1.0, 2.0, 4.0),
        "class_conditional",
        "scale_conditional",
        True,
    ),
    "B6": ExperimentSpec(
        "B6",
        "SEED-V to SEED-VII with the A6 full single-source model",
        ("seed_v",),
        (1.0, 2.0, 4.0),
        "class_conditional",
        "scale_conditional",
        True,
        target_dataset="seed_vii",
        target_subject_count=20,
        target_trials=80,
    ),
}


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
) -> tuple[DataLoader, DataLoader]:
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


class PrototypeBank:
    """EMA source/target prototypes indexed by domain, scale, and class."""

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
    ) -> None:
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.uniform_mix = float(uniform_mix)
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
    def update_target(
        self,
        embeddings: torch.Tensor,
        probability: torch.Tensor,
        confidence_threshold: float,
    ) -> float:
        probability = probability.detach()
        confidence = probability.max(dim=1).values
        confidence_weight = (
            (confidence - confidence_threshold)
            / max(1.0 - confidence_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        for class_index in range(self.target.shape[1]):
            weight = probability[:, class_index] * confidence_weight
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
        return float((confidence >= confidence_threshold).float().mean())

    @torch.no_grad()
    def joint_weights(self) -> torch.Tensor:
        domains, scales, classes, _ = self.source.shape
        result = self.source.new_full(
            (domains, scales, classes), 1.0 / (domains * scales)
        )
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
            logits = (similarity / self.temperature).masked_fill(~valid, -1e9)
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
            "joint_source_scale_class_weights": self.joint_weights().cpu().tolist(),
            "scale_class_reliability": self.scale_class_reliability().cpu().tolist(),
            "source_initialized": self.source_initialized.cpu().tolist(),
            "target_initialized": self.target_initialized.cpu().tolist(),
            "prototype_cosine_distance": distances,
        }


def _domain_loss(
    source_outputs: list[dict],
    target_output: dict,
) -> tuple[torch.Tensor, list[float]]:
    all_outputs = source_outputs + [target_output]
    if target_output["scale_domain_logits"] is not None:
        scale_losses = []
        for scale_index in range(target_output["scale_domain_logits"].shape[1]):
            domain_losses = []
            for domain_index, output in enumerate(all_outputs):
                logits = output["scale_domain_logits"][:, scale_index]
                labels = torch.full(
                    (len(logits),), domain_index, dtype=torch.long, device=logits.device
                )
                domain_losses.append(F.cross_entropy(logits, labels))
            scale_losses.append(torch.stack(domain_losses).mean())
        return torch.stack(scale_losses).mean(), [
            float(item.detach()) for item in scale_losses
        ]
    domain_losses = []
    for domain_index, output in enumerate(all_outputs):
        logits = output["fused_domain_logits"]
        labels = torch.full(
            (len(logits),), domain_index, dtype=torch.long, device=logits.device
        )
        domain_losses.append(F.cross_entropy(logits, labels))
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
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ramp = _adaptation_ramp(
        iteration, adaptation_warmup_iterations, adaptation_ramp_end
    )
    adaptation_active = iteration > adaptation_warmup_iterations
    reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None and adaptation_active
        else None
    )
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
    )

    classification_by_source = torch.stack(
        [
            F.cross_entropy(
                output["logits"], labels, label_smoothing=label_smoothing
            )
            for output, labels in zip(
                source_outputs, source_labels, strict=True
            )
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
    domain_loss, domain_by_scale = _domain_loss(source_outputs, target_output)
    if prototype_bank is not None and adaptation_active:
        prototype_loss, pseudo_coverage = (
            class_conditional_prototype_alignment_loss(
                [output["scale_embeddings"] for output in source_outputs],
                source_labels,
                target_output["scale_embeddings"],
                target_output["probability"],
                prototype_bank.joint_weights(),
                pseudo_confidence_threshold,
            )
        )
    else:
        prototype_loss = target_output["logits"].sum() * 0.0
        pseudo_coverage = float(
            (
                target_output["probability"].max(dim=1).values
                >= pseudo_confidence_threshold
            )
            .float()
            .mean()
        )
    total_loss = classification_loss + ramp * (
        spec.domain_weight * domain_loss
        + spec.prototype_weight * prototype_loss
    )
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip
    )
    optimizer.step()
    scheduler.step()

    if prototype_bank is not None:
        for domain_index, (output, labels) in enumerate(
            zip(source_outputs, source_labels, strict=True)
        ):
            prototype_bank.update_source(
                domain_index, output["scale_embeddings"], labels
            )
        if adaptation_active:
            pseudo_coverage = prototype_bank.update_target(
                target_output["scale_embeddings"],
                target_output["probability"],
                pseudo_confidence_threshold,
            )
        else:
            pseudo_coverage = 0.0
        reliability = prototype_bank.scale_class_reliability()
        source_weights = prototype_bank.source_weights(source_class_priors)

    mean_gate = target_output["scale_class_weight"].detach().mean(dim=0)
    record = {
        "iteration": iteration,
        "adaptation_ramp": ramp,
        "prototype_updates_active": (
            prototype_bank is not None and adaptation_active
        ),
        "total": float(total_loss.detach()),
        "classification": float(classification_loss.detach()),
        "classification_by_source": [
            float(value) for value in classification_by_source.detach()
        ],
        "domain": float(domain_loss.detach()),
        "domain_by_scale": domain_by_scale,
        "prototype": float(prototype_loss.detach()),
        "pseudo_label_coverage": pseudo_coverage,
        "source_weights": [float(value) for value in source_weights.detach()],
        "mean_target_scale_class_gate": mean_gate.cpu().tolist(),
        "gradient_norm": float(gradient_norm),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    if prototype_bank is not None:
        record["joint_source_scale_class_weights"] = (
            prototype_bank.joint_weights().cpu().tolist()
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


def evaluate_trials(
    model: MultiScaleMultiSourceDANN,
    loader: DataLoader,
    device: torch.device,
    description: str,
    prototype_bank: PrototypeBank | None,
) -> dict:
    model.eval()
    probabilities = []
    scale_probabilities = []
    scale_gates = []
    labels = []
    reliability = (
        prototype_bank.scale_class_reliability()
        if prototype_bank is not None
        else None
    )
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
            )
            probabilities.append(output["probability"].cpu().numpy())
            scale_probabilities.append(
                F.softmax(output["scale_logits"], dim=-1).cpu().numpy()
            )
            scale_gates.append(output["scale_class_weight"].cpu().numpy())
            labels.append(batch["y"].numpy())
    labels_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    scale_probability_array = np.concatenate(scale_probabilities)
    gate_array = np.concatenate(scale_gates)
    return {
        "fused": _classification_metrics(labels_array, probability_array),
        "by_scale": {
            scale_key(scale): _classification_metrics(
                labels_array, scale_probability_array[:, index]
            )
            for index, scale in enumerate(model.scales)
        },
        "mean_scale_class_gate": gate_array.mean(axis=0).tolist(),
    }


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
    target_loader, test_loader = _target_loaders(
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
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = _scheduler(optimizer, iterations, args.warmup_iterations)
    source_class_priors = _natural_source_class_priors(prepared, device)
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
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    source_iterators = [iter(loader) for loader in source_loaders]
    target_iterator = iter(target_loader)
    training_trace = []
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
        )
        if iteration == 1 or iteration % args.log_interval == 0 or iteration == iterations:
            training_trace.append(record)
            progress.set_postfix(
                total=f"{record['total']:.3f}",
                cls=f"{record['classification']:.3f}",
                dom=f"{record['domain']:.3f}",
                proto=f"{record['prototype']:.3f}",
            )

    # This is the only point at which target labels are accessed.
    evaluation = evaluate_trials(
        model,
        test_loader,
        device,
        f"Final eval {spec.name}/seed{seed}/S{subject:02d}",
        prototype_bank,
    )
    result = {
        "variant": VARIANT,
        "experiment": spec.name,
        "experiment_spec": asdict(spec),
        "protocol": {
            "name": (
                f"{'_'.join(prepared.domain_names)}_to_"
                f"{spec.target_dataset}_transductive_fixed1000"
            ),
            "source_domains": list(prepared.domain_names),
            "target_dataset": spec.target_dataset,
            "target_subject_count": spec.target_subject_count,
            "target_trials": spec.target_trials,
            "training_iterations": FIXED_UDA_PROTOCOL.training_iterations,
            "checkpoint_selection": FIXED_UDA_PROTOCOL.checkpoint_selection,
            "target_evaluations": FIXED_UDA_PROTOCOL.target_evaluations,
            "target_probability_correction": False,
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
        "target_subject": subject,
        "random_seed": seed,
        "model_parameters": parameter_count,
        "model": {
            "scales": list(spec.scales),
            "fusion_mode": spec.fusion_mode,
            "domain_mode": spec.domain_mode,
            "channel_attention": True,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "spatial_layers": args.spatial_layers,
            "temporal_layers": args.temporal_layers,
            "fusion_layers": args.fusion_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "spatial_topk": args.spatial_topk,
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
            "removed_losses": [
                "supervised_contrastive",
                "information_maximization",
                "one_second_teacher_consistency",
            ],
        },
        "final_prototype_bank": (
            prototype_bank.state() if prototype_bank is not None else None
        ),
        "training_trace": training_trace,
        "evaluation": evaluation,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary(experiment_dir)
    fused = evaluation["fused"]
    tqdm.write(
        f"Final {result_path}: acc={fused['accuracy']:.4f}, "
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
    if args.log_interval < 1 or args.gradient_clip <= 0:
        raise ValueError("log interval and gradient clip must be positive")
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
        "Protocol: fixed 1000 iterations, raw fused logits, no checkpoint "
        "selection, one final target evaluation"
    )
    prepared = prepare_sources(args.data_dir, spec.source_domains, spec.scales)
    for seed in args.random_seeds:
        for subject in args.target_subjects:
            run_fold(args, spec, prepared, seed, subject, device)


if __name__ == "__main__":
    main()
