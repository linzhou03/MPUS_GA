"""Train a true trial-level temporal transfer baseline.

Each sample is a complete movie trial. Electrode graphs are encoded per DE
window, a Transformer models the ordered window sequence, and attention pooling
produces one prediction per trial. Target labels are unavailable to the training
loader and are consumed only by the final evaluation after source-only model
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm.auto import tqdm

from trial_data import (
    PreparedTrialSource,
    UnlabeledTrialView,
    collate_trials,
    prepare_trial_source,
    prepare_trial_target,
)
from trial_temporal_model import TrialTemporalDANN
from window_data import NUM_BANDS, NUM_CLASSES, SEED_V_SUBJECTS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data_processed"
VARIANT = "trial-temporal-dann"


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_integer_set(value: str, minimum: int, maximum: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(minimum, maximum + 1))
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
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


def _target_subjects(value: str) -> list[int]:
    return _parse_integer_set(value, 1, SEED_V_SUBJECTS)


def _source_validation_subjects(value: str) -> list[int]:
    result = _parse_integer_set(value, 1, 20)
    if len(result) == 20:
        raise argparse.ArgumentTypeError("source validation cannot contain all subjects")
    return result


def source_trial_sampling_weights(
    labels: torch.Tensor, balance_alpha: float
) -> torch.Tensor:
    labels = labels.long()
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    if torch.any(counts == 0):
        raise ValueError(f"Source training split has an empty class: {counts.tolist()}")
    return counts.float().pow(-balance_alpha)[labels].double()


def make_loaders(
    source: PreparedTrialSource,
    target,
    batch_size: int,
    iterations: int,
    seed: int,
    source_balance_alpha: float,
    pin_memory: bool,
):
    sample_count = batch_size * iterations
    source_sampler = WeightedRandomSampler(
        source_trial_sampling_weights(
            source.train.labels, source_balance_alpha
        ),
        num_samples=sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    unlabeled_target = UnlabeledTrialView(target)
    target_sampler = RandomSampler(
        unlabeled_target,
        replacement=True,
        num_samples=sample_count,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": pin_memory,
        "collate_fn": collate_trials,
    }
    return (
        DataLoader(source.train, sampler=source_sampler, drop_last=True, **common),
        DataLoader(unlabeled_target, sampler=target_sampler, drop_last=True, **common),
        DataLoader(source.validation, shuffle=False, drop_last=False, **common),
        DataLoader(target, shuffle=False, drop_last=False, **common),
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _inputs_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return (
        batch["x"].to(device, non_blocking=True),
        batch["window_mask"].to(device, non_blocking=True),
    )


def train_step(
    model: TrialTemporalDANN,
    source_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iteration: int,
    max_iters: int,
    domain_weight: float,
) -> dict[str, float]:
    if "y" in target_batch:
        raise RuntimeError("Target training batch unexpectedly contains labels")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = (iteration - 1) / max(max_iters - 1, 1)
    grl_alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    source_x, source_mask = _inputs_to_device(source_batch, device)
    target_x, target_mask = _inputs_to_device(target_batch, device)
    source_labels = source_batch["y"].to(device, non_blocking=True)
    source_logits, _, source_domain, *_ = model(
        source_x, source_mask, grl_alpha=grl_alpha
    )
    _, _, target_domain, *_ = model(target_x, target_mask, grl_alpha=grl_alpha)

    classification = F.cross_entropy(source_logits, source_labels)
    domain_logits = torch.cat((source_domain, target_domain))
    domain_labels = torch.cat(
        (torch.zeros_like(source_domain), torch.ones_like(target_domain))
    )
    domain = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
    total = classification + domain_weight * domain
    total.backward()
    optimizer.step()
    return {
        "iteration": iteration,
        "total": float(total.detach()),
        "classification": float(classification.detach()),
        "domain": float(domain.detach()),
        "grl_alpha": grl_alpha,
    }


def _classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "pred_positive": int((prediction == 0).sum()),
        "pred_neutral": int((prediction == 1).sum()),
        "pred_negative": int((prediction == 2).sum()),
        "trials": int(len(labels)),
    }


def evaluate_trials(
    model: TrialTemporalDANN,
    loader: DataLoader,
    device: torch.device,
    description: str,
    show_progress: bool,
) -> dict:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    iterator = tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for batch in iterator:
            if "y" not in batch:
                raise RuntimeError("Evaluation requires labeled trials")
            x, mask = _inputs_to_device(batch, device)
            probability = model(x, mask)[1]
            probabilities.append(probability.cpu().numpy())
            labels.append(batch["y"].numpy())
    return _classification_metrics(np.concatenate(labels), np.concatenate(probabilities))


def _cpu_state_dict(model: TrialTemporalDANN) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _write_summary(result_dir: Path) -> None:
    results = [
        json.loads(path.read_text())
        for path in sorted(result_dir.glob("subject_*.json"))
    ]
    if not results:
        return
    metric_names = ("trial_accuracy", "trial_balanced_accuracy", "trial_macro_f1")
    with (result_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("metric", "mean", "std", "subjects")
        )
        writer.writeheader()
        for metric in metric_names:
            values = np.asarray([result[metric] for result in results], dtype=float)
            writer.writerow(
                {
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "subjects": len(values),
                }
            )


def run_fold(
    args,
    window_seconds: float,
    seed: int,
    subject: int,
    source: PreparedTrialSource,
    device: torch.device,
) -> None:
    result_dir = (
        args.result_dir
        / VARIANT
        / f"window_{window_seconds:g}s"
        / f"random_seed_{seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"subject_{subject:02d}.json"
    if result_path.exists() and not args.overwrite:
        tqdm.write(f"Skip existing {result_path}")
        return

    target = prepare_trial_target(
        args.data_dir,
        window_seconds,
        subject,
        source,
        args.target_normalization,
    )
    set_seed(seed, device)
    source_loader, target_loader, validation_loader, test_loader = make_loaders(
        source,
        target,
        args.batch_size,
        args.max_iters,
        seed,
        args.source_balance_alpha,
        device.type == "cuda",
    )
    model = TrialTemporalDANN(
        input_dim=NUM_BANDS,
        d_model=args.d_model,
        num_heads=args.num_heads,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        spatial_topk=args.spatial_topk,
        num_classes=NUM_CLASSES,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    best_score = (-math.inf, -math.inf, -math.inf)
    best_iteration = 0
    best_metrics: dict | None = None
    best_state: dict[str, torch.Tensor] | None = None

    log_path = result_dir / f"subject_{subject:02d}_train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        iterations = tqdm(
            range(1, args.max_iters + 1),
            desc=f"{window_seconds:g}s | rnd={seed} | target={subject:02d}",
            unit="iter",
            leave=False,
            dynamic_ncols=True,
        )
        for iteration in iterations:
            source_batch, source_iterator = _next_batch(
                source_iterator, source_loader
            )
            target_batch, target_iterator = _next_batch(
                target_iterator, target_loader
            )
            losses = train_step(
                model,
                source_batch,
                target_batch,
                optimizer,
                device,
                iteration,
                args.max_iters,
                args.domain_weight,
            )
            validate_now = (
                iteration == 1
                or iteration % args.validation_interval == 0
                or iteration == args.max_iters
            )
            record = dict(losses)
            if validate_now:
                validation = evaluate_trials(
                    model,
                    validation_loader,
                    device,
                    "Source validation",
                    show_progress=False,
                )
                record["source_validation"] = validation
                score = (
                    validation["balanced_accuracy"],
                    validation["macro_f1"],
                    validation["accuracy"],
                )
                if score > best_score:
                    best_score = score
                    best_iteration = iteration
                    best_metrics = validation
                    best_state = _cpu_state_dict(model)
            if validate_now or iteration % args.log_interval == 0:
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                postfix = {
                    "total": f"{losses['total']:.3f}",
                    "cls": f"{losses['classification']:.3f}",
                    "dom": f"{losses['domain']:.3f}",
                }
                if best_metrics is not None:
                    postfix["src_val_bal"] = f"{best_metrics['balanced_accuracy']:.3f}"
                    postfix["best_iter"] = best_iteration
                iterations.set_postfix(**postfix)

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    target_metrics = evaluate_trials(
        model,
        test_loader,
        device,
        f"Final target eval {window_seconds:g}s/rnd{seed}/S{subject:02d}",
        show_progress=True,
    )
    result = {
        "variant": VARIANT,
        "source": "SEED-VII",
        "target": "SEED-V",
        "window_seconds": window_seconds,
        "random_seed": seed,
        "target_subject": subject,
        "iterations": args.max_iters,
        "best_source_validation_iteration": best_iteration,
        "source_validation_subjects": args.source_validation_subjects,
        "source_validation_accuracy": best_metrics["accuracy"],
        "source_validation_balanced_accuracy": best_metrics["balanced_accuracy"],
        "source_validation_macro_f1": best_metrics["macro_f1"],
        "target_normalization": args.target_normalization,
        "source_balance_alpha": args.source_balance_alpha,
        "domain_weight": args.domain_weight,
        "trial_accuracy": target_metrics["accuracy"],
        "trial_balanced_accuracy": target_metrics["balanced_accuracy"],
        "trial_macro_f1": target_metrics["macro_f1"],
        "trial_pred_positive": target_metrics["pred_positive"],
        "trial_pred_neutral": target_metrics["pred_neutral"],
        "trial_pred_negative": target_metrics["pred_negative"],
        "trials": target_metrics["trials"],
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary(result_dir)
    tqdm.write(
        f"Final {result_path}: trial_acc={target_metrics['accuracy']:.4f}, "
        f"trial_bal={target_metrics['balanced_accuracy']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="True trial-level temporal baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--result-dir", type=Path, default=SCRIPT_DIR / "results_trial_temporal"
    )
    parser.add_argument("--window-seconds", nargs="+", type=float, default=(1,))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--target-subjects", type=_target_subjects, default="all")
    parser.add_argument(
        "--source-validation-subjects",
        type=_source_validation_subjects,
        default="17-20",
    )
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument(
        "--target-normalization", choices=("source", "domain"), default="source"
    )
    parser.add_argument("--source-balance-alpha", type=float, default=0.98)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--spatial-topk", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args) -> None:
    if isinstance(args.target_subjects, str):
        args.target_subjects = _target_subjects(args.target_subjects)
    if isinstance(args.source_validation_subjects, str):
        args.source_validation_subjects = _source_validation_subjects(
            args.source_validation_subjects
        )
    if any(scale not in {1.0, 2.0, 4.0} for scale in args.window_seconds):
        raise ValueError("window-seconds must be selected from 1, 2, 4")
    if args.batch_size < 2 or args.max_iters < 1:
        raise ValueError("batch-size >=2 and max-iters >=1 are required")
    if args.d_model % args.num_heads:
        raise ValueError("d-model must be divisible by num-heads")
    if args.spatial_layers < 1 or args.temporal_layers < 1:
        raise ValueError("spatial-layers and temporal-layers must be positive")
    if args.validation_interval < 1 or args.log_interval < 1:
        raise ValueError("validation-interval and log-interval must be positive")
    if not 0 <= args.source_balance_alpha <= 1:
        raise ValueError("source-balance-alpha must be between 0 and 1")
    if args.domain_weight < 0:
        raise ValueError("domain-weight must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.result_dir = args.result_dir.expanduser().resolve()
    device_name = args.device
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tqdm.write(
        f"Variant={VARIANT}, device={device}, windows={args.window_seconds}, "
        f"random_seeds={args.random_seeds}, targets={args.target_subjects}, "
        f"source_validation={args.source_validation_subjects}"
    )
    total_folds = (
        len(args.window_seconds)
        * len(args.random_seeds)
        * len(args.target_subjects)
    )
    with tqdm(
        total=total_folds,
        desc="Overall trial-level progress",
        unit="fold",
        dynamic_ncols=True,
    ) as overall:
        for window_seconds in args.window_seconds:
            tqdm.write(f"Loading {window_seconds:g}s trial-level SEED-VII source...")
            source = prepare_trial_source(
                args.data_dir,
                window_seconds,
                args.source_validation_subjects,
            )
            counts = torch.bincount(source.train.labels, minlength=NUM_CLASSES)
            tqdm.write(
                f"Source train trials={len(source.train)}, validation trials="
                f"{len(source.validation)}, train classes={counts.tolist()}"
            )
            for seed in args.random_seeds:
                for subject in args.target_subjects:
                    run_fold(args, window_seconds, seed, subject, source, device)
                    overall.update(1)


if __name__ == "__main__":
    main()
