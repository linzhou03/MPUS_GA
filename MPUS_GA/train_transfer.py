"""Train SEED-VII -> SEED-V transfer models on 1/2/4-second DE windows.

The entry point supports the frozen CAGA-SGA-balanced baseline and the
proposal's first-priority R-SoftSGA MVP. Target labels are touched only by the
final evaluation function after the fixed iteration budget.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from golden_style import ch_stats, gram  # noqa: E402
from graph_align import SemanticAligner  # noqa: E402
from model import GSA_CAST  # noqa: E402

from transfer_losses import (  # noqa: E402
    balanced_reliable_gate,
    entropy_mi_reliability,
    graph_ramp_weight,
    soft_graph_edge_loss,
)
from window_data import (  # noqa: E402
    NUM_BANDS,
    NUM_CHANNELS,
    NUM_CLASSES,
    SEED_V_SUBJECTS,
    prepare_source,
    prepare_target,
    summarize_labels,
)


DEFAULT_DATA_DIR = SCRIPT_DIR / "data_processed"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _subjects(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, SEED_V_SUBJECTS + 1))
    result = []
    for item in value.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        elif item:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    if not result or any(item < 1 or item > SEED_V_SUBJECTS for item in result):
        raise argparse.ArgumentTypeError("target subjects must be within 1..16")
    return result


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def make_loaders(source, target, batch_size: int, iterations: int, seed: int):
    source_labels = source.labels.long()
    counts = torch.bincount(source_labels, minlength=NUM_CLASSES)
    if torch.any(counts == 0):
        raise ValueError(f"Source has an empty class: {counts.tolist()}")
    weights = counts.float().reciprocal()[source_labels].double()
    sample_count = batch_size * iterations
    source_sampler = WeightedRandomSampler(
        weights,
        num_samples=sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    target_sampler = RandomSampler(
        target,
        replacement=True,
        num_samples=sample_count,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    common = {"batch_size": batch_size, "num_workers": 0, "pin_memory": True}
    return (
        DataLoader(source, sampler=source_sampler, drop_last=True, **common),
        DataLoader(target, sampler=target_sampler, drop_last=True, **common),
        DataLoader(target, shuffle=False, drop_last=False, **common),
    )


def bootstrap_style(model: GSA_CAST, source_batch, target_batch, device) -> None:
    source_batch = source_batch.to(device)
    target_batch = target_batch.to(device)
    model.train()
    with torch.no_grad():
        *_, source_spatial = model(
            source_batch.x,
            source_batch.edge_index,
            source_batch.batch,
            return_style_info=True,
        )
        *_, target_spatial = model(
            target_batch.x,
            target_batch.edge_index,
            target_batch.batch,
            return_style_info=True,
        )
        model(
            source_batch.x,
            source_batch.edge_index,
            source_batch.batch,
            return_style_info=True,
            style_bootstrap_pair=(source_spatial, target_spatial),
        )
    if not model.gsb.fully_bootstrapped:
        raise RuntimeError("Golden Style Bank bootstrap did not complete")


def make_teacher(student: GSA_CAST) -> GSA_CAST:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def update_teacher(teacher: GSA_CAST, student: GSA_CAST, momentum: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(momentum).add_(
                student_parameter, alpha=1.0 - momentum
            )
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(), student.buffers()
        ):
            teacher_buffer.copy_(student_buffer)


def mc_teacher_probability(
    teacher: GSA_CAST,
    target_batch,
    passes: int,
    temperature: float,
) -> torch.Tensor:
    teacher.eval()
    for module in teacher.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    predictions = []
    with torch.no_grad():
        for _ in range(passes):
            logits, *_ = teacher(
                target_batch.x, target_batch.edge_index, target_batch.batch
            )
            predictions.append(F.softmax(logits / temperature, dim=1))
    teacher.eval()
    return torch.stack(predictions)


def _mean_or_weighted(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def style_loss(
    source_style: list[dict],
    target_style: list[dict],
    target_reliability: torch.Tensor | None,
) -> torch.Tensor:
    layer_losses = []
    for source, target in zip(source_style, target_style):
        mu_s0, std_s0 = ch_stats(source["pre"])
        mu_s1, std_s1 = ch_stats(source["post"])
        mu_t0, std_t0 = ch_stats(target["pre"])
        mu_t1, std_t1 = ch_stats(target["post"])
        preserve = (
            (mu_s1 - mu_s0).abs().mean()
            + (std_s1 - std_s0).abs().mean()
            + (mu_t1 - mu_t0).abs().mean()
            + (std_t1 - std_t0).abs().mean()
            + (gram(source["post"]) - gram(source["pre"])).abs().mean()
            + (gram(target["post"]) - gram(target["pre"])).abs().mean()
        )
        mu_g = source["mu_g"].view(1, -1)
        sig_g = source["sig_g"].view(1, -1)
        source_anchor = (mu_s1 - mu_g).abs().mean(1) + (
            std_s1 - sig_g
        ).abs().mean(1)
        target_anchor = (mu_t1 - mu_g).abs().mean(1) + (
            std_t1 - sig_g
        ).abs().mean(1)
        golden = source_anchor.mean() + _mean_or_weighted(
            target_anchor, target_reliability
        )
        batch_align = (
            (mu_s1.mean(0) - mu_t1.mean(0)).abs().mean()
            + (std_s1.mean(0) - std_t1.mean(0)).abs().mean()
            + 0.1
            * (gram(source["post"]).mean(0) - gram(target["post"]).mean(0))
            .abs()
            .mean()
        )
        layer_losses.append(preserve + golden + batch_align)
    return torch.stack(layer_losses).sum()


def hard_balanced_edge_loss(
    affinity_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_probability: torch.Tensor,
    selected_target: torch.Tensor,
) -> torch.Tensor:
    target_labels = target_probability.argmax(dim=1)
    combined_labels = torch.cat((source_labels, target_labels))
    ideal = (combined_labels[:, None] == combined_labels[None, :]).to(
        affinity_logits.dtype
    )
    valid = torch.cat(
        (
            torch.ones_like(source_labels, dtype=torch.bool),
            selected_target,
        )
    )
    pairs = valid[:, None] & valid[None, :]
    return F.binary_cross_entropy_with_logits(
        affinity_logits[pairs], ideal[pairs]
    )


def train_step(
    variant: str,
    model: GSA_CAST,
    teacher: GSA_CAST | None,
    graph_module: SemanticAligner,
    source_batch,
    target_batch,
    optimizer,
    graph_optimizer,
    device,
    iteration: int,
    args,
) -> dict[str, float | int]:
    model.train()
    graph_module.train()
    source_batch = source_batch.to(device)
    target_batch = target_batch.to(device)
    source_labels = source_batch.y.view(-1).long()
    optimizer.zero_grad(set_to_none=True)
    graph_optimizer.zero_grad(set_to_none=True)

    if variant == "r-softsga":
        assert teacher is not None
        mc_probability = mc_teacher_probability(
            teacher, target_batch, args.mc_passes, args.temperature
        )
        target_probability, entropy, mi, reliability = entropy_mi_reliability(
            mc_probability, args.reliability_beta
        )
        selected, raw_counts = balanced_reliable_gate(
            target_probability,
            reliability,
            args.min_soft_confidence,
            args.min_reliability,
        )
    else:
        target_probability = entropy = mi = reliability = selected = raw_counts = None

    source_logits, _, source_domain, source_features, source_style, _ = model(
        source_batch.x,
        source_batch.edge_index,
        source_batch.batch,
        return_style_info=True,
    )
    target_logits, target_student_probability, target_domain, target_features, target_style, _ = model(
        target_batch.x,
        target_batch.edge_index,
        target_batch.batch,
        return_style_info=True,
    )

    if variant == "caga-balanced":
        target_probability = target_student_probability.detach()
        reliability = torch.ones(
            len(target_probability), device=device, dtype=target_probability.dtype
        )
        entropy = -(target_probability * target_probability.clamp_min(1e-8).log()).sum(1) / math.log(NUM_CLASSES)
        mi = torch.zeros_like(entropy)
        selected, raw_counts = balanced_reliable_gate(
            target_probability, reliability, 0.90, 0.0
        )

    classification = F.cross_entropy(source_logits, source_labels)
    domain_logits = torch.cat((source_domain, target_domain))
    domain_labels = torch.cat(
        (torch.zeros_like(source_domain), torch.ones_like(target_domain))
    )
    domain = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
    style = style_loss(
        source_style,
        target_style,
        reliability.detach() if variant == "r-softsga" else None,
    )

    combined_features = torch.cat((source_features, target_features))
    graph_logits, affinity_logits = graph_module(combined_features)
    node = F.cross_entropy(graph_logits[: len(source_labels)], source_labels)
    if variant == "r-softsga":
        ramp = graph_ramp_weight(
            iteration, args.graph_warmup_iters, args.graph_rampup_iters
        )
        edge, blocks = soft_graph_edge_loss(
            affinity_logits,
            source_labels,
            target_probability.detach(),
            reliability.detach(),
            selected,
            ramp,
        )
    else:
        ramp = 1.0
        edge = hard_balanced_edge_loss(
            affinity_logits, source_labels, target_probability, selected
        )
        zero = edge.detach() * 0.0
        blocks = {"ss": edge, "st": zero, "tt": zero}

    graph = edge + node
    total = classification + 0.1 * torch.clamp(domain, max=1.0) + graph + 0.1 * style
    total.backward()
    optimizer.step()
    graph_optimizer.step()
    if teacher is not None:
        update_teacher(teacher, model, args.teacher_ema)

    return {
        "iteration": iteration,
        "total": float(total.detach()),
        "classification": float(classification.detach()),
        "domain": float(domain.detach()),
        "style": float(style.detach()),
        "edge": float(edge.detach()),
        "node": float(node.detach()),
        "edge_ss": float(blocks["ss"].detach()),
        "edge_st": float(blocks["st"].detach()),
        "edge_tt": float(blocks["tt"].detach()),
        "graph_ramp": ramp,
        "entropy": float(entropy.mean()),
        "mutual_information": float(mi.mean()),
        "reliability": float(reliability.mean()),
        "selected_target": int(selected.sum()),
        "raw_positive": int(raw_counts[0]),
        "raw_neutral": int(raw_counts[1]),
        "raw_negative": int(raw_counts[2]),
    }


def _classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "pred_positive": int((prediction == 0).sum()),
        "pred_neutral": int((prediction == 1).sum()),
        "pred_negative": int((prediction == 2).sum()),
    }


def evaluate(model: GSA_CAST, loader: DataLoader, device, description: str) -> dict:
    """Final-only target evaluation at both window and trial levels."""

    model.eval()
    probabilities = []
    labels = []
    keys = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=description,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            labels.append(batch.y.view(-1).numpy())
            keys.append(
                np.stack(
                    (
                        batch.subject_id.view(-1).numpy(),
                        batch.session_id.view(-1).numpy(),
                        batch.trial_id.view(-1).numpy(),
                    ),
                    axis=1,
                )
            )
            batch = batch.to(device)
            _, probability, _, _ = model(batch.x, batch.edge_index, batch.batch)
            probabilities.append(probability.cpu().numpy())
    probability = np.concatenate(probabilities)
    label = np.concatenate(labels)
    key = np.concatenate(keys)
    window_metrics = _classification_metrics(label, probability)

    trial_probability = defaultdict(list)
    trial_label = {}
    for item_key, item_label, item_probability in zip(
        map(tuple, key.tolist()), label, probability
    ):
        trial_probability[item_key].append(item_probability)
        trial_label[item_key] = int(item_label)
    ordered_keys = sorted(trial_probability)
    pooled_probability = np.stack(
        [np.mean(trial_probability[item], axis=0) for item in ordered_keys]
    )
    pooled_label = np.asarray([trial_label[item] for item in ordered_keys])
    trial_metrics = _classification_metrics(pooled_label, pooled_probability)
    return {
        **{f"window_{key}": value for key, value in window_metrics.items()},
        **{f"trial_{key}": value for key, value in trial_metrics.items()},
        "windows": len(label),
        "trials": len(pooled_label),
    }


def _write_summary(result_dir: Path) -> None:
    results = [json.loads(path.read_text()) for path in sorted(result_dir.glob("subject_*.json"))]
    if not results:
        return
    metric_names = (
        "window_accuracy",
        "window_balanced_accuracy",
        "window_macro_f1",
        "trial_accuracy",
        "trial_balanced_accuracy",
        "trial_macro_f1",
    )
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std", "subjects"))
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


def run_fold(args, window_seconds: float, seed: int, subject: int, source, device):
    result_dir = (
        args.result_dir
        / args.variant
        / f"window_{window_seconds:g}s"
        / f"random_seed_{seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"subject_{subject:02d}.json"
    if result_path.exists() and not args.overwrite:
        tqdm.write(f"Skip existing {result_path}")
        return

    target = prepare_target(
        args.data_dir,
        window_seconds,
        subject,
        source,
        args.target_normalization,
    )
    fold_seed = seed
    set_seed(fold_seed)
    source_loader, target_loader, test_loader = make_loaders(
        source.dataset, target, args.batch_size, args.max_iters, fold_seed
    )
    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    source_batch, source_iterator = _next_batch(source_iterator, source_loader)
    target_batch, target_iterator = _next_batch(target_iterator, target_loader)

    model = GSA_CAST(
        input_feat_dim=NUM_BANDS,
        num_nodes=NUM_CHANNELS,
        d_model=args.d_model,
        n_head=args.num_heads,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout,
        topk=args.spatial_topk,
        num_classes=NUM_CLASSES,
    ).to(device)
    bootstrap_style(model, source_batch, target_batch, device)
    teacher = make_teacher(model) if args.variant == "r-softsga" else None
    graph_module = SemanticAligner(args.d_model, NUM_CLASSES, args.d_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    graph_optimizer = torch.optim.AdamW(
        graph_module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

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
            if iteration > 1:
                source_batch, source_iterator = _next_batch(source_iterator, source_loader)
                target_batch, target_iterator = _next_batch(target_iterator, target_loader)
            losses = train_step(
                args.variant,
                model,
                teacher,
                graph_module,
                source_batch,
                target_batch,
                optimizer,
                graph_optimizer,
                device,
                iteration,
                args,
            )
            if iteration == 1 or iteration % args.log_interval == 0:
                log_handle.write(json.dumps(losses) + "\n")
                log_handle.flush()
                iterations.set_postfix(
                    total=f"{losses['total']:.3f}",
                    cls=f"{losses['classification']:.3f}",
                    edge=f"{losses['edge']:.3f}",
                    reliability=f"{losses['reliability']:.2f}",
                    selected=losses["selected_target"],
                )

    metrics = evaluate(
        model,
        test_loader,
        device,
        description=f"Final eval {window_seconds:g}s/rnd{seed}/S{subject:02d}",
    )
    result = {
        "variant": args.variant,
        "source": "SEED-VII",
        "target": "SEED-V",
        "window_seconds": window_seconds,
        "random_seed": seed,
        "target_subject": subject,
        "iterations": args.max_iters,
        "target_normalization": args.target_normalization,
        **metrics,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary(result_dir)
    tqdm.write(
        f"Final {result_path}: trial_acc={metrics['trial_accuracy']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPUS-GA staged transfer training")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument(
        "--variant", choices=("caga-balanced", "r-softsga"), default="caga-balanced"
    )
    parser.add_argument("--window-seconds", nargs="+", type=float, default=(1, 2, 4))
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=(42,),
        help="optimization random seeds; unrelated to the SEED dataset names",
    )
    parser.add_argument("--target-subjects", type=_subjects, default="all")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-normalization", choices=("source", "domain"), default="source")
    parser.add_argument("--teacher-ema", type=float, default=0.99)
    parser.add_argument("--mc-passes", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reliability-beta", type=float, default=2.0)
    parser.add_argument("--min-soft-confidence", type=float, default=0.70)
    parser.add_argument("--min-reliability", type=float, default=0.20)
    parser.add_argument("--graph-warmup-iters", type=int, default=200)
    parser.add_argument("--graph-rampup-iters", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--spatial-topk", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args) -> None:
    if isinstance(args.target_subjects, str):
        args.target_subjects = _subjects(args.target_subjects)
    if args.batch_size < 2 or args.max_iters < 1 or args.mc_passes < 1:
        raise ValueError("batch-size >=2, max-iters >=1 and mc-passes >=1 are required")
    if args.d_model % args.num_heads:
        raise ValueError("d-model must be divisible by num-heads")
    if not 0 <= args.teacher_ema < 1:
        raise ValueError("teacher-ema must be in [0,1)")
    if any(scale not in {1.0, 2.0, 4.0} for scale in args.window_seconds):
        raise ValueError("window-seconds must be selected from 1, 2, 4")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.result_dir = args.result_dir.expanduser().resolve()
    device = torch.device(args.device)
    tqdm.write(
        f"Variant={args.variant}, device={device}, windows={args.window_seconds}, "
        f"random_seeds={args.random_seeds}, targets={args.target_subjects}"
    )
    total_folds = (
        len(args.window_seconds)
        * len(args.random_seeds)
        * len(args.target_subjects)
    )
    with tqdm(
        total=total_folds,
        desc="Overall transfer progress",
        unit="fold",
        dynamic_ncols=True,
    ) as overall:
        for window_seconds in args.window_seconds:
            tqdm.write(f"Loading {window_seconds:g}s SEED-VII source...")
            source = prepare_source(args.data_dir, window_seconds)
            tqdm.write(
                f"Source windows={len(source.dataset)}, classes="
                f"{summarize_labels(source.dataset.labels.numpy())}"
            )
            for seed in args.random_seeds:
                for subject in args.target_subjects:
                    run_fold(args, window_seconds, seed, subject, source, device)
                    overall.update(1)


if __name__ == "__main__":
    main()
