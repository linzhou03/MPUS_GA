"""Train CAGA-SGA for a supported SEED-series -> SEED-V task."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from datapipe import (
    DEFAULT_DATA_ROOT,
    INPUT_FEATURE_DIM,
    NUM_CHANNELS,
    NUM_CLASSES,
    SOURCE_DATASETS,
    SOURCE_NAMES,
    TARGET_SUBJECTS,
    create_cross_dataset_setup,
    load_cross_dataset_fold,
)
from golden_style import ch_stats, gram
from graph_align import SemanticAligner
from model import GSA_CAST


# Paper Table I: one fixed configuration for every transfer task.
DEFAULT_BATCH_SIZE = 48
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_ITERS = 1000
DEFAULT_SEED = 42

D_MODEL = 64
NUM_ENCODER_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0.3
TOP_K = 8
GCN_HIDDEN_DIM = 64

TAU = 1.0
LAMBDA_DIS = 0.1
LAMBDA_STYLE = 0.1
LAMBDA_PRESERVE = 1.0
LAMBDA_GOLD = 1.0
LAMBDA_ALIGN = 1.0
LAMBDA_ALIGN_GRAM = 0.1
LAMBDA_EDGE = 1.0
LAMBDA_NODE = 1.0
PSEUDO_LABEL_THRESHOLD = 0.90


def select_target_pseudo_labels(
    target_probabilities: torch.Tensor,
    selection: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select pseudo-labels without letting a single target class dominate SGA.

    ``balanced`` applies the confidence threshold, requires every class to be
    represented, and retains equal top-confidence counts per class.
    ``threshold`` keeps the original unbalanced selection rule for ablations.
    """

    target_confidence, target_pseudo_label = target_probabilities.max(dim=1)
    confident = target_confidence > PSEUDO_LABEL_THRESHOLD
    raw_counts = torch.bincount(
        target_pseudo_label[confident], minlength=NUM_CLASSES
    )
    if selection == "threshold":
        return target_pseudo_label, confident, raw_counts

    selected = torch.zeros_like(confident)
    if torch.all(raw_counts > 0):
        quota = int(raw_counts.min().item())
        for class_id in range(NUM_CLASSES):
            candidates = torch.where(confident & (target_pseudo_label == class_id))[0]
            class_confidence = target_confidence[candidates]
            keep = candidates[torch.topk(class_confidence, k=quota).indices]
            selected[keep] = True
    return target_pseudo_label, selected, raw_counts


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def style_loss_per_layer(
    src_pre: torch.Tensor,
    src_post: torch.Tensor,
    tgt_pre: torch.Tensor,
    tgt_post: torch.Tensor,
    mu_g: torch.Tensor,
    sig_g: torch.Tensor,
) -> torch.Tensor:
    """Equation (13)-(16): content, golden-anchor and domain alignment."""

    mu_s0, std_s0 = ch_stats(src_pre)
    mu_s1, std_s1 = ch_stats(src_post)
    mu_t0, std_t0 = ch_stats(tgt_pre)
    mu_t1, std_t1 = ch_stats(tgt_post)

    preserve = (
        (mu_s1 - mu_s0).abs().mean()
        + (std_s1 - std_s0).abs().mean()
        + (mu_t1 - mu_t0).abs().mean()
        + (std_t1 - std_t0).abs().mean()
        + (gram(src_post) - gram(src_pre)).abs().mean()
        + (gram(tgt_post) - gram(tgt_pre)).abs().mean()
    )
    golden = (
        (mu_s1 - mu_g).abs().mean()
        + (std_s1 - sig_g).abs().mean()
        + (mu_t1 - mu_g).abs().mean()
        + (std_t1 - sig_g).abs().mean()
    )
    align = (
        (mu_s1 - mu_t1).abs().mean()
        + (std_s1 - std_t1).abs().mean()
        + LAMBDA_ALIGN_GRAM * (gram(src_post) - gram(tgt_post)).abs().mean()
    )
    return LAMBDA_PRESERVE * preserve + LAMBDA_GOLD * golden + LAMBDA_ALIGN * align


def _bootstrap_pair_if_needed(
    model: GSA_CAST, source_data, target_data
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
    if model.gsb.fully_bootstrapped:
        return None
    with torch.no_grad():
        *_, source_spatial = model(
            source_data.x,
            source_data.edge_index,
            source_data.batch,
            return_style_info=True,
        )
        *_, target_spatial = model(
            target_data.x,
            target_data.edge_index,
            target_data.batch,
            return_style_info=True,
        )
    return source_spatial, target_spatial


def train_step(
    model: GSA_CAST,
    graph_module: SemanticAligner,
    source_data,
    target_data,
    class_criterion,
    domain_criterion,
    model_optimizer,
    graph_optimizer,
    device: torch.device,
    pseudo_selection: str,
) -> dict[str, float | int]:
    model.train()
    graph_module.train()
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    source_label = source_data.y.view(-1).long()

    model_optimizer.zero_grad(set_to_none=True)
    graph_optimizer.zero_grad(set_to_none=True)

    bootstrap_pair = _bootstrap_pair_if_needed(model, source_data, target_data)
    source_out, _, source_domain, source_feature, source_style, _ = model(
        source_data.x,
        source_data.edge_index,
        source_data.batch,
        return_style_info=True,
        style_bootstrap_pair=bootstrap_pair,
    )
    target_out, _, target_domain, target_feature, target_style, _ = model(
        target_data.x,
        target_data.edge_index,
        target_data.batch,
        return_style_info=True,
        style_bootstrap_pair=bootstrap_pair,
    )

    classification_loss = class_criterion(source_out, source_label)

    domain_logits = torch.cat((source_domain, target_domain), dim=0)
    domain_labels = torch.cat(
        (torch.zeros_like(source_domain), torch.ones_like(target_domain)), dim=0
    )
    raw_domain_loss = domain_criterion(domain_logits, domain_labels)
    capped_domain_loss = torch.clamp(raw_domain_loss, max=TAU)

    combined_features = torch.cat((source_feature, target_feature), dim=0)
    graph_logits, affinity_logits = graph_module(combined_features)
    with torch.no_grad():
        target_pseudo_label, confident_target, raw_pseudo_counts = (
            select_target_pseudo_labels(
                F.softmax(target_out, dim=1), selection=pseudo_selection
            )
        )
        combined_labels = torch.cat((source_label, target_pseudo_label), dim=0)
        ideal_affinity = (combined_labels[:, None] == combined_labels[None, :]).to(
            affinity_logits.dtype
        )
        valid_nodes = torch.cat(
            (
                torch.ones_like(source_label, dtype=torch.bool),
                confident_target,
            ),
            dim=0,
        )
        valid_pairs = valid_nodes[:, None] & valid_nodes[None, :]

    edge_loss = F.binary_cross_entropy_with_logits(
        affinity_logits[valid_pairs], ideal_affinity[valid_pairs]
    )
    node_loss = class_criterion(graph_logits[: source_label.numel()], source_label)
    graph_loss = LAMBDA_EDGE * edge_loss + LAMBDA_NODE * node_loss

    per_layer_style_losses = [
        style_loss_per_layer(
            source_style[layer]["pre"],
            source_style[layer]["post"],
            target_style[layer]["pre"],
            target_style[layer]["post"],
            source_style[layer]["mu_g"],
            source_style[layer]["sig_g"],
        )
        for layer in range(len(source_style))
    ]
    style_loss = torch.stack(per_layer_style_losses).sum()

    total_loss = (
        classification_loss
        + LAMBDA_DIS * capped_domain_loss
        + graph_loss
        + LAMBDA_STYLE * style_loss
    )
    total_loss.backward()
    model_optimizer.step()
    graph_optimizer.step()

    return {
        "total": total_loss.item(),
        "classification": classification_loss.item(),
        "domain": raw_domain_loss.item(),
        "graph": graph_loss.item(),
        "style": style_loss.item(),
        "pseudo_ratio": confident_target.float().mean().item(),
        "pseudo_selected": int(confident_target.sum().item()),
        "pseudo_raw_positive": int(raw_pseudo_counts[0].item()),
        "pseudo_raw_neutral": int(raw_pseudo_counts[1].item()),
        "pseudo_raw_negative": int(raw_pseudo_counts[2].item()),
    }


def evaluate(
    model: GSA_CAST, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for data in loader:
            labels.append(data.y.view(-1).cpu().numpy())
            data = data.to(device)
            _, prediction, _, _ = model(data.x, data.edge_index, data.batch)
            probabilities.append(prediction.cpu().numpy())

    probability = np.vstack(probabilities)
    label = np.concatenate(labels).astype(np.int64, copy=False)
    predicted_label = probability.argmax(axis=1)
    one_hot_label = np.eye(NUM_CLASSES, dtype=np.float32)[label]
    try:
        auc = roc_auc_score(
            one_hot_label, probability, average="macro", multi_class="ovr"
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(label, predicted_label),
        "balanced_accuracy": balanced_accuracy_score(label, predicted_label),
        "f1_score": f1_score(label, predicted_label, average="macro", zero_division=0),
        "recall": recall_score(
            label, predicted_label, average="macro", zero_division=0
        ),
        "precision": precision_score(
            label, predicted_label, average="macro", zero_division=0
        ),
        "auc": auc,
        "pred_positive": int((predicted_label == 0).sum()),
        "pred_neutral": int((predicted_label == 1).sum()),
        "pred_negative": int((predicted_label == 2).sum()),
    }


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _make_loaders(
    source_dataset,
    target_dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    source_sampling: str,
):
    if len(target_dataset) == 0 or len(source_dataset) < batch_size:
        raise ValueError("Source and target datasets must contain enough samples")

    source_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    if source_sampling == "balanced":
        source_labels = source_dataset.labels.view(-1).long()
        class_counts = torch.bincount(source_labels, minlength=NUM_CLASSES)
        if torch.any(class_counts == 0):
            raise ValueError(f"Source dataset has an empty class: {class_counts.tolist()}")
        sample_weights = class_counts.float().reciprocal()[source_labels].double()
        source_samples = batch_size * (len(source_dataset) // batch_size)
        source_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=source_samples,
            replacement=True,
            generator=source_generator,
        )
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            sampler=source_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=source_generator,
        )
    target_samples = batch_size * len(source_loader)
    target_sampler = RandomSampler(
        target_dataset,
        replacement=True,
        num_samples=target_samples,
        generator=target_generator,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        sampler=target_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return source_loader, target_loader, test_loader


def _parse_subjects(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, TARGET_SUBJECTS + 1))

    subjects = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            subjects.extend(range(int(start_text), int(end_text) + 1))
        else:
            subjects.append(int(part))
    subjects = list(dict.fromkeys(subjects))
    if not subjects or any(
        subject < 1 or subject > TARGET_SUBJECTS for subject in subjects
    ):
        raise argparse.ArgumentTypeError(
            f"Target subjects must be within 1..{TARGET_SUBJECTS}"
        )
    return subjects


def _result_paths(result_dir: Path, source_dataset: str) -> tuple[Path, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    experiment = f"{SOURCE_NAMES[source_dataset]}_to_SEED-V_CAGA-SGA"
    version = 1
    while True:
        summary = result_dir / f"Summary_{experiment}_v{version}.csv"
        detailed = result_dir / f"Detailed_{experiment}_v{version}.csv"
        if not summary.exists() and not detailed.exists():
            return summary, detailed
        version += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce CAGA-SGA -> SEED-V")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--source-dataset",
        choices=SOURCE_DATASETS,
        default="seed-vii",
        help="labeled source domain; the target domain is SEED-V",
    )
    parser.add_argument("--target-subjects", type=_parse_subjects, default="all")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--source-sampling",
        choices=("balanced", "natural"),
        default="balanced",
        help="balanced prevents the coarse negative source class from dominating",
    )
    parser.add_argument(
        "--pseudo-selection",
        choices=("balanced", "threshold"),
        default="balanced",
        help="balanced admits equal confident pseudo-label counts from all classes",
    )
    parser.add_argument(
        "--target-normalization",
        choices=("source", "domain"),
        default="domain",
        help="domain handles pre-extracted DE scale mismatch; source follows Eq. (2)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="validate and load requested folds without constructing or training a model",
    )
    return parser


def _validate_args(args) -> None:
    if args.batch_size <= 1:
        raise ValueError("batch-size must be greater than 1")
    if args.max_iters <= 0:
        raise ValueError("max-iters must be positive")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {args.device}"
        )


def _subjects_from_arg(value: str | Iterable[int]) -> list[int]:
    return _parse_subjects(value) if isinstance(value, str) else list(value)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.target_subjects = _subjects_from_arg(args.target_subjects)
    _validate_args(args)
    create_cross_dataset_setup(args.data_root, source_dataset=args.source_dataset)

    if args.validate_data_only:
        for subject_id in args.target_subjects:
            load_cross_dataset_fold(
                subject_id - 1,
                args.data_root,
                target_normalization=args.target_normalization,
                source_dataset=args.source_dataset,
            )
        print("Data validation completed; no model was constructed or trained.")
        return

    device = torch.device(args.device)
    summary_path, detailed_path = _result_paths(args.result_dir, args.source_dataset)
    print(f"Transfer: {SOURCE_NAMES[args.source_dataset]} -> SEED-V")
    print(f"Device: {device}")
    print(
        f"Anti-collapse modes: source_sampling={args.source_sampling}, "
        f"pseudo_selection={args.pseudo_selection}, "
        f"target_normalization={args.target_normalization}"
    )
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results: {summary_path}")

    detailed_results = []
    class_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.BCEWithLogitsLoss()

    for subject_id in args.target_subjects:
        fold_seed = args.seed + subject_id - 1
        set_random_seed(fold_seed)
        print(f"\n--- Target subject {subject_id:02d} (seed={fold_seed}) ---")
        source_dataset, target_dataset = load_cross_dataset_fold(
            subject_id - 1,
            args.data_root,
            target_normalization=args.target_normalization,
            source_dataset=args.source_dataset,
        )
        source_loader, target_loader, test_loader = _make_loaders(
            source_dataset,
            target_dataset,
            args.batch_size,
            args.num_workers,
            fold_seed,
            pin_memory=device.type == "cuda",
            source_sampling=args.source_sampling,
        )

        model = GSA_CAST(
            input_feat_dim=INPUT_FEATURE_DIM,
            num_nodes=NUM_CHANNELS,
            d_model=D_MODEL,
            n_head=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT,
            topk=TOP_K,
            num_classes=NUM_CLASSES,
        ).to(device)
        graph_module = SemanticAligner(
            in_features=D_MODEL,
            num_classes=NUM_CLASSES,
            gcn_out_features=GCN_HIDDEN_DIM,
        ).to(device)
        model_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        graph_optimizer = torch.optim.AdamW(
            graph_module.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)
        for iteration in range(1, args.max_iters + 1):
            source_batch, source_iterator = _next_batch(source_iterator, source_loader)
            target_batch, target_iterator = _next_batch(target_iterator, target_loader)
            losses = train_step(
                model,
                graph_module,
                source_batch,
                target_batch,
                class_criterion,
                domain_criterion,
                model_optimizer,
                graph_optimizer,
                device,
                args.pseudo_selection,
            )
            if iteration == 1 or iteration % args.log_interval == 0:
                print(
                    f"S{subject_id:02d}|I{iteration:04d}/{args.max_iters} "
                    f"total={losses['total']:.4f} "
                    f"cls={losses['classification']:.4f} "
                    f"domain={losses['domain']:.4f} "
                    f"graph={losses['graph']:.4f} "
                    f"style={losses['style']:.4f} "
                    f"pseudo={losses['pseudo_selected']}/{args.batch_size} "
                    "raw="
                    f"{losses['pseudo_raw_positive']}/"
                    f"{losses['pseudo_raw_neutral']}/"
                    f"{losses['pseudo_raw_negative']}"
                )

        # Target labels are accessed only here, after the fixed training budget.
        metrics = evaluate(model, test_loader, device)
        detailed_results.append(
            {
                "subject": subject_id,
                "source_dataset": args.source_dataset,
                "iterations": args.max_iters,
                "seed": fold_seed,
                "source_sampling": args.source_sampling,
                "pseudo_selection": args.pseudo_selection,
                "target_normalization": args.target_normalization,
                **metrics,
            }
        )
        pd.DataFrame(detailed_results).to_csv(detailed_path, index=False)
        print(
            f"Subject {subject_id:02d} final: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            "predicted(pos/neu/neg)="
            f"{metrics['pred_positive']}/{metrics['pred_neutral']}/"
            f"{metrics['pred_negative']}"
        )

    detailed_frame = pd.DataFrame(detailed_results)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "recall",
        "precision",
        "auc",
    ]
    summary_frame = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean": [detailed_frame[column].mean() for column in metric_columns],
            "std": [detailed_frame[column].std(ddof=1) for column in metric_columns],
            "subjects": len(detailed_frame),
        }
    )
    summary_frame.to_csv(summary_path, index=False)
    print(
        "\nFinal accuracy: "
        f"{detailed_frame['accuracy'].mean():.4f} +/- "
        f"{detailed_frame['accuracy'].std(ddof=1):.4f}"
    )


if __name__ == "__main__":
    main()
