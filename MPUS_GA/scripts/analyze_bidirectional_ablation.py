#!/usr/bin/env python3
"""Paired subject-level analysis for the bidirectional ablation suite.

Each target subject is the statistical unit. Repeated random seeds are averaged
within a subject before any candidate/reference difference is computed. This
avoids treating repeated runs of the same target subject as independent data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.stats import wilcoxon


CLASS_NAMES = ("positive", "neutral", "negative")
EXPECTED_EXPERIMENTS = tuple(
    name
    for ablation in (*map(str, range(7)), "main")
    for name in (
        f"A_{ablation}" if ablation == "main" else f"A{ablation}",
        f"B_{ablation}" if ablation == "main" else f"B{ablation}",
    )
)


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    extract: Callable[[dict], float]
    larger_is_better: bool | None = True


@dataclass(frozen=True)
class Comparison:
    candidate: str
    reference: str
    family: str
    label: str

    @property
    def direction(self) -> str:
        return self.candidate[0]


def _fused(result: dict) -> dict:
    return result["evaluation"]["fused"]


METRICS = (
    Metric("accuracy", "Accuracy", lambda result: _fused(result)["accuracy"]),
    Metric(
        "balanced_accuracy",
        "Balanced Accuracy",
        lambda result: _fused(result)["balanced_accuracy"],
    ),
    Metric("macro_f1", "Macro-F1", lambda result: _fused(result)["macro_f1"]),
    Metric(
        "worst_class_recall",
        "Worst-class recall",
        lambda result: _fused(result)["worst_class_recall"],
    ),
    Metric(
        "recall_gap",
        "Recall gap",
        lambda result: _fused(result)["recall_gap"],
        larger_is_better=False,
    ),
    *(
        Metric(
            f"recall_{class_name}",
            f"Recall/{class_name}",
            lambda result, name=class_name: _fused(result)["per_class_recall"][
                name
            ],
        )
        for class_name in CLASS_NAMES
    ),
    *(
        Metric(
            f"prediction_rate_{class_name}",
            f"Prediction rate/{class_name}",
            lambda result, name=class_name: (
                _fused(result)["prediction_counts"][name]
                / _fused(result)["trials"]
            ),
            larger_is_better=None,
        )
        for class_name in CLASS_NAMES
    ),
)


COMPARISONS = tuple(
    Comparison(
        f"{direction}3",
        f"{direction}{scale_index}",
        "uniform_multiscale_vs_single_scale",
        f"uniform multiscale vs {scale_seconds} s",
    )
    for direction in "AB"
    for scale_index, scale_seconds in enumerate((1, 2, 4))
) + tuple(
    comparison
    for direction in "AB"
    for comparison in (
        Comparison(
            f"{direction}_main",
            f"{direction}3",
            "full_vs_uniform",
            "full class-conditional method vs uniform multiscale",
        ),
        Comparison(
            f"{direction}_main",
            f"{direction}4",
            "feature_pyramid",
            "feature-pyramid contribution",
        ),
        Comparison(
            f"{direction}_main",
            f"{direction}5",
            "boundary_attractor",
            "boundary-attractor contribution",
        ),
        Comparison(
            f"{direction}5",
            f"{direction}6",
            "source_excess",
            "source-excess contribution with boundary disabled",
        ),
        Comparison(
            f"{direction}_main",
            f"{direction}6",
            "full_vs_no_suppression",
            "full method vs both suppressors disabled",
        ),
    )
)


def _validate_fixed_final(result: dict, path: Path) -> None:
    protocol = result.get("protocol", {})
    expected = {
        "training_iterations": 1000,
        "checkpoint_selection": "none_final_iteration",
        "target_evaluations": 1,
        "selected_iteration": 1000,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Non-fixed-final result in {path}: {mismatches}")


def load_results(result_root: Path) -> dict[str, dict[tuple[int, int], dict]]:
    """Load and validate one result per (experiment, subject, seed)."""
    loaded: dict[str, dict[tuple[int, int], dict]] = {}
    for experiment in EXPECTED_EXPERIMENTS:
        experiment_dir = result_root / experiment
        paths = sorted(experiment_dir.glob("seed_*_subject_*.json"))
        if not paths:
            raise FileNotFoundError(f"No fold JSON files in {experiment_dir}")
        folds: dict[tuple[int, int], dict] = {}
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("experiment") != experiment:
                raise ValueError(
                    f"Experiment mismatch in {path}: {result.get('experiment')!r}"
                )
            _validate_fixed_final(result, path)
            key = (int(result["target_subject"]), int(result["random_seed"]))
            if key in folds:
                raise ValueError(f"Duplicate subject/seed {key} in {experiment_dir}")
            folds[key] = result
        loaded[experiment] = folds
    return loaded


def aggregate_subjects(
    results: dict[str, dict[tuple[int, int], dict]],
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[int, tuple[int, ...]]]]:
    """Average all metrics over seeds inside each target subject."""
    aggregated: dict[str, dict[int, dict[str, float]]] = {}
    seed_sets: dict[str, dict[int, tuple[int, ...]]] = {}
    for experiment, folds in results.items():
        by_subject: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for (subject, seed), result in folds.items():
            by_subject[subject].append((seed, result))
        aggregated[experiment] = {}
        seed_sets[experiment] = {}
        for subject, seeded_results in sorted(by_subject.items()):
            seeded_results.sort(key=lambda item: item[0])
            seed_sets[experiment][subject] = tuple(
                seed for seed, _ in seeded_results
            )
            aggregated[experiment][subject] = {
                metric.name: float(
                    np.mean([metric.extract(result) for _, result in seeded_results])
                )
                for metric in METRICS
            }
    return aggregated, seed_sets


def _paired_values(
    subject_metrics: dict[str, dict[int, dict[str, float]]],
    seed_sets: dict[str, dict[int, tuple[int, ...]]],
    comparison: Comparison,
    metric: Metric,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    candidate_subjects = set(subject_metrics[comparison.candidate])
    reference_subjects = set(subject_metrics[comparison.reference])
    if candidate_subjects != reference_subjects:
        raise ValueError(
            f"Subject mismatch for {comparison.candidate}/{comparison.reference}: "
            f"candidate={sorted(candidate_subjects)}, "
            f"reference={sorted(reference_subjects)}"
        )
    subjects = tuple(sorted(candidate_subjects))
    for subject in subjects:
        candidate_seeds = seed_sets[comparison.candidate][subject]
        reference_seeds = seed_sets[comparison.reference][subject]
        if candidate_seeds != reference_seeds:
            raise ValueError(
                f"Seed mismatch for subject {subject}, "
                f"{comparison.candidate}/{comparison.reference}: "
                f"{candidate_seeds} != {reference_seeds}"
            )
    candidate = np.asarray(
        [subject_metrics[comparison.candidate][s][metric.name] for s in subjects],
        dtype=float,
    )
    reference = np.asarray(
        [subject_metrics[comparison.reference][s][metric.name] for s in subjects],
        dtype=float,
    )
    return candidate, reference, subjects


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _wilcoxon_p(values: np.ndarray) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _rank_biserial(values: np.ndarray) -> float:
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.empty(len(nonzero), dtype=float)
    order = np.argsort(np.abs(nonzero), kind="mergesort")
    sorted_absolute = np.abs(nonzero)[order]
    position = 0
    while position < len(nonzero):
        end = position + 1
        while end < len(nonzero) and np.isclose(
            sorted_absolute[end], sorted_absolute[position]
        ):
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def _cohen_dz(values: np.ndarray) -> float:
    standard_deviation = float(values.std(ddof=1))
    if math.isclose(standard_deviation, 0.0):
        return 0.0 if math.isclose(float(values.mean()), 0.0) else math.nan
    return float(values.mean() / standard_deviation)


def _holm_adjust(rows: list[dict]) -> None:
    """Apply Holm correction separately inside every metric."""
    by_metric: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_metric[row["metric"]].append(index)
    for indices in by_metric.values():
        ordered = sorted(indices, key=lambda index: rows[index]["wilcoxon_p"])
        running_maximum = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * rows[index]["wilcoxon_p"])
            running_maximum = max(running_maximum, adjusted)
            rows[index]["wilcoxon_p_holm"] = running_maximum


def analyze(
    subject_metrics: dict[str, dict[int, dict[str, float]]],
    seed_sets: dict[str, dict[int, tuple[int, ...]]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []
    for comparison in COMPARISONS:
        for metric in METRICS:
            candidate, reference, subjects = _paired_values(
                subject_metrics, seed_sets, comparison, metric
            )
            raw_delta = candidate - reference
            if metric.larger_is_better is False:
                reported_change = -raw_delta
                change_orientation = "gap_reduction_positive"
            elif metric.larger_is_better is True:
                reported_change = raw_delta
                change_orientation = "higher_is_better"
            else:
                reported_change = raw_delta
                change_orientation = "descriptive_raw_change"
            ci_low, ci_high = _bootstrap_mean_ci(
                reported_change, rng, bootstrap_samples
            )
            rows.append(
                {
                    "direction": comparison.direction,
                    "family": comparison.family,
                    "label": comparison.label,
                    "candidate": comparison.candidate,
                    "reference": comparison.reference,
                    "metric": metric.name,
                    "metric_label": metric.label,
                    "subjects": len(subjects),
                    "candidate_mean": float(candidate.mean()),
                    "reference_mean": float(reference.mean()),
                    "candidate_minus_reference": float(raw_delta.mean()),
                    "reported_change": float(reported_change.mean()),
                    "reported_change_ci95_low": ci_low,
                    "reported_change_ci95_high": ci_high,
                    "change_orientation": change_orientation,
                    "wilcoxon_p": _wilcoxon_p(reported_change),
                    "cohen_dz": _cohen_dz(reported_change),
                    "rank_biserial": _rank_biserial(reported_change),
                }
            )
    _holm_adjust(rows)
    return rows


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_subject_metrics(
    path: Path,
    subject_metrics: dict[str, dict[int, dict[str, float]]],
    seed_sets: dict[str, dict[int, tuple[int, ...]]],
) -> None:
    rows = []
    for experiment in EXPECTED_EXPERIMENTS:
        for subject, values in sorted(subject_metrics[experiment].items()):
            rows.append(
                {
                    "experiment": experiment,
                    "direction": experiment[0],
                    "target_subject": subject,
                    "seeds": " ".join(map(str, seed_sets[experiment][subject])),
                    **values,
                }
            )
    _write_csv(path, rows)


def _format_estimate(row: dict) -> str:
    change = 100.0 * row["reported_change"]
    low = 100.0 * row["reported_change_ci95_low"]
    high = 100.0 * row["reported_change_ci95_high"]
    return f"{change:+.2f} [{low:+.2f}, {high:+.2f}]"


def _write_report(
    path: Path,
    rows: list[dict],
    result_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    primary_metrics = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "worst_class_recall",
        "recall_gap",
    }
    lookup = {
        (row["candidate"], row["reference"], row["metric"]): row for row in rows
    }
    lines = [
        "# Paired bidirectional ablation analysis",
        "",
        f"Result root: `{result_root}`",
        "",
        "The target subject is the statistical unit. The three random seeds are "
        "averaged within each subject before paired differences are calculated. "
        "All results were validated as fixed-final 1000-iteration runs with one "
        "target evaluation and no checkpoint selection.",
        "",
        f"Confidence intervals are percentile paired-bootstrap intervals "
        f"({bootstrap_samples:,} resamples, seed {bootstrap_seed}). Wilcoxon tests "
        "are two-sided. Holm correction is applied across all planned comparisons "
        "separately for each metric. Performance changes are oriented so that "
        "positive is always better; for recall gap it means a reduction. Prediction "
        "rates in the CSV are descriptive raw candidate-minus-reference changes.",
        "",
    ]
    for direction, description in (
        ("A", "SEED-VII to SEED-V"),
        ("B", "SEED-V to SEED-VII"),
    ):
        lines.extend(
            [
                f"## Direction {direction}: {description}",
                "",
                "| Comparison | Metric | Improvement pp [95% CI] | p | Holm p | dz | Rank-biserial |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in (item for item in COMPARISONS if item.direction == direction):
            for metric in (item for item in METRICS if item.name in primary_metrics):
                row = lookup[(comparison.candidate, comparison.reference, metric.name)]
                lines.append(
                    "| "
                    f"{comparison.candidate} − {comparison.reference} "
                    f"({comparison.label}) | {metric.label} | "
                    f"{_format_estimate(row)} | {row['wilcoxon_p']:.4g} | "
                    f"{row['wilcoxon_p_holm']:.4g} | {row['cohen_dz']:.3f} | "
                    f"{row['rank_biserial']:.3f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subject-level paired statistics for the bidirectional full ablation suite"
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results_bidirectional_full_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/bidirectional_full_ablation"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("--bootstrap-samples must be at least 1000")
    result_root = args.result_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(result_root)
    subject_metrics, seed_sets = aggregate_subjects(results)
    rows = analyze(
        subject_metrics,
        seed_sets,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_subject_metrics(
        output_dir / "subject_seed_averaged_metrics.csv",
        subject_metrics,
        seed_sets,
    )
    _write_csv(output_dir / "paired_comparisons.csv", rows)
    _write_report(
        output_dir / "paired_analysis.md",
        rows,
        result_root,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(
        f"Analyzed {sum(len(items) for items in subject_metrics.values())} "
        f"experiment-subject aggregates across {len(EXPECTED_EXPERIMENTS)} "
        f"experiments. Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
