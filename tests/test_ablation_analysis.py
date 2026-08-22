from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_bidirectional_ablation import (
    METRICS,
    Comparison,
    _holm_adjust,
    _paired_values,
    aggregate_subjects,
)


def _result(value: float) -> dict:
    return {
        "evaluation": {
            "fused": {
                "accuracy": value,
                "balanced_accuracy": value,
                "macro_f1": value,
                "worst_class_recall": value,
                "recall_gap": 1.0 - value,
                "per_class_recall": {
                    "positive": value,
                    "neutral": value,
                    "negative": value,
                },
                "prediction_counts": {
                    "positive": int(100 * value),
                    "neutral": 0,
                    "negative": 100 - int(100 * value),
                },
                "trials": 100,
            }
        }
    }


def test_seeds_are_averaged_before_subject_pairing() -> None:
    results = {
        "A_main": {
            (1, 42): _result(0.3),
            (1, 43): _result(0.6),
            (1, 44): _result(0.9),
            (2, 42): _result(0.4),
            (2, 43): _result(0.4),
            (2, 44): _result(0.4),
        },
        "A3": {
            (1, 42): _result(0.2),
            (1, 43): _result(0.2),
            (1, 44): _result(0.2),
            (2, 42): _result(0.1),
            (2, 43): _result(0.2),
            (2, 44): _result(0.3),
        },
    }
    subject_metrics, seed_sets = aggregate_subjects(results)
    assert subject_metrics["A_main"][1]["accuracy"] == pytest.approx(0.6)
    assert seed_sets["A_main"][1] == (42, 43, 44)

    accuracy = next(metric for metric in METRICS if metric.name == "accuracy")
    candidate, reference, subjects = _paired_values(
        subject_metrics,
        seed_sets,
        Comparison("A_main", "A3", "test", "test"),
        accuracy,
    )
    np.testing.assert_allclose(candidate - reference, (0.4, 0.2))
    assert subjects == (1, 2)


def test_pairing_rejects_different_seed_sets() -> None:
    subject_metrics = {
        "A_main": {1: {"accuracy": 0.5}},
        "A3": {1: {"accuracy": 0.4}},
    }
    seed_sets = {
        "A_main": {1: (42, 43, 44)},
        "A3": {1: (42, 43)},
    }
    accuracy = next(metric for metric in METRICS if metric.name == "accuracy")
    with pytest.raises(ValueError, match="Seed mismatch"):
        _paired_values(
            subject_metrics,
            seed_sets,
            Comparison("A_main", "A3", "test", "test"),
            accuracy,
        )


def test_holm_adjustment_is_monotone_in_p_value_order() -> None:
    rows = [
        {"metric": "accuracy", "wilcoxon_p": 0.01},
        {"metric": "accuracy", "wilcoxon_p": 0.04},
        {"metric": "accuracy", "wilcoxon_p": 0.03},
    ]
    _holm_adjust(rows)
    assert rows[0]["wilcoxon_p_holm"] == pytest.approx(0.03)
    assert rows[2]["wilcoxon_p_holm"] == pytest.approx(0.06)
    assert rows[1]["wilcoxon_p_holm"] == pytest.approx(0.06)
