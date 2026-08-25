from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MPUS_GA.trial_temporal.losses import (  # noqa: E402
    class_conditional_prototype_alignment_loss,
)
from MPUS_GA.trial_temporal.data import _artifact_files  # noqa: E402
from MPUS_GA.trial_temporal.model import (  # noqa: E402
    EEGChannelAttention,
    MultiScaleMultiSourceDANN,
)
from MPUS_GA.trial_temporal.train import (  # noqa: E402
    EVALUATION_PROTOCOL_CAGA_TARGET_BEST,
    EVALUATION_PROTOCOL_FIXED_FINAL,
    EXPERIMENT_ORDER,
    EXPERIMENTS,
    PrototypeBank,
    TargetPriorEstimator,
    _adaptation_ramp,
    _class_balanced_domain_ce,
    _gate_supervision_loss,
    _pyramid_gate_supervision_loss,
    _should_evaluate_target,
    _target_evaluation_is_better,
    _target_subjects,
    collect_unlabeled_target_evidence,
    independent_scale_consensus,
    train_step,
)


def _small_model(**overrides) -> MultiScaleMultiSourceDANN:
    arguments = {
        "scales": (1.0,),
        "num_domains": 3,
        "d_model": 32,
        "num_heads": 4,
        "spatial_layers": 1,
        "temporal_layers": 1,
        "fusion_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "spatial_topk": 4,
        "use_channel_attention": True,
        "fusion_mode": "class_conditional",
        "domain_mode": "scale_conditional",
        "detach_domain_probability": True,
    }
    arguments.update(overrides)
    return MultiScaleMultiSourceDANN(**arguments)


def test_channel_attention_shape_range_and_gradient() -> None:
    attention = EEGChannelAttention(num_channels=62, reduction=4)
    x = torch.randn(5, 62, 5, requires_grad=True)
    output, weights = attention(x)
    assert output.shape == x.shape
    assert weights.shape == (5, 62)
    assert torch.all((weights > 0.5) & (weights < 1.5))
    output.square().mean().backward()
    assert x.grad is not None


def test_class_conditional_fusion_uses_external_scale_class_reliability() -> None:
    model = _small_model(scales=(1.0, 2.0), num_domains=2)
    for parameter in model.class_scale_gate.parameters():
        torch.nn.init.zeros_(parameter)
    embeddings = torch.randn(3, 2, 32)
    logits = torch.randn(3, 2, 3)
    reliability = torch.tensor([[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]])
    fusion = model._fuse(embeddings, logits, reliability)
    class_weight = fusion["scale_class_weight"]
    torch.testing.assert_close(
        class_weight.sum(dim=1), torch.ones(3, 3), atol=1e-6, rtol=1e-6
    )
    assert torch.all(class_weight[:, 0, 0] > class_weight[:, 1, 0])
    assert torch.all(class_weight[:, 1, 1] > class_weight[:, 0, 1])


def test_weighted_pyramid_keeps_one_feature_per_class() -> None:
    model = _small_model(scales=(1.0, 2.0, 4.0), num_domains=2)
    embeddings = torch.randn(4, 3, 32)
    logits = torch.randn(4, 3, 3)
    reliability = torch.tensor(
        [[0.7, 0.2, 0.3], [0.2, 0.7, 0.2], [0.1, 0.1, 0.5]]
    )
    fusion = model._fuse(embeddings, logits, reliability)
    fused_logits = fusion["logits"]
    fused_embedding = fusion["embedding"]
    class_weight = fusion["scale_class_weight"]
    pyramid_class_weight = fusion["pyramid_scale_class_weight"]
    class_features = fusion["pyramid_class_features"]
    pyramid_logits = fusion["pyramid_logits"]
    residual_gate = fusion["pyramid_gate"]
    assert fused_logits.shape == (4, 3)
    assert fused_embedding.shape == (4, 32)
    assert class_weight.shape == (4, 3, 3)
    assert pyramid_class_weight.shape == (4, 3, 3)
    assert class_features.shape == (4, 3, 32)
    assert pyramid_logits.shape == (4, 3)
    torch.testing.assert_close(class_weight.sum(dim=1), torch.ones(4, 3))
    torch.testing.assert_close(
        pyramid_class_weight.sum(dim=1), torch.ones(4, 3)
    )
    torch.testing.assert_close(
        residual_gate, torch.full((4, 3), 0.05), atol=1e-6, rtol=1e-6
    )
    expected_anchor = (class_weight * logits).sum(dim=1)
    torch.testing.assert_close(
        fused_logits,
        (1.0 - residual_gate) * expected_anchor
        + residual_gate * pyramid_logits,
    )


def test_feature_pyramid_ablation_uses_only_relation_weighted_logits() -> None:
    model = _small_model(
        scales=(1.0, 2.0, 4.0),
        num_domains=2,
        use_feature_pyramid=False,
    )
    embeddings = torch.randn(4, 3, 32)
    logits = torch.randn(4, 3, 3)
    reliability = torch.tensor(
        [[0.7, 0.2, 0.3], [0.2, 0.7, 0.2], [0.1, 0.1, 0.5]]
    )
    fusion = model._fuse(embeddings, logits, reliability)
    fused_logits = fusion["logits"]
    class_weight = fusion["scale_class_weight"]
    pyramid_class_weight = fusion["pyramid_scale_class_weight"]
    pyramid_logits = fusion["pyramid_logits"]
    residual_gate = fusion["pyramid_gate"]
    expected = (class_weight * logits).sum(dim=1)
    torch.testing.assert_close(fused_logits, expected)
    torch.testing.assert_close(pyramid_logits, expected)
    torch.testing.assert_close(pyramid_class_weight, class_weight)
    torch.testing.assert_close(residual_gate, torch.zeros(4, 3))
    assert len(model.pyramid_projections) == 0
    assert not model.use_feature_pyramid


def test_class_static_pyramid_gate_has_safe_anchor_warmup() -> None:
    model = _small_model(
        scales=(1.0, 2.0, 4.0),
        num_domains=2,
        pyramid_gate_mode="class_static",
        pyramid_residual_clip=0.5,
    )
    desired_gate = torch.tensor([0.2, 0.5, 0.8])
    with torch.no_grad():
        model.pyramid_class_gate_logit.copy_(torch.logit(desired_gate))
    embeddings = torch.randn(4, 3, 32)
    logits = torch.randn(4, 3, 3)
    reliability = torch.full((3, 3), 1.0 / 3.0)

    warmup = model._fuse(
        embeddings, logits, reliability, pyramid_gate_ramp=0.0
    )
    anchor = (
        warmup["scale_class_weight"] * logits
    ).sum(dim=1)
    torch.testing.assert_close(warmup["logits"], anchor)
    torch.testing.assert_close(warmup["pyramid_gate"], torch.zeros(4, 3))

    active = model._fuse(
        embeddings, logits, reliability, pyramid_gate_ramp=1.0
    )
    torch.testing.assert_close(
        active["pyramid_raw_gate"], desired_gate.expand(4, -1)
    )
    assert torch.all(active["pyramid_logit_delta"].abs() <= 0.5)


def test_sample_class_gate_starts_from_class_prior() -> None:
    model = _small_model(
        scales=(1.0, 2.0, 4.0),
        num_domains=2,
        pyramid_gate_mode="sample_class",
    )
    embeddings = torch.randn(4, 3, 32)
    logits = torch.randn(4, 3, 3)
    reliability = torch.full((3, 3), 1.0 / 3.0)
    fusion = model._fuse(embeddings, logits, reliability)
    assert fusion["pyramid_raw_gate"].shape == (4, 3)
    torch.testing.assert_close(
        fusion["pyramid_raw_gate"],
        torch.full((4, 3), 0.05),
        atol=1e-6,
        rtol=1e-6,
    )


def test_pyramid_bias_guard_suppresses_only_risky_class() -> None:
    model = _small_model(
        scales=(1.0, 2.0, 4.0),
        num_domains=2,
        pyramid_gate_mode="sample_class",
        use_pyramid_bias_guard=True,
        pyramid_guard_strength=4.0,
        pyramid_guard_tolerance=1.0,
    )
    embeddings = torch.randn(4, 3, 32)
    logits = torch.randn(4, 3, 3)
    reliability = torch.full((3, 3), 1.0 / 3.0)
    fusion = model._fuse(
        embeddings,
        logits,
        reliability,
        pyramid_bias_risk=torch.tensor([0.0, 1.0, 0.0]),
    )
    guard = fusion["pyramid_guard_factor"]
    torch.testing.assert_close(guard[:, 0], torch.ones(4))
    torch.testing.assert_close(guard[:, 2], torch.ones(4))
    torch.testing.assert_close(
        guard[:, 1], torch.full((4,), math.exp(-4.0))
    )
    assert torch.all(
        fusion["pyramid_gate"][:, 1]
        < fusion["pyramid_gate"][:, [0, 2]].min(dim=1).values
    )


def test_pyramid_gate_supervision_rewards_useful_class_changes() -> None:
    relation = torch.zeros(2, 3)
    pyramid = torch.tensor(
        [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]]
    )
    labels = torch.tensor([0, 1])
    matching_gate = torch.full((2, 3), 0.9, requires_grad=True)
    opposing_gate = torch.full((2, 3), 0.1, requires_grad=True)
    matching_loss = _pyramid_gate_supervision_loss(
        relation, pyramid, matching_gate, labels, 0.1, 0.05
    )
    opposing_loss = _pyramid_gate_supervision_loss(
        relation, pyramid, opposing_gate, labels, 0.1, 0.05
    )
    assert matching_loss < opposing_loss
    matching_loss.backward()
    assert matching_gate.grad is not None


def test_uniform_fusion_has_no_inactive_gate_supervision_constant() -> None:
    logits = torch.randn(4, 3, 3, requires_grad=True)
    uniform_weight = torch.full((4, 3, 3), 1.0 / 3.0)
    loss = _gate_supervision_loss(
        logits, uniform_weight, torch.tensor([0, 1, 2, 0]), 0.25
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_target_prior_estimator_recovers_multiscale_label_shift() -> None:
    estimator = TargetPriorEstimator(
        domain_count=1,
        scale_count=3,
        class_count=3,
        source_prior=torch.tensor([0.2, 0.2, 0.6]),
        device=torch.device("cpu"),
        momentum=0.0,
        ridge=0.0,
        prior_floor=0.001,
    )
    labels = torch.tensor([0, 1, 2])
    source_logits = torch.full((3, 3, 3), -8.0)
    for class_index in range(3):
        source_logits[class_index, :, class_index] = 8.0
    estimator.update_source(0, source_logits, labels)

    target_labels = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 2, 2])
    target_logits = torch.full((10, 3, 3), -8.0)
    for row, class_index in enumerate(target_labels.tolist()):
        target_logits[row, :, class_index] = 8.0
    estimator.update_target(target_logits)
    torch.testing.assert_close(
        estimator.estimated_prior,
        torch.tensor([0.3, 0.1, 0.6]),
        atol=1e-4,
        rtol=1e-4,
    )
    adjustment = estimator.logit_adjustment(0.5)
    assert adjustment[0] > 0
    assert adjustment[1] < 0
    assert abs(float(adjustment[2])) < 1e-4


def test_common_bias_suppression_requires_shared_excess_and_false_positive_risk() -> None:
    estimator = TargetPriorEstimator(
        domain_count=1,
        scale_count=3,
        class_count=3,
        source_prior=torch.tensor([0.25, 0.25, 0.50]),
        natural_source_prior=torch.tensor([0.20, 0.20, 0.60]),
        device=torch.device("cpu"),
        momentum=0.0,
    )
    soft_confusion = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.70, 0.30],
            [0.10, 0.20, 0.60],
        ]
    )
    estimator.source_confusion[0] = soft_confusion.unsqueeze(0).expand(3, -1, -1)
    estimator.source_hard_confusion[0] = soft_confusion.unsqueeze(0).expand(
        3, -1, -1
    )
    estimator.source_initialized.fill_(True)
    estimator.target_initialized = True
    estimator.target_mean_probability.copy_(
        torch.tensor(
            [
                [0.20, 0.50, 0.30],
                [0.20, 0.50, 0.30],
                [0.30, 0.30, 0.40],
            ]
        )
    )
    estimator.target_hard_frequency.copy_(estimator.target_mean_probability)
    adjustment = estimator.common_bias_adjustment(2.0, 0.10, 0.50)
    assert adjustment[1] < 0
    assert adjustment[0] == 0
    assert adjustment[2] == 0
    assert adjustment.min() >= -0.50

    # A single noisy scale cannot trigger the median cross-scale detector.
    estimator.target_mean_probability.copy_(
        torch.tensor(
            [
                [0.20, 0.50, 0.30],
                [0.25, 0.30, 0.45],
                [0.25, 0.30, 0.45],
            ]
        )
    )
    estimator.target_hard_frequency.copy_(estimator.target_mean_probability)
    isolated = estimator.common_bias_adjustment(2.0, 0.10, 0.50)
    torch.testing.assert_close(isolated, torch.zeros(3))

    # Common low-confidence hard decisions are caught even when mean soft
    # probabilities stay below the excess threshold.
    estimator.target_hard_frequency.copy_(
        torch.tensor(
            [
                [0.20, 0.50, 0.30],
                [0.20, 0.50, 0.30],
                [0.30, 0.30, 0.40],
            ]
        )
    )
    hard_vote_bias = estimator.common_bias_adjustment(2.0, 0.10, 0.50)
    assert hard_vote_bias[1] < 0


def test_boundary_attractor_requires_shared_hard_soft_gap() -> None:
    estimator = TargetPriorEstimator(
        domain_count=1,
        scale_count=3,
        class_count=3,
        source_prior=torch.tensor([0.25, 0.25, 0.50]),
        natural_source_prior=torch.tensor([0.20, 0.20, 0.60]),
        device=torch.device("cpu"),
        momentum=0.0,
    )
    confusion = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.70, 0.30],
            [0.10, 0.20, 0.60],
        ]
    )
    estimator.source_confusion[0] = confusion.unsqueeze(0).expand(3, -1, -1)
    estimator.source_hard_confusion[0] = confusion.unsqueeze(0).expand(
        3, -1, -1
    )
    estimator.source_initialized.fill_(True)
    estimator.target_initialized = True
    probability = torch.tensor(
        [
            [0.25, 0.35, 0.40],
            [0.25, 0.35, 0.40],
            [0.25, 0.35, 0.40],
        ]
    )
    estimator.target_mean_probability.copy_(probability)
    estimator.target_hard_frequency.copy_(probability)

    shared_hard = torch.tensor(
        [
            [0.15, 0.55, 0.30],
            [0.20, 0.50, 0.30],
            [0.25, 0.35, 0.40],
        ]
    )
    shared = estimator.common_bias_adjustments(
        source_excess_strength=0.0,
        source_relative_tolerance=0.10,
        boundary_strength=2.0,
        boundary_ratio_tolerance=0.15,
        maximum_adjustment=0.50,
        boundary_mean_probability=probability,
        boundary_hard_frequency=shared_hard,
    )
    assert shared["boundary_attractor"][1] < 0
    assert shared["combined"][1] < 0
    assert shared["combined"].min() >= -0.50
    assert shared["boundary_attractor"][0] == 0
    assert shared["boundary_attractor"][2] == 0

    # One noisy temporal scale is rejected by the cross-scale median.
    isolated_hard = probability.clone()
    isolated_hard[0] = torch.tensor([0.15, 0.55, 0.30])
    isolated = estimator.common_bias_adjustments(
        source_excess_strength=0.0,
        source_relative_tolerance=0.10,
        boundary_strength=2.0,
        boundary_ratio_tolerance=0.15,
        maximum_adjustment=0.50,
        boundary_mean_probability=probability,
        boundary_hard_frequency=isolated_hard,
    )
    torch.testing.assert_close(isolated["combined"], torch.zeros(3))

    # A confident target class shift has hard rates matching probability mass
    # and is not mistaken for a decision-boundary attractor.
    legitimate = estimator.common_bias_adjustments(
        source_excess_strength=0.0,
        source_relative_tolerance=0.10,
        boundary_strength=2.0,
        boundary_ratio_tolerance=0.15,
        maximum_adjustment=0.50,
        boundary_mean_probability=probability,
        boundary_hard_frequency=probability,
    )
    torch.testing.assert_close(legitimate["combined"], torch.zeros(3))


def test_final_unlabeled_refresh_changes_only_boundary_component() -> None:
    estimator = TargetPriorEstimator(
        1,
        3,
        3,
        torch.tensor([0.25, 0.25, 0.50]),
        torch.device("cpu"),
        momentum=0.0,
    )
    confusion = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.70, 0.30],
            [0.10, 0.20, 0.60],
        ]
    )
    estimator.source_confusion[0] = confusion.unsqueeze(0).expand(3, -1, -1)
    estimator.source_hard_confusion[0] = confusion.unsqueeze(0).expand(
        3, -1, -1
    )
    estimator.source_initialized.fill_(True)
    estimator.target_initialized = True
    probability = torch.tensor(
        [[0.25, 0.35, 0.40], [0.25, 0.35, 0.40], [0.25, 0.35, 0.40]]
    )
    estimator.target_mean_probability.copy_(probability)
    estimator.target_hard_frequency.copy_(probability)
    arguments = {
        "source_excess_strength": 2.0,
        "source_relative_tolerance": 0.10,
        "boundary_strength": 2.0,
        "boundary_ratio_tolerance": 0.15,
        "maximum_adjustment": 0.50,
    }
    ema = estimator.common_bias_adjustments(**arguments)
    final = estimator.common_bias_adjustments(
        **arguments,
        boundary_mean_probability=probability,
        boundary_hard_frequency=torch.tensor(
            [[0.15, 0.55, 0.30], [0.20, 0.50, 0.30], [0.25, 0.35, 0.40]]
        ),
    )
    torch.testing.assert_close(ema["source_excess"], final["source_excess"])
    assert ema["boundary_attractor"][1] == 0
    assert final["boundary_attractor"][1] < 0


def test_complete_target_evidence_refresh_is_unlabeled() -> None:
    class FixedScaleModel:
        def eval(self):
            return self

        def __call__(self, x, mask, compute_domain=False):
            batch_size = next(iter(x.values())).shape[0]
            logits = torch.tensor(
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
            )
            return {"scale_logits": logits.unsqueeze(0).expand(batch_size, -1, -1)}

    batch = {
        "x": {"1s": torch.zeros(2, 1, 62, 5)},
        "mask": {"1s": torch.ones(2, 1, dtype=torch.bool)},
    }
    snapshot = collect_unlabeled_target_evidence(
        FixedScaleModel(), [batch], torch.device("cpu"), "test evidence"
    )
    assert snapshot.trials == 2
    assert snapshot.mean_probability.shape == (3, 3)
    assert snapshot.hard_frequency.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    labeled_batch = dict(batch)
    labeled_batch["y"] = torch.tensor([0, 1])
    raised = False
    try:
        collect_unlabeled_target_evidence(
            FixedScaleModel(), [labeled_batch], torch.device("cpu"), "test"
        )
    except RuntimeError as error:
        raised = "unlabeled target view" in str(error)
    assert raised


def test_scale_logits_are_independent_before_cross_scale_context() -> None:
    model = _small_model(scales=(1.0, 2.0), num_domains=2)
    model.eval()
    first = {
        "1s": torch.randn(2, 3, 62, 5),
        "2s": torch.randn(2, 2, 62, 5),
    }
    second = {"1s": first["1s"].clone(), "2s": torch.randn(2, 2, 62, 5)}
    mask = {
        "1s": torch.ones(2, 3, dtype=torch.bool),
        "2s": torch.ones(2, 2, dtype=torch.bool),
    }
    with torch.no_grad():
        first_output = model(first, mask, compute_domain=False)
        second_output = model(second, mask, compute_domain=False)
    torch.testing.assert_close(
        first_output["scale_embeddings"][:, 0],
        second_output["scale_embeddings"][:, 0],
    )
    torch.testing.assert_close(
        first_output["scale_logits"][:, 0],
        second_output["scale_logits"][:, 0],
    )


def test_relation_graph_changes_only_fusion_not_independent_scale_logits() -> None:
    model = _small_model(scales=(1.0, 2.0), num_domains=2)
    model.eval()
    x = {
        "1s": torch.randn(3, 2, 62, 5),
        "2s": torch.randn(3, 2, 62, 5),
    }
    mask = {
        "1s": torch.ones(3, 2, dtype=torch.bool),
        "2s": torch.ones(3, 2, dtype=torch.bool),
    }
    first_relation = torch.tensor([[0.95, 0.05, 0.50], [0.05, 0.95, 0.50]])
    second_relation = torch.tensor([[0.05, 0.95, 0.50], [0.95, 0.05, 0.50]])
    with torch.no_grad():
        first = model(
            x,
            mask,
            compute_domain=False,
            scale_class_reliability=first_relation,
        )
        second = model(
            x,
            mask,
            compute_domain=False,
            scale_class_reliability=second_relation,
        )
    torch.testing.assert_close(first["scale_embeddings"], second["scale_embeddings"])
    torch.testing.assert_close(first["scale_logits"], second["scale_logits"])
    assert not torch.allclose(
        first["pyramid_class_features"], second["pyramid_class_features"]
    )
    assert not torch.allclose(first["logits"], second["logits"])


def test_prior_adjustment_changes_only_calibrated_independent_evidence() -> None:
    model = _small_model(scales=(1.0, 2.0), num_domains=2)
    model.eval()
    x = {
        "1s": torch.randn(2, 2, 62, 5),
        "2s": torch.randn(2, 2, 62, 5),
    }
    mask = {
        "1s": torch.ones(2, 2, dtype=torch.bool),
        "2s": torch.ones(2, 2, dtype=torch.bool),
    }
    with torch.no_grad():
        raw = model(x, mask, compute_domain=False)
        corrected = model(
            x,
            mask,
            compute_domain=False,
            class_logit_adjustment=torch.tensor([0.4, -0.6, 0.0]),
        )
    torch.testing.assert_close(raw["scale_logits"], corrected["scale_logits"])
    assert not torch.allclose(
        raw["calibrated_scale_logits"], corrected["calibrated_scale_logits"]
    )
    assert not torch.allclose(raw["logits"], corrected["logits"])


def test_scale_conditional_domain_head_detaches_classifier_probability() -> None:
    model = _small_model()
    x = {"1s": torch.randn(3, 2, 62, 5)}
    mask = {"1s": torch.ones(3, 2, dtype=torch.bool)}
    output = model(x, mask, compute_domain=True)
    assert output["scale_domain_logits"].shape == (3, 1, 3)
    output["scale_domain_logits"].square().mean().backward()
    classifier_gradient = model.classifier.weight.grad
    assert classifier_gradient is None or torch.count_nonzero(
        classifier_gradient
    ) == 0
    assert any(
        parameter.grad is not None
        for parameter in model.spatial.channel_attention.parameters()
    )


def test_prototype_bank_retains_missing_classes_and_normalizes_weights() -> None:
    bank = PrototypeBank(
        domain_count=2,
        scale_count=2,
        class_count=3,
        feature_dim=4,
        momentum=0.9,
        temperature=0.15,
        uniform_mix=0.1,
        device=torch.device("cpu"),
    )
    class_zero = torch.randn(4, 2, 4)
    bank.update_source(0, class_zero, torch.zeros(4, dtype=torch.long))
    saved = bank.source[0, :, 0].clone()
    bank.update_source(0, torch.randn(4, 2, 4), torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(bank.source[0, :, 0], saved)
    target_probability = torch.tensor(
        [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02]]
    )
    bank.update_target(torch.randn(2, 2, 4), target_probability, 0.6)
    joint = bank.joint_weights()
    torch.testing.assert_close(joint.sum(dim=(0, 1)), torch.ones(3))
    torch.testing.assert_close(
        bank.scale_class_reliability().sum(dim=0), torch.ones(3)
    )


def test_source_scale_class_relation_is_class_normalized_and_count_invariant() -> None:
    device = torch.device("cpu")
    labels = torch.tensor([0, 1, 2])
    logits = torch.zeros(3, 3, 3)
    logits[0, 0, 0] = 5.0
    logits[1, 1, 1] = 5.0
    logits[2, 2, 2] = 5.0
    first = PrototypeBank(1, 3, 3, 4, 0.9, 0.15, 0.1, device)
    second = PrototypeBank(1, 3, 3, 4, 0.9, 0.15, 0.1, device)
    first.update_source_relation(0, logits, labels)
    duplicated_logits = torch.cat((logits, logits[2:].repeat(8, 1, 1)))
    duplicated_labels = torch.cat((labels, labels[2:].repeat(8)))
    second.update_source_relation(0, duplicated_logits, duplicated_labels)
    first_relation = first.source_relation_weights()[0]
    second_relation = second.source_relation_weights()[0]
    torch.testing.assert_close(first_relation.sum(dim=0), torch.ones(3))
    torch.testing.assert_close(first_relation, second_relation)
    assert torch.equal(first_relation.argmax(dim=0), torch.tensor([0, 1, 2]))


def test_independent_scale_consensus_uses_votes_without_fused_predictions() -> None:
    logits = torch.tensor(
        [
            [[8.0, 0.0, 0.0], [7.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            [[0.0, 8.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    consensus = independent_scale_consensus(
        logits,
        confidence_threshold=0.6,
        js_divergence_threshold=1.0,
        minimum_votes=2,
    )
    assert consensus.valid_mask.tolist() == [True, False, True]
    assert consensus.pseudo_label.tolist() == [0, 0, 1]
    assert consensus.vote_count.tolist() == [3, 1, 2]


def test_class_balanced_domain_loss_ignores_majority_duplication() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [1.0, -0.5], [0.5, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 2])
    original = _class_balanced_domain_ce(logits, 0, labels)
    duplicated_logits = torch.cat((logits, logits[2:].repeat(8, 1)))
    duplicated_labels = torch.cat((labels, labels[2:].repeat(8)))
    duplicated = _class_balanced_domain_ce(
        duplicated_logits, 0, duplicated_labels
    )
    assert original is not None and duplicated is not None
    torch.testing.assert_close(original, duplicated)


def test_prototype_alignment_is_differentiable() -> None:
    source = [torch.randn(6, 2, 8, requires_grad=True)]
    labels = [torch.tensor([0, 1, 2, 0, 1, 2])]
    target = torch.randn(6, 2, 8, requires_grad=True)
    probability = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    joint = torch.full((1, 2, 3), 0.5)
    loss, coverage = class_conditional_prototype_alignment_loss(
        source, labels, target, probability, joint, 0.6
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert coverage == 1.0
    assert source[0].grad is not None
    assert target.grad is not None


def test_experiment_matrix_and_adaptation_schedule() -> None:
    assert EXPERIMENT_ORDER == (
        "A0",
        "B0",
        "A1",
        "B1",
        "A2",
        "B2",
        "A3",
        "B3",
        "A4",
        "B4",
        "A5",
        "B5",
        "A6",
        "B6",
        "A_main",
        "B_main",
        "A_G0",
        "B_G0",
        "A_G1",
        "B_G1",
        "A_G2",
        "B_G2",
        "A_G3",
        "B_G3",
    )
    assert set(EXPERIMENTS) == set(EXPERIMENT_ORDER)
    assert EXPERIMENTS["A0"].scales == (1.0,)
    assert EXPERIMENTS["A1"].scales == (2.0,)
    assert EXPERIMENTS["A2"].scales == (4.0,)
    assert not EXPERIMENTS["A0"].use_feature_pyramid
    assert EXPERIMENTS["A3"].fusion_mode == "uniform"
    assert not EXPERIMENTS["A4"].use_feature_pyramid
    assert EXPERIMENTS["A5"].use_source_excess_suppression
    assert not EXPERIMENTS["A5"].use_boundary_attractor_suppression
    assert not EXPERIMENTS["A6"].use_source_excess_suppression
    assert not EXPERIMENTS["A6"].use_boundary_attractor_suppression
    assert EXPERIMENTS["A_main"].use_feature_pyramid
    assert EXPERIMENTS["A_main"].use_source_excess_suppression
    assert EXPERIMENTS["A_main"].use_boundary_attractor_suppression
    assert EXPERIMENTS["A_main"].source_domains == ("seed_vii",)
    assert EXPERIMENTS["B6"].source_domains == ("seed_v",)
    assert EXPERIMENTS["B6"].target_dataset == "seed_vii"
    assert EXPERIMENTS["B6"].target_subject_count == 20
    assert EXPERIMENTS["B6"].target_trials == 80
    assert EXPERIMENTS["B6"].scales == EXPERIMENTS["A6"].scales
    assert EXPERIMENTS["B6"].fusion_mode == EXPERIMENTS["A6"].fusion_mode
    assert not EXPERIMENTS["A_G0"].use_feature_pyramid
    assert EXPERIMENTS["A_G1"].pyramid_gate_mode == "class_static"
    assert EXPERIMENTS["A_G1"].use_pyramid_gate_warmup
    assert EXPERIMENTS["A_G2"].pyramid_gate_mode == "sample_class"
    assert not EXPERIMENTS["A_G2"].use_pyramid_bias_guard
    assert EXPERIMENTS["A_G3"].pyramid_gate_mode == "sample_class"
    assert EXPERIMENTS["A_G3"].use_pyramid_bias_guard
    paired_fields = (
        "ablation",
        "scales",
        "fusion_mode",
        "domain_mode",
        "use_prototypes",
        "use_feature_pyramid",
        "pyramid_gate_mode",
        "use_pyramid_bias_guard",
        "use_pyramid_gate_warmup",
        "use_source_excess_suppression",
        "use_boundary_attractor_suppression",
        "domain_weight",
        "prototype_weight",
    )
    for ablation in (*map(str, range(7)), "main"):
        a_name = f"A_{ablation}" if ablation == "main" else f"A{ablation}"
        b_name = f"B_{ablation}" if ablation == "main" else f"B{ablation}"
        for field in paired_fields:
            assert getattr(EXPERIMENTS[a_name], field) == getattr(
                EXPERIMENTS[b_name], field
            )
    for variant in ("G0", "G1", "G2", "G3"):
        for field in paired_fields:
            assert getattr(EXPERIMENTS[f"A_{variant}"], field) == getattr(
                EXPERIMENTS[f"B_{variant}"], field
            )
    assert _target_subjects("all", 20) == list(range(1, 21))
    assert _adaptation_ramp(300, 300, 600) == 0.0
    assert 0.0 < _adaptation_ramp(450, 300, 600) < 1.0
    assert _adaptation_ramp(600, 300, 600) == 1.0


def test_target_evaluation_schedule_keeps_fixed_and_caga_protocols_separate() -> None:
    assert not _should_evaluate_target(
        EVALUATION_PROTOCOL_FIXED_FINAL, 50, 1000, 50
    )
    assert _should_evaluate_target(
        EVALUATION_PROTOCOL_FIXED_FINAL, 1000, 1000, 50
    )
    assert _should_evaluate_target(
        EVALUATION_PROTOCOL_CAGA_TARGET_BEST, 50, 1000, 50
    )
    assert not _should_evaluate_target(
        EVALUATION_PROTOCOL_CAGA_TARGET_BEST, 51, 1000, 50
    )
    assert _should_evaluate_target(
        EVALUATION_PROTOCOL_CAGA_TARGET_BEST, 1000, 1000, 50
    )


def test_caga_target_selection_uses_accuracy_and_retains_earliest_tie() -> None:
    incumbent = {"fused": {"accuracy": 0.6, "balanced_accuracy": 0.7}}
    lower_accuracy = {"fused": {"accuracy": 0.59, "balanced_accuracy": 0.9}}
    tied_accuracy = {"fused": {"accuracy": 0.6, "balanced_accuracy": 0.8}}
    higher_accuracy = {"fused": {"accuracy": 0.61, "balanced_accuracy": 0.5}}
    assert _target_evaluation_is_better(incumbent, None)
    assert not _target_evaluation_is_better(lower_accuracy, incumbent)
    assert not _target_evaluation_is_better(tied_accuracy, incumbent)
    assert _target_evaluation_is_better(higher_accuracy, incumbent)


def test_full_train_step_uses_unlabeled_target_and_updates_bank() -> None:
    device = torch.device("cpu")
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    labels = torch.tensor([0, 1, 2])

    def source_batch(domain_index: int) -> dict:
        return {
            "x": {"1s": torch.randn(3, 2, 62, 5)},
            "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
            "y": labels.clone(),
            "domain_id": torch.full((3,), domain_index, dtype=torch.long),
        }

    target_batch = {
        "x": {"1s": torch.randn(3, 2, 62, 5)},
        "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
        "domain_id": torch.full((3,), 2, dtype=torch.long),
    }
    bank = PrototypeBank(2, 1, 3, 32, 0.9, 0.15, 0.1, device)
    record = train_step(
        model,
        [source_batch(0), source_batch(1)],
        target_batch,
        optimizer,
        scheduler,
        device,
        iteration=600,
        spec=EXPERIMENTS["A0"],
        source_class_priors=torch.tensor(
            [[0.25, 0.20, 0.55], [0.20, 0.20, 0.60]]
        ),
        prototype_bank=bank,
        label_smoothing=0.1,
        gradient_clip=5.0,
        adaptation_warmup_iterations=300,
        adaptation_ramp_end=600,
        pseudo_confidence_threshold=0.6,
    )
    assert "y" not in target_batch
    assert math.isfinite(record["total"])
    assert bank.source_initialized.all()


def test_multiscale_relation_graph_train_step_updates_class_edges() -> None:
    device = torch.device("cpu")
    model = _small_model(scales=(1.0, 2.0, 4.0), num_domains=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    labels = torch.tensor([0, 1, 2])

    def features() -> dict:
        return {
            key: torch.randn(3, 2, 62, 5)
            for key in ("1s", "2s", "4s")
        }

    masks = {
        key: torch.ones(3, 2, dtype=torch.bool)
        for key in ("1s", "2s", "4s")
    }
    source_batch = {
        "x": features(),
        "mask": masks,
        "y": labels,
        "domain_id": torch.zeros(3, dtype=torch.long),
    }
    target_batch = {
        "x": features(),
        "mask": masks,
        "domain_id": torch.ones(3, dtype=torch.long),
    }
    bank = PrototypeBank(1, 3, 3, 32, 0.9, 0.15, 0.1, device)
    prior = TargetPriorEstimator(
        1,
        3,
        3,
        torch.tensor([0.3, 0.1, 0.6]),
        device,
        momentum=0.9,
    )
    record = train_step(
        model,
        [source_batch],
        target_batch,
        optimizer,
        scheduler,
        device,
        iteration=301,
        spec=EXPERIMENTS["A6"],
        source_class_priors=torch.tensor([[0.3, 0.1, 0.6]]),
        prototype_bank=bank,
        label_smoothing=0.1,
        gradient_clip=5.0,
        adaptation_warmup_iterations=300,
        adaptation_ramp_end=600,
        pseudo_confidence_threshold=0.0,
        consensus_jsd_threshold=1.0,
        consensus_minimum_votes=1,
        target_prior_estimator=prior,
    )
    assert bank.source_relation_initialized.all()
    torch.testing.assert_close(
        bank.source_relation_weights().sum(dim=1), torch.ones(1, 3)
    )
    assert record["scale_classification"] > 0
    assert record["gate_supervision"] >= 0
    assert record["prototype_updates_active"]
    assert "estimated_target_prior" in record
    assert record["prior_logit_adjustment"] == [0.0, 0.0, 0.0]
    assert record["common_bias_logit_adjustment"] == [0.0, 0.0, 0.0]
    assert record["boundary_attractor_logit_adjustment"] == [0.0, 0.0, 0.0]
    assert prior.source_initialized.all()
    assert prior.target_updates == 1


def test_g3_gate_guard_and_supervision_run_in_one_train_step() -> None:
    device = torch.device("cpu")
    model = _small_model(
        scales=(1.0, 2.0, 4.0),
        num_domains=2,
        pyramid_gate_mode="sample_class",
        use_pyramid_bias_guard=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    labels = torch.tensor([0, 1, 2])

    def features() -> dict:
        return {
            key: torch.randn(3, 2, 62, 5)
            for key in ("1s", "2s", "4s")
        }

    masks = {
        key: torch.ones(3, 2, dtype=torch.bool)
        for key in ("1s", "2s", "4s")
    }
    source_batch = {
        "x": features(),
        "mask": masks,
        "y": labels,
        "domain_id": torch.zeros(3, dtype=torch.long),
    }
    target_batch = {
        "x": features(),
        "mask": masks,
        "domain_id": torch.ones(3, dtype=torch.long),
    }
    prior = TargetPriorEstimator(
        1,
        3,
        3,
        torch.tensor([0.25, 0.25, 0.50]),
        device,
        natural_source_prior=torch.tensor([0.20, 0.20, 0.60]),
        momentum=0.9,
    )
    confusion = torch.tensor(
        [[0.80, 0.10, 0.10], [0.10, 0.70, 0.30], [0.10, 0.20, 0.60]]
    )
    prior.source_confusion[0] = confusion.unsqueeze(0).expand(3, -1, -1)
    prior.source_hard_confusion[0] = confusion.unsqueeze(0).expand(3, -1, -1)
    prior.source_initialized.fill_(True)
    prior.target_initialized = True
    prior.target_mean_probability.copy_(
        torch.tensor(
            [[0.20, 0.50, 0.30], [0.20, 0.50, 0.30], [0.30, 0.30, 0.40]]
        )
    )
    prior.target_hard_frequency.copy_(prior.target_mean_probability)
    before = model.pyramid_sample_gate[-1].bias.detach().clone()
    record = train_step(
        model,
        [source_batch],
        target_batch,
        optimizer,
        scheduler,
        device,
        iteration=600,
        spec=EXPERIMENTS["A_G3"],
        source_class_priors=torch.tensor([[0.20, 0.20, 0.60]]),
        prototype_bank=None,
        label_smoothing=0.1,
        gradient_clip=5.0,
        adaptation_warmup_iterations=300,
        adaptation_ramp_end=600,
        pseudo_confidence_threshold=0.6,
        target_prior_estimator=prior,
    )
    assert math.isfinite(record["total"])
    assert record["pyramid_gate_supervision"] >= 0
    assert record["pyramid_bias_risk"][1] > 0
    assert record["mean_target_pyramid_guard_by_class"][1] < 1.0
    assert not torch.equal(before, model.pyramid_sample_gate[-1].bias)


def test_train_step_rejects_target_labels_before_model_access() -> None:
    raised = False
    try:
        train_step(
            None,
            [],
            {"y": torch.tensor([0])},
            None,
            None,
            torch.device("cpu"),
            1,
            EXPERIMENTS["A6"],
            torch.ones(1, 3) / 3,
            None,
            0.1,
            5.0,
            300,
            600,
            0.6,
        )
    except RuntimeError as error:
        raised = "Target adaptation batch" in str(error)
    assert raised


def test_target_prototype_updates_start_after_adaptation_warmup() -> None:
    device = torch.device("cpu")
    model = _small_model(num_domains=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    source_batch = {
        "x": {"1s": torch.randn(3, 2, 62, 5)},
        "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
        "y": torch.tensor([0, 1, 2]),
        "domain_id": torch.zeros(3, dtype=torch.long),
    }
    target_batch = {
        "x": {"1s": torch.randn(3, 2, 62, 5)},
        "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
        "domain_id": torch.ones(3, dtype=torch.long),
    }
    bank = PrototypeBank(1, 1, 3, 32, 0.9, 0.15, 0.1, device)

    def run(iteration: int) -> dict:
        return train_step(
            model,
            [source_batch],
            target_batch,
            optimizer,
            scheduler,
            device,
            iteration=iteration,
            spec=EXPERIMENTS["B_main"],
            source_class_priors=torch.tensor([[0.20, 0.20, 0.60]]),
            prototype_bank=bank,
            label_smoothing=0.1,
            gradient_clip=5.0,
            adaptation_warmup_iterations=300,
            adaptation_ramp_end=600,
            pseudo_confidence_threshold=0.0,
        )

    warmup_record = run(1)
    assert bank.source_initialized.all()
    assert not bank.target_initialized.any()
    assert not warmup_record["prototype_updates_active"]
    assert warmup_record["pseudo_label_coverage"] == 0.0

    adaptation_record = run(301)
    assert bank.target_initialized.any()
    assert bank.target_class_updates.sum() > 0
    assert adaptation_record["prototype_updates_active"]


def test_seed_v_and_seed_vii_support_source_and_target_roles() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for dataset, subjects, sessions in (
            ("seed_v", 16, 3),
            ("seed_vii", 20, 4),
        ):
            scale_dir = root / dataset / "window_1s"
            scale_dir.mkdir(parents=True)
            for subject in range(1, subjects + 1):
                for session in range(1, sessions + 1):
                    (scale_dir / (
                        f"subject_{subject:02d}_session_{session}.npz"
                    )).touch()
            assert len(_artifact_files(root, dataset, 1.0, None)) == (
                subjects * sessions
            )
            assert len(_artifact_files(root, dataset, 1.0, 1)) == sessions


def test_b6_single_source_training_step() -> None:
    device = torch.device("cpu")
    model = _small_model(num_domains=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    labels = torch.tensor([0, 1, 2])
    source_batch = {
        "x": {"1s": torch.randn(3, 2, 62, 5)},
        "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
        "y": labels,
        "domain_id": torch.zeros(3, dtype=torch.long),
    }
    target_batch = {
        "x": {"1s": torch.randn(3, 2, 62, 5)},
        "mask": {"1s": torch.ones(3, 2, dtype=torch.bool)},
        "domain_id": torch.ones(3, dtype=torch.long),
    }
    bank = PrototypeBank(1, 1, 3, 32, 0.9, 0.15, 0.1, device)
    record = train_step(
        model,
        [source_batch],
        target_batch,
        optimizer,
        scheduler,
        device,
        iteration=600,
        spec=EXPERIMENTS["B_main"],
        source_class_priors=torch.tensor([[0.20, 0.20, 0.60]]),
        prototype_bank=bank,
        label_smoothing=0.1,
        gradient_clip=5.0,
        adaptation_warmup_iterations=300,
        adaptation_ramp_end=600,
        pseudo_confidence_threshold=0.6,
    )
    assert math.isfinite(record["total"])
    assert len(record["source_weights"]) == 1
    assert record["source_weights"][0] == 1.0
