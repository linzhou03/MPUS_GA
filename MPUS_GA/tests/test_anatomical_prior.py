from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MPUS_GA.trial_temporal.anatomical_prior import (  # noqa: E402
    ANATOMICAL_REGIONS,
    CHANNEL_NAMES,
    COMPACT_PHYSIOLOGY_DIM,
    PHYSIOLOGY_DIM,
    physiology_descriptor,
    region_weight_matrix_numpy,
)
from MPUS_GA.trial_temporal.data import (  # noqa: E402
    ScaleArrays,
    fit_physiology_stats,
    robust_standardize_physiology_by_subject,
)
from MPUS_GA.trial_temporal.model import MultiScaleMultiSourceDANN  # noqa: E402


def _model(**overrides) -> MultiScaleMultiSourceDANN:
    arguments = {
        "scales": (1.0,),
        "num_domains": 2,
        "d_model": 16,
        "num_heads": 4,
        "spatial_layers": 1,
        "temporal_layers": 1,
        "fusion_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "spatial_topk": 4,
        "use_anatomical_regions": True,
    }
    arguments.update(overrides)
    return MultiScaleMultiSourceDANN(**arguments)


def _arrays(offset: float) -> ScaleArrays:
    features = np.zeros((4, 62, 5), dtype=np.float32) + offset
    return ScaleArrays(
        features=features,
        labels=np.asarray([0, 0, 1, 1]),
        subjects=np.ones(4, dtype=np.int64),
        sessions=np.ones(4, dtype=np.int64),
        trials=np.asarray([1, 1, 2, 2]),
        window_ids=np.asarray([0, 1, 0, 1]),
    )


def test_anatomical_partition_is_exhaustive_disjoint_and_region_balanced() -> None:
    matrix = region_weight_matrix_numpy()
    assert matrix.shape == (8, 62)
    assert sum(len(channels) for channels in ANATOMICAL_REGIONS.values()) == 62
    assert len({name for region in ANATOMICAL_REGIONS.values() for name in region}) == 62
    assert set(name for region in ANATOMICAL_REGIONS.values() for name in region) == set(CHANNEL_NAMES)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(8))
    assert np.all((matrix > 0).sum(axis=0) == 1)


def test_raw_de_descriptor_uses_right_minus_left_frontal_alpha() -> None:
    raw = np.zeros((3, 62, 5), dtype=np.float32)
    right = ("FP2", "AF4", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6")
    for name in right:
        raw[:, CHANNEL_NAMES.index(name), 2] = 0.5
    descriptor = physiology_descriptor(raw)
    assert descriptor.shape == (PHYSIOLOGY_DIM,)
    assert descriptor[40] == pytest.approx(1.0)


def test_physiology_statistics_are_fitted_from_supplied_source_only() -> None:
    source_mean, source_std = fit_physiology_stats([_arrays(0.0)])
    repeated_mean, repeated_std = fit_physiology_stats([_arrays(0.0)])
    np.testing.assert_array_equal(source_mean, repeated_mean)
    np.testing.assert_array_equal(source_std, repeated_std)
    hypothetical_target = _arrays(0.0)
    hypothetical_target.features[:, CHANNEL_NAMES.index("FP2"), 2] = 10.0
    contaminated_mean, _ = fit_physiology_stats(
        [_arrays(0.0), hypothetical_target]
    )
    assert contaminated_mean[40] != source_mean[40]


def test_subject_robust_physiology_removes_subject_offsets_without_labels() -> None:
    descriptors = np.asarray(
        [[10.0, 1.0, 2.0, 3.0], [11.0, 2.0, 3.0, 4.0],
         [12.0, 3.0, 4.0, 5.0], [100.0, 8.0, 9.0, 10.0],
         [101.0, 9.0, 10.0, 11.0], [102.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    keys = tuple(
        (1 if index < 3 else 2, 1, index + 1)
        for index in range(len(descriptors))
    )
    standardised = robust_standardize_physiology_by_subject(
        descriptors, keys
    )
    np.testing.assert_allclose(np.median(standardised[:3], axis=0), 0.0)
    np.testing.assert_allclose(np.median(standardised[3:], axis=0), 0.0)


def test_n1_keeps_shared_r2_initialisation_and_adds_anatomical_signal() -> None:
    torch.manual_seed(17)
    baseline = _model(use_anatomical_regions=False)
    torch.manual_seed(17)
    anatomy = _model()
    baseline_state = baseline.state_dict()
    anatomy_state = anatomy.state_dict()
    for key, value in baseline_state.items():
        torch.testing.assert_close(value, anatomy_state[key])

    x = {"1s": torch.randn(2, 3, 62, 5)}
    mask = {"1s": torch.ones(2, 3, dtype=torch.bool)}
    baseline.eval()
    anatomy.eval()
    with torch.no_grad():
        baseline_logits = baseline(x, mask, compute_domain=False)["scale_logits"]
        anatomy_output = anatomy(x, mask, compute_domain=False)
    assert not torch.equal(baseline_logits, anatomy_output["scale_logits"])
    assert anatomy_output["anatomical_region_gate"].item() == pytest.approx(0.05)


def test_n2_requires_44d_descriptors_and_preserves_global_rng_stream() -> None:
    torch.manual_seed(23)
    _model()
    n1_next_random = torch.rand(4)
    torch.manual_seed(23)
    model = _model(use_physiology_prior=True)
    n2_next_random = torch.rand(4)
    torch.testing.assert_close(n1_next_random, n2_next_random)

    x = {"1s": torch.randn(2, 3, 62, 5)}
    mask = {"1s": torch.ones(2, 3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="requires descriptors"):
        model(x, mask, compute_domain=False)
    with pytest.raises(ValueError, match=r"\[batch,scales,44\]"):
        model(
            x,
            mask,
            compute_domain=False,
            physiology_by_scale={"1s": torch.zeros(2, PHYSIOLOGY_DIM - 1)},
        )
    output = model(
        x,
        mask,
        compute_domain=False,
        physiology_by_scale={"1s": torch.zeros(2, PHYSIOLOGY_DIM)},
    )
    assert output["physiology_gate"].shape == (1,)
    assert output["physiology_gate"].item() == pytest.approx(0.05)


def test_n3_reliability_head_cannot_change_initial_r2_logits() -> None:
    torch.manual_seed(31)
    baseline = _model(use_anatomical_regions=False)
    torch.manual_seed(31)
    reliability = _model(
        use_anatomical_regions=False,
        use_physiology_reliability=True,
    )
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, reliability.state_dict()[key])
    x = {"1s": torch.randn(3, 2, 62, 5)}
    mask = {"1s": torch.ones(3, 2, dtype=torch.bool)}
    physiology = {"1s": torch.randn(3, COMPACT_PHYSIOLOGY_DIM)}
    baseline.eval()
    reliability.eval()
    baseline_output = baseline(x, mask, compute_domain=False)
    reliability_output = reliability(
        x,
        mask,
        compute_domain=False,
        physiology_by_scale=physiology,
    )
    torch.testing.assert_close(
        baseline_output["scale_logits"], reliability_output["scale_logits"]
    )
    torch.testing.assert_close(
        baseline_output["logits"], reliability_output["logits"]
    )
    assert reliability_output["physiology_logits"].shape == (3, 1, 3)
