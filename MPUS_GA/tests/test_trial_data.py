import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trial_data import (  # noqa: E402
    TrialSequenceDataset,
    UnlabeledTrialView,
    collate_trials,
)
from window_data import WindowArrays  # noqa: E402


def _arrays() -> WindowArrays:
    values = np.arange(5 * 62 * 5, dtype=np.float32).reshape(5, 62, 5)
    return WindowArrays(
        features=values,
        labels=np.array([1, 0, 1, 0, 0]),
        subjects=np.array([1, 1, 1, 1, 1]),
        sessions=np.array([1, 1, 1, 1, 1]),
        trials=np.array([2, 1, 2, 1, 1]),
        window_ids=np.array([1, 2, 0, 1, 0]),
    )


def test_trial_dataset_groups_and_orders_windows() -> None:
    arrays = _arrays()
    dataset = TrialSequenceDataset(
        arrays, np.zeros((62, 5), dtype=np.float32), np.ones((62, 5), dtype=np.float32)
    )
    assert len(dataset) == 2
    assert dataset.labels.tolist() == [0, 1]
    first = dataset[0]
    expected = torch.from_numpy(arrays.features[[4, 3, 1]])
    assert torch.equal(first["x"], expected)


def test_collate_trials_builds_padding_mask() -> None:
    arrays = _arrays()
    dataset = TrialSequenceDataset(
        arrays, np.zeros((62, 5), dtype=np.float32), np.ones((62, 5), dtype=np.float32)
    )
    batch = collate_trials([dataset[0], dataset[1]])
    assert batch["x"].shape == (2, 3, 62, 5)
    assert batch["window_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert torch.count_nonzero(batch["x"][1, 2]) == 0


def test_unlabeled_view_does_not_return_target_label() -> None:
    arrays = _arrays()
    dataset = TrialSequenceDataset(
        arrays, np.zeros((62, 5), dtype=np.float32), np.ones((62, 5), dtype=np.float32)
    )
    batch = collate_trials([UnlabeledTrialView(dataset)[0]])
    assert "y" not in batch
