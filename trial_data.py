"""Trial-level datasets built from the independently computed DE windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from window_data import (
    NUM_BANDS,
    NUM_CHANNELS,
    SEED_V_SUBJECTS,
    WindowArrays,
    fit_source_stats,
    load_window_arrays,
)


@dataclass(frozen=True)
class PreparedTrialSource:
    train: "TrialSequenceDataset"
    validation: "TrialSequenceDataset"
    mean: np.ndarray
    std: np.ndarray


class TrialSequenceDataset(Dataset):
    """One item is one complete, chronologically ordered movie trial.

    Normalization is applied lazily per trial. This avoids materialising a
    second copy of the several-hundred-megabyte 1-second source array.
    """

    def __init__(
        self,
        arrays: WindowArrays,
        mean: np.ndarray,
        std: np.ndarray,
        window_indices: np.ndarray | None = None,
    ) -> None:
        if arrays.features.ndim != 3 or arrays.features.shape[1:] != (
            NUM_CHANNELS,
            NUM_BANDS,
        ):
            raise ValueError(
                f"Expected [N,{NUM_CHANNELS},{NUM_BANDS}], got "
                f"{arrays.features.shape}"
            )
        count = len(arrays.features)
        for name in ("labels", "subjects", "sessions", "trials", "window_ids"):
            if len(getattr(arrays, name)) != count:
                raise ValueError(f"{name} length does not match features")

        self.features = arrays.features
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        expected_stats_shape = (NUM_CHANNELS, NUM_BANDS)
        if self.mean.shape != expected_stats_shape or self.std.shape != expected_stats_shape:
            raise ValueError(
                f"mean/std must have shape {expected_stats_shape}, got "
                f"{self.mean.shape}/{self.std.shape}"
            )
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive")

        if window_indices is None:
            selected = np.arange(count, dtype=np.int64)
        else:
            selected = np.asarray(window_indices, dtype=np.int64)
            if selected.ndim != 1:
                raise ValueError("window_indices must be one-dimensional")
            if len(selected) == 0:
                raise ValueError("TrialSequenceDataset cannot be empty")
            if selected.min() < 0 or selected.max() >= count:
                raise IndexError("window_indices contains an out-of-range index")

        order = np.lexsort(
            (
                arrays.window_ids[selected],
                arrays.trials[selected],
                arrays.sessions[selected],
                arrays.subjects[selected],
            )
        )
        selected = selected[order]
        keys = np.stack(
            (
                arrays.subjects[selected],
                arrays.sessions[selected],
                arrays.trials[selected],
            ),
            axis=1,
        )
        boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
        groups = np.split(selected, boundaries)

        labels: list[int] = []
        subjects: list[int] = []
        sessions: list[int] = []
        trials: list[int] = []
        for indices in groups:
            group_labels = np.unique(arrays.labels[indices])
            if len(group_labels) != 1:
                key = (
                    int(arrays.subjects[indices[0]]),
                    int(arrays.sessions[indices[0]]),
                    int(arrays.trials[indices[0]]),
                )
                raise ValueError(f"Trial {key} contains inconsistent labels")
            window_ids = arrays.window_ids[indices]
            if len(np.unique(window_ids)) != len(window_ids):
                raise ValueError("A trial contains duplicate window_id values")
            labels.append(int(group_labels[0]))
            subjects.append(int(arrays.subjects[indices[0]]))
            sessions.append(int(arrays.sessions[indices[0]]))
            trials.append(int(arrays.trials[indices[0]]))

        self.groups = tuple(groups)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subjects = torch.tensor(subjects, dtype=torch.long)
        self.sessions = torch.tensor(sessions, dtype=torch.long)
        self.trials = torch.tensor(trials, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.groups)

    def _item(
        self, index: int, include_label: bool
    ) -> dict[str, torch.Tensor]:
        indices = self.groups[index]
        features = (self.features[indices] - self.mean) / self.std
        item = {
            "x": torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            "subject_id": self.subjects[index],
            "session_id": self.sessions[index],
            "trial_id": self.trials[index],
        }
        if include_label:
            item["y"] = self.labels[index]
        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._item(index, include_label=True)


class UnlabeledTrialView(Dataset):
    """Training view that does not retrieve target labels from the dataset."""

    def __init__(self, dataset: TrialSequenceDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset._item(index, include_label=False)


def collate_trials(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length trials and return True for real window positions."""

    if not items:
        raise ValueError("Cannot collate an empty trial batch")
    max_windows = max(len(item["x"]) for item in items)
    batch_size = len(items)
    features = items[0]["x"].new_zeros(
        (batch_size, max_windows, NUM_CHANNELS, NUM_BANDS)
    )
    window_mask = torch.zeros((batch_size, max_windows), dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item["x"])
        if length == 0:
            raise ValueError("A trial must contain at least one window")
        features[row, :length] = item["x"]
        window_mask[row, :length] = True
    batch = {
        "x": features,
        "window_mask": window_mask,
        "subject_id": torch.stack([item["subject_id"] for item in items]),
        "session_id": torch.stack([item["session_id"] for item in items]),
        "trial_id": torch.stack([item["trial_id"] for item in items]),
    }
    label_presence = ["y" in item for item in items]
    if any(label_presence) and not all(label_presence):
        raise ValueError("A batch cannot mix labeled and unlabeled trials")
    if all(label_presence):
        batch["y"] = torch.stack([item["y"] for item in items])
    return batch


def _fit_stats_subset(
    features: np.ndarray, indices: np.ndarray, chunk_size: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Fit source-only statistics without copying the complete source array."""

    total = np.zeros((NUM_CHANNELS, NUM_BANDS), dtype=np.float64)
    squared = np.zeros_like(total)
    for start in range(0, len(indices), chunk_size):
        chunk = features[indices[start : start + chunk_size]].astype(
            np.float64, copy=False
        )
        total += chunk.sum(axis=0)
        squared += np.square(chunk).sum(axis=0)
    mean = total / len(indices)
    variance = np.maximum(squared / len(indices) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return mean.astype(np.float32), np.maximum(std.astype(np.float32), 1e-5)


def prepare_trial_source(
    data_dir: Path,
    window_seconds: float,
    validation_subjects: Sequence[int] = (17, 18, 19, 20),
) -> PreparedTrialSource:
    arrays = load_window_arrays(data_dir, "seed_vii", window_seconds)
    validation_subjects = tuple(int(value) for value in validation_subjects)
    if not validation_subjects or any(value < 1 or value > 20 for value in validation_subjects):
        raise ValueError("validation_subjects must be a non-empty subset of 1..20")
    validation_mask = np.isin(arrays.subjects, validation_subjects)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("The source split must contain both training and validation windows")
    mean, std = _fit_stats_subset(arrays.features, train_indices)
    return PreparedTrialSource(
        train=TrialSequenceDataset(arrays, mean, std, train_indices),
        validation=TrialSequenceDataset(arrays, mean, std, validation_indices),
        mean=mean,
        std=std,
    )


def prepare_trial_target(
    data_dir: Path,
    window_seconds: float,
    target_subject: int,
    source: PreparedTrialSource,
    target_normalization: str,
) -> TrialSequenceDataset:
    if not 1 <= target_subject <= SEED_V_SUBJECTS:
        raise ValueError(f"target_subject must be within 1..{SEED_V_SUBJECTS}")
    arrays = load_window_arrays(
        data_dir, "seed_v", window_seconds, subject=target_subject
    )
    if target_normalization == "source":
        mean, std = source.mean, source.std
    elif target_normalization == "domain":
        mean, std = fit_source_stats(arrays.features)
    else:
        raise ValueError("target_normalization must be source or domain")
    return TrialSequenceDataset(arrays, mean, std)
