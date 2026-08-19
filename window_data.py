"""Leakage-safe graph datasets for the newly recomputed DE windows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


NUM_CHANNELS = 62
NUM_BANDS = 5
NUM_CLASSES = 3
SEED_VII_SESSIONS = 4
SEED_V_SESSIONS = 3
SEED_VII_SUBJECTS = 20
SEED_V_SUBJECTS = 16


@dataclass(frozen=True)
class WindowArrays:
    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    trials: np.ndarray
    window_ids: np.ndarray


@dataclass(frozen=True)
class PreparedSource:
    dataset: "WindowGraphDataset"
    mean: np.ndarray
    std: np.ndarray


class WindowGraphDataset(Dataset):
    """One DE window is one graph with 62 electrode nodes and five features."""

    def __init__(self, arrays: WindowArrays) -> None:
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

        self.features = torch.from_numpy(
            np.ascontiguousarray(arrays.features, dtype=np.float32)
        )
        self.labels = torch.from_numpy(np.asarray(arrays.labels, dtype=np.int64))
        self.subjects = torch.from_numpy(np.asarray(arrays.subjects, dtype=np.int64))
        self.sessions = torch.from_numpy(np.asarray(arrays.sessions, dtype=np.int64))
        self.trials = torch.from_numpy(np.asarray(arrays.trials, dtype=np.int64))
        self.window_ids = torch.from_numpy(
            np.asarray(arrays.window_ids, dtype=np.int64)
        )
        nodes = torch.arange(NUM_CHANNELS, dtype=torch.long)
        self.edge_index = torch.stack((nodes, nodes), dim=0)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Data:
        return Data(
            x=self.features[index],
            edge_index=self.edge_index,
            y=self.labels[index].view(1),
            subject_id=self.subjects[index].view(1),
            session_id=self.sessions[index].view(1),
            trial_id=self.trials[index].view(1),
            window_id=self.window_ids[index].view(1),
        )


def _scale_name(window_seconds: float) -> str:
    value = float(window_seconds)
    return f"{int(value) if value.is_integer() else value:g}s".replace(".", "p")


def _files_for_domain(
    data_dir: Path,
    dataset: str,
    window_seconds: float,
    subject: int | None,
) -> list[Path]:
    scale_dir = data_dir / dataset / f"window_{_scale_name(window_seconds)}"
    pattern = "*.npz" if subject is None else f"subject_{subject:02d}_session_*.npz"
    files = sorted(scale_dir.glob(pattern))
    expected = (
        SEED_VII_SUBJECTS * SEED_VII_SESSIONS
        if dataset == "seed_vii" and subject is None
        else SEED_V_SESSIONS
    )
    if len(files) != expected:
        scope = "all source subjects" if subject is None else f"subject {subject}"
        raise FileNotFoundError(
            f"Expected {expected} {dataset} files for {scope} at "
            f"{window_seconds:g}s, found {len(files)} in {scale_dir}"
        )
    return files


def load_window_arrays(
    data_dir: Path,
    dataset: str,
    window_seconds: float,
    subject: int | None = None,
) -> WindowArrays:
    """Load one source scale or one target subject without using labels for fit."""

    if dataset not in {"seed_vii", "seed_v"}:
        raise ValueError("dataset must be seed_vii or seed_v")
    if dataset == "seed_vii" and subject is not None:
        raise ValueError("SEED-VII is loaded as the complete source domain")
    if dataset == "seed_v" and subject is None:
        raise ValueError("SEED-V must be loaded one target subject at a time")

    fields: dict[str, list[np.ndarray]] = {
        "features": [],
        "labels": [],
        "subjects": [],
        "sessions": [],
        "trials": [],
        "window_ids": [],
    }
    files = _files_for_domain(data_dir, dataset, window_seconds, subject)
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"]
            if features.ndim != 3 or features.shape[1:] != (
                NUM_CHANNELS,
                NUM_BANDS,
            ):
                raise ValueError(f"Invalid features in {path}: {features.shape}")
            recorded_scale = float(archive["window_seconds"])
            if not np.isclose(recorded_scale, window_seconds):
                raise ValueError(
                    f"{path} records {recorded_scale:g}s, requested "
                    f"{window_seconds:g}s"
                )
            fields["features"].append(features.astype(np.float32, copy=False))
            fields["labels"].append(archive["label_3class"].astype(np.int64))
            fields["subjects"].append(archive["subject_id"].astype(np.int64))
            fields["sessions"].append(archive["session_id"].astype(np.int64))
            fields["trials"].append(archive["trial_id"].astype(np.int64))
            fields["window_ids"].append(archive["window_id"].astype(np.int64))

    return WindowArrays(**{key: np.concatenate(value) for key, value in fields.items()})


def fit_source_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-electrode/per-band statistics on labeled source features only."""

    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-5))


def normalize_arrays(
    arrays: WindowArrays, mean: np.ndarray, std: np.ndarray
) -> WindowArrays:
    normalized = ((arrays.features - mean) / std).astype(np.float32, copy=False)
    return WindowArrays(
        features=normalized,
        labels=arrays.labels,
        subjects=arrays.subjects,
        sessions=arrays.sessions,
        trials=arrays.trials,
        window_ids=arrays.window_ids,
    )


def load_transfer_fold(
    data_dir: Path,
    window_seconds: float,
    target_subject: int,
    target_normalization: str = "source",
    source_arrays: WindowArrays | None = None,
) -> tuple[WindowGraphDataset, WindowGraphDataset, WindowArrays]:
    """Return source and one transductive target fold.

    Target labels stay attached solely for the final evaluation. Neither
    normalization mode reads them: ``source`` uses source statistics and
    ``domain`` fits statistics on unlabeled target features.
    """

    if not 1 <= target_subject <= SEED_V_SUBJECTS:
        raise ValueError(f"target_subject must be within 1..{SEED_V_SUBJECTS}")
    if target_normalization not in {"source", "domain"}:
        raise ValueError("target_normalization must be source or domain")
    if source_arrays is None:
        source_arrays = load_window_arrays(
            data_dir, "seed_vii", window_seconds, subject=None
        )
    target_arrays = load_window_arrays(
        data_dir, "seed_v", window_seconds, subject=target_subject
    )
    source_mean, source_std = fit_source_stats(source_arrays.features)
    normalized_source = normalize_arrays(source_arrays, source_mean, source_std)
    if target_normalization == "domain":
        target_mean, target_std = fit_source_stats(target_arrays.features)
    else:
        target_mean, target_std = source_mean, source_std
    normalized_target = normalize_arrays(target_arrays, target_mean, target_std)
    return (
        WindowGraphDataset(normalized_source),
        WindowGraphDataset(normalized_target),
        source_arrays,
    )


def prepare_source(data_dir: Path, window_seconds: float) -> PreparedSource:
    arrays = load_window_arrays(data_dir, "seed_vii", window_seconds)
    mean, std = fit_source_stats(arrays.features)
    normalized = normalize_arrays(arrays, mean, std)
    return PreparedSource(WindowGraphDataset(normalized), mean, std)


def prepare_target(
    data_dir: Path,
    window_seconds: float,
    target_subject: int,
    source: PreparedSource,
    target_normalization: str,
) -> WindowGraphDataset:
    arrays = load_window_arrays(
        data_dir, "seed_v", window_seconds, subject=target_subject
    )
    if target_normalization == "domain":
        mean, std = fit_source_stats(arrays.features)
    elif target_normalization == "source":
        mean, std = source.mean, source.std
    else:
        raise ValueError("target_normalization must be source or domain")
    return WindowGraphDataset(normalize_arrays(arrays, mean, std))


def summarize_labels(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in labels)
    return {"positive": counts[0], "neutral": counts[1], "negative": counts[2]}
