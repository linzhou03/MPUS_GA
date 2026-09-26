"""Aligned multiscale trial datasets for multi-source UDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .anatomical_prior import (
    PHYSIOLOGY_DIM,
    physiology_descriptor,
    validate_channel_names,
)


NUM_CHANNELS = 62
NUM_BANDS = 5
NUM_CLASSES = 3
ALL_SCALES = (1.0, 2.0, 4.0)
DATASET_LAYOUT = {
    "seed_v": {"subjects": 16, "sessions_per_subject": 3, "files": 48},
    "seed_vii": {"subjects": 20, "sessions_per_subject": 4, "files": 80},
    "seed_iv": {"subjects": 15, "sessions_per_subject": 3, "files": 45},
}


def scale_key(scale: float) -> str:
    value = float(scale)
    return f"{int(value) if value.is_integer() else value:g}s".replace(".", "p")


@dataclass(frozen=True)
class ScaleArrays:
    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    trials: np.ndarray
    window_ids: np.ndarray


@dataclass(frozen=True)
class PreparedMultiSource:
    domain_names: tuple[str, ...]
    datasets: tuple["MultiScaleTrialDataset", ...]
    stats: dict[float, tuple[np.ndarray, np.ndarray]]
    physiology_stats: dict[float, tuple[np.ndarray, np.ndarray]] | None = None
    physiology_enabled: bool = False
    compact_physiology: bool = False
    subject_robust_physiology: bool = False


def _artifact_files(
    data_dir: Path,
    dataset: str,
    scale: float,
    subject: int | None,
) -> list[Path]:
    if dataset not in DATASET_LAYOUT:
        raise ValueError(f"Unsupported dataset: {dataset}")
    scale_dir = data_dir / dataset / f"window_{scale_key(scale)}"
    layout = DATASET_LAYOUT[dataset]
    if subject is not None:
        if not 1 <= subject <= layout["subjects"]:
            raise ValueError(
                f"{dataset} subject must be within 1..{layout['subjects']}"
            )
        files = sorted(scale_dir.glob(f"subject_{subject:02d}_session_*.npz"))
        expected = layout["sessions_per_subject"]
    else:
        files = sorted(scale_dir.glob("subject_*_session_*.npz"))
        expected = layout["files"]
    if len(files) != expected:
        raise FileNotFoundError(
            f"Expected {expected} {dataset} artifacts at {scale:g}s, "
            f"found {len(files)} in {scale_dir}"
        )
    return files


def load_scale_arrays(
    data_dir: Path,
    dataset: str,
    scale: float,
    subject: int | None = None,
) -> ScaleArrays:
    fields: dict[str, list[np.ndarray]] = {
        "features": [],
        "labels": [],
        "subjects": [],
        "sessions": [],
        "trials": [],
        "window_ids": [],
    }
    for path in _artifact_files(data_dir, dataset, scale, subject):
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"]
            if features.ndim != 3 or features.shape[1:] != (
                NUM_CHANNELS,
                NUM_BANDS,
            ):
                raise ValueError(f"Invalid feature shape in {path}: {features.shape}")
            if not np.isclose(float(archive["window_seconds"]), scale):
                raise ValueError(f"Scale mismatch in {path}")
            if "channel_names" in archive:
                validate_channel_names(archive["channel_names"].tolist())
            fields["features"].append(features.astype(np.float32, copy=False))
            fields["labels"].append(archive["label_3class"].astype(np.int64))
            fields["subjects"].append(archive["subject_id"].astype(np.int64))
            fields["sessions"].append(archive["session_id"].astype(np.int64))
            fields["trials"].append(archive["trial_id"].astype(np.int64))
            fields["window_ids"].append(archive["window_id"].astype(np.int64))
    return ScaleArrays(
        **{name: np.concatenate(values) for name, values in fields.items()}
    )


def fit_combined_stats(
    arrays: Sequence[ScaleArrays],
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit source-only electrode/band statistics without a large concatenation."""

    if not arrays:
        raise ValueError("At least one source array is required")
    total = 0
    total_sum = np.zeros((NUM_CHANNELS, NUM_BANDS), dtype=np.float64)
    total_square = np.zeros_like(total_sum)
    for item in arrays:
        for start in range(0, len(item.features), chunk_size):
            block = item.features[start : start + chunk_size]
            total += len(block)
            total_sum += block.sum(axis=0, dtype=np.float64)
            total_square += np.square(block).sum(axis=0, dtype=np.float64)
    mean64 = total_sum / total
    variance = np.maximum(total_square / total - np.square(mean64), 0.0)
    mean = mean64.astype(np.float32)
    std = np.maximum(np.sqrt(variance).astype(np.float32), np.float32(1e-5))
    return mean, std


def _group_scale(arrays: ScaleArrays) -> dict[tuple[int, int, int], np.ndarray]:
    selected = np.arange(len(arrays.features), dtype=np.int64)
    order = np.lexsort(
        (
            arrays.window_ids,
            arrays.trials,
            arrays.sessions,
            arrays.subjects,
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
    groups: dict[tuple[int, int, int], np.ndarray] = {}
    for indices in np.split(selected, boundaries):
        first = int(indices[0])
        key = (
            int(arrays.subjects[first]),
            int(arrays.sessions[first]),
            int(arrays.trials[first]),
        )
        window_ids = arrays.window_ids[indices]
        if not np.array_equal(window_ids, np.arange(len(indices))):
            raise ValueError(f"Trial {key} has non-contiguous window IDs")
        groups[key] = indices
    return groups


def _trial_physiology_matrix(arrays: ScaleArrays) -> tuple[tuple, np.ndarray]:
    groups = _group_scale(arrays)
    keys = tuple(sorted(groups))
    descriptors = np.stack(
        [
            physiology_descriptor(arrays.features[groups[key]])
            for key in keys
        ],
        axis=0,
    )
    return keys, descriptors


def fit_physiology_stats(
    arrays: Sequence[ScaleArrays],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit trial-descriptor statistics using source trials only."""

    if not arrays:
        raise ValueError("At least one source array is required")
    descriptor_blocks = [_trial_physiology_matrix(item)[1] for item in arrays]
    descriptors = np.concatenate(descriptor_blocks, axis=0).astype(
        np.float64, copy=False
    )
    mean = descriptors.mean(axis=0).astype(np.float32)
    std = np.maximum(
        descriptors.std(axis=0).astype(np.float32), np.float32(1e-5)
    )
    if mean.shape != (PHYSIOLOGY_DIM,) or std.shape != (PHYSIOLOGY_DIM,):
        raise RuntimeError("Unexpected physiology-statistic shape")
    return mean, std


def robust_standardize_physiology_by_subject(
    descriptors: np.ndarray,
    keys: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Median/MAD normalisation using only trials from the same EEG subject."""

    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(keys):
        raise ValueError("Descriptors and trial keys must align")
    result = np.empty_like(values)
    subject_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    for subject_id in np.unique(subject_ids):
        selected = subject_ids == subject_id
        block = values[selected].astype(np.float64, copy=False)
        median = np.median(block, axis=0)
        mad = 1.4826 * np.median(np.abs(block - median), axis=0)
        scale = np.maximum(mad, 1e-5)
        result[selected] = np.clip(
            (block - median) / scale, -5.0, 5.0
        ).astype(np.float32)
    return result


class MultiScaleTrialDataset(Dataset):
    """One item contains aligned 1/2/4-second sequences for one movie trial."""

    def __init__(
        self,
        arrays_by_scale: Mapping[float, ScaleArrays],
        stats: Mapping[float, tuple[np.ndarray, np.ndarray]],
        domain_id: int,
        physiology_stats: Mapping[
            float, tuple[np.ndarray, np.ndarray]
        ] | None = None,
        use_physiology: bool = False,
        compact_physiology: bool = False,
        subject_robust_physiology: bool = False,
    ) -> None:
        self.scales = tuple(sorted(map(float, arrays_by_scale)))
        if not self.scales:
            raise ValueError("At least one scale is required")
        self.arrays = {float(scale): arrays_by_scale[scale] for scale in self.scales}
        self.stats = {float(scale): stats[scale] for scale in self.scales}
        self.domain_id = int(domain_id)
        groups = {scale: _group_scale(self.arrays[scale]) for scale in self.scales}
        reference_keys = set(groups[self.scales[0]])
        for scale in self.scales[1:]:
            if set(groups[scale]) != reference_keys:
                raise ValueError(f"Trial keys do not align at scale {scale:g}s")
        self.keys = tuple(sorted(reference_keys))
        self.groups = groups
        use_physiology = bool(use_physiology or physiology_stats is not None)
        if subject_robust_physiology and physiology_stats is not None:
            raise ValueError(
                "Subject-robust physiology must not use source-global statistics"
            )
        self.physiology: dict[float, torch.Tensor] | None = None
        if use_physiology:
            self.physiology = {}
            for scale in self.scales:
                descriptor_keys, descriptors = _trial_physiology_matrix(
                    self.arrays[scale]
                )
                if descriptor_keys != self.keys:
                    raise ValueError(
                        f"Physiology trial keys do not align at scale {scale:g}s"
                    )
                if compact_physiology:
                    descriptors = descriptors[:, -4:]
                if subject_robust_physiology:
                    standardised = robust_standardize_physiology_by_subject(
                        descriptors, descriptor_keys
                    )
                else:
                    if physiology_stats is None:
                        raise ValueError(
                            "Source-global physiology statistics are required"
                        )
                    mean, std = physiology_stats[scale]
                    if mean.shape != descriptors.shape[1:]:
                        raise ValueError(
                            "Physiology descriptor and statistic dimensions differ"
                        )
                    standardised = (descriptors - mean) / std
                self.physiology[scale] = torch.from_numpy(
                    np.ascontiguousarray(standardised, dtype=np.float32)
                )

        labels = []
        for key in self.keys:
            observed = set()
            for scale in self.scales:
                arrays = self.arrays[scale]
                values = np.unique(arrays.labels[groups[scale][key]])
                if len(values) != 1:
                    raise ValueError(f"Trial {key} mixes labels at {scale:g}s")
                observed.add(int(values[0]))
            if len(observed) != 1:
                raise ValueError(f"Trial {key} has inconsistent labels across scales")
            labels.append(observed.pop())
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.keys)

    def _item(self, index: int, include_label: bool) -> dict:
        key = self.keys[index]
        features: dict[str, torch.Tensor] = {}
        for scale in self.scales:
            arrays = self.arrays[scale]
            indices = self.groups[scale][key]
            mean, std = self.stats[scale]
            normalized = (arrays.features[indices] - mean) / std
            features[scale_key(scale)] = torch.from_numpy(
                np.ascontiguousarray(normalized, dtype=np.float32)
            )
        item = {
            "x": features,
            "trial_key_by_scale": {
                scale_key(scale): torch.tensor([
                    self.arrays[scale].subjects[self.groups[scale][key][0]],
                    self.arrays[scale].sessions[self.groups[scale][key][0]],
                    self.arrays[scale].trials[self.groups[scale][key][0]],
                ], dtype=torch.long) for scale in self.scales
            },
            "subject_id": torch.tensor(key[0], dtype=torch.long),
            "session_id": torch.tensor(key[1], dtype=torch.long),
            "trial_id": torch.tensor(key[2], dtype=torch.long),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
        }
        if self.physiology is not None:
            item["physiology"] = {
                scale_key(scale): self.physiology[scale][index]
                for scale in self.scales
            }
        if include_label:
            item["y"] = self.labels[index]
        return item

    def __getitem__(self, index: int) -> dict:
        return self._item(index, include_label=True)


class UnlabeledMultiScaleView(Dataset):
    def __init__(self, dataset: MultiScaleTrialDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return self.dataset._item(index, include_label=False)


def collate_multiscale(items: Sequence[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    keys = tuple(items[0]["x"])
    if any(tuple(item["x"]) != keys for item in items):
        raise ValueError("Batch items do not contain identical scales")
    features: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for key in keys:
        max_windows = max(len(item["x"][key]) for item in items)
        batch = items[0]["x"][key].new_zeros(
            (len(items), max_windows, NUM_CHANNELS, NUM_BANDS)
        )
        mask = torch.zeros((len(items), max_windows), dtype=torch.bool)
        for row, item in enumerate(items):
            length = len(item["x"][key])
            if length == 0:
                raise ValueError("Every scale must contain at least one window")
            batch[row, :length] = item["x"][key]
            mask[row, :length] = True
        features[key] = batch
        masks[key] = mask
    result = {
        "x": features,
        "mask": masks,
        "subject_id": torch.stack([item["subject_id"] for item in items]),
        "session_id": torch.stack([item["session_id"] for item in items]),
        "trial_id": torch.stack([item["trial_id"] for item in items]),
        "domain_id": torch.stack([item["domain_id"] for item in items]),
    }
    if any('trial_key_by_scale' in item for item in items):
        if not all('trial_key_by_scale' in item for item in items):
            raise ValueError('Batch mixes trials with and without per-scale IDs')
        result['trial_key_by_scale'] = {key: torch.stack([item['trial_key_by_scale'][key] for item in items]) for key in keys}
        from .multiscale_evidence import assert_scale_trial_ids
        assert_scale_trial_ids(result)
    physiology_presence = ["physiology" in item for item in items]
    if any(physiology_presence) and not all(physiology_presence):
        raise ValueError("A batch cannot mix physiology-enabled and plain trials")
    if all(physiology_presence):
        if any(tuple(item["physiology"]) != keys for item in items):
            raise ValueError("Physiology and feature scales do not match")
        result["physiology"] = {
            key: torch.stack([item["physiology"][key] for item in items])
            for key in keys
        }
    label_presence = ["y" in item for item in items]
    if any(label_presence) and not all(label_presence):
        raise ValueError("A batch cannot mix labeled and unlabeled trials")
    if all(label_presence):
        result["y"] = torch.stack([item["y"] for item in items])
    return result


def prepare_sources(
    data_dir: Path,
    source_domains: Sequence[str],
    scales: Sequence[float],
    use_physiology: bool = False,
    compact_physiology: bool = False,
    subject_robust_physiology: bool = False,
) -> PreparedMultiSource:
    domains = tuple(source_domains)
    scales = tuple(sorted(map(float, scales)))
    if len(set(domains)) != len(domains):
        raise ValueError("Source domain names must be unique")
    if (compact_physiology or subject_robust_physiology) and not use_physiology:
        raise ValueError("Physiology options require use_physiology=True")
    loaded = {
        domain: {
            scale: load_scale_arrays(data_dir, domain, scale)
            for scale in scales
        }
        for domain in domains
    }
    stats = {
        scale: fit_combined_stats([loaded[domain][scale] for domain in domains])
        for scale in scales
    }
    physiology_stats = (
        {
            scale: fit_physiology_stats(
                [loaded[domain][scale] for domain in domains]
            )
            for scale in scales
        }
        if use_physiology and not subject_robust_physiology
        else None
    )
    datasets = tuple(
        MultiScaleTrialDataset(
            loaded[domain],
            stats,
            domain_id=index,
            physiology_stats=physiology_stats,
            use_physiology=use_physiology,
            compact_physiology=compact_physiology,
            subject_robust_physiology=subject_robust_physiology,
        )
        for index, domain in enumerate(domains)
    )
    return PreparedMultiSource(
        domain_names=domains,
        datasets=datasets,
        stats=stats,
        physiology_stats=physiology_stats,
        physiology_enabled=use_physiology,
        compact_physiology=compact_physiology,
        subject_robust_physiology=subject_robust_physiology,
    )


def prepare_target(
    data_dir: Path,
    target_dataset: str,
    subject: int,
    prepared_sources: PreparedMultiSource,
    scales: Sequence[float],
    target_normalization: str = "source",
) -> MultiScaleTrialDataset:
    if target_normalization not in {"source", "std_only", "domain"}:
        raise ValueError(f"Unsupported target normalization: {target_normalization}")
    scales = tuple(sorted(map(float, scales)))
    arrays = {
        scale: load_scale_arrays(
            data_dir, target_dataset, scale, subject=subject
        )
        for scale in scales
    }
    if target_normalization == "source":
        stats = prepared_sources.stats
    else:
        target_stats = {
            scale: fit_combined_stats([arrays[scale]]) for scale in scales
        }
        if target_normalization == "std_only":
            stats = {
                scale: (prepared_sources.stats[scale][0], target_stats[scale][1])
                for scale in scales
            }
        else:
            stats = target_stats
    return MultiScaleTrialDataset(
        arrays,
        stats,
        domain_id=len(prepared_sources.domain_names),
        physiology_stats=prepared_sources.physiology_stats,
        use_physiology=prepared_sources.physiology_enabled,
        compact_physiology=prepared_sources.compact_physiology,
        subject_robust_physiology=(
            prepared_sources.subject_robust_physiology
        ),
    )
