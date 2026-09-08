"""Fixed anatomical regions and continuous raw-DE physiology descriptors."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch


CHANNEL_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
)
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")

# The extra midline and cerebellar groups make the partition exhaustive.  The
# order is part of the model definition and must stay stable across datasets.
ANATOMICAL_REGIONS = OrderedDict(
    (
        ("left_frontal", ("FP1", "AF3", "F7", "F5", "F3", "F1", "FC5", "FC3", "FC1")),
        ("right_frontal", ("FP2", "AF4", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6")),
        ("midline_frontal", ("FPZ", "FZ", "FCZ")),
        ("left_temporal", ("FT7", "T7", "TP7")),
        ("right_temporal", ("FT8", "T8", "TP8")),
        ("central", ("C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6")),
        ("parietal_occipital", ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2")),
        ("cerebellar", ("CB1", "CB2")),
    )
)

FRONTAL_ASYMMETRY_PAIRS = (
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F1", "F2"),
    ("F3", "F4"),
    ("F5", "F6"),
    ("F7", "F8"),
    ("FC1", "FC2"),
    ("FC3", "FC4"),
    ("FC5", "FC6"),
)

REGION_NAMES = tuple(ANATOMICAL_REGIONS)
PHYSIOLOGY_DESCRIPTOR_NAMES = tuple(
    f"{region}_{band}_contrast"
    for region in REGION_NAMES
    for band in BAND_NAMES
) + (
    "frontal_alpha_asymmetry_log_power_right_minus_left",
    "temporal_beta_gamma_contrast",
    "parietal_occipital_beta_gamma_contrast",
    "global_alpha_dominance",
)
PHYSIOLOGY_DIM = len(PHYSIOLOGY_DESCRIPTOR_NAMES)
COMPACT_PHYSIOLOGY_DESCRIPTOR_NAMES = PHYSIOLOGY_DESCRIPTOR_NAMES[-4:]
COMPACT_PHYSIOLOGY_DIM = len(COMPACT_PHYSIOLOGY_DESCRIPTOR_NAMES)


def validate_channel_names(names: Sequence[str]) -> None:
    observed = tuple(str(name).upper() for name in names)
    if observed != CHANNEL_NAMES:
        raise ValueError(
            "EEG channel order does not match the fixed 62-channel prior"
        )


def _validate_partition() -> None:
    flattened = [name for values in ANATOMICAL_REGIONS.values() for name in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Anatomical regions overlap")
    if set(flattened) != set(CHANNEL_NAMES):
        missing = sorted(set(CHANNEL_NAMES) - set(flattened))
        extra = sorted(set(flattened) - set(CHANNEL_NAMES))
        raise RuntimeError(
            f"Anatomical partition mismatch; missing={missing}, extra={extra}"
        )


_validate_partition()
_CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}


def region_weight_matrix_numpy() -> np.ndarray:
    """Return fixed equal-within-region weights with shape [regions, channels]."""

    matrix = np.zeros((len(REGION_NAMES), len(CHANNEL_NAMES)), dtype=np.float32)
    for region_index, names in enumerate(ANATOMICAL_REGIONS.values()):
        weight = np.float32(1.0 / len(names))
        for name in names:
            matrix[region_index, _CHANNEL_INDEX[name]] = weight
    return matrix


def region_weight_matrix_torch() -> torch.Tensor:
    return torch.from_numpy(region_weight_matrix_numpy())


def physiology_descriptor(raw_trial: np.ndarray) -> np.ndarray:
    """Compute a 44-D descriptor from unnormalised DE windows of one trial.

    DE is proportional to log variance, so twice the right-minus-left alpha-DE
    difference is a log-power-ratio proxy.  The other regional values are
    within-trial contrasts, which reduces global subject/session offsets.
    """

    raw = np.asarray(raw_trial, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (len(CHANNEL_NAMES), len(BAND_NAMES)):
        raise ValueError("raw_trial must have shape [windows,62,5]")
    if len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("raw_trial must be nonempty and finite")

    channel_band = raw.mean(axis=0)
    regional_band = region_weight_matrix_numpy().astype(np.float64) @ channel_band
    global_band = channel_band.mean(axis=0)
    regional_contrast = regional_band - global_band[None, :]

    alpha_index = BAND_NAMES.index("alpha")
    paired_alpha = [
        channel_band[_CHANNEL_INDEX[right], alpha_index]
        - channel_band[_CHANNEL_INDEX[left], alpha_index]
        for left, right in FRONTAL_ASYMMETRY_PAIRS
    ]
    frontal_asymmetry = 2.0 * float(np.mean(paired_alpha))
    high_frequency = [BAND_NAMES.index("beta"), BAND_NAMES.index("gamma")]
    temporal_rows = [
        REGION_NAMES.index("left_temporal"),
        REGION_NAMES.index("right_temporal"),
    ]
    temporal_high = float(
        regional_band[np.ix_(temporal_rows, high_frequency)].mean()
        - global_band[high_frequency].mean()
    )
    posterior_high = float(
        regional_band[REGION_NAMES.index("parietal_occipital"), high_frequency].mean()
        - global_band[high_frequency].mean()
    )
    non_alpha = [index for index in range(len(BAND_NAMES)) if index != alpha_index]
    alpha_dominance = float(global_band[alpha_index] - global_band[non_alpha].mean())

    descriptor = np.concatenate(
        (
            regional_contrast.reshape(-1),
            np.asarray(
                [frontal_asymmetry, temporal_high, posterior_high, alpha_dominance],
                dtype=np.float64,
            ),
        )
    ).astype(np.float32)
    if descriptor.shape != (PHYSIOLOGY_DIM,):
        raise RuntimeError("Unexpected physiology descriptor size")
    return descriptor


def compact_physiology_descriptor(raw_trial: np.ndarray) -> np.ndarray:
    """Return only the four explicit, low-capacity physiology indicators."""

    return physiology_descriptor(raw_trial)[-COMPACT_PHYSIOLOGY_DIM:]


def physiology_metadata() -> dict:
    return {
        "channel_order": list(CHANNEL_NAMES),
        "bands": list(BAND_NAMES),
        "regions": {
            name: list(channels) for name, channels in ANATOMICAL_REGIONS.items()
        },
        "descriptor_names": list(PHYSIOLOGY_DESCRIPTOR_NAMES),
        "compact_descriptor_names": list(COMPACT_PHYSIOLOGY_DESCRIPTOR_NAMES),
        "frontal_asymmetry_pairs": [list(pair) for pair in FRONTAL_ASYMMETRY_PAIRS],
        "frontal_asymmetry_definition": (
            "2 * mean(DE_alpha_right - DE_alpha_left), approximating "
            "ln(alpha_power_right / alpha_power_left)"
        ),
        "target_labels_used_for_descriptor_or_standardisation": False,
    }
