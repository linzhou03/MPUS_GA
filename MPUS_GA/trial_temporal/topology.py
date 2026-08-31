"""Sensor-space topology priors for the canonical 62-channel SEED montage.

The processed SEED-IV/V/VII artifacts share this exact channel order.  The
matrix below is deliberately an electrode-topology prior rather than a claim
of source-localized cortical connectivity: it combines local montage
neighbours, coarse scalp regions, and bilateral homologues.
"""

from __future__ import annotations

import torch


MONTAGE_ROWS = (
    ("FP1", "FPZ", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"),
    ("FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"),
    ("T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"),
    ("TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"),
    ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"),
    ("PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"),
    ("CB1", "O1", "OZ", "O2", "CB2"),
)

SEED_62_CHANNEL_NAMES = tuple(
    channel for row in MONTAGE_ROWS for channel in row
)

REGION_NAMES = (
    "prefrontal",
    "frontal",
    "frontocentral",
    "central_temporal",
    "centroparietal",
    "parietal",
    "occipital",
)

REGION_CHANNELS = {
    "prefrontal": MONTAGE_ROWS[0] + MONTAGE_ROWS[1],
    "frontal": MONTAGE_ROWS[2],
    "frontocentral": MONTAGE_ROWS[3],
    "central_temporal": MONTAGE_ROWS[4],
    "centroparietal": MONTAGE_ROWS[5],
    "parietal": MONTAGE_ROWS[6],
    "occipital": MONTAGE_ROWS[7] + MONTAGE_ROWS[8],
}

HOMOTOPIC_PAIRS = (
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FT7", "FT8"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("TP7", "TP8"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("CB1", "CB2"),
    ("O1", "O2"),
)


def channel_region_ids() -> torch.Tensor:
    """Return the seven-region membership index in canonical channel order."""

    channel_to_region = {
        channel: region_index
        for region_index, region in enumerate(REGION_NAMES)
        for channel in REGION_CHANNELS[region]
    }
    if set(channel_to_region) != set(SEED_62_CHANNEL_NAMES):
        raise RuntimeError("Region definitions do not cover the SEED montage")
    return torch.tensor(
        [channel_to_region[channel] for channel in SEED_62_CHANNEL_NAMES],
        dtype=torch.long,
    )


def _montage_coordinates() -> torch.Tensor:
    """Create a deterministic 2-D layout preserving the standard row order."""

    coordinates: dict[str, tuple[float, float]] = {}
    for row_index, row in enumerate(MONTAGE_ROWS):
        if len(row) == 1:
            x_positions = (0.0,)
        else:
            half_width = min(4.0, max(1.0, (len(row) - 1) / 2.0))
            step = 2.0 * half_width / (len(row) - 1)
            x_positions = tuple(-half_width + step * index for index in range(len(row)))
        y_position = float(len(MONTAGE_ROWS) - 1 - row_index)
        coordinates.update(
            {
                channel: (x_position, y_position)
                for channel, x_position in zip(row, x_positions, strict=True)
            }
        )
    return torch.tensor(
        [coordinates[channel] for channel in SEED_62_CHANNEL_NAMES],
        dtype=torch.float32,
    )


def build_seed_topology_prior(
    local_neighbors: int = 4,
    region_bonus: float = 0.20,
    homotopic_weight: float = 0.90,
    permuted: bool = False,
) -> torch.Tensor:
    """Build a symmetric ``[62,62]`` soft topology-bias matrix.

    Non-prior edges stay at zero and remain available to the dynamic graph.
    The prior therefore encourages plausible edges without acting as a hard
    anatomical mask.  ``permuted`` deterministically relabels the same graph
    and is intended only as a negative-control ablation.
    """

    if not 1 <= local_neighbors < len(SEED_62_CHANNEL_NAMES):
        raise ValueError("local_neighbors must be within [1, channels)")
    if not 0 <= region_bonus <= 1 or not 0 <= homotopic_weight <= 1:
        raise ValueError("topology weights must be within [0,1]")

    coordinates = _montage_coordinates()
    distance = torch.cdist(coordinates, coordinates)
    positive_distance = distance[distance > 0]
    scale = positive_distance.median().clamp_min(1e-6)
    proximity = torch.exp(-0.5 * (distance / scale).square())

    count = len(SEED_62_CHANNEL_NAMES)
    prior = torch.zeros((count, count), dtype=torch.float32)
    nearest = distance.masked_fill(
        torch.eye(count, dtype=torch.bool), float("inf")
    ).topk(local_neighbors, dim=1, largest=False).indices
    row = torch.arange(count).unsqueeze(1).expand_as(nearest)
    prior[row, nearest] = proximity[row, nearest]
    prior = torch.maximum(prior, prior.transpose(0, 1))

    region = channel_region_ids()
    same_region = region[:, None] == region[None, :]
    prior = torch.maximum(
        prior,
        same_region.to(prior.dtype) * float(region_bonus) * proximity,
    )

    channel_index = {
        channel: index for index, channel in enumerate(SEED_62_CHANNEL_NAMES)
    }
    for left, right in HOMOTOPIC_PAIRS:
        left_index = channel_index[left]
        right_index = channel_index[right]
        prior[left_index, right_index] = max(
            float(prior[left_index, right_index]), homotopic_weight
        )
        prior[right_index, left_index] = prior[left_index, right_index]

    prior.fill_diagonal_(1.0)
    prior = prior.clamp(0.0, 1.0)
    if permuted:
        permutation = torch.arange(count).roll(7)
        prior = prior[permutation][:, permutation]
    return prior

