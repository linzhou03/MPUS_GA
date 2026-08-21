"""Validate multiscale DE artifacts without consulting target labels externally."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_DIR / "data_processed"


def validate_file(path: Path) -> dict:
    required = {
        "features",
        "label_original",
        "label_3class",
        "subject_id",
        "session_id",
        "trial_id",
        "session_trial_id",
        "window_id",
        "start_second",
        "channel_names",
        "band_names",
        "band_edges_hz",
        "sampling_rate_hz",
        "window_seconds",
        "source_file",
        "feature_units",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        features = archive["features"]
        if features.ndim != 3 or features.shape[1:] != (62, 5):
            raise ValueError(f"{path} has invalid feature shape {features.shape}")
        if features.dtype != np.float32:
            raise ValueError(f"{path} features must be float32, got {features.dtype}")
        if not np.isfinite(features).all():
            raise ValueError(f"{path} contains NaN or infinite DE values")

        count = features.shape[0]
        metadata_keys = (
            "label_original",
            "label_3class",
            "subject_id",
            "session_id",
            "trial_id",
            "session_trial_id",
            "window_id",
            "start_second",
        )
        for key in metadata_keys:
            if archive[key].shape != (count,):
                raise ValueError(
                    f"{path} key {key} has shape {archive[key].shape}; "
                    f"expected ({count},)"
                )
        labels = archive["label_3class"]
        if not set(np.unique(labels)).issubset({0, 1, 2}):
            raise ValueError(f"{path} contains invalid three-class labels")
        if archive["channel_names"].shape != (62,):
            raise ValueError(f"{path} does not contain 62 channel names")
        if archive["band_names"].shape != (5,):
            raise ValueError(f"{path} does not contain five band names")
        if archive["band_edges_hz"].shape != (5, 2):
            raise ValueError(f"{path} has invalid band edge metadata")

        composite = np.stack(
            (
                archive["subject_id"],
                archive["session_id"],
                archive["trial_id"],
                archive["window_id"],
            ),
            axis=1,
        )
        if np.unique(composite, axis=0).shape[0] != count:
            raise ValueError(f"{path} contains duplicate window identifiers")

        subjects = np.unique(archive["subject_id"])
        sessions = np.unique(archive["session_id"])
        if subjects.size != 1 or sessions.size != 1:
            raise ValueError(f"{path} mixes subjects or sessions")

        scale = float(archive["window_seconds"])
        trial_counts = {}
        trial_labels = {}
        for trial_id in np.unique(archive["trial_id"]):
            mask = archive["trial_id"] == trial_id
            expected_starts = np.arange(mask.sum(), dtype=np.float32) * scale
            observed_starts = archive["start_second"][mask]
            if not np.allclose(observed_starts, expected_starts, atol=1e-5):
                raise ValueError(
                    f"{path} trial {trial_id} has invalid window start times"
                )
            unique_labels = np.unique(labels[mask])
            if unique_labels.size != 1:
                raise ValueError(f"{path} trial {trial_id} mixes class labels")
            trial_counts[str(int(trial_id))] = int(mask.sum())
            trial_labels[str(int(trial_id))] = int(unique_labels[0])

        return {
            "path": str(path),
            "samples": count,
            "shape": list(features.shape),
            "window_seconds": scale,
            "subject": int(subjects[0]),
            "session": int(sessions[0]),
            "source_file": str(archive["source_file"]),
            "feature_min": float(features.min()),
            "feature_max": float(features.max()),
            "feature_mean": float(features.mean()),
            "feature_std": float(features.std()),
            "class_counts_3class": {
                str(label): int((labels == label).sum()) for label in range(3)
            },
            "trial_counts": trial_counts,
            "trial_labels": trial_labels,
        }


def validate_cross_scale(reports: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for report in reports:
        path = Path(report["path"])
        key = (path.parents[1].name, report["subject"], report["session"])
        scale = report["window_seconds"]
        if scale in grouped[key]:
            raise ValueError(f"Duplicate {scale:g}s file for {key}")
        grouped[key][scale] = report

    results = []
    expected_scales = {1.0, 2.0, 4.0}
    for key, by_scale in sorted(grouped.items()):
        if set(by_scale) != expected_scales:
            raise ValueError(
                f"{key} has scales {sorted(by_scale)}, expected [1, 2, 4]"
            )
        one = by_scale[1.0]
        for scale in (2.0, 4.0):
            other = by_scale[scale]
            if other["source_file"] != one["source_file"]:
                raise ValueError(f"{key} has inconsistent source filenames")
            if other["trial_labels"] != one["trial_labels"]:
                raise ValueError(f"{key} has inconsistent labels across scales")
            expected_counts = {
                trial: count // int(scale)
                for trial, count in one["trial_counts"].items()
            }
            if other["trial_counts"] != expected_counts:
                raise ValueError(
                    f"{key} has inconsistent {scale:g}s per-trial window counts"
                )
        results.append(
            {
                "dataset": key[0],
                "subject": key[1],
                "session": key[2],
                "status": "ok",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed MPUS-GA DE files")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    files = sorted(data_dir.glob("seed_*/window_*/*.npz"))
    if not files:
        raise FileNotFoundError(f"No processed NPZ files found under {data_dir}")

    reports = [validate_file(path) for path in files]
    cross_scale = validate_cross_scale(reports)
    totals = Counter()
    for report in reports:
        dataset = Path(report["path"]).parents[1].name
        key = f"{dataset}:{report['window_seconds']:g}s"
        totals[key] += report["samples"]
    output = {
        "files": len(reports),
        "sessions": len(cross_scale),
        "sample_totals": dict(totals),
        "cross_scale": cross_scale,
        "details": reports,
    }
    report_path = data_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(reports),
                "sessions": len(cross_scale),
                "sample_totals": dict(totals),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
