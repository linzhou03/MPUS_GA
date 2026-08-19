"""Build 1 s, 2 s, and 4 s DE features for SEED-VII -> SEED-V."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dataset_metadata import (
    EMOTION_TO_ORIGINAL_SEED_V,
    EMOTION_TO_ORIGINAL_SEED_VII,
    EMOTION_TO_THREE_CLASS,
    SEED_V_EMOTIONS,
    SEED_V_END_SECONDS,
    SEED_V_START_SECONDS,
    SEED_VII_SPECIAL_TRIGGER_START,
    load_seed_vii_emotions,
    parse_seed_v_filename,
    parse_seed_vii_filename,
    read_special_trigger_samples,
)
from de_features import BANDS, extract_multiscale_de, validate_window_seconds


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("CAGA_SGA_DATA_ROOT", "/dataset/gzw/seed_series")
)
DEFAULT_OUTPUT = PROJECT_DIR / "data_processed"
TARGET_SFREQ = 200.0
DROP_CHANNELS = {"M1", "M2", "ECG", "HEO", "VEO"}


class MissingTriggerError(ValueError):
    """Raised when a recording cannot be segmented without invented boundaries."""


@dataclass(frozen=True)
class Trial:
    data: np.ndarray
    subject: int
    session: int
    global_trial: int
    session_trial: int
    emotion: str
    source_file: str


def _jsonable_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _scale_name(value: float) -> str:
    return f"{_jsonable_number(value)}s".replace(".", "p")


def _parse_subjects(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    subjects: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(part))
    if not subjects or min(subjects) < 1:
        raise argparse.ArgumentTypeError("subjects must be positive IDs or 'all'")
    return subjects


def _find_existing(explicit: Path | None, candidates: Iterable[Path], name: str) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{name} does not exist: {resolved}")
        return resolved
    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not discover {name}. Checked:\n  " + "\n  ".join(checked)
    )


def _raw_dir_candidates(root: Path, dataset: str) -> tuple[Path, ...]:
    if dataset == "seed-vii":
        return (
            root / "eeg_raw/SEED_VII",
            root / "raw/SEED_VII/EEG_raw",
            root / "raw/seed_vii/EEG_raw",
            root / "feature/seed_vii/EEG_raw",
            root / "SEED-VII/EEG_raw",
        )
    return (
        root / "eeg_raw/SEED_V",
        root / "raw/SEED_V/EEG_raw",
        root / "raw/seed_v/EEG_raw",
        root / "feature/seed_v/EEG_raw",
        root / "SEED-V/EEG_raw",
    )


def _save_info_candidates(root: Path, raw_dir: Path) -> tuple[Path, ...]:
    return (
        raw_dir.parent / "save_info",
        raw_dir.parent / "save_info.zip",
        root / "raw/SEED_VII/save_info",
        root / "raw/SEED_VII/save_info.zip",
        root / "labels/SEED_VII/save_info",
        root / "labels/SEED_VII/save_info.zip",
    )


def _load_cnt(path: Path) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MNE is required. Use the server BCI environment or install the "
            "packages listed in MPUS_GA/requirements.txt."
        ) from exc

    raw = mne.io.read_raw_cnt(path, preload=False, verbose="ERROR")
    removable = [channel for channel in raw.ch_names if channel.upper() in DROP_CHANNELS]
    if removable:
        raw.drop_channels(removable)
    if len(raw.ch_names) != 62:
        raise ValueError(
            f"{path.name} has {len(raw.ch_names)} channels after dropping "
            f"{removable}; expected 62. Remaining channels: {raw.ch_names}"
        )

    raw.load_data()
    raw.filter(l_freq=0.1, h_freq=70.0, n_jobs=1, verbose="ERROR")
    raw.notch_filter(freqs=[50.0], n_jobs=1, verbose="ERROR")
    raw.resample(TARGET_SFREQ, n_jobs=1, verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # Trial boundaries are annotations named 1 and 2. One official recording
    # also contains two KeyPad Response events which must not enter the pairs.
    boundary_ids = {event_id[name] for name in ("1", "2") if name in event_id}
    if boundary_ids:
        boundary_events = events[np.isin(events[:, 2], list(boundary_ids))]
        if len(boundary_events) >= 40:
            events = boundary_events
    trigger_samples = events[:, 0].astype(int).tolist()
    data = raw.get_data(units="uV").astype(np.float32, copy=False)
    channel_names = tuple(raw.ch_names)
    return data, channel_names, trigger_samples


def _seed_v_trials(path: Path) -> tuple[list[Trial], tuple[str, ...]]:
    subject, session = parse_seed_v_filename(path)
    data, channel_names, _ = _load_cnt(path)
    starts = SEED_V_START_SECONDS[session]
    ends = SEED_V_END_SECONDS[session]
    emotions = SEED_V_EMOTIONS[session]
    trials = []
    for offset, (start_sec, end_sec, emotion) in enumerate(
        zip(starts, ends, emotions), start=1
    ):
        start = int(round(start_sec * TARGET_SFREQ))
        end = int(round(end_sec * TARGET_SFREQ))
        if end > data.shape[1] or start >= end:
            raise ValueError(
                f"Invalid SEED-V trial boundary in {path.name}: "
                f"trial={offset}, start={start_sec}, end={end_sec}, "
                f"recording_seconds={data.shape[1] / TARGET_SFREQ:.2f}"
            )
        trials.append(
            Trial(
                data=data[:, start:end],
                subject=subject,
                session=session,
                global_trial=(session - 1) * 15 + offset,
                session_trial=offset,
                emotion=emotion,
                source_file=path.name,
            )
        )
    return trials, channel_names


def _discover_save_info(root: Path, raw_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return _find_existing(explicit, (), "SEED-VII save_info")
    return next((path.resolve() for path in _save_info_candidates(root, raw_dir) if path.exists()), None)


def _seed_vii_trials(
    path: Path,
    emotions: dict[int, str],
    save_info: Path | None,
) -> tuple[list[Trial], tuple[str, ...]]:
    subject, session = parse_seed_vii_filename(path)
    data, channel_names, trigger_samples = _load_cnt(path)
    if path.name in SEED_VII_SPECIAL_TRIGGER_START and save_info is not None:
        trigger_samples = read_special_trigger_samples(
            save_info, path.name, TARGET_SFREQ
        )
    if len(trigger_samples) < 40:
        raise MissingTriggerError(
            f"{path.name} has {len(trigger_samples)} usable trial-boundary "
            "triggers; expected 40. A manual trigger CSV is required."
        )

    trials = []
    for offset in range(1, 21):
        global_trial = (session - 1) * 20 + offset
        start = int(trigger_samples[2 * (offset - 1)])
        end = int(trigger_samples[2 * (offset - 1) + 1])
        if start < 0 or end > data.shape[1] or start >= end:
            raise ValueError(
                f"Invalid SEED-VII trigger pair in {path.name}: "
                f"trial={offset}, start={start}, end={end}, "
                f"recording_samples={data.shape[1]}"
            )
        trials.append(
            Trial(
                data=data[:, start:end],
                subject=subject,
                session=session,
                global_trial=global_trial,
                session_trial=offset,
                emotion=emotions[global_trial],
                source_file=path.name,
            )
        )
    return trials, channel_names


def _atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _session_payload(
    trials: list[Trial],
    channel_names: tuple[str, ...],
    scales: tuple[float, ...],
    original_mapping: dict[str, int],
) -> dict[float, dict[str, np.ndarray]]:
    fields: dict[float, dict[str, list[np.ndarray]]] = {
        scale: defaultdict(list) for scale in scales
    }
    for trial in trials:
        extracted = extract_multiscale_de(trial.data, TARGET_SFREQ, scales)
        for scale, (features, starts) in extracted.items():
            count = features.shape[0]
            if count == 0:
                raise ValueError(
                    f"{trial.source_file} trial {trial.global_trial} produced no "
                    f"{scale}s windows"
                )
            item = fields[scale]
            item["features"].append(features)
            item["label_original"].append(
                np.full(count, original_mapping[trial.emotion], dtype=np.int8)
            )
            item["label_3class"].append(
                np.full(count, EMOTION_TO_THREE_CLASS[trial.emotion], dtype=np.int8)
            )
            item["subject_id"].append(
                np.full(count, trial.subject, dtype=np.int16)
            )
            item["session_id"].append(
                np.full(count, trial.session, dtype=np.int8)
            )
            item["trial_id"].append(
                np.full(count, trial.global_trial, dtype=np.int16)
            )
            item["session_trial_id"].append(
                np.full(count, trial.session_trial, dtype=np.int8)
            )
            item["window_id"].append(np.arange(count, dtype=np.int16))
            item["start_second"].append(starts.astype(np.float32, copy=False))

    result = {}
    band_names = np.asarray([band[0] for band in BANDS], dtype="U16")
    band_edges = np.asarray([[band[1], band[2]] for band in BANDS], dtype=np.float32)
    for scale, item in fields.items():
        payload = {key: np.concatenate(values, axis=0) for key, values in item.items()}
        payload.update(
            {
                "channel_names": np.asarray(channel_names, dtype="U16"),
                "band_names": band_names,
                "band_edges_hz": band_edges,
                "sampling_rate_hz": np.asarray(TARGET_SFREQ, dtype=np.float32),
                "window_seconds": np.asarray(scale, dtype=np.float32),
                "source_file": np.asarray(trials[0].source_file),
                "feature_units": np.asarray("natural-log differential entropy of uV^2"),
            }
        )
        result[scale] = payload
    return result


def _existing_file_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["label_3class"]
        return {
            "path": str(path),
            "samples": int(archive["features"].shape[0]),
            "shape": list(archive["features"].shape),
            "class_counts_3class": {
                str(label): int((labels == label).sum()) for label in range(3)
            },
        }


def _process_dataset(
    dataset: str,
    raw_dir: Path,
    output_root: Path,
    scales: tuple[float, ...],
    subjects: set[int] | None,
    overwrite: bool,
    seed_vii_emotions: dict[int, str] | None,
    seed_vii_save_info: Path | None,
    missing_trigger_policy: str,
) -> dict[str, Any]:
    if dataset == "seed-vii":
        parser = parse_seed_vii_filename
        original_mapping = EMOTION_TO_ORIGINAL_SEED_VII
    else:
        parser = parse_seed_v_filename
        original_mapping = EMOTION_TO_ORIGINAL_SEED_V

    raw_files = []
    for path in raw_dir.glob("*.cnt"):
        if "_repaired" in path.stem:
            continue
        try:
            subject, session = parser(path)
        except ValueError:
            continue
        if subjects is None or subject in subjects:
            raw_files.append((subject, session, path))
    raw_files.sort(key=lambda item: (item[0], item[1], item[2].name))
    if not raw_files:
        raise FileNotFoundError(
            f"No matching {dataset} CNT files found in {raw_dir}"
        )

    manifest_files = []
    excluded_files = []
    canonical_channels: tuple[str, ...] | None = None
    for file_index, (subject, session, raw_path) in enumerate(raw_files, start=1):
        print(
            f"[{dataset} {file_index}/{len(raw_files)}] {raw_path.name}",
            flush=True,
        )
        expected_outputs = {
            scale: output_root
            / dataset.replace("-", "_")
            / f"window_{_scale_name(scale)}"
            / f"subject_{subject:02d}_session_{session}.npz"
            for scale in scales
        }
        if not overwrite and all(path.is_file() for path in expected_outputs.values()):
            for scale, path in expected_outputs.items():
                summary = _existing_file_summary(path)
                summary.update(
                    {
                        "window_seconds": _jsonable_number(scale),
                        "subject": subject,
                        "session": session,
                        "source_file": raw_path.name,
                    }
                )
                manifest_files.append(summary)
                if canonical_channels is None:
                    with np.load(path, allow_pickle=False) as archive:
                        canonical_channels = tuple(archive["channel_names"].tolist())
            print("  outputs already exist; validated and skipped", flush=True)
            continue

        try:
            if dataset == "seed-vii":
                assert seed_vii_emotions is not None
                trials, channel_names = _seed_vii_trials(
                    raw_path, seed_vii_emotions, seed_vii_save_info
                )
            else:
                trials, channel_names = _seed_v_trials(raw_path)
        except MissingTriggerError as exc:
            if missing_trigger_policy == "error":
                raise
            excluded_files.append(
                {
                    "subject": subject,
                    "session": session,
                    "source_file": raw_path.name,
                    "reason": str(exc),
                }
            )
            print(f"  EXCLUDED: {exc}", flush=True)
            continue

        if canonical_channels is None:
            canonical_channels = channel_names
        elif channel_names != canonical_channels:
            raise ValueError(
                f"Channel order mismatch in {raw_path.name}.\n"
                f"Expected: {canonical_channels}\nObserved: {channel_names}"
            )

        payloads = _session_payload(
            trials, channel_names, scales, original_mapping
        )
        for scale, path in expected_outputs.items():
            if overwrite or not path.exists():
                _atomic_save_npz(path, payloads[scale])
            summary = _existing_file_summary(path)
            summary.update(
                {
                    "window_seconds": _jsonable_number(scale),
                    "subject": subject,
                    "session": session,
                    "source_file": raw_path.name,
                    "source_size_bytes": raw_path.stat().st_size,
                    "source_mtime_ns": raw_path.stat().st_mtime_ns,
                }
            )
            manifest_files.append(summary)
            print(
                f"  {scale:g}s: {summary['samples']} samples -> {path}",
                flush=True,
            )

    sample_counts = Counter()
    for item in manifest_files:
        sample_counts[str(item["window_seconds"])] += item["samples"]
    manifest = {
        "dataset": dataset,
        "raw_dir": str(raw_dir),
        "sampling_rate_hz": TARGET_SFREQ,
        "windows_seconds": [_jsonable_number(scale) for scale in scales],
        "bands": [
            {"name": name, "low_hz": low, "high_hz": high}
            for name, low, high in BANDS
        ],
        "channels": list(canonical_channels or ()),
        "raw_files": len(raw_files),
        "processed_raw_files": len({item["source_file"] for item in manifest_files}),
        "excluded_raw_files": len(excluded_files),
        "excluded_files": excluded_files,
        "sample_counts": dict(sample_counts),
        "original_label_mapping": original_mapping,
        "three_class_mapping": EMOTION_TO_THREE_CLASS,
        "files": manifest_files,
    }
    manifest_path = output_root / dataset.replace("-", "_") / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute SEED-VII and SEED-V multiscale DE from raw CNT EEG"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("seed-vii", "seed-v"),
        default=("seed-vii", "seed-v"),
    )
    parser.add_argument(
        "--window-seconds", nargs="+", type=float, default=(1.0, 2.0, 4.0)
    )
    parser.add_argument("--subjects", type=_parse_subjects, default=None)
    parser.add_argument("--seed-vii-raw-dir", type=Path)
    parser.add_argument("--seed-v-raw-dir", type=Path)
    parser.add_argument("--seed-vii-label-file", type=Path)
    parser.add_argument("--seed-vii-save-info", type=Path)
    parser.add_argument(
        "--missing-trigger-policy",
        choices=("skip", "error"),
        default="skip",
        help="Skip and document CNT files without recoverable boundaries, or fail.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scales = validate_window_seconds(args.window_seconds, TARGET_SFREQ)
    subjects = args.subjects

    resolved: dict[str, Path] = {}
    if "seed-vii" in args.datasets:
        resolved["seed-vii"] = _find_existing(
            args.seed_vii_raw_dir,
            _raw_dir_candidates(data_root, "seed-vii"),
            "SEED-VII raw directory",
        )
    if "seed-v" in args.datasets:
        resolved["seed-v"] = _find_existing(
            args.seed_v_raw_dir,
            _raw_dir_candidates(data_root, "seed-v"),
            "SEED-V raw directory",
        )

    seed_vii_emotions = None
    seed_vii_save_info = None
    label_file = None
    if "seed-vii" in args.datasets:
        label_file = _find_existing(
            args.seed_vii_label_file,
            (
                data_root / "labels/SEED_VII/emotion_label_and_stimuli_order.xlsx",
                data_root / "feature/seed_vii/emotion_label_and_stimuli_order.xlsx",
                resolved["seed-vii"].parent / "emotion_label_and_stimuli_order.xlsx",
            ),
            "SEED-VII label file",
        )
        seed_vii_emotions = load_seed_vii_emotions(label_file)
        seed_vii_save_info = _discover_save_info(
            data_root, resolved["seed-vii"], args.seed_vii_save_info
        )

    config = {
        "data_root": str(data_root),
        "output_dir": str(output_root),
        "datasets": list(args.datasets),
        "window_seconds": [_jsonable_number(value) for value in scales],
        "subjects": "all" if subjects is None else sorted(subjects),
        "seed_vii_raw_dir": str(resolved.get("seed-vii", "")),
        "seed_v_raw_dir": str(resolved.get("seed-v", "")),
        "seed_vii_label_file": str(label_file or ""),
        "seed_vii_save_info": str(seed_vii_save_info or ""),
        "sampling_rate_hz": TARGET_SFREQ,
        "broadband_filter_hz": [0.1, 70.0],
        "notch_hz": 50.0,
        "de_formula": "0.5 * ln(2*pi*e*(variance+1e-8))",
        "lds_smoothing": False,
        "overwrite": bool(args.overwrite),
        "missing_trigger_policy": args.missing_trigger_policy,
    }
    (output_root / "processing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifests = {}
    for dataset in args.datasets:
        manifests[dataset] = _process_dataset(
            dataset=dataset,
            raw_dir=resolved[dataset],
            output_root=output_root,
            scales=scales,
            subjects=subjects,
            overwrite=args.overwrite,
            seed_vii_emotions=seed_vii_emotions,
            seed_vii_save_info=seed_vii_save_info,
            missing_trigger_policy=args.missing_trigger_policy,
        )

    summary = {
        dataset: {
            "raw_files": manifest["raw_files"],
            "processed_raw_files": manifest["processed_raw_files"],
            "excluded_raw_files": manifest["excluded_raw_files"],
            "sample_counts": manifest["sample_counts"],
        }
        for dataset, manifest in manifests.items()
    }
    (output_root / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
