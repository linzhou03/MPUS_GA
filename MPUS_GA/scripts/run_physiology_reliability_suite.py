"""Detached two-GPU runner for bidirectional N3 physiology reliability."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


GPU_EXPERIMENTS = {0: "A_N3", 1: "B_N3"}


def command(
    python: str,
    experiment: str,
    data_dir: Path,
    result_root: Path,
    seeds: list[int],
    subjects: str,
) -> list[str]:
    return [
        python,
        "-u",
        "-m",
        "MPUS_GA.trial_temporal.train",
        "--experiment",
        experiment,
        "--data-dir",
        str(data_dir),
        "--result-root",
        str(result_root),
        "--random-seeds",
        *map(str, seeds),
        "--target-subjects",
        subjects,
        "--source-batch-size",
        "24",
        "--target-batch-size",
        "16",
        "--evaluation-protocol",
        "fixed_final",
        "--device",
        "cuda:0",
    ]


def _code_hashes(package: Path) -> dict[str, str]:
    paths = [package / "layers.py", package / "__init__.py"]
    for directory in ("trial_temporal", "protocols", "preprocessing", "scripts"):
        paths.extend(sorted((package / directory).glob("*.py")))
    return {
        str(path.relative_to(package)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--random-seeds", nargs="+", type=int, default=[42, 43, 44]
    )
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("Provide two distinct physical GPU indices")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("Seeds must be distinct")

    package = Path(__file__).resolve().parents[1]
    run_name = args.run_name or (
        datetime.now().strftime("physiology_reliability_%Y%m%d_%H%M%S_")
        + str(os.getpid())
    )
    if (
        not run_name.startswith("physiology_reliability_")
        or Path(run_name).name != run_name
    ):
        parser.error(
            "Run name must be physiology_reliability_<unique-name>"
        )
    data_dir = (args.data_dir or package / "data_processed").resolve()
    result_root = package / f"results_{run_name}"
    log_dir = package / "logs" / run_name
    result_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.worker:
        with (log_dir / "suite.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "MPUS_GA.scripts.run_physiology_reliability_suite",
                    "--worker",
                    "--run-name",
                    run_name,
                    "--gpus",
                    *args.gpus,
                    "--data-dir",
                    str(data_dir),
                    "--random-seeds",
                    *map(str, args.random_seeds),
                    "--target-subjects",
                    args.target_subjects,
                ],
                cwd=package.parent,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started PID={process.pid}")
        print(f"Results: {result_root}")
        print(f"Logs: {log_dir}")
        print(f"GPU {args.gpus[0]}: A_N3")
        print(f"GPU {args.gpus[1]}: B_N3")
        return

    lock_handle = (result_root / ".suite.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another suite already owns this result root")

    data_files = sorted(
        path
        for domain in ("seed_v", "seed_vii")
        for path in (data_dir / domain).glob("window_*/*.npz")
    )
    if not data_files:
        raise SystemExit("No source/target DE files found")
    manifest = {
        "experiments": ["A_N3", "B_N3"],
        "gpu_mapping": {
            args.gpus[index]: GPU_EXPERIMENTS[index] for index in (0, 1)
        },
        "seeds": args.random_seeds,
        "subjects": args.target_subjects,
        "data_dir": str(data_dir),
        "python": sys.executable,
        "protocol": "cross_dataset_transductive_fixed1000",
        "code_sha256": _code_hashes(package),
        "data_identity": [
            (
                str(path.relative_to(data_dir)),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in data_files
        ],
    }
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    saved_manifest = result_root / "suite_manifest.json"
    if (
        saved_manifest.exists()
        and saved_manifest.read_text(encoding="utf-8") != serialized
    ):
        raise SystemExit("Configuration/code/data changed; use a new run name")
    if not saved_manifest.exists() and list(
        result_root.glob("*/seed_*_subject_*.json")
    ):
        raise SystemExit("Refusing to resume unmanifested results")
    saved_manifest.write_text(serialized, encoding="utf-8")

    def run_one(index: int) -> str | None:
        gpu = args.gpus[index]
        experiment = GPU_EXPERIMENTS[index]
        print(
            f"{datetime.now().isoformat(timespec='seconds')} START "
            f"{experiment} GPU={gpu}",
            flush=True,
        )
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gpu,
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        with (log_dir / f"{experiment}.log").open(
            "a", encoding="utf-8"
        ) as log:
            exit_code = subprocess.call(
                command(
                    sys.executable,
                    experiment,
                    data_dir,
                    result_root,
                    args.random_seeds,
                    args.target_subjects,
                ),
                cwd=package.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(
            f"{datetime.now().isoformat(timespec='seconds')} END "
            f"{experiment} GPU={gpu} exit={exit_code}",
            flush=True,
        )
        return experiment if exit_code else None

    print(f"SUITE START {run_name}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failures = [item for item in pool.map(run_one, (0, 1)) if item]
    print(
        f"SUITE END status={'failed' if failures else 'completed'} "
        f"failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
