"""Detached two-GPU queue for R2, N1 anatomy, and N2 physiology experiments."""

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


GPU_EXPERIMENTS = {
    0: ("A_R2", "A_N1", "A_N2"),
    1: ("B_R2", "B_N1", "B_N2"),
}
EXPERIMENTS = tuple(
    experiment
    for queue in GPU_EXPERIMENTS.values()
    for experiment in queue
)


def command(
    python: str,
    experiment: str,
    data: Path,
    result: Path,
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
        str(data),
        "--result-root",
        str(result),
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


def _code_paths(package: Path) -> list[Path]:
    paths = [package / "layers.py", package / "__init__.py"]
    for directory in ("trial_temporal", "protocols", "preprocessing", "scripts"):
        paths.extend(sorted((package / directory).glob("*.py")))
    return sorted(path for path in paths if path.is_file())


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
        datetime.now().strftime("anatomical_physiology_%Y%m%d_%H%M%S_")
        + str(os.getpid())
    )
    if (
        not run_name.startswith("anatomical_physiology_")
        or Path(run_name).name != run_name
    ):
        parser.error(
            "Run name must be anatomical_physiology_<unique-name>, without separators"
        )
    data = (args.data_dir or package / "data_processed").resolve()
    result = package / f"results_{run_name}"
    logs = package / "logs" / run_name
    result.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    if not args.worker:
        with (logs / "suite.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "MPUS_GA.scripts.run_anatomical_physiology_suite",
                    "--worker",
                    "--run-name",
                    run_name,
                    "--gpus",
                    *args.gpus,
                    "--data-dir",
                    str(data),
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
        print(f"Started PID={child.pid}")
        print(f"Run name: {run_name}")
        print(f"Results: {result}")
        print(f"Logs: {logs}")
        print("GPU 0 queue: A_R2 -> A_N1 -> A_N2")
        print("GPU 1 queue: B_R2 -> B_N1 -> B_N2")
        print("Each GPU runs at most one training process.")
        return

    lock_handle = (result / ".suite.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another suite already owns this result root")

    data_files = [
        path
        for domain in ("seed_v", "seed_vii")
        for path in (data / domain).glob("window_*/*.npz")
    ]
    if not data_files:
        raise SystemExit("No source/target DE files found")
    manifest = {
        "experiments": list(EXPERIMENTS),
        "gpu_queues": {
            args.gpus[index]: list(GPU_EXPERIMENTS[index]) for index in (0, 1)
        },
        "seeds": args.random_seeds,
        "subjects": args.target_subjects,
        "data_dir": str(data),
        "python": sys.executable,
        "protocol": "cross_dataset_transductive_fixed1000",
        "code_sha256": {
            str(path.relative_to(package)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in _code_paths(package)
        },
        "data_identity": [
            (str(path.relative_to(data)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(data_files)
        ],
    }
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    saved_manifest = result / "suite_manifest.json"
    if saved_manifest.exists() and saved_manifest.read_text(encoding="utf-8") != serialized:
        raise SystemExit("Configuration/code/data changed; use a new run name")
    if not saved_manifest.exists() and list(
        result.glob("*/seed_*_subject_*.json")
    ):
        raise SystemExit("Refusing to resume unmanifested results")
    saved_manifest.write_text(serialized, encoding="utf-8")

    def run_gpu_queue(queue_index: int) -> list[str]:
        physical_gpu = args.gpus[queue_index]
        failures = []
        for experiment in GPU_EXPERIMENTS[queue_index]:
            print(
                f"{datetime.now().isoformat(timespec='seconds')} START "
                f"{experiment} GPU={physical_gpu}",
                flush=True,
            )
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=physical_gpu,
                PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            with (logs / f"{experiment}.log").open(
                "a", encoding="utf-8"
            ) as log:
                exit_code = subprocess.call(
                    command(
                        sys.executable,
                        experiment,
                        data,
                        result,
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
                f"{experiment} GPU={physical_gpu} exit={exit_code}",
                flush=True,
            )
            if exit_code:
                failures.append(experiment)
        return failures

    print(f"SUITE START {run_name}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failures = sum(list(pool.map(run_gpu_queue, (0, 1))), [])
    print(
        f"SUITE END status={'failed' if failures else 'completed'} "
        f"failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
