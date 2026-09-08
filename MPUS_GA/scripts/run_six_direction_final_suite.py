"""Run the final A-F cross-dataset suite sequentially on one physical GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = ("A", "B", "C", "D", "E", "F")
DIRECTIONS = {
    "A": "SEED-VII -> SEED-V",
    "B": "SEED-V -> SEED-VII",
    "C": "SEED-IV -> SEED-V",
    "D": "SEED-V -> SEED-IV",
    "E": "SEED-IV -> SEED-VII",
    "F": "SEED-VII -> SEED-IV",
}
RANDOM_SEED = 43


def resolve_gpu_uuid(physical_gpu: str) -> str:
    """Resolve by PCI identity so a failed lower-index GPU cannot poison CUDA."""

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            physical_gpu,
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    gpu_uuid = output.splitlines()[0].strip() if output else ""
    if not gpu_uuid.startswith("GPU-"):
        raise RuntimeError(
            f"Could not resolve physical GPU {physical_gpu} to a GPU UUID"
        )
    return gpu_uuid


def command(
    python: str,
    experiment: str,
    data_dir: Path,
    result_root: Path,
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
        str(RANDOM_SEED),
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
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("GPU must be a physical CUDA index")

    package = Path(__file__).resolve().parents[1]
    run_name = args.run_name or (
        datetime.now().strftime("six_direction_final_%Y%m%d_%H%M%S_")
        + str(os.getpid())
    )
    if (
        not run_name.startswith("six_direction_final_")
        or Path(run_name).name != run_name
    ):
        parser.error("Run name must be six_direction_final_<unique-name>")
    data_dir = (args.data_dir or package / "data_processed").resolve()
    gpu_uuid = resolve_gpu_uuid(args.gpu)
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
                    "MPUS_GA.scripts.run_six_direction_final_suite",
                    "--worker",
                    "--gpu",
                    args.gpu,
                    "--run-name",
                    run_name,
                    "--data-dir",
                    str(data_dir),
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
        print(f"Physical GPU: {args.gpu}")
        print(f"CUDA GPU UUID: {gpu_uuid}")
        print(f"Sequential experiments: {' -> '.join(EXPERIMENTS)}")
        return

    lock_handle = (result_root / ".suite.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another suite already owns this result root")

    data_files = sorted(
        path
        for domain in ("seed_iv", "seed_v", "seed_vii")
        for path in (data_dir / domain).glob("window_*/*.npz")
    )
    if not data_files:
        raise SystemExit("No SEED-IV/V/VII DE files found")
    manifest = {
        "experiments": list(EXPERIMENTS),
        "directions": DIRECTIONS,
        "physical_gpu": args.gpu,
        "cuda_visible_device": gpu_uuid,
        "seed": RANDOM_SEED,
        "subjects": args.target_subjects,
        "data_dir": str(data_dir),
        "python": sys.executable,
        "protocol": "pairwise_cross_dataset_transductive_fixed1000",
        "execution": "strictly_sequential_one_experiment_at_a_time",
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

    environment = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    failures = []
    print(
        f"SUITE START {run_name} GPU={args.gpu} seed={RANDOM_SEED}",
        flush=True,
    )
    for experiment in EXPERIMENTS:
        print(
            f"{datetime.now().isoformat(timespec='seconds')} START "
            f"{experiment} {DIRECTIONS[experiment]}",
            flush=True,
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
            f"{experiment} exit={exit_code}",
            flush=True,
        )
        if exit_code:
            failures.append(experiment)
            break
    print(
        f"SUITE END status={'failed' if failures else 'completed'} "
        f"failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
