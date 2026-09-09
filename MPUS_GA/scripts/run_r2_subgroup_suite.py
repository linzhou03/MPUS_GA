"""Run R2 subgroup A-F on two GPUs, one direction per GPU at a time."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from MPUS_GA.trial_temporal.subgroup_alignment import SubgroupConfig
from .run_six_direction_final_suite import DIRECTIONS, _code_hashes, resolve_gpu_uuid


GPU_EXPERIMENTS = (("A", "C", "E"), ("B", "D", "F"))
MODULE = "MPUS_GA.scripts.run_r2_subgroup_suite"


def subgroup_arguments(config: SubgroupConfig) -> list[str]:
    return [part for name, value in asdict(config).items()
            for part in ("--subgroup-" + name.replace("_", "-"), str(value))]


def command(python, experiment, data_dir, result_root, seeds, subjects, config):
    return [python, "-u", "-m", "MPUS_GA.trial_temporal.train",
            "--experiment", experiment, "--method", "r2_subgroup",
            "--data-dir", str(data_dir), "--result-root", str(result_root),
            "--random-seeds", *map(str, seeds), "--target-subjects", subjects,
            "--source-batch-size", "24", "--target-batch-size", "16",
            "--evaluation-protocol", "fixed_final", "--device", "cuda:0",
            *subgroup_arguments(config)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--run-name")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[43, 42])
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name, default in asdict(SubgroupConfig()).items():
        parser.add_argument("--subgroup-" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    config = SubgroupConfig(**{name: getattr(args, f"subgroup_{name}")
                              for name in SubgroupConfig.__dataclass_fields__})
    config.validate()
    if len(set(args.gpus)) != 2 or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("Provide two distinct physical GPU indices")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("Seeds must be distinct")
    package = Path(__file__).resolve().parents[1]
    run_name = args.run_name or datetime.now().strftime("r2_subgroup_%Y%m%d_%H%M%S")
    if not run_name.startswith("r2_subgroup_") or Path(run_name).name != run_name:
        parser.error("Run name must be r2_subgroup_<unique-name>")
    data_dir = (args.data_dir or package / "data_processed").resolve()
    result_root = package / f"results_{run_name}"
    log_dir = package / "logs" / run_name
    if args.dry_run:
        for gpu, experiments in zip(args.gpus, GPU_EXPERIMENTS, strict=True):
            for experiment in experiments:
                print(json.dumps({"physical_gpu": gpu, "direction": DIRECTIONS[experiment],
                                  "command": command(sys.executable, experiment, data_dir,
                                                     result_root, args.random_seeds,
                                                     args.target_subjects, config)}))
        return

    uuids = [resolve_gpu_uuid(gpu) for gpu in args.gpus]
    data_files = []
    for domain in ("seed_iv", "seed_v", "seed_vii"):
        files = sorted((data_dir / domain).glob("window_*/*.npz"))
        if not files:
            raise SystemExit(f"Missing processed data: {domain}")
        data_files.extend(files)
    result_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "r2_subgroup", "directions": DIRECTIONS, "seeds": args.random_seeds,
        "subgroup_config": asdict(config), "subjects": args.target_subjects,
        "gpu_mapping": {gpu: list(exps) for gpu, exps in zip(args.gpus, GPU_EXPERIMENTS)},
        "gpu_uuids": uuids, "python": sys.executable, "data_dir": str(data_dir),
        "protocol": "fixed_final_1000_no_target_label_selection",
        "code_sha256": _code_hashes(package),
        "data_identity": [(str(p.relative_to(data_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in data_files],
    }
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    saved = result_root / "suite_manifest.json"
    if saved.exists() and saved.read_text() != serialized:
        raise SystemExit("Configuration/code/data changed; use a new run name")
    if not saved.exists() and list(result_root.glob("*/seed_*_subject_*.json")):
        raise SystemExit("Refusing to resume unmanifested results")

    if not args.worker:
        with (log_dir / "suite.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", MODULE, "--worker", "--run-name", run_name,
                 "--gpus", *args.gpus, "--data-dir", str(data_dir),
                 "--random-seeds", *map(str, args.random_seeds),
                 "--target-subjects", args.target_subjects, *subgroup_arguments(config)],
                cwd=package.parent, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(f"Started PID={process.pid}\nResults: {result_root}\nLogs: {log_dir}")
        for gpu, experiments in zip(args.gpus, GPU_EXPERIMENTS):
            print(f"GPU {gpu}: {' -> '.join(experiments)}; seeds={args.random_seeds}")
        return

    lock = (result_root / ".suite.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another suite owns this result root")
    saved.write_text(serialized, encoding="utf-8")

    def run_queue(index):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[index],
                           CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                           PYTHONDONTWRITEBYTECODE="1")
        for experiment in GPU_EXPERIMENTS[index]:
            print(f"{datetime.now().isoformat(timespec='seconds')} START {experiment} "
                  f"GPU={args.gpus[index]} seeds={args.random_seeds}", flush=True)
            with (log_dir / f"{experiment}.log").open("a", encoding="utf-8") as log:
                result = subprocess.call(command(sys.executable, experiment, data_dir,
                                                 result_root, args.random_seeds,
                                                 args.target_subjects, config),
                                         cwd=package.parent, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT)
            print(f"{datetime.now().isoformat(timespec='seconds')} END {experiment} exit={result}",
                  flush=True)
            if result:
                return [experiment]
        return []

    print(f"SUITE START {run_name}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failures = [name for result in pool.map(run_queue, (0, 1)) for name in result]
    print(f"SUITE END status={'failed' if failures else 'completed'} failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
