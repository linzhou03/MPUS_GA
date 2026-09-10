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


def gpu_experiments(gpus, reverse=False):
    order = tuple("FEDCBA" if reverse else "ABCDEF")
    if len(gpus) == 1:
        return (order,)
    if len(gpus) == 2:
        return (order[::2], order[1::2])
    raise ValueError("Provide one or two physical GPUs")


def subgroup_arguments(config: SubgroupConfig) -> list[str]:
    return [part for name, value in asdict(config).items()
            for part in ("--subgroup-" + name.replace("_", "-"), str(value))]


def command(python, experiment, data_dir, result_root, seeds, subjects, config, method="r2_subgroup"):
    return [python, "-u", "-m", "MPUS_GA.trial_temporal.train",
            "--experiment", experiment, "--method", method,
            "--data-dir", str(data_dir), "--result-root", str(result_root),
            "--random-seeds", *map(str, seeds), "--target-subjects", subjects,
            "--source-batch-size", "24", "--target-batch-size", "16",
            "--evaluation-protocol", "fixed_final", "--device", "cuda:0",
            *subgroup_arguments(config)]


def main(method="r2_subgroup", config_type=SubgroupConfig, module=MODULE, default_gpus=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=default_gpus or ["0", "1"])
    parser.add_argument("--run-name")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[43, 42])
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--reverse", action="store_true", help="Run directions F to A; seed order stays unchanged")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name, default in asdict(config_type()).items():
        parser.add_argument("--subgroup-" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    config = config_type(**{name: getattr(args, f"subgroup_{name}")
                           for name in config_type.__dataclass_fields__})
    config.validate()
    if len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus) or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("Provide one or two distinct physical GPU indices")
    queues = gpu_experiments(args.gpus, reverse=args.reverse)
    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("Seeds must be distinct")
    package = Path(__file__).resolve().parents[1]
    run_name = args.run_name or method + datetime.now().strftime("_%Y%m%d_%H%M%S")
    if not run_name.startswith(method + "_") or Path(run_name).name != run_name:
        parser.error(f"Run name must be {method}_<unique-name>")
    data_dir = (args.data_dir or package / "data_processed").resolve()
    result_root = package / f"results_{run_name}"
    log_dir = package / "logs" / run_name
    if args.dry_run:
        for gpu, experiments in zip(args.gpus, queues, strict=True):
            for experiment in experiments:
                print(json.dumps({"physical_gpu": gpu, "direction": DIRECTIONS[experiment],
                                  "command": command(sys.executable, experiment, data_dir,
                                                     result_root, args.random_seeds,
                                                     args.target_subjects, config, method)}))
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
        "method": method, "directions": DIRECTIONS, "seeds": args.random_seeds,
        "subgroup_config": asdict(config), "subjects": args.target_subjects,
        "gpu_mapping": {gpu: list(exps) for gpu, exps in zip(args.gpus, queues)},
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
                [sys.executable, "-u", "-m", module, "--worker", "--run-name", run_name,
                 "--gpus", *args.gpus, "--data-dir", str(data_dir),
                 "--random-seeds", *map(str, args.random_seeds),
                 "--target-subjects", args.target_subjects,
                 *(["--reverse"] if args.reverse else []), *subgroup_arguments(config)],
                cwd=package.parent, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(f"Started PID={process.pid}\nResults: {result_root}\nLogs: {log_dir}")
        for gpu, experiments in zip(args.gpus, queues):
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
        for experiment in queues[index]:
            print(f"{datetime.now().isoformat(timespec='seconds')} START {experiment} "
                  f"GPU={args.gpus[index]} seeds={args.random_seeds}", flush=True)
            with (log_dir / f"{experiment}.log").open("a", encoding="utf-8") as log:
                result = subprocess.call(command(sys.executable, experiment, data_dir,
                                                 result_root, args.random_seeds,
                                                 args.target_subjects, config, method),
                                         cwd=package.parent, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT)
            print(f"{datetime.now().isoformat(timespec='seconds')} END {experiment} exit={result}",
                  flush=True)
            if result:
                return [experiment]
        return []

    print(f"SUITE START {run_name}", flush=True)
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        failures = [name for result in pool.map(run_queue, range(len(queues))) for name in result]
    print(f"SUITE END status={'failed' if failures else 'completed'} failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
