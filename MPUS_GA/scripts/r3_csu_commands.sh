#!/usr/bin/env bash
# GPU 1 only, as requested on 2026-09-11. plan/status never mutate training processes.
set -euo pipefail
mpus_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mpus_release_root="$(cd -- "$mpus_script_dir/../.." && pwd)"
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_run_name="${MPUS_R3_RUN_NAME:-r3_six_direction_full_s43_s42_ablation_s43_20260910}"
cd "$mpus_release_root"
export PYTHONPATH="$mpus_release_root"
export PYTHONDONTWRITEBYTECODE=1

case "${1:-status}" in
  status)
    nvidia-smi
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run_name"
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name r2_coteaching_six_direction_s43_s42_20260910
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --gpu 1
    ;;
  stop-old)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name r2_coteaching_six_direction_s43_s42_20260910 --execute
    ;;
  stop-gpu1)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --gpu 1 --execute
    ;;
  stop-r3)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run_name" --execute
    ;;
  start|plan)
    mpus_extra=()
    if [[ "$1" == plan ]]; then mpus_extra+=(--dry-run); fi
    "$mpus_python" -u -m MPUS_GA.scripts.run_r3_suite \
      --gpu 1 --allow-scheduler-update --run-name "$mpus_run_name" \
      --data-dir "$mpus_output/data_processed" --output-root "$mpus_output" \
      --full-seeds 43 42 --ablation-seed 43 --target-subjects all "${mpus_extra[@]}"
    ;;
  *)
    echo 'Usage: bash r3_csu_commands.sh {status|plan|stop-old|stop-gpu1|stop-r3|start}' >&2
    exit 2
    ;;
esac
