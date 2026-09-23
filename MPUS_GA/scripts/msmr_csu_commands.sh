#!/usr/bin/env bash
set -euo pipefail
mpus_parent="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_run="${MPUS_MSMR_RUN_NAME:-msmr_main_s42_s43_s44_20260913_v1}"
mpus_action="${1:-plan}"
cd "$mpus_parent"
export PYTHONPATH="$mpus_parent"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$mpus_action" in
  start|plan)
    mpus_extra=()
    if [[ "$mpus_action" == plan ]]; then mpus_extra=(--dry-run); fi
    "$mpus_python" -u -m MPUS_GA.scripts.run_msmr_suite --gpus 0 1 \
      --run-name "$mpus_run" --data-dir "$mpus_output/data_processed" \
      --output-root "$mpus_output" "${mpus_extra[@]}"
    ;;
  stop)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run" --execute
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "${mpus_run}_full" --execute
    ;;
  *) echo 'Usage: bash msmr_csu_commands.sh {start|plan|stop}' >&2; exit 2 ;;
esac
