#!/usr/bin/env bash
set -euo pipefail
mpus_parent="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_run="${MPUS_PCDIAG_RUN_NAME:-r2_original_pcdiag_s42_s43_s44_20260914_v1}"
cd "$mpus_parent"
export PYTHONPATH="$mpus_parent"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "${1:-plan}" in
  start|plan)
    mpus_extra=()
    if [[ "${1:-plan}" == plan ]]; then mpus_extra=(--dry-run); fi
    "$mpus_python" -u -m MPUS_GA.scripts.run_r2_pcdiag_suite --gpus 0 1 \
      --run-name "$mpus_run" --data-dir "$mpus_output/data_processed" \
      --output-root "$mpus_output" "${mpus_extra[@]}"
    ;;
  stop)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run" --execute
    ;;
  *) echo 'Usage: bash r2_pcdiag_csu_commands.sh {start|plan|stop}' >&2; exit 2 ;;
esac
