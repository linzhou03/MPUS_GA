#!/usr/bin/env bash
set -euo pipefail
mpus_host=xju
mpus_python=/home/gzw/anaconda3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_parent=/home/gzw/projects/MPUS_GA_releases/neighbor_soft_20260920_v1
mpus_run="${MPUS_NEIGHBOR_RUN_NAME:-neighbor_soft_20260920_v1}"
cd "$mpus_parent"
export PYTHONPATH="$mpus_parent" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
case "${1:-plan}" in
  start|plan|status)
    mpus_extra=()
    if [[ "${1:-plan}" == plan ]]; then mpus_extra=(--dry-run); fi
    if [[ "${1:-plan}" == status ]]; then mpus_extra=(--status); fi
    "$mpus_python" -u -m MPUS_GA.scripts.run_neighbor_soft_suite --host "$mpus_host" \
      --run-name "$mpus_run" --data-dir "$mpus_output/data_processed" --output-root "$mpus_output" "${mpus_extra[@]}"
    ;;
  stop)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run" --execute
    ;;
  *) echo 'Usage: bash neighbor_soft_xju_commands.sh {start|plan|status|stop}' >&2; exit 2 ;;
esac
