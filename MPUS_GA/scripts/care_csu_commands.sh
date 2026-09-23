#!/usr/bin/env bash
set -euo pipefail
mpus_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_run="${MPUS_CARE_RUN_NAME:-care_r2_s43_s42_ablation_s43_20260911_v1}"
cd "$mpus_dir"
export PYTHONPATH="$mpus_dir"
export PYTHONDONTWRITEBYTECODE=1
case "${1:-plan}" in
  start|plan)
    mpus_extra=()
    if [[ "$1" == plan ]]; then mpus_extra=(--dry-run); fi
    "$mpus_python" -u -m MPUS_GA.scripts.run_care_suite --gpus 0 1 \
      --run-name "$mpus_run" --data-dir "$mpus_output/data_processed" --output-root "$mpus_output" "${mpus_extra[@]}"
    ;;
  stop)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run" --execute
    for mpus_variant in full r2 anchor_ce no_geometry fixed_selection; do
      "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "${mpus_run}_${mpus_variant}" --execute
    done
    ;;
  *) echo 'Usage: bash care_csu_commands.sh {start|plan|stop}' >&2; exit 2 ;;
esac
