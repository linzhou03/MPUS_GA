#!/usr/bin/env bash
set -euo pipefail
mpus_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mpus_release_root="$(cd -- "$mpus_script_dir/../.." && pwd)"
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_output=/home/gzw/projects/MPUS_GA
mpus_run_name="${MPUS_R4_RUN_NAME:-r4_fullpseudo_s43_s42_ablation_s43_20260911}"
cd "$mpus_release_root"
export PYTHONPATH="$mpus_release_root"
export PYTHONDONTWRITEBYTECODE=1
case "${1:-plan}" in
  start0|start1|plan)
    mpus_gpus=("${1#start}")
    mpus_extra=()
    if [[ "$1" == plan ]]; then mpus_gpus=(0 1); mpus_extra=(--dry-run); fi
    for mpus_gpu in "${mpus_gpus[@]}"; do
      "$mpus_python" -u -m MPUS_GA.scripts.run_r4_suite --gpu "$mpus_gpu" \
        --run-name "$mpus_run_name" --data-dir "$mpus_output/data_processed" \
        --output-root "$mpus_output" "${mpus_extra[@]}"
    done
    ;;
  stop)
    "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run_name" --execute
    ;;
  *) echo 'Usage: bash r4_csu_commands.sh {start0|start1|plan|stop}' >&2; exit 2 ;;
esac
