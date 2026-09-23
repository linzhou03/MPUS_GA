#!/usr/bin/env bash
set -euo pipefail
mpus_run=cbst_quota_xju_fixed_20260922_s43_v1
mpus_parent=/home/gzw/projects/MPUS_GA_releases/cbst_quota_20260922_s43_v1
mpus_output=/home/gzw/projects/MPUS_GA
mpus_python=/home/gzw/anaconda3/envs/BCI/bin/python
cd "$mpus_parent"
export PYTHONPATH="$mpus_parent" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mpus_args=(-u -m MPUS_GA.scripts.run_cbst_suite --host xju --run-name "$mpus_run" --data-dir "$mpus_output/data_processed" --output-root "$mpus_output")
case "${1:-plan}" in
  plan) "$mpus_python" "${mpus_args[@]}" --dry-run ;;
  status) "$mpus_python" "${mpus_args[@]}" --status ;;
  report) "$mpus_python" "${mpus_args[@]}" --report ;;
  worker) exec "$mpus_python" "${mpus_args[@]}" ;;
  start)
    mkdir -p "$mpus_output/logs/$mpus_run"
    screen -dmS "$mpus_run" bash -c "exec bash '$mpus_parent/MPUS_GA/scripts/cbst_xju_commands.sh' worker >> '$mpus_output/logs/$mpus_run/suite.log' 2>&1"
    for mpus_attempt in $(seq 1 45); do
      sleep 1
      if "$mpus_python" -c 'import json,os,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="running"; os.kill(r["pid"],0)' "$mpus_output/results_$mpus_run/suite_state.json" 2>/dev/null; then
        echo "Running $mpus_run; logs: $mpus_output/logs/$mpus_run/suite.log"
        exit 0
      fi
    done
    echo "Startup not confirmed; inspect $mpus_output/logs/$mpus_run/suite.log" >&2;exit 1 ;;
  stop) "$mpus_python" -m MPUS_GA.scripts.stop_owned_training --suite-run-name "$mpus_run" --execute ;;
  *) echo 'Usage: cbst_xju_commands.sh {plan|start|status|report|stop|worker}' >&2;exit 2 ;;
esac
