#!/usr/bin/env bash
set -euo pipefail

mpus_run=r3_posgate_csu_targetnorm_c01_post300bal_s43_20260924_v1
mpus_parent=/home/gzw/projects/MPUS_GA_releases/r3_posgate_targetnorm_c01_20260924_v1
mpus_output=/home/gzw/projects/MPUS_GA
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_gpu_uuid=GPU-9ec3f7d5-d9b8-c2be-97a6-9429d99f311e
mpus_root="$mpus_output/results_$mpus_run"
mpus_log_root="$mpus_output/logs/$mpus_run"
mpus_mode=starting

cd "$mpus_parent"
export PYTHONPATH="$mpus_parent" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$mpus_gpu_uuid"

mpus_state() {
  mkdir -p "$mpus_root"
  printf '{"status":"%s","mode":"%s","gpu_uuid":"%s","seed":43,"direction":"C","subject":1,"selection":"post300_bal_best"}\n' \
    "$1" "$mpus_mode" "$mpus_gpu_uuid" > "$mpus_root/.state.tmp"
  mv "$mpus_root/.state.tmp" "$mpus_root/suite_state.json"
}

case "${1:-plan}" in
  plan)
    echo "run=$mpus_run; GPU=0; direction=C; subject=01; seed=43; variant=positive_gate"
    echo 'Sequence: std_only, domain; target-test BalAcc best among steps 301..1000'
    echo "Results: $mpus_root/{std_only,domain}/C/"
    echo "Logs: $mpus_log_root/{std_only,domain}.log" ;;
  status)
    if test -f "$mpus_root/suite_state.json"; then cat "$mpus_root/suite_state.json"; else echo 'Not started'; fi
    for mpus_item in std_only domain; do
      test ! -f "$mpus_root/$mpus_item/C/seed_43_subject_01.offline.json" || echo "Completed: $mpus_item"
    done ;;
  worker)
    mkdir -p "$mpus_root" "$mpus_log_root"
    exec 9>"$mpus_root/.suite.lock"
    flock -n 9 || { echo 'Suite already active'; exit 0; }
    trap 'mpus_state failed' ERR
    for mpus_mode in std_only domain; do
      mpus_state running
      echo "$(date -Is) START $mpus_mode on GPU 0"
      "$mpus_python" -u -m MPUS_GA.trial_temporal.train_cbst \
        --direction C --variant positive_gate --data-dir "$mpus_output/data_processed" \
        --result-root "$mpus_root/$mpus_mode" --seed 43 --subjects 1 \
        --selection post300_bal_best --target-normalization "$mpus_mode" \
        >> "$mpus_log_root/$mpus_mode.log" 2>&1
      test -f "$mpus_root/$mpus_mode/C/seed_43_subject_01.offline.json"
      echo "$(date -Is) COMPLETE $mpus_mode"
    done
    mpus_mode=all
    mpus_state complete ;;
  start)
    mkdir -p "$mpus_log_root"
    if test -f "$mpus_root/suite_state.json" && grep -q '"status":"running"' "$mpus_root/suite_state.json"; then
      echo 'Suite already marked running'; exit 0
    fi
    nohup bash "$mpus_parent/MPUS_GA/scripts/target_normalization_c01_csu_commands.sh" worker \
      >> "$mpus_log_root/suite.log" 2>&1 < /dev/null &
    mpus_pid=$!
    for mpus_attempt in $(seq 1 60); do
      sleep 1
      if test -f "$mpus_root/suite_state.json" && grep -q '"status":"running"' "$mpus_root/suite_state.json"; then
        echo "Running PID=$mpus_pid; log=$mpus_log_root/suite.log"
        exit 0
      fi
      kill -0 "$mpus_pid" 2>/dev/null || break
    done
    echo "Startup failed; inspect $mpus_log_root/suite.log" >&2; exit 1 ;;
  *) echo 'Usage: target_normalization_c01_csu_commands.sh {plan|start|status|worker}' >&2; exit 2 ;;
esac
