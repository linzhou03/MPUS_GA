#!/usr/bin/env bash
set -euo pipefail

mpus_run=r3_posgate_csu_domain_ac_post300bal_s43_20260924_v2
mpus_parent=/home/gzw/projects/MPUS_GA_releases/r3_posgate_domain_ac_dualgpu_20260924_v2
mpus_output=/home/gzw/projects/MPUS_GA
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_gpu0_uuid=GPU-9ec3f7d5-d9b8-c2be-97a6-9429d99f311e
mpus_gpu1_uuid=GPU-07872aeb-d8e3-abe6-9f21-ad76fd39c412
mpus_root="$mpus_output/results_$mpus_run"
mpus_log_root="$mpus_output/logs/$mpus_run"

cd "$mpus_parent"
export PYTHONPATH="$mpus_parent" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mpus_state() {
  local state="$1"
  local temp="$mpus_root/.suite_state.$$.tmp"
  printf '{"status":"%s","run":"%s","directions":["A","C"],"expected_folds":{"A":16,"C":16},"variant":"positive_gate","normalization":"domain","selection":"post300_bal_best","seed":43,"gpu_mapping":{"A":"0","C":"1"},"gpu_uuids":{"A":"%s","C":"%s"},"worker_pid":%s,"A_pid":%s,"C_pid":%s}\n' \
    "$state" "$mpus_run" "$mpus_gpu0_uuid" "$mpus_gpu1_uuid" "$$" "${mpus_a_pid:-null}" "${mpus_c_pid:-null}" > "$temp"
  mv "$temp" "$mpus_root/suite_state.json"
}

mpus_worker() {
  mkdir -p "$mpus_root" "$mpus_log_root"
  exec 9>"$mpus_root/.suite.lock"
  flock -n 9 || { echo 'Suite already active'; exit 1; }
  mpus_a_pid=null
  mpus_c_pid=null
  mpus_state starting
  echo "$(date -Is) launching A on GPU 0 and C on GPU 1 concurrently"
  CUDA_VISIBLE_DEVICES="$mpus_gpu0_uuid" "$mpus_python" -u -m MPUS_GA.trial_temporal.train_cbst \
    --direction A --variant positive_gate --data-dir "$mpus_output/data_processed" \
    --result-root "$mpus_root" --seed 43 --subjects all \
    --selection post300_bal_best --target-normalization domain \
    >> "$mpus_log_root/A.log" 2>&1 &
  mpus_a_pid=$!
  CUDA_VISIBLE_DEVICES="$mpus_gpu1_uuid" "$mpus_python" -u -m MPUS_GA.trial_temporal.train_cbst \
    --direction C --variant positive_gate --data-dir "$mpus_output/data_processed" \
    --result-root "$mpus_root" --seed 43 --subjects all \
    --selection post300_bal_best --target-normalization domain \
    >> "$mpus_log_root/C.log" 2>&1 &
  mpus_c_pid=$!
  mpus_state running
  echo "$(date -Is) A pid=$mpus_a_pid C pid=$mpus_c_pid"

  local a_code=0 c_code=0 direction subject count
  wait "$mpus_a_pid" || a_code=$?
  wait "$mpus_c_pid" || c_code=$?
  if ((a_code != 0 || c_code != 0)); then
    echo "$(date -Is) FAILED: A exit=$a_code C exit=$c_code"
    mpus_state failed
    return 1
  fi
  for direction in A C; do
    count=0
    for subject in $(seq -w 1 16); do
      test -s "$mpus_root/$direction/seed_43_subject_$subject.offline.json" || {
        echo "Missing $direction subject $subject offline report" >&2
        mpus_state failed
        return 1
      }
      count=$((count+1))
    done
    echo "$(date -Is) $direction complete: $count/16 folds"
  done
  mpus_state complete
}

case "${1:-plan}" in
  plan)
    echo "run=$mpus_run; GPU0=A; GPU1=C; simultaneous; 16 folds each; seed=43"
    echo 'variant=positive_gate; target-normalization=domain; target-test BalAcc best among steps 301..1000'
    echo "Results: $mpus_root/{A,C}/"
    echo "Logs: $mpus_log_root/{A,C,suite}.log" ;;
  status)
    if test -f "$mpus_root/suite_state.json"; then cat "$mpus_root/suite_state.json"; else echo 'Not started'; fi
    for direction in A C; do
      count=0
      if test -d "$mpus_root/$direction"; then
        count=$(find "$mpus_root/$direction" -maxdepth 1 -name 'seed_43_subject_*.offline.json' | wc -l)
      fi
      echo "$direction $count/16"
    done ;;
  worker) mpus_worker ;;
  start)
    mkdir -p "$mpus_log_root" "$mpus_root"
    if test -s "$mpus_root/suite_state.json" && grep -q '"status":"running"' "$mpus_root/suite_state.json"; then
      echo 'Suite already marked running'; exit 1
    fi
    nohup bash "$mpus_parent/MPUS_GA/scripts/target_normalization_ac_dualgpu_csu_commands.sh" worker \
      >> "$mpus_log_root/suite.log" 2>&1 < /dev/null &
    mpus_pid=$!
    for mpus_attempt in $(seq 1 30); do
      sleep 1
      if test -s "$mpus_root/suite_state.json" && grep -q '"status":"running"' "$mpus_root/suite_state.json"; then
        echo "Running worker PID=$mpus_pid"
        cat "$mpus_root/suite_state.json"
        exit 0
      fi
      kill -0 "$mpus_pid" 2>/dev/null || break
    done
    echo "Startup failed; inspect $mpus_log_root/suite.log" >&2; exit 1 ;;
  *) echo 'Usage: target_normalization_ac_dualgpu_csu_commands.sh {plan|start|status|worker}' >&2; exit 2 ;;
esac
