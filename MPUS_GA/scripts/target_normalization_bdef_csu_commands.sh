#!/usr/bin/env bash
set -euo pipefail

mpus_run=r3_posgate_csu_domain_bdef_post300bal_s43_20260924_v1
mpus_parent=/home/gzw/projects/MPUS_GA_releases/r3_posgate_domain_bdef_20260924_v1
mpus_output=/home/gzw/projects/MPUS_GA
mpus_python=/home/gzw/miniforge3/envs/BCI/bin/python
mpus_gpu0_uuid=GPU-9ec3f7d5-d9b8-c2be-97a6-9429d99f311e
mpus_gpu1_uuid=GPU-07872aeb-d8e3-abe6-9f21-ad76fd39c412
mpus_root="$mpus_output/results_$mpus_run"
mpus_log_root="$mpus_output/logs/$mpus_run"
mpus_ac_root="$mpus_output/results_r3_posgate_csu_domain_ac_post300bal_s43_20260924_v2"

cd "$mpus_parent"
export PYTHONPATH="$mpus_parent" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mpus_state() {
  local state="$1" temp="$mpus_root/.suite_state.$$.tmp"
  printf '{"status":"%s","run":"%s","variant":"positive_gate","target_normalization":"domain","selection":"post300_bal_best","seed":43,"queues":{"gpu0":["B","D"],"gpu1":["E","F"]},"expected_folds":{"B":20,"D":15,"E":20,"F":15},"A_C_reused_from":"%s","worker_pid":%s,"gpu0_queue_pid":%s,"gpu1_queue_pid":%s}\n' \
    "$state" "$mpus_run" "$mpus_ac_root" "$$" "${mpus_gpu0_pid:-null}" "${mpus_gpu1_pid:-null}" > "$temp"
  mv "$temp" "$mpus_root/suite_state.json"
}

mpus_queue() {
  local gpu_uuid="$1" direction
  shift
  for direction in "$@"; do
    echo "$(date -Is) START direction=$direction gpu=$gpu_uuid"
    CUDA_VISIBLE_DEVICES="$gpu_uuid" "$mpus_python" -u -m MPUS_GA.trial_temporal.train_cbst \
      --direction "$direction" --variant positive_gate \
      --data-dir "$mpus_output/data_processed" --result-root "$mpus_root" \
      --seed 43 --subjects all --selection post300_bal_best --target-normalization domain \
      >> "$mpus_log_root/$direction.log" 2>&1 || return $?
    echo "$(date -Is) COMPLETE direction=$direction"
  done
}

mpus_worker() {
  mkdir -p "$mpus_root" "$mpus_log_root"
  exec 9>"$mpus_root/.suite.lock"
  flock -n 9 || { echo 'Suite already active'; exit 1; }
  mpus_gpu0_pid=null
  mpus_gpu1_pid=null
  mpus_state starting
  cat > "$mpus_root/six_direction_sources.json" <<EOF
{"A":"$mpus_ac_root/A","B":"$mpus_root/B","C":"$mpus_ac_root/C","D":"$mpus_root/D","E":"$mpus_root/E","F":"$mpus_root/F","seed":43,"selection":"post300_bal_best","target_normalization":"domain","variant":"positive_gate"}
EOF
  echo "$(date -Is) parallel queues: GPU0 B->D; GPU1 E->F; A/C reused"
  mpus_queue "$mpus_gpu0_uuid" B D &
  mpus_gpu0_pid=$!
  mpus_queue "$mpus_gpu1_uuid" E F &
  mpus_gpu1_pid=$!
  mpus_state running
  echo "$(date -Is) gpu0_queue_pid=$mpus_gpu0_pid gpu1_queue_pid=$mpus_gpu1_pid"

  local gpu0_code=0 gpu1_code=0 direction expected count subject
  wait "$mpus_gpu0_pid" || gpu0_code=$?
  wait "$mpus_gpu1_pid" || gpu1_code=$?
  if ((gpu0_code != 0 || gpu1_code != 0)); then
    echo "$(date -Is) FAILED gpu0=$gpu0_code gpu1=$gpu1_code" >&2
    mpus_state failed
    return 1
  fi
  for direction in B D E F; do
    case "$direction" in B|E) expected=20 ;; D|F) expected=15 ;; esac
    count=0
    for subject in $(seq -w 1 "$expected"); do
      test -s "$mpus_root/$direction/seed_43_subject_$subject.offline.json" || {
        echo "Missing $direction subject $subject offline report" >&2
        mpus_state failed
        return 1
      }
      count=$((count+1))
    done
    echo "$(date -Is) $direction complete: $count/$expected folds"
  done
  mpus_state complete
}

case "${1:-plan}" in
  plan)
    echo "run=$mpus_run; seed=43; GPU0 B->D, GPU1 E->F, two queues in parallel"
    echo "A/C reused: $mpus_ac_root/{A,C}/"
    echo "Results: $mpus_root/{B,D,E,F}/"
    echo "Logs: $mpus_log_root/{B,D,E,F,suite}.log" ;;
  status)
    if test -s "$mpus_root/suite_state.json"; then cat "$mpus_root/suite_state.json"; else echo 'Not started'; fi
    for direction in B D E F; do
      count=0
      if test -d "$mpus_root/$direction"; then
        count=$(find "$mpus_root/$direction" -maxdepth 1 -name 'seed_43_subject_*.offline.json' | wc -l)
      fi
      echo "$direction $count"
    done ;;
  worker) mpus_worker ;;
  start)
    mkdir -p "$mpus_log_root" "$mpus_root"
    if test -s "$mpus_root/suite_state.json" && grep -q '"status":"running"' "$mpus_root/suite_state.json"; then
      echo 'Suite already marked running'; exit 1
    fi
    nohup bash "$mpus_parent/MPUS_GA/scripts/target_normalization_bdef_csu_commands.sh" worker \
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
  *) echo 'Usage: target_normalization_bdef_csu_commands.sh {plan|start|status|worker}' >&2; exit 2 ;;
esac
