#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
SCRIPT_PATH="${SCRIPT_DIR}/run_bidirectional_full_ablation_suite.sh"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/gzw/miniforge3/envs/BCI/bin/python ]]; then
    PYTHON_BIN=/home/gzw/miniforge3/envs/BCI/bin/python
  elif [[ -x /home/gzw/anaconda3/envs/BCI/bin/python ]]; then
    PYTHON_BIN=/home/gzw/anaconda3/envs/BCI/bin/python
  else
    PYTHON_BIN=python
  fi
fi

RANDOM_SEEDS="${RANDOM_SEEDS:-42 43 44}"
TARGET_SUBJECTS="${TARGET_SUBJECTS:-all}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-24}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-16}"
DATA_DIR="${DATA_DIR:-${PACKAGE_DIR}/data_processed}"
RESULT_ROOT="${RESULT_ROOT:-${PACKAGE_DIR}/results_bidirectional_full_ablation}"
LOG_DIR="${LOG_DIR:-${PACKAGE_DIR}/logs/bidirectional_full_ablation}"
SUITE_LOG="${LOG_DIR}/suite.log"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-3}"

EXPERIMENTS=(
  A0 B0 A1 B1 A2 B2 A3 B3
  A4 B4 A5 B5 A6 B6 A_main B_main
)

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"
if [[ "${SUITE_DETACHED:-0}" != "1" ]]; then
  : > "${SUITE_LOG}"
  for experiment in "${EXPERIMENTS[@]}"; do
    mkdir -p "${RESULT_ROOT}/${experiment}"
    : > "${LOG_DIR}/${experiment}.log"
  done
  nohup env \
    SUITE_DETACHED=1 \
    PACKAGE_DIR="${PACKAGE_DIR}" \
    PROJECT_DIR="${PROJECT_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    RANDOM_SEEDS="${RANDOM_SEEDS}" \
    TARGET_SUBJECTS="${TARGET_SUBJECTS}" \
    SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE}" \
    TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE}" \
    DATA_DIR="${DATA_DIR}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    START_DELAY_SECONDS="${START_DELAY_SECONDS}" \
    OVERWRITE="${OVERWRITE:-0}" \
    bash "${SCRIPT_PATH}" >> "${SUITE_LOG}" 2>&1 < /dev/null &
  echo "Suite started with nohup; this SSH session may now be closed."
  echo "Suite log: ${SUITE_LOG}"
  echo "Results: ${RESULT_ROOT}/{A0,B0,...,A6,B6,A_main,B_main}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "$(date '+%F %T') SUITE END status=failed reason=python_unavailable"
  exit 2
fi
if (( BASH_VERSINFO[0] < 5 )); then
  echo "$(date '+%F %T') SUITE END status=failed reason=bash_5_required"
  exit 2
fi

cd "${PROJECT_DIR}" || exit 2
read -r -a SEED_ARGS <<< "${RANDOM_SEEDS}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

declare -A PID_TO_GPU=()
declare -A PID_TO_EXPERIMENT=()
next_index=0
active_jobs=0
failures=0

launch_next() {
  local physical_gpu="$1"
  if (( next_index >= ${#EXPERIMENTS[@]} )); then
    return 1
  fi
  local experiment="${EXPERIMENTS[next_index]}"
  local experiment_log="${LOG_DIR}/${experiment}.log"
  next_index=$((next_index + 1))
  local -a command=(
    "${PYTHON_BIN}"
    -m MPUS_GA.trial_temporal.train
    --experiment "${experiment}"
    --data-dir "${DATA_DIR}"
    --result-root "${RESULT_ROOT}"
    --random-seeds "${SEED_ARGS[@]}"
    --target-subjects "${TARGET_SUBJECTS}"
    --source-batch-size "${SOURCE_BATCH_SIZE}"
    --target-batch-size "${TARGET_BATCH_SIZE}"
    --evaluation-protocol fixed_final
    --device cuda:0
    "${OVERWRITE_ARGS[@]}"
  )

  echo "$(date '+%F %T') START experiment=${experiment} physical_gpu=${physical_gpu}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" > "${experiment_log}" 2>&1 < /dev/null &
  local child_process=$!
  PID_TO_GPU["${child_process}"]="${physical_gpu}"
  PID_TO_EXPERIMENT["${child_process}"]="${experiment}"
  active_jobs=$((active_jobs + 1))
}

echo "$(date '+%F %T') SUITE START experiments=${#EXPERIMENTS[@]}"
launch_next 0
launch_next 1

while (( active_jobs > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid "${!PID_TO_GPU[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  physical_gpu="${PID_TO_GPU[${finished_pid}]}"
  experiment="${PID_TO_EXPERIMENT[${finished_pid}]}"
  echo "$(date '+%F %T') END experiment=${experiment} physical_gpu=${physical_gpu}"
  unset 'PID_TO_GPU['"${finished_pid}"']'
  unset 'PID_TO_EXPERIMENT['"${finished_pid}"']'
  active_jobs=$((active_jobs - 1))
  if (( exit_code != 0 )); then
    failures=$((failures + 1))
  fi
  if (( next_index < ${#EXPERIMENTS[@]} )); then
    sleep "${START_DELAY_SECONDS}"
    launch_next "${physical_gpu}"
  fi
done

if (( failures > 0 )); then
  echo "$(date '+%F %T') SUITE END status=failed failures=${failures}"
  exit 1
fi
echo "$(date '+%F %T') SUITE END status=completed failures=0"
