#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
SCRIPT_PATH="${SCRIPT_DIR}/run_main_A0_A7_suite.sh"

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
RESULT_ROOT="${RESULT_ROOT:-${PACKAGE_DIR}/results_class_conditional_multiscale}"
LOG_DIR="${LOG_DIR:-${PACKAGE_DIR}/logs/class_conditional_multiscale}"
SUITE_LOG="${LOG_DIR}/suite.log"

EXPERIMENTS=(A0 A1 A2 A3 A4 A5 A6 A7 main)
GPU0_QUEUE=(A0 A2 A4 A6 main)
GPU1_QUEUE=(A1 A3 A5 A7)

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"
for experiment in "${EXPERIMENTS[@]}"; do
  mkdir -p "${RESULT_ROOT}/${experiment}"
  : > "${LOG_DIR}/${experiment}.log"
done

if [[ "${SUITE_DETACHED:-0}" != "1" ]]; then
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
    OVERWRITE="${OVERWRITE:-0}" \
    bash "${SCRIPT_PATH}" > "${SUITE_LOG}" 2>&1 < /dev/null &
  echo "Suite started with nohup; this SSH session may now be closed."
  echo "Suite log: ${SUITE_LOG}"
  echo "Results: ${RESULT_ROOT}/{A0,A1,A2,A3,A4,A5,A6,A7,main}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "$(date '+%F %T') Python is unavailable: ${PYTHON_BIN}"
  exit 2
fi

cd "${PROJECT_DIR}"
read -r -a SEED_ARGS <<< "${RANDOM_SEEDS}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

run_experiment() {
  local physical_gpu="$1"
  local experiment="$2"
  local experiment_log="${LOG_DIR}/${experiment}.log"
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
    --device cuda:0
    "${OVERWRITE_ARGS[@]}"
  )

  echo "$(date '+%F %T') START experiment=${experiment} physical_gpu=${physical_gpu} log=${experiment_log}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" > "${experiment_log}" 2>&1 < /dev/null &
  local child_process=$!
  wait "${child_process}"
  local exit_code=$?
  echo "$(date '+%F %T') END experiment=${experiment} physical_gpu=${physical_gpu} exit_code=${exit_code}"
  return "${exit_code}"
}

run_queue() {
  local physical_gpu="$1"
  shift
  local failures=0
  local experiment
  for experiment in "$@"; do
    if ! run_experiment "${physical_gpu}" "${experiment}"; then
      failures=1
    fi
  done
  return "${failures}"
}

echo "$(date '+%F %T') SUITE START"
echo "gpu0_queue=${GPU0_QUEUE[*]}"
echo "gpu1_queue=${GPU1_QUEUE[*]}"
echo "random_seeds=${RANDOM_SEEDS} target_subjects=${TARGET_SUBJECTS}"
echo "result_root=${RESULT_ROOT}"

run_queue 0 "${GPU0_QUEUE[@]}" &
gpu0_worker=$!
run_queue 1 "${GPU1_QUEUE[@]}" &
gpu1_worker=$!

wait "${gpu0_worker}"
gpu0_status=$?
wait "${gpu1_worker}"
gpu1_status=$?

if [[ "${gpu0_status}" -ne 0 || "${gpu1_status}" -ne 0 ]]; then
  echo "$(date '+%F %T') SUITE END status=failed gpu0=${gpu0_status} gpu1=${gpu1_status}"
  exit 1
fi
echo "$(date '+%F %T') SUITE END status=completed"
