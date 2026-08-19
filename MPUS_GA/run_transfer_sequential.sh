#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gzw/anaconda3/envs/BCI/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_ITERS="${MAX_ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data_processed}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/training}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

BASELINE_LOG="${LOG_DIR}/${RUN_ID}_caga-balanced.log"
RSOFT_LOG="${LOG_DIR}/${RUN_ID}_r-softsga-v2.log"
VALIDATION_LOG="${LOG_DIR}/${RUN_ID}_data_validation.log"

export PYTHONUNBUFFERED=1
export MNE_DONTWRITE_HOME=true
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

check_file_count() {
  local dataset="$1"
  local scale="$2"
  local expected="$3"
  local directory="${DATA_DIR}/${dataset}/window_${scale}s"
  local observed

  if [[ ! -d "${directory}" ]]; then
    echo "[$(timestamp)] ERROR: missing directory ${directory}" >&2
    exit 1
  fi
  observed="$(find "${directory}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "[$(timestamp)] ERROR: ${dataset} ${scale}s has ${observed}/${expected} files" >&2
    exit 1
  fi
  echo "[$(timestamp)] OK: ${dataset} ${scale}s has ${observed}/${expected} files"
}

run_variant() {
  local variant="$1"
  local log_file="$2"

  echo "[$(timestamp)] START ${variant}; detailed log: ${log_file}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_transfer.py" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${RESULT_DIR}" \
    --variant "${variant}" \
    --window-seconds 1 2 4 \
    --random-seeds "${RANDOM_SEED}" \
    --target-subjects all \
    --max-iters "${MAX_ITERS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    >"${log_file}" 2>&1
  echo "[$(timestamp)] FINISH ${variant}"
}

on_error() {
  local exit_code="$?"
  echo "[$(timestamp)] FAILED with exit code ${exit_code}." >&2
  echo "[$(timestamp)] Check ${BASELINE_LOG} and ${RSOFT_LOG}." >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

echo "[$(timestamp)] MPUS-GA sequential transfer run ${RUN_ID}"
echo "[$(timestamp)] physical_gpu=${PHYSICAL_GPU}, visible_device=${DEVICE}, random_seed=${RANDOM_SEED}, max_iters=${MAX_ITERS}, batch_size=${BATCH_SIZE}"

for scale in 1 2 4; do
  check_file_count seed_vii "${scale}" 80
  check_file_count seed_v "${scale}" 48
done

echo "[$(timestamp)] Running processed-data validation; log: ${VALIDATION_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_processed.py" \
  --data-dir "${DATA_DIR}" \
  >"${VALIDATION_LOG}" 2>&1
echo "[$(timestamp)] Data validation passed"

run_variant caga-balanced "${BASELINE_LOG}"
run_variant r-softsga-v2 "${RSOFT_LOG}"

echo "[$(timestamp)] ALL EXPERIMENTS FINISHED"
echo "[$(timestamp)] Results: ${RESULT_DIR}"
