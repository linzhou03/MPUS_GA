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
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results/ablations}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/ablations}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RELIABILITY_REFERENCES="${RELIABILITY_REFERENCES:-0.10 0.20}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export MNE_DONTWRITE_HOME=true
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

on_error() {
  local exit_code="$?"
  echo "[$(timestamp)] FAILED with exit code ${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

echo "[$(timestamp)] R-SoftSGA v2 target-graph weight ablation ${RUN_ID}"
echo "[$(timestamp)] physical_gpu=${PHYSICAL_GPU}, seed=${RANDOM_SEED}, windows=1s/2s"

for reference in ${RELIABILITY_REFERENCES}; do
  reference_tag="${reference/./}"
  result_dir="${RESULT_ROOT}/reliability_ref_${reference_tag}"
  detail_log="${LOG_DIR}/${RUN_ID}_reliability_ref_${reference_tag}.log"

  echo "[$(timestamp)] START reliability_reference=${reference}"
  echo "[$(timestamp)] result_dir=${result_dir}"
  echo "[$(timestamp)] detail_log=${detail_log}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_transfer.py" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${result_dir}" \
    --variant r-softsga-v2 \
    --window-seconds 1 2 \
    --random-seeds "${RANDOM_SEED}" \
    --target-subjects all \
    --max-iters "${MAX_ITERS}" \
    --batch-size "${BATCH_SIZE}" \
    --reliability-reference "${reference}" \
    --device "${DEVICE}" \
    >"${detail_log}" 2>&1

  echo "[$(timestamp)] FINISH reliability_reference=${reference}"
done

echo "[$(timestamp)] ALL ABLATIONS FINISHED"
echo "[$(timestamp)] Results: ${RESULT_ROOT}"
