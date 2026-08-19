#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gzw/anaconda3/envs/BCI/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
RANDOM_SEED="${RANDOM_SEED:?Set RANDOM_SEED before running this worker}"
MAX_ITERS="${MAX_ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data_processed}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/multiseed}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RELIABILITY_REFERENCES="${RELIABILITY_REFERENCES:-0.05 0.10 0.20}"

mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export MNE_DONTWRITE_HOME=true
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

on_error() {
  local exit_code="$?"
  echo "[$(timestamp)] seed=${RANDOM_SEED} FAILED with exit code ${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

echo "[$(timestamp)] R-SoftSGA v2 multiseed worker"
echo "[$(timestamp)] seed=${RANDOM_SEED}, physical_gpu=${PHYSICAL_GPU}, windows=1s/2s"

for reference in ${RELIABILITY_REFERENCES}; do
  reference_tag="${reference/./}"
  case "${reference}" in
    0.05)
      result_dir="${SCRIPT_DIR}/results"
      ;;
    0.10|0.20)
      result_dir="${SCRIPT_DIR}/results/ablations/reliability_ref_${reference_tag}"
      ;;
    *)
      echo "[$(timestamp)] Unsupported reliability_reference=${reference}" >&2
      exit 2
      ;;
  esac

  detail_log="${LOG_DIR}/${RUN_ID}_seed${RANDOM_SEED}_reliability_ref_${reference_tag}.log"
  mkdir -p "${result_dir}"

  echo "[$(timestamp)] START seed=${RANDOM_SEED}, reliability_reference=${reference}"
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

  echo "[$(timestamp)] FINISH seed=${RANDOM_SEED}, reliability_reference=${reference}"
done

echo "[$(timestamp)] seed=${RANDOM_SEED} ALL EXPERIMENTS FINISHED"
