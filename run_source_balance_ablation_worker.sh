#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gzw/anaconda3/envs/BCI/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
SOURCE_BALANCE_ALPHA="${SOURCE_BALANCE_ALPHA:?Set SOURCE_BALANCE_ALPHA before running}"
VARIANT="${VARIANT:-caga-balanced}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 43 44}"
MAX_ITERS="${MAX_ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data_processed}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/sampler_ablation}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

alpha_tag="${SOURCE_BALANCE_ALPHA/./}"
seed_tag="${RANDOM_SEEDS// /_}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/sampler_ablation/alpha_${alpha_tag}}"
DETAIL_LOG="${LOG_DIR}/${RUN_ID}_seeds_${seed_tag}_alpha_${alpha_tag}.log"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

export PYTHONUNBUFFERED=1
export MNE_DONTWRITE_HOME=true
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

on_error() {
  local exit_code="$?"
  echo "[$(timestamp)] alpha=${SOURCE_BALANCE_ALPHA} FAILED with exit code ${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

echo "[$(timestamp)] START source sampler ablation"
echo "[$(timestamp)] variant=${VARIANT}, alpha=${SOURCE_BALANCE_ALPHA}, seeds=${RANDOM_SEEDS}, physical_gpu=${PHYSICAL_GPU}"
echo "[$(timestamp)] result_dir=${RESULT_DIR}"
echo "[$(timestamp)] detail_log=${DETAIL_LOG}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_transfer.py" \
  --data-dir "${DATA_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --variant "${VARIANT}" \
  --window-seconds 1 2 \
  --random-seeds ${RANDOM_SEEDS} \
  --target-subjects all \
  --max-iters "${MAX_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --source-balance-alpha "${SOURCE_BALANCE_ALPHA}" \
  --device "${DEVICE}" \
  >"${DETAIL_LOG}" 2>&1

echo "[$(timestamp)] alpha=${SOURCE_BALANCE_ALPHA} ALL EXPERIMENTS FINISHED"
