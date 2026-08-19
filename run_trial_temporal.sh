#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gzw/projects/CAGA-SGA}"
PYTHON_BIN="${PYTHON_BIN:-/home/gzw/anaconda3/envs/BCI/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
WINDOW_SECONDS="${WINDOW_SECONDS:-1}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 43 44}"
TARGET_SUBJECTS="${TARGET_SUBJECTS:-all}"
MAX_ITERS="${MAX_ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/MPUS_GA/results_trial_temporal}"

read -r -a WINDOW_ARGS <<< "${WINDOW_SECONDS}"
read -r -a SEED_ARGS <<< "${RANDOM_SEEDS}"

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export MNE_DONTWRITE_HOME=true

echo "Trial temporal run: physical_gpu=${PHYSICAL_GPU} windows=${WINDOW_SECONDS} seeds=${RANDOM_SEEDS} targets=${TARGET_SUBJECTS}"
"${PYTHON_BIN}" MPUS_GA/train_trial_temporal.py \
  --window-seconds "${WINDOW_ARGS[@]}" \
  --random-seeds "${SEED_ARGS[@]}" \
  --target-subjects "${TARGET_SUBJECTS}" \
  --max-iters "${MAX_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --result-dir "${RESULT_DIR}" \
  --device cuda:0
