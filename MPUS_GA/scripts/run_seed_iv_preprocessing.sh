#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/gzw/anaconda3/envs/BCI/bin/python ]]; then
    PYTHON_BIN=/home/gzw/anaconda3/envs/BCI/bin/python
  elif [[ -x /home/gzw/miniforge3/envs/BCI/bin/python ]]; then
    PYTHON_BIN=/home/gzw/miniforge3/envs/BCI/bin/python
  else
    PYTHON_BIN=python
  fi
fi
DATA_ROOT="${DATA_ROOT:-/dataset/gzw/seed_series}"
OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_DIR}/data_processed}"
SUBJECTS="${SUBJECTS:-all}"

cd "${PROJECT_DIR}"
export MNE_DONTWRITE_HOME=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "$(date '+%F %T') starting SEED-IV preprocessing"
echo "data_root=${DATA_ROOT} output_dir=${OUTPUT_DIR} subjects=${SUBJECTS}"
"${PYTHON_BIN}" -m MPUS_GA.preprocessing.preprocess \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --datasets seed-iv \
  --subjects "${SUBJECTS}" \
  --window-seconds 1 2 4

echo "$(date '+%F %T') preprocessing completed; validating all artifacts"
"${PYTHON_BIN}" -m MPUS_GA.preprocessing.validate_processed \
  --data-dir "${OUTPUT_DIR}"
echo "$(date '+%F %T') SEED-IV preprocessing and validation completed"
