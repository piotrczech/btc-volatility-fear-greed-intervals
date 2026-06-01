#!/bin/bash
#SBATCH -N 1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV="${VENV:?VENV is required}"
SLURM_MODE="${SLURM_MODE:?SLURM_MODE is required}"
GRID="${GRID:?GRID is required}"
CSV="${CSV:-btc.csv}"
OUT="${OUT:-results}"
DEVICE="${DEVICE:-auto}"
REGIME_MODE="${REGIME_MODE:-fixed}"

if [ "${CLUSTER:-}" = "wcss" ]; then
  # shellcheck disable=SC1091
  source /usr/local/sbin/modules.sh
  module purge
  module load Python/3.12.3-GCCcore-13.3.0
fi

cd "$PROJECT_DIR"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python run_task.py \
  --mode "$SLURM_MODE" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --grid "$GRID" \
  --csv "$CSV" \
  --out "$OUT" \
  --trials "${TRIALS:-20}" \
  --seq-trials "${SEQ_TRIALS:-20}" \
  --epochs-cap "${EPOCHS_CAP:-60}" \
  --device "$DEVICE" \
  --regime-mode "$REGIME_MODE" \
  --fear-threshold "${FEAR_THRESHOLD:-33}" \
  --greed-threshold "${GREED_THRESHOLD:-67}" \
  --train-min-days "${TRAIN_MIN_DAYS:-900}" \
  --calib-days "${CALIB_DAYS:-60}" \
  --test-days "${TEST_DAYS:-60}" \
  --step-days "${STEP_DAYS:-60}" \
  --embargo-days "${EMBARGO_DAYS:--1}" \
  --min-regime-calib-n "${MIN_REGIME_CALIB_N:-20}" \
  ${HOLDOUT_START:+--holdout-start "$HOLDOUT_START"} \
  ${HOLDOUT_END:+--holdout-end "$HOLDOUT_END"} \
  --progress-every "${PROGRESS_EVERY:-3}"
