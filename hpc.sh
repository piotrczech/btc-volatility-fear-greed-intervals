#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./hpc.sh setup serverai cpu|gpu
  ./hpc.sh setup wcss cpu|gpu
  ./hpc.sh smoke serverai cpu|gpu
  ./hpc.sh smoke wcss cpu|gpu
  ./hpc.sh submit serverai returns|classical|neural [--dry-run]
  ./hpc.sh submit wcss returns|classical|neural [--dry-run]
  ./hpc.sh logs
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

project_dir() {
  printf '%s\n' "${PROJECT_DIR:-$(pwd)}"
}

venv_path() {
  local cluster="$1"
  local device="$2"
  local project
  project="$(project_dir)"
  if [ -n "${VENV:-}" ]; then
    printf '%s\n' "$VENV"
  elif [ "$cluster" = "wcss" ]; then
    printf '%s\n' "$project/venvs/btc-vol-$device"
  else
    printf '%s\n' "$project/.venv-$device"
  fi
}

load_cluster_modules() {
  local cluster="$1"
  if [ "$cluster" = "wcss" ]; then
    # shellcheck disable=SC1091
    source /usr/local/sbin/modules.sh
    module purge
    module load Python/3.12.3-GCCcore-13.3.0
  fi
}

setup_env() {
  local cluster="$1"
  local device="$2"
  local project
  local venv
  project="$(project_dir)"
  venv="$(venv_path "$cluster" "$device")"
  mkdir -p "$(dirname "$venv")"
  cd "$project"
  load_cluster_modules "$cluster"

  if [ "$cluster" = "serverai" ]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "uv not found; installing uv to ~/.local/bin"
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install "${PYTHON_VERSION:-3.12}"
    uv venv "$venv" --python "${PYTHON_VERSION:-3.12}"
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    uv pip install --upgrade pip setuptools wheel
    uv pip install -r requirements-cpu.txt
    if [ "$device" = "gpu" ]; then
      uv pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    fi
  else
    python3 -m venv "$venv"
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    if [ "$device" = "gpu" ] && [ -n "${TORCH_INDEX_URL:-}" ]; then
      pip install -r requirements-cpu.txt
      pip install torch --index-url "$TORCH_INDEX_URL"
    else
      pip install -r "requirements-$device.txt"
    fi
  fi

  python make_grids.py
  python - <<'PY'
import sys
print("python:", sys.version)
import numpy, pandas, sklearn, scipy, statsmodels, optuna
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch not available:", exc)
PY
}

activate_env() {
  local cluster="$1"
  local device="$2"
  local project
  local venv
  project="$(project_dir)"
  venv="$(venv_path "$cluster" "$device")"
  cd "$project"
  load_cluster_modules "$cluster"
  if [ -f "$venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
  else
    echo "Venv not found at $venv; using current shell Python."
  fi
}

smoke_run() {
  local cluster="$1"
  local device="$2"
  local project
  local smoke_csv
  project="$(project_dir)"
  activate_env "$cluster" "$device"
  mkdir -p results/smoke
  if [ "${SMOKE_SYNTHETIC:-0}" = "1" ] || { [ -z "${SMOKE_CSV:-}" ] && [ ! -f btc.csv ]; }; then
    smoke_csv="${SMOKE_CSV:-results/smoke/btc_smoke.csv}"
    python -m src.data --synthetic-csv "$smoke_csv" --days "${SMOKE_DAYS:-520}"
  else
    smoke_csv="${SMOKE_CSV:-btc.csv}"
  fi
  echo "Smoke input: $smoke_csv"
  python make_grids.py --out-dir results/smoke/configs --seeds 333 --blocks price_only --vol-targets gk_future_7

  if [ "$device" = "gpu" ]; then
    python run_task.py \
      --mode neural --task-index "${TASK_INDEX:-0}" \
      --grid results/smoke/configs/grid_neural.csv \
      --csv "$smoke_csv" --out results/smoke \
      --seq-trials "${SEQ_TRIALS:-1}" --epochs-cap "${EPOCHS_CAP:-2}" \
      --device "${DEVICE:-cuda}" \
      --train-min-days 240 --calib-days 30 --test-days 30 --step-days 30 \
      --progress-every 1 --overwrite
  else
    python run_task.py \
      --mode returns --task-index 0 \
      --grid results/smoke/configs/grid_returns_baseline.csv \
      --csv "$smoke_csv" --out results/smoke \
      --models ridge --trials "${TRIALS:-1}" \
      --train-min-days 240 --calib-days 30 --test-days 30 --step-days 30 \
      --progress-every 1 --overwrite
    python run_task.py \
      --mode classical --task-index 0 \
      --grid results/smoke/configs/grid_classical.csv \
      --csv "$smoke_csv" --out results/smoke \
      --models ridge --trials "${TRIALS:-1}" \
      --train-min-days 240 --calib-days 30 --test-days 30 --step-days 30 \
      --progress-every 1 --overwrite
  fi
}

grid_for_kind() {
  case "$1" in
    returns) printf '%s\n' "${GRID:-configs/grid_returns_baseline.csv}" ;;
    classical) printf '%s\n' "${GRID:-configs/grid_classical.csv}" ;;
    neural) printf '%s\n' "${GRID:-configs/grid_neural.csv}" ;;
    *) die "Unknown task kind: $1" ;;
  esac
}

submit_array() {
  local cluster="$1"
  local kind="$2"
  local dry_run="${3:-}"
  local project
  local grid
  local n
  local last
  local concurrency
  local partition
  local account_arg=()
  local gres_arg=()
  local mem
  local time_limit
  local device
  local venv
  local job

  project="$(project_dir)"
  cd "$project"
  [ -f configs/grid_returns_baseline.csv ] || python make_grids.py
  grid="$(grid_for_kind "$kind")"
  [ -f "$grid" ] || die "Grid not found: $grid"
  n=$(( $(wc -l < "$grid") - 1 ))
  [ "$n" -gt 0 ] || die "Empty grid: $grid"
  last=$((n - 1))

  if [ "$kind" = "neural" ]; then
    device="gpu"
  else
    device="cpu"
  fi
  venv="$(venv_path "$cluster" "$device")"
  job="btc-$kind"

  if [ "$cluster" = "serverai" ]; then
    partition="${PARTITION:-serverai}"
    concurrency="${CONCURRENCY:-4}"
    mem="${MEM:-$([ "$kind" = "neural" ] && echo 48G || echo 32G)}"
    time_limit="${TIME:-24:00:00}"
  elif [ "$cluster" = "wcss" ]; then
    account_arg=(-A "${ACCOUNT:-hpc-piotrczech-1779870483}")
    if [ "$kind" = "neural" ]; then
      partition="${PARTITION:-lem-gpu-short}"
      concurrency="${CONCURRENCY:-8}"
    else
      partition="${PARTITION:-bem2-cpu-short}"
      concurrency="${CONCURRENCY:-24}"
    fi
    mem="${MEM:-32G}"
    time_limit="${TIME:-08:00:00}"
  else
    die "Unknown cluster: $cluster"
  fi

  if [ "$kind" = "neural" ]; then
    gres_arg=(--gres="${GRES:-gpu:1}")
  fi

  cmd=(
    sbatch
    -J "$job"
    "${account_arg[@]}"
    -p "$partition"
    -N 1
    -c "${CPUS:-8}"
    --mem "$mem"
    --time "$time_limit"
    --array "0-${last}%${concurrency}"
    "${gres_arg[@]}"
    --export "ALL,CLUSTER=$cluster,SLURM_MODE=$kind,GRID=$grid,CSV=${CSV:-btc.csv},OUT=${OUT:-results},PROJECT_DIR=$project,VENV=$venv,DEVICE=${DEVICE:-$([ "$kind" = "neural" ] && echo cuda || echo auto)},REGIME_MODE=${REGIME_MODE:-fixed},TRIALS=${TRIALS:-20},SEQ_TRIALS=${SEQ_TRIALS:-20},EPOCHS_CAP=${EPOCHS_CAP:-60},PROGRESS_EVERY=${PROGRESS_EVERY:-3}"
    slurm/array.sbatch
  )

  echo "Submitting $kind on $cluster: array 0-${last}%${concurrency}"
  if [ "$dry_run" = "--dry-run" ]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

log_note() {
  cat <<'EOF'
SLURM logs are cluster-managed. This project does not create a logs/ directory.
Use cluster defaults such as:
  squeue -u "$USER"
  sacct -j <jobid>
  tail -f slurm-<jobid>.out
EOF
}

cmd="${1:-}"
case "$cmd" in
  setup)
    [ "$#" -eq 3 ] || { usage; exit 1; }
    setup_env "$2" "$3"
    ;;
  smoke)
    [ "$#" -eq 3 ] || { usage; exit 1; }
    smoke_run "$2" "$3"
    ;;
  submit)
    [ "$#" -ge 3 ] || { usage; exit 1; }
    submit_array "$2" "$3" "${4:-}"
    ;;
  logs)
    [ "$#" -eq 1 ] || { usage; exit 1; }
    log_note
    ;;
  *)
    usage
    exit 1
    ;;
esac
