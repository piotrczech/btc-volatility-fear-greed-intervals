#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./hpc.sh setup serverai cpu|gpu [--dry-run]
  ./hpc.sh setup wcss cpu|gpu [--dry-run]
  ./hpc.sh smoke serverai cpu|gpu [--dry-run]
  ./hpc.sh smoke wcss cpu|gpu [--dry-run]
  ./hpc.sh submit serverai returns|classical|neural [--dry-run]
  ./hpc.sh submit wcss returns|classical|neural [--dry-run]
  ./hpc.sh aggregate serverai|wcss [--dry-run]
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

validate_cluster() {
  case "$1" in
    serverai|wcss) ;;
    *) die "Unknown cluster: $1" ;;
  esac
}

validate_device() {
  case "$1" in
    cpu|gpu) ;;
    *) die "Unknown device: $1" ;;
  esac
}

parse_dry_run_arg() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  if [ "$#" -eq 1 ] && [ "$1" = "--dry-run" ]; then
    printf '%s\n' "$1"
    return 0
  fi
  if [ "$#" -eq 1 ]; then
    die "Unknown option: $1"
  fi
  usage
  exit 1
}

in_slurm_context() {
  [ -n "${HPC_BATCH_EXEC:-}" ] || [ -n "${SLURM_JOB_ID:-}" ] || [ -n "${SLURM_JOBID:-}" ]
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

aggregate_run() {
  local cluster="$1"
  local results_root
  local raw_dir
  local merged_dir

  results_root="${OUT:-results}"
  raw_dir="${RAW_DIR:-$results_root/raw_predictions}"
  merged_dir="${MERGED_DIR:-$results_root/merged}"

  activate_env "$cluster" cpu
  python aggregate_results.py --raw-dir "$raw_dir" --out-dir "$merged_dir"
}

submit_single_job() {
  local action="$1"
  local cluster="$2"
  local device="$3"
  local dry_run="${4:-}"
  local project
  local script
  local partition
  local account_arg=()
  local gres_arg=()
  local cpus
  local mem
  local time_limit
  local job
  local cmd=()

  project="$(project_dir)"
  script="$project/hpc.sh"
  cd "$project"

  if [ "$cluster" = "serverai" ]; then
    partition="${PARTITION:-serverai}"
    if [ "$device" = "gpu" ]; then
      cpus="${CPUS:-8}"
      mem="${MEM:-24G}"
      if [ "$action" = "setup" ]; then
        time_limit="${TIME:-00:30:00}"
      else
        time_limit="${TIME:-00:30:00}"
      fi
    else
      cpus="${CPUS:-4}"
      mem="${MEM:-8G}"
      if [ "$action" = "setup" ]; then
        time_limit="${TIME:-01:00:00}"
      else
        time_limit="${TIME:-00:30:00}"
      fi
    fi
  elif [ "$cluster" = "wcss" ]; then
    account_arg=(-A "${ACCOUNT:-hpc-piotrczech-1779870483}")
    cpus="${CPUS:-8}"
    if [ "$device" = "gpu" ]; then
      partition="${PARTITION:-lem-gpu-short}"
    else
      partition="${PARTITION:-bem2-cpu-short}"
    fi
    mem="${MEM:-32G}"
    time_limit="${TIME:-08:00:00}"
  else
    die "Unknown cluster: $cluster"
  fi

  if [ "$device" = "gpu" ]; then
    gres_arg=(--gres="${GRES:-gpu:1}")
  fi

  job="btc-$action-$device"
  cmd=(
    sbatch
    -J "$job"
    "${account_arg[@]}"
    -p "$partition"
    -N 1
    -c "$cpus"
    --mem "$mem"
    --time "$time_limit"
    "${gres_arg[@]}"
    --export "ALL,HPC_BATCH_EXEC=1,PROJECT_DIR=$project"
    "$script" "$action" "$cluster" "$device"
  )

  echo "Submitting $action on $cluster/$device"
  if [ "$dry_run" = "--dry-run" ]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

submit_aggregate_job() {
  local cluster="$1"
  local dry_run="${2:-}"
  local project
  local script
  local partition
  local account_arg=()
  local cpus
  local mem
  local time_limit
  local cmd=()

  project="$(project_dir)"
  script="$project/hpc.sh"
  cd "$project"

  if [ "$cluster" = "serverai" ]; then
    partition="${PARTITION:-serverai}"
  elif [ "$cluster" = "wcss" ]; then
    account_arg=(-A "${ACCOUNT:-hpc-piotrczech-1779870483}")
    partition="${PARTITION:-bem2-cpu-short}"
  else
    die "Unknown cluster: $cluster"
  fi

  cpus="${CPUS:-2}"
  mem="${MEM:-8G}"
  time_limit="${TIME:-00:30:00}"

  cmd=(
    sbatch
    -J btc-aggregate
    "${account_arg[@]}"
    -p "$partition"
    -N 1
    -c "$cpus"
    --mem "$mem"
    --time "$time_limit"
    --export "ALL,HPC_BATCH_EXEC=1,PROJECT_DIR=$project"
    "$script" aggregate "$cluster"
  )

  echo "Submitting aggregate on $cluster"
  if [ "$dry_run" = "--dry-run" ]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

run_setup_or_smoke() {
  local action="$1"
  local cluster="$2"
  local device="$3"
  local dry_run="${4:-}"

  if [ "$dry_run" = "--dry-run" ] || ! in_slurm_context; then
    submit_single_job "$action" "$cluster" "$device" "$dry_run"
    return
  fi

  case "$action" in
    setup) setup_env "$cluster" "$device" ;;
    smoke) smoke_run "$cluster" "$device" ;;
    *) die "Unknown action: $action" ;;
  esac
}

run_aggregate() {
  local cluster="$1"
  local dry_run="${2:-}"

  if [ "$dry_run" = "--dry-run" ] || ! in_slurm_context; then
    submit_aggregate_job "$cluster" "$dry_run"
    return
  fi

  aggregate_run "$cluster"
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
  local cmd=()

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
    [ "$#" -ge 3 ] || { usage; exit 1; }
    dry_run="$(parse_dry_run_arg "${@:4}")"
    validate_cluster "$2"
    validate_device "$3"
    run_setup_or_smoke setup "$2" "$3" "$dry_run"
    ;;
  smoke)
    [ "$#" -ge 3 ] || { usage; exit 1; }
    dry_run="$(parse_dry_run_arg "${@:4}")"
    validate_cluster "$2"
    validate_device "$3"
    run_setup_or_smoke smoke "$2" "$3" "$dry_run"
    ;;
  submit)
    [ "$#" -ge 3 ] || { usage; exit 1; }
    dry_run="$(parse_dry_run_arg "${@:4}")"
    validate_cluster "$2"
    submit_array "$2" "$3" "$dry_run"
    ;;
  aggregate)
    [ "$#" -ge 2 ] || { usage; exit 1; }
    dry_run="$(parse_dry_run_arg "${@:3}")"
    validate_cluster "$2"
    run_aggregate "$2" "$dry_run"
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
