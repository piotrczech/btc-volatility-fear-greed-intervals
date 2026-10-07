# BTC Volatility Interval Benchmark

HPC-ready experiments for a paper on Bitcoin prediction intervals under Fear/Neutral/Greed regimes. The core claim is narrow: globally calibrated intervals can still fail inside market regimes, especially for short-horizon volatility risk.

## Experiments

Baseline:

- `ret_future_1`
- `price_only`
- HAR/Ridge/RF/GBR
- `sym_conformal`, `regime_sym_conformal`

Main volatility benchmark:

- targets: `rv_future_7`, `gk_future_7`, `rs_future_7`
- emphasis: `gk_future_7`, `rs_future_7`
- feature blocks: `price_only`, `price_onchain`, `full`
- models: HAR/Ridge/RF/GBR/LSTM/GRU
- intervals: `raw_quantile`, `sym_conformal`, `regime_sym_conformal`, `cqr`, `regime_cqr`
- evaluation: purged walk-forward with horizon-aware embargo
- regimes: fixed thresholds by default, `fear <= 33`, `neutral 34-66`, `greed >= 67`

Use percentile regimes only as robustness: `REGIME_MODE=percentile`.

## Project Layout

```text
run_task.py              # one grid task: returns/classical/neural
make_grids.py            # creates configs/*.csv
aggregate_results.py     # merges predictions and summary tables
hpc.sh                   # setup/smoke/submit helper
slurm/array.sbatch       # single SLURM worker
src/                     # data, models, intervals, runner
configs/                 # grid CSVs
data/btc_sample.csv      # 40-line public schema/sample file
results/merged/          # selected public aggregate summaries
```

Heavy generated outputs go to `results/` or `results_*` and are ignored by git. The repository keeps only selected aggregate CSV summaries under `results/merged/`.

## Input

The full `btc.csv` input dataset is not distributed in this repository because it combines third-party time series that may have redistribution limits. Keep the full file private and place it in the project root when running experiments locally or on HPC. The root `btc.csv` path is ignored by git.

The repository includes `data/btc_sample.csv`, a 40-line sample with the expected schema. It is intended only for schema inspection and lightweight loader checks; it is not sufficient to reproduce the paper results.

Required columns:

```text
date, open, high, low, close, volume,
avg-block-size, n-transactions-per-block, n-payments-per-block,
transactions-per-second, blocks-size, hash-rate, difficulty,
usa-inflation-monthly, eu-inflation, tweets-volume,
wikipedia-reads, fear-and-greed
```

Targets, OHLC volatility estimators, lagged/rolling features and regime labels are created internally.

## Data Availability

The code, experiment grids, small aggregate summaries, and a minimal input sample are public. Full reproduction requires reconstructing the private `btc.csv` file from the original data providers using the schema above, then running the benchmark commands with `--csv btc.csv`.

The committed aggregate summaries are:

```text
results/merged/interval_summary_global.csv
results/merged/interval_summary_by_seed.csv
results/merged/interval_summary_by_regime.csv
results/merged/regime_signed_gap_pivot.csv
results/merged/candidate_ranking.csv
```

The full merged prediction table, raw predictions, smoke outputs, robustness outputs, notebooks, and the private root `btc.csv` are intentionally excluded from git.

## Run

Generate grids:

```bash
python make_grids.py
```

Task counts:

```text
returns:    6
classical: 54
neural:   108
```

Local single-task examples:

```bash
python run_task.py --mode returns --task-index 0 --grid configs/grid_returns_baseline.csv --csv btc.csv --out results
python run_task.py --mode classical --task-index 0 --grid configs/grid_classical.csv --csv btc.csv --out results
```

HPC setup and smoke submit one-off SLURM jobs when run outside an existing SLURM job/allocation. Smoke uses `btc.csv` when present; set `SMOKE_SYNTHETIC=1` only for an environment-only test without real data.

```bash
./hpc.sh setup serverai cpu
./hpc.sh smoke serverai cpu

./hpc.sh setup serverai gpu
./hpc.sh smoke serverai gpu
```

Observed `serverai` GPU layout from `sinfo -o "%P %l %D %c %G"`:

```text
PARTITION TIMELIMIT NODES CPUS GRES
serverai* infinite 1 64 gpu:gpu0:2(S:0),gpu:gpu1:2(S:1)
```

GPU commands default to `GRES=${GRES:-gpu:1}`. If a real submission fails because the cluster requires typed GRES, retry with `gpu:gpu0:1` or `gpu:gpu1:1`:

```bash
GRES=gpu:gpu0:1 ./hpc.sh setup serverai gpu
GRES=gpu:gpu0:1 ./hpc.sh smoke serverai gpu
GRES=gpu:gpu0:1 ./hpc.sh submit serverai neural
GRES=gpu:gpu1:1 ./hpc.sh submit serverai neural
```

Submit arrays:

```bash
./hpc.sh submit serverai returns
./hpc.sh submit serverai classical
./hpc.sh submit serverai neural

./hpc.sh submit wcss returns
./hpc.sh submit wcss classical
./hpc.sh submit wcss neural
```

Useful variants:

```bash
CONCURRENCY=4 ./hpc.sh submit serverai classical
REGIME_MODE=percentile OUT=results_percentile ./hpc.sh submit serverai classical
HOLDOUT_START=2026-01-01 HOLDOUT_END=2026-05-31 OUT=results_holdout ./hpc.sh submit serverai classical
./hpc.sh setup serverai gpu --dry-run
./hpc.sh smoke serverai gpu --dry-run
./hpc.sh submit serverai neural --dry-run
./hpc.sh submit serverai classical --dry-run
```

If those dry-runs show `--gres=gpu:1` but a real GPU submission is rejected, verify the typed request before resubmitting:

```bash
GRES=gpu:gpu0:1 ./hpc.sh setup serverai gpu --dry-run
GRES=gpu:gpu0:1 ./hpc.sh smoke serverai gpu --dry-run
GRES=gpu:gpu0:1 ./hpc.sh submit serverai neural --dry-run
```

SLURM logs are left to the cluster defaults. The project does not create a persistent `logs/` directory.

## Outputs

Raw predictions:

```text
results/raw_predictions/returns/*.csv
results/raw_predictions/classical/*.csv
results/raw_predictions/neural/*.csv
```

Per-task artifacts:

```text
*.json
*__tuning.csv
*__point_metrics.csv
*__model_diagnostics.csv
*__training_history.csv
```

Aggregate:

```bash
./hpc.sh aggregate serverai
./hpc.sh aggregate serverai --dry-run
```

Aggregation runs as a short CPU SLURM job outside existing SLURM jobs/allocations. By default it reads `${OUT:-results}/raw_predictions` and writes `${OUT:-results}/merged`; override exact paths with `RAW_DIR=...` and `MERGED_DIR=...`.

Main summaries:

```text
interval_summary_global.csv
interval_summary_by_seed.csv
interval_summary_by_regime.csv
regime_signed_gap_pivot.csv
candidate_ranking.csv
```

Key metrics: coverage, signed coverage error, upper/lower miss rates, average width, interval score, pinball losses and regime instability. SHAP and deeper interpretation should be run posthoc for selected configurations only.

## Evaluate saved predictions

Rebuild descriptive summaries without retraining:

```bash
python aggregate_results.py --predictions results/merged/interval_predictions_all.csv
```

Aggregation rejects duplicate configuration/seed/date records and unequal seed
panels. Summaries include `n_records`, `n_dates` and `n_seeds`; `n` is an alias
for `n_records`. Seeds share realized dates, so pooled summaries are descriptive
and omit classical calibration p-values.

`src.diagnostics` provides point evaluation by forecast engine and a date-block
bootstrap that keeps seeds together and pairs methods on the same sampled dates.
Point evaluation reports both volatility- and variance-scale QLIKE. Variance
QLIKE clips volatility at `1e-6` before squaring, preserving a variance floor of
`1e-12`. Quantile medians remain separate from point-residual forecasts.
Optuna's volatility-scale objective is unchanged. Full evaluation requires saved
predictions; the public data sample does not reproduce the complete benchmark.
