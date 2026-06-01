#!/usr/bin/env python3
from __future__ import annotations

import argparse

DEFAULT_ALPHA = 0.05
DEFAULT_TRAIN_MIN_DAYS = 900
DEFAULT_CALIB_DAYS = 60
DEFAULT_TEST_DAYS = 60
DEFAULT_STEP_DAYS = 60
DEFAULT_MIN_REGIME_CALIB_N = 20


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one BTC interval benchmark grid task.")
    p.add_argument("--mode", choices=["returns", "classical", "neural"], required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--grid", type=str, required=True)
    p.add_argument("--csv", type=str, default="btc.csv")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seq-trials", type=int, default=20)
    p.add_argument("--epochs-cap", type=int, default=60)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--models", type=str, default=None, help="Comma-separated classical models.")
    p.add_argument("--min-regime-calib-n", type=int, default=DEFAULT_MIN_REGIME_CALIB_N)
    p.add_argument("--train-min-days", type=int, default=DEFAULT_TRAIN_MIN_DAYS)
    p.add_argument("--calib-days", type=int, default=DEFAULT_CALIB_DAYS)
    p.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    p.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    p.add_argument("--embargo-days", type=int, default=-1, help="Use -1 for automatic target-horizon embargo.")
    p.add_argument("--holdout-start", type=str, default=None, help="Optional fixed test start date, e.g. 2026-01-01.")
    p.add_argument("--holdout-end", type=str, default=None, help="Optional fixed test end date, e.g. 2026-05-31.")
    p.add_argument("--regime-mode", choices=["fixed", "percentile"], default="fixed")
    p.add_argument("--fear-threshold", type=float, default=33.0)
    p.add_argument("--greed-threshold", type=float, default=67.0)
    p.add_argument("--seed-override", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    from src.runner import run_classical_task, run_neural_task

    if args.mode in ["returns", "classical"]:
        run_classical_task(args)
    else:
        run_neural_task(args)


if __name__ == "__main__":
    main()
