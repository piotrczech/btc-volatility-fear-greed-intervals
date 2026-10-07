#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


CONFIG_KEY = ["target", "block", "model", "method"]
RECORD_KEY = CONFIG_KEY + ["seed", "date"]


def validate_panel(df: pd.DataFrame, expected_seeds=None) -> pd.DataFrame:
    """Reject duplicates and unequal panels; seeds are repetitions of dates."""
    from src.data import SEEDS

    expected = set(SEEDS if expected_seeds is None else expected_seeds)
    required = RECORD_KEY + ["fg_regime", "y_true", "y_pred", "interval_engine", "nominal_coverage"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if not len(df) or df[required].isna().any().any():
        raise ValueError("Empty panel or null prediction keys/values")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df.duplicated(RECORD_KEY).any():
        bad = df.loc[df.duplicated(RECORD_KEY, keep=False), RECORD_KEY].head()
        raise ValueError(f"Duplicate prediction records:\n{bad.to_string(index=False)}")
    if not np.isfinite(df[["y_true", "y_pred", "nominal_coverage"]].to_numpy(float)).all():
        raise ValueError("Nonfinite prediction values")
    # The realized target and state belong to a date, not to a model or seed.
    common = df.groupby(["target", "date"], observed=True)[["fg_regime", "y_true"]].nunique()
    if common.gt(1).any().any():
        raise ValueError("Conflicting states or target realizations on common dates")
    reference_dates = {}
    for key, grp in df.groupby(CONFIG_KEY, observed=True, sort=True):
        found = set(grp["seed"].unique())
        if found != expected:
            raise ValueError(f"Incomplete seeds for {key}: expected {sorted(expected)}, found {sorted(found)}")
        counts = grp.groupby("date", observed=True)["seed"].nunique()
        if not counts.eq(len(expected)).all():
            raise ValueError(f"Unequal date/seed panel for {key}")
        dates = frozenset(counts.index)
        target = key[0]
        if target in reference_dates and dates != reference_dates[target]:
            raise ValueError(f"Unequal configuration dates for {key}")
        reference_dates[target] = dates
        if grp["nominal_coverage"].nunique() != 1 or grp["interval_engine"].nunique() != 1:
            raise ValueError(f"Mixed nominal coverage or interval engines for {key}")
    return df.sort_values(RECORD_KEY, kind="stable").reset_index(drop=True)


def read_prediction_files(raw_dir: Path) -> pd.DataFrame:
    artifact_suffixes = (
        "__tuning.csv",
        "__point_metrics.csv",
        "__model_diagnostics.csv",
        "__training_history.csv",
    )
    files = [
        path for path in sorted(raw_dir.glob("**/*.csv"))
        if not path.name.endswith(artifact_suffixes)
    ]
    if not files:
        raise FileNotFoundError(f"No raw prediction CSV files found under {raw_dir}")
    frames = []
    for path in files:
        tmp = pd.read_csv(path, parse_dates=["date"])
        tmp["source_file"] = str(path)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def summarize_group(grp: pd.DataFrame) -> pd.Series:
    nominal = float(grp["nominal_coverage"].iloc[0]) if "nominal_coverage" in grp.columns else 0.95

    coverage = float(grp["hit"].mean())
    return pd.Series({
        "n": int(len(grp)),  # Historical alias for n_records.
        "n_records": int(len(grp)),
        "n_dates": int(grp["date"].nunique()),
        "n_seeds": int(grp["seed"].nunique()),
        "coverage": coverage,
        "signed_coverage_error": coverage - nominal,
        "abs_coverage_gap": abs(coverage - nominal),
        "below_lower_rate": float(grp["below_lower"].mean()),
        "above_upper_rate": float(grp["above_upper"].mean()),
        "lower_miss_rate": float(grp["below_lower"].mean()),
        "upper_miss_rate": float(grp["above_upper"].mean()),
        "avg_width": float(grp["width"].mean()),
        "median_width": float(grp["width"].median()),
        "interval_score": float(grp["interval_score"].mean()) if "interval_score" in grp.columns else np.nan,
        "pinball_lower": float(grp["pinball_lower"].mean()),
        "pinball_median": float(grp["pinball_median"].mean()),
        "pinball_upper": float(grp["pinball_upper"].mean()),
        "point_loss": float(grp["point_loss"].mean()),
        "nominal_coverage": nominal,
        "point_loss_scale": "return_squared_error" if str(grp["target"].iloc[0]) == "ret_future_1" else "volatility_qlike",
        "inference": "descriptive_only",
    })


def summarize_panel(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    # Explicit iteration keeps key columns available with both pandas 2 and 3.
    return pd.DataFrame([
        {**dict(zip(keys, key)), **summarize_group(grp).to_dict()}
        for key, grp in df.groupby(keys, observed=True, dropna=False, sort=True)
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate raw HPC interval prediction CSVs.")
    p.add_argument("--raw-dir", default="results/raw_predictions")
    p.add_argument("--out-dir", default="results/merged")
    p.add_argument("--predictions", type=Path, help="Read an archived merged CSV instead of raw files.")
    p.add_argument("--expected-seeds", nargs="+", type=int, default=[111, 222, 333, 444, 555, 666])
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (pd.read_csv(args.predictions, parse_dates=["date"]) if args.predictions
          else read_prediction_files(raw_dir))
    df = validate_panel(df, args.expected_seeds)
    output_predictions = out_dir / "interval_predictions_all.csv"
    if not args.predictions or args.predictions.resolve() != output_predictions.resolve():
        df.to_csv(output_predictions, index=False)

    key = ["target", "block", "model", "method", "seed"]
    seed_summary = summarize_panel(df, key)
    seed_summary.to_csv(out_dir / "interval_summary_by_seed.csv", index=False)

    global_key = ["target", "block", "model", "method"]
    global_summary = summarize_panel(df, global_key)
    global_summary = global_summary.sort_values(["target", "abs_coverage_gap", "avg_width"])
    global_summary.to_csv(out_dir / "interval_summary_global.csv", index=False)

    regime_key = ["target", "block", "model", "method", "fg_regime"]
    regime_summary = summarize_panel(df, regime_key)
    regime_summary = regime_summary.sort_values(["target", "block", "model", "method", "fg_regime"])
    regime_summary.to_csv(out_dir / "interval_summary_by_regime.csv", index=False)

    # Compare signed coverage errors across states.
    pivot = regime_summary.pivot_table(
        index=["target", "block", "model", "method"],
        columns="fg_regime",
        values="signed_coverage_error",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for col in ["fear", "neutral", "greed"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["max_abs_regime_gap"] = pivot[["fear", "neutral", "greed"]].abs().max(axis=1)
    pivot["fear_undercoverage"] = pivot["fear"].clip(upper=0)
    pivot = pivot.sort_values(["target", "max_abs_regime_gap"])
    pivot.to_csv(out_dir / "regime_signed_gap_pivot.csv", index=False)

    ranking = global_summary.merge(
        pivot[["target", "block", "model", "method", "fear", "neutral", "greed", "max_abs_regime_gap"]],
        on=["target", "block", "model", "method"],
        how="left",
    )
    ranking["global_undercoverage_penalty"] = (-ranking["signed_coverage_error"]).clip(lower=0)
    ranking["fear_undercoverage_penalty"] = (-ranking["fear"]).clip(lower=0)
    ranking["undercoverage_penalty"] = ranking[[
        "global_undercoverage_penalty",
        "fear_undercoverage_penalty",
    ]].max(axis=1)
    ranking = ranking.sort_values([
        "target",
        "undercoverage_penalty",
        "max_abs_regime_gap",
        "interval_score",
        "avg_width",
    ])
    ranking.to_csv(out_dir / "candidate_ranking.csv", index=False)

    print(f"Wrote merged predictions and summaries to {out_dir}")
    print(f"Raw rows: {len(df)}")


if __name__ == "__main__":
    main()
