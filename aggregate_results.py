#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_uc_test(hit: np.ndarray, nominal: float):
    hit = np.asarray(hit, dtype=int)
    n = len(hit)
    if n < 30:
        return np.nan, np.nan
    n1 = int(hit.sum())
    n0 = n - n1
    p = np.clip(nominal, 1e-8, 1 - 1e-8)
    phat = np.clip(n1 / n, 1e-8, 1 - 1e-8)
    ll0 = n0 * np.log(1 - p) + n1 * np.log(p)
    ll1 = n0 * np.log(1 - phat) + n1 * np.log(phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence_test(hit: np.ndarray):
    hit = np.asarray(hit, dtype=int)
    if len(hit) < 30:
        return np.nan, np.nan
    x = hit[:-1]
    y = hit[1:]
    n00 = int(np.sum((x == 0) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n11 = int(np.sum((x == 1) & (y == 1)))
    denom0 = max(n00 + n01, 1)
    denom1 = max(n10 + n11, 1)
    pi01 = np.clip(n01 / denom0, 1e-8, 1 - 1e-8)
    pi11 = np.clip(n11 / denom1, 1e-8, 1 - 1e-8)
    pi1 = np.clip((n01 + n11) / max(n00 + n01 + n10 + n11, 1), 1e-8, 1 - 1e-8)
    ll_iid = ((n00 + n10) * np.log(1 - pi1)) + ((n01 + n11) * np.log(pi1))
    ll_markov = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    lr = -2 * (ll_iid - ll_markov)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


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
    hit = grp["hit"].astype(int).to_numpy()
    uc_lr, uc_p = kupiec_uc_test(hit, nominal=nominal)
    ind_lr, ind_p = christoffersen_independence_test(hit)

    coverage = float(grp["hit"].mean())
    return pd.Series({
        "n": int(len(grp)),
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
        "uc_lr": uc_lr,
        "uc_p": uc_p,
        "ind_lr": ind_lr,
        "ind_p": ind_p,
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate raw HPC interval prediction CSVs.")
    p.add_argument("--raw-dir", default="results/raw_predictions")
    p.add_argument("--out-dir", default="results/merged")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_prediction_files(raw_dir)
    df.to_csv(out_dir / "interval_predictions_all.csv", index=False)

    key = ["target", "block", "model", "method", "seed"]
    seed_summary = df.groupby(key, dropna=False).apply(summarize_group).reset_index()
    seed_summary.to_csv(out_dir / "interval_summary_by_seed.csv", index=False)

    global_key = ["target", "block", "model", "method"]
    global_summary = df.groupby(global_key, dropna=False).apply(summarize_group).reset_index()
    global_summary = global_summary.sort_values(["target", "abs_coverage_gap", "avg_width"])
    global_summary.to_csv(out_dir / "interval_summary_global.csv", index=False)

    regime_key = ["target", "block", "model", "method", "fg_regime"]
    regime_summary = df.groupby(regime_key, dropna=False).apply(summarize_group).reset_index()
    regime_summary = regime_summary.sort_values(["target", "block", "model", "method", "fg_regime"])
    regime_summary.to_csv(out_dir / "interval_summary_by_regime.csv", index=False)

    # Coverage pivot helpful for article inspection.
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
