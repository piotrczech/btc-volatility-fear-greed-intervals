from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

TARGETS = ["rv_future_7", "gk_future_7", "rs_future_7"]
VOL_TARGETS = ["rv_future_7", "gk_future_7", "rs_future_7"]
RETURNS_TARGET = "ret_future_1"
BLOCKS = ["price_only", "price_onchain", "full"]
SEEDS = [111, 222, 333, 444, 555, 666]

DEFAULT_ALPHA = 0.05
DEFAULT_TRAIN_MIN_DAYS = 900
DEFAULT_CALIB_DAYS = 60
DEFAULT_TEST_DAYS = 60
DEFAULT_STEP_DAYS = 60
DEFAULT_MIN_REGIME_CALIB_N = 20

def rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


def future_window_sum(x: pd.Series, horizon: int) -> pd.Series:
    return x.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def infer_target_horizon_days(target: str) -> int:
    """Infer forecast horizon from names such as ret_future_1 or gk_future_7."""
    try:
        return int(str(target).split("_future_")[-1])
    except Exception:
        return 1


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV misses required columns: " + ", ".join(missing)
        )


def load_df(csv_path: Path, regime_mode: str = "fixed", fear_threshold: float = 33.0, greed_threshold: float = 67.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    require_columns(df, [
        "date", "open", "high", "low", "close", "volume",
        "avg-block-size", "n-transactions-per-block", "n-payments-per-block",
        "transactions-per-second", "blocks-size", "hash-rate", "difficulty",
        "usa-inflation-monthly", "eu-inflation", "tweets-volume",
        "wikipedia-reads", "fear-and-greed",
    ])

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    price_cols = ["open", "close", "high", "low"]
    metric_cols = [
        "volume",
        "avg-block-size",
        "n-transactions-per-block",
        "n-payments-per-block",
        "transactions-per-second",
        "blocks-size",
        "hash-rate",
        "difficulty",
        "usa-inflation-monthly",
        "eu-inflation",
        "tweets-volume",
        "wikipedia-reads",
        "fear-and-greed",
    ]

    df[price_cols + metric_cols] = df[price_cols + metric_cols].astype(float)

    df["log_close"] = np.log(df["close"])
    df["ret_1"] = df["log_close"].diff()

    for lag in [2, 5, 7, 14]:
        df[f"ret_{lag}"] = df["log_close"].diff(lag)

    df["abs_ret_1"] = df["ret_1"].abs()

    hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
    hc = np.log(df["high"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    ho = np.log(df["high"] / df["open"]).replace([np.inf, -np.inf], np.nan)
    lc = np.log(df["low"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    lo = np.log(df["low"] / df["open"]).replace([np.inf, -np.inf], np.nan)

    df["park_vol_1"] = np.sqrt((hl ** 2) / (4 * np.log(2)))

    gk_core = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
    rs_core = hc * ho + lc * lo

    df["gk_vol_1"] = np.sqrt(np.clip(gk_core, 0, None))
    df["rs_vol_1"] = np.sqrt(np.clip(rs_core, 0, None))

    horizon = 7
    df["ret_future_1"] = df["ret_1"].shift(-1)
    df["rv_future_7"] = np.sqrt(future_window_sum(df["ret_1"] ** 2, horizon))
    df["gk_future_7"] = np.sqrt(future_window_sum(df["gk_vol_1"] ** 2, horizon))
    df["rs_future_7"] = np.sqrt(future_window_sum(df["rs_vol_1"] ** 2, horizon))

    core_cols = [
        "gk_vol_1",
        "rs_vol_1",
        "park_vol_1",
        "abs_ret_1",
        "volume",
        "hash-rate",
        "difficulty",
    ]

    for col in core_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_avg7"] = df[col].rolling(7).mean()
        df[f"{col}_avg30"] = df[col].rolling(30).mean()

    aux_cols = [
        "volume",
        "avg-block-size",
        "n-transactions-per-block",
        "n-payments-per-block",
        "transactions-per-second",
        "blocks-size",
        "hash-rate",
        "difficulty",
        "tweets-volume",
        "wikipedia-reads",
        "fear-and-greed",
    ]

    for col in aux_cols:
        s = df[col].replace([np.inf, -np.inf], np.nan)
        # Some attention/sentiment variables may contain zero. Use log1p on
        # non-negative series to avoid brittle failures while keeping the
        # transformation monotone. Historical branch selection examines the
        # full snapshot, including future rows. Freeze this choice in advance
        # or on training data in future experiments; preserve it for this run.
        if (s > 0).all():
            df[f"{col}_logdiff1"] = np.log(s).diff()
        elif (s >= 0).all():
            df[f"{col}_logdiff1"] = np.log1p(s).diff()
        else:
            df[f"{col}_logdiff1"] = s.diff()
        df[f"{col}_z30"] = rolling_zscore(s, 30)
        df[f"{col}_avg7"] = s.rolling(7).mean()
        df[f"{col}_avg30"] = s.rolling(30).mean()

    fg = df["fear-and-greed"]
    if regime_mode == "fixed":
        # Pre-declared thresholds on the published 0-100 index scale.
        # This avoids using the future test distribution to define regimes.
        q_low = float(fear_threshold)
        q_high = float(greed_threshold)
    elif regime_mode == "percentile":
        # Robustness mode: define regimes by full-sample 33/67 percentiles.
        # The main experiment uses fixed thresholds on the published 0-100 scale.
        q_low = float(fg.quantile(0.33))
        q_high = float(fg.quantile(0.67))
    else:
        raise ValueError(f"Unknown regime_mode: {regime_mode}")

    df.attrs["regime_mode"] = regime_mode
    df.attrs["fear_threshold"] = q_low
    df.attrs["greed_threshold"] = q_high

    df["fear_low"] = (fg <= q_low).astype(int)
    df["greed_high"] = (fg >= q_high).astype(int)
    df["fg_regime"] = np.where(
        df["fear_low"] == 1,
        "fear",
        np.where(df["greed_high"] == 1, "greed", "neutral"),
    )

    interaction_cols = [
        "tweets-volume_z30",
        "wikipedia-reads_z30",
        "fear-and-greed_z30",
        "volume_z30",
        "hash-rate_z30",
        "difficulty_z30",
        "transactions-per-second_z30",
        "n-payments-per-block_z30",
        "n-transactions-per-block_z30",
        "blocks-size_z30",
    ]

    for col in interaction_cols:
        if col in df.columns:
            df[f"{col}_x_fear"] = df[col] * df["fear_low"]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def pick_existing(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def define_feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    price_only = pick_existing(df, [
        "ret_1", "ret_2", "ret_5", "ret_7", "ret_14", "abs_ret_1",
        "park_vol_1", "gk_vol_1", "rs_vol_1",
        "gk_vol_1_lag1", "gk_vol_1_avg7", "gk_vol_1_avg30",
        "rs_vol_1_lag1", "rs_vol_1_avg7", "rs_vol_1_avg30",
        "park_vol_1_lag1", "park_vol_1_avg7", "park_vol_1_avg30",
        "abs_ret_1_lag1", "abs_ret_1_avg7", "abs_ret_1_avg30",
        "volume_logdiff1", "volume_z30", "volume_avg7", "volume_avg30",
    ])

    onchain = pick_existing(df, [
        "avg-block-size_logdiff1", "avg-block-size_z30",
        "n-transactions-per-block_logdiff1", "n-transactions-per-block_z30",
        "n-payments-per-block_logdiff1", "n-payments-per-block_z30",
        "transactions-per-second_logdiff1", "transactions-per-second_z30",
        "blocks-size_logdiff1", "blocks-size_z30",
        "hash-rate_logdiff1", "hash-rate_z30", "hash-rate_avg7", "hash-rate_avg30",
        "difficulty_logdiff1", "difficulty_z30", "difficulty_avg7", "difficulty_avg30",
    ])

    attention_macro = pick_existing(df, [
        "tweets-volume_logdiff1", "tweets-volume_z30", "tweets-volume_avg7", "tweets-volume_avg30",
        "wikipedia-reads_logdiff1", "wikipedia-reads_z30", "wikipedia-reads_avg7", "wikipedia-reads_avg30",
        "fear-and-greed_z30", "fear-and-greed_avg7", "fear-and-greed_avg30",
        "usa-inflation-monthly", "eu-inflation", "fear_low", "greed_high",
    ])
    attention_macro += [c for c in df.columns if c.endswith("_x_fear")]
    attention_macro = list(dict.fromkeys(attention_macro))

    return {
        "price_only": price_only,
        "price_onchain": list(dict.fromkeys(price_only + onchain)),
        "full": list(dict.fromkeys(price_only + onchain + attention_macro)),
    }


@dataclass
class SplitDef:
    name: str
    train_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    df: pd.DataFrame,
    train_min_days: int = DEFAULT_TRAIN_MIN_DAYS,
    calib_days: int = DEFAULT_CALIB_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    embargo_days: int = 0,
) -> List[SplitDef]:
    dates = pd.to_datetime(df["date"])
    start_date = dates.min()
    min_train_end = start_date + pd.Timedelta(days=train_min_days)

    splits: List[SplitDef] = []
    split_no = 1
    calib_start = min_train_end + pd.Timedelta(days=1)

    while True:
        calib_end = calib_start + pd.Timedelta(days=calib_days - 1)
        # Purged walk-forward: for h-day-ahead targets, the last h forecast origins
        # before the calibration/test boundary are skipped so labels do not overlap
        # the next block.
        test_start = calib_end + pd.Timedelta(days=embargo_days + 1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_end > dates.max():
            break

        train_end = calib_start - pd.Timedelta(days=embargo_days + 1)
        train_idx = df.index[dates <= train_end].to_numpy()
        calib_idx = df.index[(dates >= calib_start) & (dates <= calib_end)].to_numpy()
        test_idx = df.index[(dates >= test_start) & (dates <= test_end)].to_numpy()

        if len(train_idx) >= 200 and len(calib_idx) >= 20 and len(test_idx) >= 20:
            splits.append(SplitDef(
                name=f"split_{split_no:02d}_{test_start.date()}_{test_end.date()}",
                train_idx=train_idx,
                calib_idx=calib_idx,
                test_idx=test_idx,
                test_start=test_start,
                test_end=test_end,
            ))
            split_no += 1

        calib_start += pd.Timedelta(days=step_days)

    if not splits:
        raise RuntimeError("No walk-forward splits were created. Check date range and window sizes.")
    return splits


def build_holdout_split(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    calib_days: int = DEFAULT_CALIB_DAYS,
    embargo_days: int = 0,
) -> List[SplitDef]:
    dates = pd.to_datetime(df["date"])
    holdout_start = pd.Timestamp(test_start)
    holdout_end = pd.Timestamp(test_end)
    if holdout_end < holdout_start:
        raise ValueError("holdout test_end must be on or after test_start")

    calib_end = holdout_start - pd.Timedelta(days=embargo_days + 1)
    calib_start = calib_end - pd.Timedelta(days=calib_days - 1)
    train_end = calib_start - pd.Timedelta(days=embargo_days + 1)

    train_idx = df.index[dates <= train_end].to_numpy()
    calib_idx = df.index[(dates >= calib_start) & (dates <= calib_end)].to_numpy()
    test_idx = df.index[(dates >= holdout_start) & (dates <= holdout_end)].to_numpy()

    if len(train_idx) < 200 or len(calib_idx) < 20 or len(test_idx) < 20:
        raise RuntimeError(
            "Holdout split is too small. Check date range, calibration days and embargo."
        )

    return [SplitDef(
        name=f"holdout_{holdout_start.date()}_{holdout_end.date()}",
        train_idx=train_idx,
        calib_idx=calib_idx,
        test_idx=test_idx,
        test_start=holdout_start,
        test_end=holdout_end,
    )]




def make_synthetic_btc_df(days: int = 520, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=int(days), freq="D")
    ret = rng.normal(0.0005, 0.035, size=len(dates))
    close = 20000 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0.0, 0.01, size=len(dates)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.015, size=len(dates))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.015, size=len(dates))))
    volume = rng.lognormal(16, 0.35, size=len(dates))
    fear = np.clip(50 - 450 * ret + rng.normal(0, 18, size=len(dates)), 0, 100)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "avg-block-size": rng.normal(1.4, 0.08, size=len(dates)).clip(0.8, None),
        "n-transactions-per-block": rng.normal(2400, 180, size=len(dates)).clip(100, None),
        "n-payments-per-block": rng.normal(5200, 500, size=len(dates)).clip(100, None),
        "transactions-per-second": rng.normal(4.2, 0.4, size=len(dates)).clip(0.1, None),
        "blocks-size": np.linspace(300000, 500000, len(dates)) + rng.normal(0, 5000, size=len(dates)),
        "hash-rate": np.linspace(1.2e8, 2.6e8, len(dates)) + rng.normal(0, 5e6, size=len(dates)),
        "difficulty": np.linspace(1.4e13, 4.8e13, len(dates)) + rng.normal(0, 8e11, size=len(dates)),
        "usa-inflation-monthly": rng.normal(0.25, 0.15, size=len(dates)),
        "eu-inflation": rng.normal(0.22, 0.14, size=len(dates)),
        "tweets-volume": rng.poisson(120000, size=len(dates)),
        "wikipedia-reads": rng.poisson(45000, size=len(dates)),
        "fear-and-greed": fear,
    })


def write_synthetic_btc_csv(path: Path, days: int = 520, seed: int = 123) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_btc_df(days=days, seed=seed).to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Data helpers for smoke tests.")
    p.add_argument("--synthetic-csv", type=Path, required=True)
    p.add_argument("--days", type=int, default=520)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    write_synthetic_btc_csv(args.synthetic_csv, days=args.days, seed=args.seed)
    print(f"Wrote synthetic BTC CSV to {args.synthetic_csv}")


if __name__ == "__main__":
    main()
