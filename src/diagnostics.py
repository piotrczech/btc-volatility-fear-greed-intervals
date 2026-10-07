"""Evaluation of archived forecasts, with dates as the resampling unit."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .intervals import qlike_loss, summarize_point_metrics

POINT_KEY = ["target", "block", "model", "seed", "date"]
REGIMES = ["fear", "neutral", "greed"]


def select_point_predictions(df: pd.DataFrame, engine: str = "point_residual") -> pd.DataFrame:
    """Validate method copies, then select one per engine/seed/date.

    Quantile medians and point-residual predictions are separate families.
    """
    methods = (["sym_conformal", "regime_sym_conformal"] if engine == "point_residual"
               else ["raw_quantile", "cqr", "regime_cqr"])
    mask = df["method"].isin(methods)
    mask &= df["interval_engine"].eq(engine)
    chosen = df.loc[mask].copy()
    if chosen.empty:
        raise ValueError(f"No predictions for engine {engine}")
    if chosen.duplicated(POINT_KEY + ["method"]).any():
        raise ValueError("Duplicate point prediction within a method")
    consistency = chosen.groupby(POINT_KEY, observed=True)[["y_pred", "y_true", "fg_regime"]].nunique()
    if consistency.gt(1).any().any():
        raise ValueError("Interval methods disagree on their underlying forecasts")
    chosen["method_priority"] = chosen["method"].map(dict(zip(methods, range(len(methods)))))
    chosen = chosen.sort_values(POINT_KEY + ["method_priority"], kind="stable")
    return chosen.drop_duplicates(POINT_KEY).drop(columns="method_priority")


def point_evaluation(df: pd.DataFrame, engine: str = "point_residual") -> pd.DataFrame:
    rows = []
    for key, grp in select_point_predictions(df, engine).groupby(["target", "block", "model"], observed=True):
        y, pred = grp["y_true"].to_numpy(float), grp["y_pred"].to_numpy(float)
        metrics = summarize_point_metrics(y, pred, key[0])
        # A variance floor of 1e-12 equals a positive volatility floor of 1e-6
        # applied before squaring. Volatility QLIKE uses its own floor of 1e-12.
        metrics.pop("qlike", None)
        metrics.pop("qlike_scale", None)
        if key[0] != "ret_future_1":
            metrics["qlike_volatility"] = qlike_loss(y, pred, scale="volatility")
            metrics["qlike_variance"] = qlike_loss(y, pred, eps=1e-6, scale="variance")
        rows.append({**dict(zip(["target", "block", "model"], key)), **metrics,
                     "n_records": len(grp), "n_dates": grp["date"].nunique(),
                     "n_seeds": grp["seed"].nunique(), "interval_engine": engine,
                     "input_scale": "return" if key[0] == "ret_future_1" else "volatility",
                     "evaluation_scale": "return_squared_error" if key[0] == "ret_future_1" else "variance_qlike",
                     "tuning_scale": "return_squared_error" if key[0] == "ret_future_1" else "volatility_qlike",
                     "qlike_volatility_input_epsilon": 1e-12,
                     "qlike_variance_input_epsilon": 1e-6,
                     "qlike_variance_epsilon": 1e-12})
    return pd.DataFrame(rows)


def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n < 1 or block_len < 1:
        raise ValueError("Positive series and block lengths required")
    starts = rng.integers(0, n, size=math.ceil(n / block_len))
    return ((starts[:, None] + np.arange(block_len)[None, :]) % n).ravel()[:n]


def date_panel(grp: pd.DataFrame) -> dict:
    """Retain target, state, and full seed vectors in chronological order."""
    if grp.duplicated(["date", "seed"]).any():
        raise ValueError("Duplicate date/seed in bootstrap configuration")
    if grp.groupby("date")[["fg_regime", "y_true"]].nunique().gt(1).any().any():
        raise ValueError("Inconsistent date-level state/target")
    panel = {}
    for name in ["hit", "below_lower", "above_upper", "width"]:
        matrix = grp.pivot(index="date", columns="seed", values=name).sort_index().sort_index(axis=1)
        if matrix.isna().any().any():
            raise ValueError("Unequal bootstrap seed panel")
        panel[name] = matrix.to_numpy(float)
    dates = matrix.index
    origin = grp.drop_duplicates("date").set_index("date").loc[dates]
    panel.update(dates=dates, seeds=matrix.columns.to_numpy(),
                 regimes=origin["fg_regime"].to_numpy(), y_true=origin["y_true"].to_numpy(float),
                 nominal=float(grp["nominal_coverage"].iloc[0]))
    return panel


def origin_metrics(panel: dict, idx: np.ndarray) -> dict:
    # Every selected date brings ALL seed outcomes together. On a balanced
    # panel this equals the archived bootstrap of date-level seed means.
    hit = panel["hit"].mean(axis=1)[idx]
    regimes = panel["regimes"][idx]
    out = {"global_gap_pp": 100 * (hit.mean() - panel["nominal"]),
           "avg_width": panel["width"][idx].mean(),
           "miss_imbalance_pp": 100 * (panel["above_upper"][idx] - panel["below_lower"][idx]).mean()}
    gaps = []
    for regime in REGIMES:
        mask = regimes == regime
        gap = 100 * (hit[mask].mean() - panel["nominal"]) if mask.any() else np.nan
        out[f"{regime}_gap_pp"] = gap
        gaps.append(abs(gap))
    out["max_regime_gap_pp"] = np.nanmax(gaps)
    out["hidden_fragility_pp"] = out["max_regime_gap_pp"] - abs(out["global_gap_pp"])
    return out


def bootstrap_metrics(panel: dict, *, block_len: int, reps: int, rng: np.random.Generator) -> dict:
    point = origin_metrics(panel, np.arange(len(panel["dates"])))
    names = [k for k in point if k != "avg_width"]
    samples = {name: [] for name in names}
    for _ in range(reps):
        value = origin_metrics(panel, circular_block_indices(len(panel["dates"]), block_len, rng))
        for name in names:
            samples[name].append(value[name])
    out = {f"{name}_point": point[name] for name in names}
    for name in names:
        lo, hi = np.nanpercentile(samples[name], [2.5, 97.5])
        out.update({f"{name}_ci_low": lo, f"{name}_ci_high": hi})
    worst = max(REGIMES, key=lambda r: abs(point[f"{r}_gap_pp"]))
    out["worst_regime"] = worst
    for suffix in ["point", "ci_low", "ci_high"]:
        out[f"worst_regime_gap_pp_{suffix}"] = out[f"{worst}_gap_pp_{suffix}"]
    return out


def paired_bootstrap(a: dict, b: dict, *, block_len: int, reps: int,
                     rng: np.random.Generator) -> dict:
    for name in ["dates", "seeds", "regimes", "y_true", "nominal"]:
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"Unmatched paired bootstrap {name}")
    values = []
    for _ in range(reps):
        idx = circular_block_indices(len(a["dates"]), block_len, rng)
        left, right = origin_metrics(a, idx), origin_metrics(b, idx)
        values.append(right["max_regime_gap_pp"] - left["max_regime_gap_pp"])
    point_idx = np.arange(len(a["dates"]))
    point = origin_metrics(b, point_idx)["max_regime_gap_pp"] - origin_metrics(a, point_idx)["max_regime_gap_pp"]
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return {"delta_max_gap_pp": point, "ci_low": lo, "ci_high": hi}
