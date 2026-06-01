from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import DEFAULT_MIN_REGIME_CALIB_N

def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=float), eps, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    return float(np.mean(np.log(y_pred) + y_true / y_pred))


def qlike_vector(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y_true = np.clip(np.asarray(y_true, dtype=float), eps, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    return np.log(y_pred) + y_true / y_pred


def pinball_loss_vec(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else np.nan


def score_for_target(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> float:
    if target == "ret_future_1":
        return -float(mean_squared_error(y_true, y_pred))
    return -qlike_loss(y_true, np.clip(y_pred, 1e-8, None))



def prepare_quantile_outputs(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.column_stack([lower, median, upper]).astype(float)
    q = np.sort(q, axis=1)
    lower, median, upper = q[:, 0], q[:, 1], q[:, 2]

    if target != "ret_future_1":
        lower = np.clip(lower, 0.0, None)
        median = np.clip(median, 1e-8, None)
        upper = np.clip(upper, 0.0, None)
        upper = np.maximum(upper, lower)
    return lower, median, upper



def conformal_quantile_adjustment(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if mask.sum() == 0:
        return 0.0
    scores = np.maximum(lower[mask] - y_true[mask], y_true[mask] - upper[mask])
    scores = np.maximum(scores, 0.0)
    return float(np.quantile(scores, 1 - alpha, method="higher"))


def symmetric_conformal_radius(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return 0.0
    scores = np.abs(y_true[mask] - y_pred[mask])
    return float(np.quantile(scores, 1 - alpha, method="higher"))


def apply_regime_radius(
    calib_df: pd.DataFrame,
    score_col: str,
    alpha: float,
    min_n: int = DEFAULT_MIN_REGIME_CALIB_N,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    scores_global = calib_df[score_col].dropna().to_numpy(dtype=float)
    global_q = 0.0 if len(scores_global) == 0 else float(np.quantile(scores_global, 1 - alpha, method="higher"))

    for regime, grp in calib_df.groupby("fg_regime"):
        scores = grp[score_col].dropna().to_numpy(dtype=float)
        out[str(regime)] = global_q if len(scores) < min_n else float(np.quantile(scores, 1 - alpha, method="higher"))
    out["__global__"] = global_q
    return out


def point_loss_vector(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> np.ndarray:
    if target == "ret_future_1":
        return (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    return qlike_vector(y_true, y_pred)


def append_raw_rows(
    rows: List[Dict],
    test: pd.DataFrame,
    *,
    split_name: str,
    seed: int,
    target: str,
    block: str,
    model: str,
    method: str,
    interval_engine: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    calib_n: int,
    train_n: int,
    extra: Optional[Dict] = None,
) -> None:
    extra = extra or {}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    median = np.asarray(median, dtype=float)
    upper = np.asarray(upper, dtype=float)

    hit = (y_true >= lower) & (y_true <= upper)
    below = y_true < lower
    above = y_true > upper
    width = upper - lower
    loss = point_loss_vector(y_true, y_pred, target)
    interval_score = width.copy()
    interval_score = interval_score + (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    interval_score = interval_score + (2.0 / alpha) * np.maximum(y_true - upper, 0.0)

    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    pb_lower = pinball_loss_vec(y_true, lower, lower_q)
    pb_median = pinball_loss_vec(y_true, median, 0.5)
    pb_upper = pinball_loss_vec(y_true, upper, upper_q)

    for i, (_, rec) in enumerate(test.iterrows()):
        row = {
            "date": rec["date"],
            "split": split_name,
            "seed": int(seed),
            "target": target,
            "block": block,
            "model": model,
            "method": method,
            "interval_engine": interval_engine,
            "fg_regime": rec["fg_regime"],
            "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]),
            "lower": float(lower[i]),
            "median": float(median[i]),
            "upper": float(upper[i]),
            "hit": int(hit[i]),
            "below_lower": int(below[i]),
            "above_upper": int(above[i]),
            "width": float(width[i]),
            "interval_score": float(interval_score[i]),
            "point_loss": float(loss[i]),
            "pinball_lower": float(pb_lower[i]),
            "pinball_median": float(pb_median[i]),
            "pinball_upper": float(pb_upper[i]),
            "alpha": float(alpha),
            "nominal_coverage": float(1 - alpha),
            "train_n": int(train_n),
            "calib_n": int(calib_n),
        }
        row.update(extra)
        rows.append(row)


def summarize_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {}
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "qlike": np.nan,
        "hit_rate": np.nan,
    }
    if target == "ret_future_1":
        out["hit_rate"] = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    else:
        out["qlike"] = qlike_loss(y_true, np.clip(y_pred, 1e-8, None))
    return out

