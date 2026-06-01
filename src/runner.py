from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import (
    RETURNS_TARGET,
    VOL_TARGETS,
    define_feature_blocks,
    build_holdout_split,
    build_walk_forward_splits,
    infer_target_horizon_days,
    load_df,
)
from .intervals import (
    append_raw_rows,
    apply_regime_radius,
    conformal_quantile_adjustment,
    prepare_quantile_outputs,
    summarize_point_metrics,
    symmetric_conformal_radius,
)
from .models import (
    CLASSICAL_MODELS,
    SEQUENCE_MODELS,
    SeqRegressor,
    build_classical_model,
    build_sequence_sets,
    extract_model_diagnostics,
    fit_classical_quantile_bundle,
    predict_classical_quantile_bundle,
    predict_seq_model,
    predict_seq_quantiles,
    resolve_device,
    set_global_seed,
    torch,
    train_seq_model,
    train_seq_quantile_model,
    tune_classical_model,
    tune_sequence_model,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_list_arg(value: Optional[str], default: List[str]) -> List[str]:
    if not value:
        return default
    return [x.strip() for x in str(value).split(",") if x.strip()]


def output_file_for_task(out_dir: Path, mode: str, grid_row: pd.Series) -> Path:
    target = str(grid_row["target"])
    block = str(grid_row["block"])
    seed = int(grid_row["seed"])
    if mode in ["returns", "classical"]:
        name = f"target={target}__block={block}__seed={seed}.csv"
        return out_dir / "raw_predictions" / mode / name
    model = str(grid_row["model"])
    name = f"target={target}__block={block}__model={model}__seed={seed}.csv"
    return out_dir / "raw_predictions" / "neural" / name


def artifact_path(raw_path: Path, label: str, suffix: str = ".csv") -> Path:
    return raw_path.with_name(f"{raw_path.stem}__{label}{suffix}")


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return str(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default))


def load_task(grid_path: Path, task_index: int) -> pd.Series:
    grid = pd.read_csv(grid_path)
    if task_index < 0 or task_index >= len(grid):
        raise IndexError(f"task_index={task_index} outside grid length={len(grid)}")
    return grid.iloc[int(task_index)]


def make_splits(args: argparse.Namespace, df: pd.DataFrame, embargo_days: int):
    if args.holdout_start or args.holdout_end:
        if not args.holdout_start or not args.holdout_end:
            raise ValueError("--holdout-start and --holdout-end must be provided together.")
        return build_holdout_split(
            df,
            test_start=args.holdout_start,
            test_end=args.holdout_end,
            calib_days=args.calib_days,
            embargo_days=embargo_days,
        )
    return build_walk_forward_splits(
        df,
        train_min_days=args.train_min_days,
        calib_days=args.calib_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=embargo_days,
    )


def run_classical_task(args: argparse.Namespace) -> None:
    set_global_seed(args.seed_override if args.seed_override is not None else 333)

    grid_row = load_task(Path(args.grid), args.task_index)
    target = str(grid_row["target"])
    block_name = str(grid_row["block"])
    seed = int(grid_row["seed"])
    mode_name = "returns" if args.mode == "returns" else "classical"

    if mode_name == "returns":
        if target != RETURNS_TARGET or block_name != "price_only":
            raise ValueError("returns mode is only for ret_future_1 with price_only features.")
    elif target == RETURNS_TARGET:
        raise ValueError("ret_future_1 belongs to the lightweight returns grid, not the main volatility grid.")

    set_global_seed(seed)

    out_path = output_file_for_task(Path(args.out), mode_name, grid_row)
    if out_path.exists() and not args.overwrite:
        log(f"[skip] output exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    df = load_df(Path(args.csv), regime_mode=args.regime_mode, fear_threshold=args.fear_threshold, greed_threshold=args.greed_threshold)
    blocks = define_feature_blocks(df)
    if block_name not in blocks:
        raise ValueError(f"Unknown block {block_name}. Available: {list(blocks)}")
    cols = blocks[block_name]
    if not cols:
        raise ValueError(f"No feature columns for block {block_name}")

    embargo_days = infer_target_horizon_days(target) if args.embargo_days < 0 else int(args.embargo_days)
    splits = make_splits(args, df, embargo_days)

    models = parse_list_arg(args.models, CLASSICAL_MODELS)
    quantile_intervals = mode_name == "classical" and target in VOL_TARGETS
    alpha = float(args.alpha)
    rows: List[Dict] = []
    params_by_model: Dict[str, Dict] = {}
    tuning_rows: List[Dict] = []
    point_metric_rows: List[Dict] = []
    diagnostic_rows: List[Dict] = []

    log(
        f"[task] mode={mode_name} task={args.task_index} target={target} block={block_name} "
        f"seed={seed} models={','.join(models)} splits={len(splits)} features={len(cols)} "
        f"embargo_days={embargo_days} regime={args.regime_mode} thresholds=({df.attrs.get('fear_threshold')},{df.attrs.get('greed_threshold')})"
    )

    base = df.loc[:, ["date", "fg_regime"] + cols + [target]].copy()
    first_split = splits[0]
    train0 = base.iloc[first_split.train_idx].dropna(subset=[target])
    calib0 = base.iloc[first_split.calib_idx].dropna(subset=[target])

    for model_name in models:
        if model_name not in CLASSICAL_MODELS:
            raise ValueError(f"Unsupported classical model: {model_name}")

        log(f"[{mode_name}] tuning {target} | {block_name} | {model_name} | seed={seed}")
        params, trial_rows = tune_classical_model(
            model_name=model_name,
            X_train=train0[cols],
            y_train=train0[target],
            X_val=calib0[cols],
            y_val=calib0[target],
            target=target,
            trials=args.trials,
            seed=seed,
        )
        params_by_model[model_name] = params
        for trial in trial_rows:
            row = {
                "target": target,
                "block": block_name,
                "model": model_name,
                "seed": seed,
            }
            row.update(trial)
            tuning_rows.append(row)

        for split_no, split in enumerate(splits):
            if split_no == 0 or (split_no + 1) % max(1, args.progress_every) == 0 or (split_no + 1) == len(splits):
                log(f"[progress] {mode_name} {target}|{block_name}|{model_name}|seed={seed} split {split_no+1}/{len(splits)}")
            train = base.iloc[split.train_idx].dropna(subset=[target])
            calib = base.iloc[split.calib_idx].dropna(subset=[target])
            test = base.iloc[split.test_idx].dropna(subset=[target])
            if len(train) < 120 or len(calib) < 20 or len(test) < 20:
                continue

            # Point model and symmetric conformal intervals.
            mdl = build_classical_model(model_name, params, seed + split_no)
            mdl.fit(train[cols], train[target])
            pred_cal = mdl.predict(calib[cols])
            pred_te = mdl.predict(test[cols])

            if target != "ret_future_1":
                pred_cal = np.clip(pred_cal, 1e-8, None)
                pred_te = np.clip(pred_te, 1e-8, None)

            y_cal = calib[target].to_numpy(dtype=float)
            y_te = test[target].to_numpy(dtype=float)
            point_metrics = summarize_point_metrics(y_te, pred_te, target)
            if point_metrics:
                point_metric_rows.append({
                    "target": target,
                    "block": block_name,
                    "model": model_name,
                    "seed": seed,
                    "split": split.name,
                    "split_no": split_no,
                    **point_metrics,
                })

            for diag in extract_model_diagnostics(mdl, model_name, cols):
                diagnostic_rows.append({
                    "target": target,
                    "block": block_name,
                    "model": model_name,
                    "seed": seed,
                    "split": split.name,
                    "split_no": split_no,
                    **diag,
                })

            radius = symmetric_conformal_radius(y_cal, pred_cal, alpha)
            low = pred_te - radius
            high = pred_te + radius
            if target != "ret_future_1":
                low = np.clip(low, 0.0, None)
            append_raw_rows(
                rows, test,
                split_name=split.name, seed=seed, target=target, block=block_name,
                model=model_name, method="sym_conformal", interval_engine="point_residual",
                y_true=y_te, y_pred=pred_te, lower=low, median=pred_te, upper=high,
                alpha=alpha, calib_n=len(calib), train_n=len(train),
                extra={"split_no": split_no, "global_radius": radius},
            )

            calib_scores = pd.DataFrame({
                "fg_regime": calib["fg_regime"].to_numpy(),
                "score": np.abs(y_cal - pred_cal),
            })
            radius_map = apply_regime_radius(
                calib_scores, "score", alpha=alpha, min_n=args.min_regime_calib_n,
            )
            rad = np.array([radius_map.get(r, radius_map["__global__"]) for r in test["fg_regime"].to_numpy()])
            low_r = pred_te - rad
            high_r = pred_te + rad
            if target != "ret_future_1":
                low_r = np.clip(low_r, 0.0, None)
            append_raw_rows(
                rows, test,
                split_name=split.name, seed=seed, target=target, block=block_name,
                model=model_name, method="regime_sym_conformal", interval_engine="point_residual",
                y_true=y_te, y_pred=pred_te, lower=low_r, median=pred_te, upper=high_r,
                alpha=alpha, calib_n=len(calib), train_n=len(train),
                extra={"split_no": split_no, "global_radius": radius_map["__global__"]},
            )

            # Quantile intervals: raw quantile, CQR, regime CQR.
            if quantile_intervals:
                try:
                    bundle = fit_classical_quantile_bundle(
                        model_name=model_name,
                        X_train=train[cols],
                        y_train=train[target],
                        alpha=alpha,
                        seed=seed + split_no,
                        params=params,
                    )
                    lo_cal, md_cal, hi_cal = predict_classical_quantile_bundle(bundle, calib[cols], target)
                    lo_te, md_te, hi_te = predict_classical_quantile_bundle(bundle, test[cols], target)
                except Exception as exc:
                    log(f"[warn] quantile skipped {target}|{block_name}|{model_name}|{split.name}: {exc}")
                    continue

                append_raw_rows(
                    rows, test,
                    split_name=split.name, seed=seed, target=target, block=block_name,
                    model=model_name, method="raw_quantile", interval_engine=bundle["kind"],
                    y_true=y_te, y_pred=md_te, lower=lo_te, median=md_te, upper=hi_te,
                    alpha=alpha, calib_n=len(calib), train_n=len(train),
                    extra={"split_no": split_no, "global_cqr_q": 0.0},
                )

                q = conformal_quantile_adjustment(y_cal, lo_cal, hi_cal, alpha)
                lo_cqr = lo_te - q
                hi_cqr = hi_te + q
                lo_cqr, md_cqr, hi_cqr = prepare_quantile_outputs(lo_cqr, md_te, hi_cqr, target)
                append_raw_rows(
                    rows, test,
                    split_name=split.name, seed=seed, target=target, block=block_name,
                    model=model_name, method="cqr", interval_engine=bundle["kind"],
                    y_true=y_te, y_pred=md_cqr, lower=lo_cqr, median=md_cqr, upper=hi_cqr,
                    alpha=alpha, calib_n=len(calib), train_n=len(train),
                    extra={"split_no": split_no, "global_cqr_q": q},
                )

                scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
                scores = np.maximum(scores, 0.0)
                q_map = apply_regime_radius(
                    pd.DataFrame({"fg_regime": calib["fg_regime"].to_numpy(), "score": scores}),
                    "score",
                    alpha=alpha,
                    min_n=args.min_regime_calib_n,
                )
                q_test = np.array([q_map.get(r, q_map["__global__"]) for r in test["fg_regime"].to_numpy()], dtype=float)
                lo_rcqr = lo_te - q_test
                hi_rcqr = hi_te + q_test
                lo_rcqr, md_rcqr, hi_rcqr = prepare_quantile_outputs(lo_rcqr, md_te, hi_rcqr, target)
                append_raw_rows(
                    rows, test,
                    split_name=split.name, seed=seed, target=target, block=block_name,
                    model=model_name, method="regime_cqr", interval_engine=bundle["kind"],
                    y_true=y_te, y_pred=md_rcqr, lower=lo_rcqr, median=md_rcqr, upper=hi_rcqr,
                    alpha=alpha, calib_n=len(calib), train_n=len(train),
                    extra={"split_no": split_no, "global_cqr_q": q_map["__global__"]},
                )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    elapsed = perf_counter() - t0

    meta = {
        "mode": mode_name,
        "target": target,
        "block": block_name,
        "seed": seed,
        "models": models,
        "alpha": alpha,
        "n_rows": int(len(out)),
        "n_splits": int(len(splits)),
        "feature_count": int(len(cols)),
        "params_by_model": params_by_model,
        "elapsed_seconds": elapsed,
        "output_csv": str(out_path),
    }
    write_json(out_path.with_suffix(".json"), meta)
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(artifact_path(out_path, "tuning"), index=False)
    if point_metric_rows:
        pd.DataFrame(point_metric_rows).to_csv(artifact_path(out_path, "point_metrics"), index=False)
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(artifact_path(out_path, "model_diagnostics"), index=False)
    log(f"[done] wrote {len(out)} rows to {out_path} in {elapsed/60:.2f} min")



def run_neural_task(args: argparse.Namespace) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for --mode neural.")

    grid_row = load_task(Path(args.grid), args.task_index)
    target = str(grid_row["target"])
    block_name = str(grid_row["block"])
    model_name = str(grid_row["model"])
    seed = int(grid_row["seed"])

    if model_name not in SEQUENCE_MODELS:
        raise ValueError(f"Unsupported sequence model: {model_name}")
    if target == RETURNS_TARGET:
        raise ValueError("ret_future_1 is a lightweight classical baseline and is not run in neural mode.")

    set_global_seed(seed)
    device = resolve_device(args.device)

    out_path = output_file_for_task(Path(args.out), "neural", grid_row)
    if out_path.exists() and not args.overwrite:
        log(f"[skip] output exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    df = load_df(Path(args.csv), regime_mode=args.regime_mode, fear_threshold=args.fear_threshold, greed_threshold=args.greed_threshold)
    blocks = define_feature_blocks(df)
    cols = blocks[block_name]
    embargo_days = infer_target_horizon_days(target) if args.embargo_days < 0 else int(args.embargo_days)
    splits = make_splits(args, df, embargo_days)

    base = df.loc[:, ["date", "fg_regime"] + cols + [target]].copy()
    first_split = splits[0]

    log(
        f"[task] mode=neural task={args.task_index} target={target} block={block_name} model={model_name} "
        f"seed={seed} splits={len(splits)} features={len(cols)} device={device} "
        f"embargo_days={embargo_days} regime={args.regime_mode} thresholds=({df.attrs.get('fear_threshold')},{df.attrs.get('greed_threshold')})"
    )
    log(f"[neural] tuning {target} | {block_name} | {model_name} | seed={seed} | device={device}")
    params, trial_rows = tune_sequence_model(
        model_name=model_name,
        X_full=base.loc[:, cols],
        y_full=base.loc[:, target],
        train_idx=first_split.train_idx,
        calib_idx=first_split.calib_idx,
        test_idx=first_split.test_idx,
        target=target,
        seq_trials=args.seq_trials,
        epochs_cap=args.epochs_cap,
        seed=seed,
        device=device,
    )

    alpha = float(args.alpha)
    quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
    rows: List[Dict] = []
    tuning_rows: List[Dict] = []
    point_metric_rows: List[Dict] = []
    history_rows: List[Dict] = []
    for trial in trial_rows:
        row = {
            "target": target,
            "block": block_name,
            "model": model_name,
            "seed": seed,
        }
        row.update(trial)
        tuning_rows.append(row)

    for split_no, split in enumerate(splits):
        if split_no == 0 or (split_no + 1) % max(1, args.progress_every) == 0 or (split_no + 1) == len(splits):
            log(f"[progress] neural {target}|{block_name}|{model_name}|seed={seed} split {split_no+1}/{len(splits)}")
        torch.manual_seed(seed + split_no)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + split_no)

        seq = build_sequence_sets(
            X_full=base.loc[:, cols],
            y_full=base.loc[:, target],
            train_idx=split.train_idx,
            calib_idx=split.calib_idx,
            test_idx=split.test_idx,
            lookback=int(params["lookback"]),
        )

        X_train, y_train = seq["X_train"], seq["y_train"]
        X_cal, y_cal = seq["X_calib"], seq["y_calib"]
        X_test, y_test = seq["X_test"], seq["y_test"]
        idx_cal = seq["idx_calib"]
        idx_test = seq["idx_test"]

        if len(X_train) < 80 or len(X_cal) < 20 or len(X_test) < 20:
            continue

        cut = max(50, int(len(X_train) * 0.85))
        if len(X_train) - cut < 20:
            cut = len(X_train) - 20
        if cut <= 0 or len(X_train) - cut < 10:
            continue

        X_tr, y_tr = X_train[:cut], y_train[:cut]
        X_val, y_val = X_train[cut:], y_train[cut:]

        test = base.iloc[idx_test].copy()
        calib = base.iloc[idx_cal].copy()

        # Point sequence model.
        point_model = SeqRegressor(
            rnn_type=model_name,
            input_dim=X_train.shape[2],
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            output_dim=1,
        )
        history = train_seq_model(
            point_model,
            X_tr, y_tr, X_val, y_val,
            lr=float(params["lr"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            device=device,
            patience=int(params["patience"]),
            weight_decay=float(params.get("weight_decay", 0.0)),
            grad_clip=float(params.get("grad_clip", 1.0)),
        )
        for rec in history:
            history_rows.append({
                "target": target,
                "block": block_name,
                "model": model_name,
                "seed": seed,
                "split": split.name,
                "split_no": split_no,
                "head": "point",
                **rec,
            })
        pred_cal = predict_seq_model(point_model, X_cal, device)
        pred_te = predict_seq_model(point_model, X_test, device)
        if target != "ret_future_1":
            pred_cal = np.clip(pred_cal, 1e-8, None)
            pred_te = np.clip(pred_te, 1e-8, None)

        point_metrics = summarize_point_metrics(y_test, pred_te, target)
        if point_metrics:
            point_metric_rows.append({
                "target": target,
                "block": block_name,
                "model": model_name,
                "seed": seed,
                "split": split.name,
                "split_no": split_no,
                **point_metrics,
            })

        radius = symmetric_conformal_radius(y_cal, pred_cal, alpha)
        low = pred_te - radius
        high = pred_te + radius
        if target != "ret_future_1":
            low = np.clip(low, 0.0, None)

        append_raw_rows(
            rows, test,
            split_name=split.name, seed=seed, target=target, block=block_name,
            model=model_name, method="sym_conformal", interval_engine="point_residual",
            y_true=y_test, y_pred=pred_te, lower=low, median=pred_te, upper=high,
            alpha=alpha, calib_n=len(X_cal), train_n=len(X_train),
            extra={"split_no": split_no, "global_radius": radius},
        )

        calib_scores = pd.DataFrame({
            "fg_regime": calib["fg_regime"].to_numpy(),
            "score": np.abs(y_cal - pred_cal),
        })
        radius_map = apply_regime_radius(calib_scores, "score", alpha=alpha, min_n=args.min_regime_calib_n)
        rad = np.array([radius_map.get(r, radius_map["__global__"]) for r in test["fg_regime"].to_numpy()])
        low_r = pred_te - rad
        high_r = pred_te + rad
        if target != "ret_future_1":
            low_r = np.clip(low_r, 0.0, None)

        append_raw_rows(
            rows, test,
            split_name=split.name, seed=seed, target=target, block=block_name,
            model=model_name, method="regime_sym_conformal", interval_engine="point_residual",
            y_true=y_test, y_pred=pred_te, lower=low_r, median=pred_te, upper=high_r,
            alpha=alpha, calib_n=len(X_cal), train_n=len(X_train),
            extra={"split_no": split_no, "global_radius": radius_map["__global__"]},
        )

        if target in VOL_TARGETS:
            q_model = SeqRegressor(
                rnn_type=model_name,
                input_dim=X_train.shape[2],
                hidden_dim=int(params["hidden_dim"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                output_dim=3,
            )
            q_history = train_seq_quantile_model(
                q_model,
                X_tr, y_tr, X_val, y_val,
                quantiles=quantiles,
                lr=float(params["lr"]),
                batch_size=int(params["batch_size"]),
                epochs=int(params["epochs"]),
                device=device,
                patience=int(params["patience"]),
                weight_decay=float(params.get("weight_decay", 0.0)),
                grad_clip=float(params.get("grad_clip", 1.0)),
            )
            for rec in q_history:
                history_rows.append({
                    "target": target,
                    "block": block_name,
                    "model": model_name,
                    "seed": seed,
                    "split": split.name,
                    "split_no": split_no,
                    "head": "quantile",
                    **rec,
                })
            lo_cal, md_cal, hi_cal = predict_seq_quantiles(q_model, X_cal, target, device)
            lo_te, md_te, hi_te = predict_seq_quantiles(q_model, X_test, target, device)

            append_raw_rows(
                rows, test,
                split_name=split.name, seed=seed, target=target, block=block_name,
                model=model_name, method="raw_quantile", interval_engine="seq_quantile",
                y_true=y_test, y_pred=md_te, lower=lo_te, median=md_te, upper=hi_te,
                alpha=alpha, calib_n=len(X_cal), train_n=len(X_train),
                extra={"split_no": split_no, "global_cqr_q": 0.0},
            )

            q = conformal_quantile_adjustment(y_cal, lo_cal, hi_cal, alpha)
            lo_cqr = lo_te - q
            hi_cqr = hi_te + q
            lo_cqr, md_cqr, hi_cqr = prepare_quantile_outputs(lo_cqr, md_te, hi_cqr, target)
            append_raw_rows(
                rows, test,
                split_name=split.name, seed=seed, target=target, block=block_name,
                model=model_name, method="cqr", interval_engine="seq_quantile",
                y_true=y_test, y_pred=md_cqr, lower=lo_cqr, median=md_cqr, upper=hi_cqr,
                alpha=alpha, calib_n=len(X_cal), train_n=len(X_train),
                extra={"split_no": split_no, "global_cqr_q": q},
            )

            scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
            scores = np.maximum(scores, 0.0)
            q_map = apply_regime_radius(
                pd.DataFrame({"fg_regime": calib["fg_regime"].to_numpy(), "score": scores}),
                "score",
                alpha=alpha,
                min_n=args.min_regime_calib_n,
            )
            q_test = np.array([q_map.get(r, q_map["__global__"]) for r in test["fg_regime"].to_numpy()], dtype=float)
            lo_rcqr = lo_te - q_test
            hi_rcqr = hi_te + q_test
            lo_rcqr, md_rcqr, hi_rcqr = prepare_quantile_outputs(lo_rcqr, md_te, hi_rcqr, target)
            append_raw_rows(
                rows, test,
                split_name=split.name, seed=seed, target=target, block=block_name,
                model=model_name, method="regime_cqr", interval_engine="seq_quantile",
                y_true=y_test, y_pred=md_rcqr, lower=lo_rcqr, median=md_rcqr, upper=hi_rcqr,
                alpha=alpha, calib_n=len(X_cal), train_n=len(X_train),
                extra={"split_no": split_no, "global_cqr_q": q_map["__global__"]},
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    elapsed = perf_counter() - t0

    meta = {
        "mode": "neural",
        "target": target,
        "block": block_name,
        "model": model_name,
        "seed": seed,
        "device": device,
        "alpha": alpha,
        "n_rows": int(len(out)),
        "n_splits": int(len(splits)),
        "feature_count": int(len(cols)),
        "params": params,
        "elapsed_seconds": elapsed,
        "output_csv": str(out_path),
    }
    write_json(out_path.with_suffix(".json"), meta)
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(artifact_path(out_path, "tuning"), index=False)
    if point_metric_rows:
        pd.DataFrame(point_metric_rows).to_csv(artifact_path(out_path, "point_metrics"), index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(artifact_path(out_path, "training_history"), index=False)
    log(f"[done] wrote {len(out)} rows to {out_path} in {elapsed/60:.2f} min")
