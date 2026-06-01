from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge

try:
    from sklearn.linear_model import QuantileRegressor
except ImportError:
    QuantileRegressor = None

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .intervals import prepare_quantile_outputs, score_for_target

optuna.logging.set_verbosity(optuna.logging.WARNING)

CLASSICAL_MODELS = ["har_ols", "ridge", "rf", "gbr"]
SEQUENCE_MODELS = ["lstm", "gru"]


def allocated_cpus(default: int = 1) -> int:
    for key in ["SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"]:
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, int(default))

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)



def make_numeric_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_tree_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def build_classical_model(model_name: str, params: Dict, seed: int) -> Pipeline:
    if model_name == "har_ols":
        return make_numeric_pipeline(LinearRegression())

    if model_name == "ridge":
        return make_numeric_pipeline(Ridge(alpha=float(params.get("alpha", 1.0))))

    if model_name == "rf":
        return make_tree_pipeline(RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") in [None, "None"] else int(params.get("max_depth")),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=allocated_cpus(),
        ))

    if model_name == "gbr":
        return make_tree_pipeline(GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=seed,
        ))

    raise ValueError(f"Unknown classical model: {model_name}")

def tune_classical_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target: str,
    trials: int,
    seed: int,
) -> Tuple[Dict, List[Dict]]:
    if model_name == "har_ols":
        return {}, []

    trial_rows: List[Dict] = []

    def objective(params: Dict) -> float:
        mdl = build_classical_model(model_name, params, seed)
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_val)
        if target != "ret_future_1":
            pred = np.clip(pred, 1e-8, None)
        return score_for_target(y_val.to_numpy(), pred, target)

    def optuna_objective(trial):
        if model_name == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 500]),
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, 12, None]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            }
        elif model_name == "gbr":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.2, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [1, 2, 3]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 5, 10, 20]),
                "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
            }
        else:
            params = {}
        value = objective(params)
        row = {"trial": int(trial.number), "value": float(value)}
        row.update(params)
        trial_rows.append(row)
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(optuna_objective, n_trials=max(1, int(trials)), show_progress_bar=False)
    return dict(study.best_params), trial_rows


class LinearQuantileModel:
    """Linear quantile regression wrapper with sklearn first and statsmodels fallback."""

    def __init__(self, quantile: float, alpha: float = 0.0, solver: str = "highs") -> None:
        self.quantile = float(quantile)
        self.alpha = float(alpha)
        self.solver = solver
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None
        self.backend = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_imp = self.imputer.fit_transform(X)
        X_std = self.scaler.fit_transform(X_imp)
        y_arr = np.asarray(y, dtype=float)

        if QuantileRegressor is not None:
            self.backend = "sklearn"
            self.model = QuantileRegressor(
                quantile=self.quantile,
                alpha=self.alpha,
                solver=self.solver,
            )
            self.model.fit(X_std, y_arr)
        else:
            self.backend = "statsmodels"
            X_sm = sm.add_constant(X_std, has_constant="add")
            self.model = sm.QuantReg(y_arr, X_sm).fit(q=self.quantile, max_iter=1000, disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_std = self.scaler.transform(X_imp)
        if self.backend == "statsmodels":
            return np.asarray(self.model.predict(sm.add_constant(X_std, has_constant="add")), dtype=float)
        return np.asarray(self.model.predict(X_std), dtype=float)


def quantile_regularization_for_model(model_name: str, params: Optional[Dict] = None) -> float:
    params = params or {}
    if model_name == "ridge":
        ridge_alpha = float(params.get("alpha", 1.0))
        return float(np.clip(ridge_alpha * 1e-3, 1e-5, 1.0))
    return 0.0


def fit_quantile_gbr(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    seed: int,
    params: Optional[Dict] = None,
) -> Pipeline:
    params = params or {}
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=float(alpha),
        n_estimators=int(params.get("n_estimators", 250)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        max_depth=int(params.get("max_depth", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
        subsample=float(params.get("subsample", 0.8)),
        random_state=seed,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    pipe.fit(X, y)
    return pipe


def predict_rf_tree_quantiles(pipe: Pipeline, X: pd.DataFrame, quantiles: List[float]) -> np.ndarray:
    imputer = pipe.named_steps["imputer"]
    forest = pipe.named_steps["model"]
    X_imp = imputer.transform(X)
    tree_preds = np.column_stack([tree.predict(X_imp) for tree in forest.estimators_])
    return np.quantile(tree_preds, quantiles, axis=1).T


def fit_classical_quantile_bundle(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    seed: int,
    params: Optional[Dict] = None,
) -> Dict:
    params = params or {}
    quantiles = [alpha / 2, 0.5, 1 - alpha / 2]

    if model_name in ["har_ols", "ridge"]:
        q_alpha = quantile_regularization_for_model(model_name, params)
        models = [LinearQuantileModel(q, alpha=q_alpha).fit(X_train, y_train) for q in quantiles]
        return {"kind": "linear_quantile", "models": models, "quantiles": quantiles}

    if model_name == "rf":
        pipe = build_classical_model("rf", params, seed)
        pipe.fit(X_train, y_train)
        return {"kind": "rf_tree_quantile", "model": pipe, "quantiles": quantiles}

    if model_name == "gbr":
        models = [fit_quantile_gbr(X_train, y_train, q, seed, params=params) for q in quantiles]
        return {"kind": "gbr_quantile", "models": models, "quantiles": quantiles}

    raise ValueError(f"No classical quantile bundle for model: {model_name}")



def predict_classical_quantile_bundle(bundle: Dict, X: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle["kind"] == "rf_tree_quantile":
        preds = predict_rf_tree_quantiles(bundle["model"], X, bundle["quantiles"])
        lower, median, upper = preds[:, 0], preds[:, 1], preds[:, 2]
    else:
        lower, median, upper = [m.predict(X) for m in bundle["models"]]
    return prepare_quantile_outputs(lower, median, upper, target)



class SeqRegressor(nn.Module if nn is not None else object):
    def __init__(
        self,
        rnn_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.output_dim = int(output_dim)
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = self.dropout(out[:, -1, :])
        pred = self.head(h)
        if self.output_dim == 1:
            return pred.squeeze(-1)
        return pred


def fit_impute_scale(X_train: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    X_std = scaler.fit_transform(X_imp)
    return imputer, scaler, X_std


def apply_impute_scale(imputer: SimpleImputer, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return scaler.transform(imputer.transform(X))


def build_sequences_from_array(X_arr: np.ndarray, y_arr: np.ndarray, idx_arr: np.ndarray, lookback: int):
    seq_X, seq_y, seq_idx = [], [], []
    for pos in range(lookback - 1, len(X_arr)):
        window = X_arr[pos - lookback + 1: pos + 1]
        target = y_arr[pos]
        if np.isfinite(target) and np.isfinite(window).all():
            seq_X.append(window)
            seq_y.append(float(target))
            seq_idx.append(int(idx_arr[pos]))
    if not seq_X:
        return np.empty((0, lookback, X_arr.shape[1])), np.array([]), np.array([], dtype=int)
    return np.stack(seq_X), np.asarray(seq_y), np.asarray(seq_idx, dtype=int)


def build_sequence_sets(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    lookback: int,
):
    start_idx = max(0, int(train_idx[0]) - lookback + 1)
    context_idx = np.arange(start_idx, int(test_idx[-1]) + 1)

    X_context = X_full.iloc[context_idx]
    y_context = y_full.iloc[context_idx].to_numpy()

    imputer, scaler, _ = fit_impute_scale(X_full.iloc[train_idx])
    X_context_std = apply_impute_scale(imputer, scaler, X_context)

    seq_X, seq_y, seq_idx = build_sequences_from_array(X_context_std, y_context, context_idx, lookback)

    return {
        "X_train": seq_X[np.isin(seq_idx, train_idx)],
        "y_train": seq_y[np.isin(seq_idx, train_idx)],
        "X_calib": seq_X[np.isin(seq_idx, calib_idx)],
        "y_calib": seq_y[np.isin(seq_idx, calib_idx)],
        "idx_calib": seq_idx[np.isin(seq_idx, calib_idx)],
        "X_test": seq_X[np.isin(seq_idx, test_idx)],
        "y_test": seq_y[np.isin(seq_idx, test_idx)],
        "idx_test": seq_idx[np.isin(seq_idx, test_idx)],
    }


def train_seq_model(
    model: SeqRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    epochs: int,
    device: str,
    patience: int = 8,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
) -> List[Dict]:
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = np.inf
    patience_left = patience
    history: List[Dict] = []

    for epoch in range(int(epochs)):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_t), y_val_t).item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        history.append({"epoch": int(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_seq_model(model: SeqRegressor, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(X_t).detach().cpu().numpy()
    return pred


def torch_quantile_loss(pred, target, quantiles: List[float]):
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    target = target.view(-1, 1)
    err = target - pred
    return torch.maximum((q - 1.0) * err, q * err).mean()


def train_seq_quantile_model(
    model: SeqRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: List[float],
    lr: float,
    batch_size: int,
    epochs: int,
    device: str,
    patience: int = 8,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
) -> List[Dict]:
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = np.inf
    patience_left = patience
    history: List[Dict] = []

    for epoch in range(int(epochs)):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch_quantile_loss(model(xb), yb, quantiles)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(torch_quantile_loss(model(X_val_t), y_val_t, quantiles).item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        history.append({"epoch": int(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_seq_quantiles(model: SeqRegressor, X: np.ndarray, target: str, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(X_t).detach().cpu().numpy()
    return prepare_quantile_outputs(pred[:, 0], pred[:, 1], pred[:, 2], target)


def tune_sequence_model(
    model_name: str,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    target: str,
    seq_trials: int,
    epochs_cap: int,
    seed: int,
    device: str,
) -> Tuple[Dict, List[Dict]]:
    fallback = {
        "lookback": 30,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": min(50, epochs_cap),
        "patience": 8,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
    }
    trial_rows: List[Dict] = []

    def objective(trial):
        params = {
            "lookback": trial.suggest_categorical("lookback", [7, 14, 30, 60, 90]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_categorical("epochs", [min(30, epochs_cap), min(50, epochs_cap), min(70, epochs_cap)]),
            "patience": trial.suggest_categorical("patience", [6, 8, 12]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0]),
        }
        params["epochs"] = int(max(1, params["epochs"]))

        torch.manual_seed(seed + trial.number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + trial.number)

        seq = build_sequence_sets(
            X_full=X_full,
            y_full=y_full,
            train_idx=train_idx,
            calib_idx=calib_idx,
            test_idx=test_idx,
            lookback=int(params["lookback"]),
        )

        X_train, y_train = seq["X_train"], seq["y_train"]
        X_cal, y_cal = seq["X_calib"], seq["y_calib"]

        if len(X_train) < 100 or len(X_cal) < 20:
            raise optuna.TrialPruned()

        cut = max(50, int(len(X_train) * 0.85))
        if len(X_train) - cut < 20:
            cut = len(X_train) - 20
        if cut <= 0 or len(X_train) - cut < 10:
            raise optuna.TrialPruned()

        X_tr, y_tr = X_train[:cut], y_train[:cut]
        X_val, y_val = X_train[cut:], y_train[cut:]

        model = SeqRegressor(
            rnn_type=model_name,
            input_dim=X_train.shape[2],
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            output_dim=1,
        )

        train_seq_model(
            model, X_tr, y_tr, X_val, y_val,
            lr=float(params["lr"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            device=device,
            patience=int(params["patience"]),
            weight_decay=float(params["weight_decay"]),
            grad_clip=float(params["grad_clip"]),
        )

        pred_cal = predict_seq_model(model, X_cal, device)
        if target != "ret_future_1":
            pred_cal = np.clip(pred_cal, 1e-8, None)

        value = score_for_target(y_cal, pred_cal, target)
        row = {"trial": int(trial.number), "value": float(value)}
        row.update(params)
        trial_rows.append(row)
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, int(seq_trials)), show_progress_bar=False)

    try:
        best_params = dict(study.best_params)
    except ValueError:
        return fallback, trial_rows

    best_params["epochs"] = int(max(1, best_params.get("epochs", fallback["epochs"])))
    best_params["patience"] = int(best_params.get("patience", fallback["patience"]))
    return {**fallback, **best_params}, trial_rows


def resolve_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    return requested




def extract_model_diagnostics(pipe: Pipeline, model_name: str, features: List[str]) -> List[Dict]:
    if not hasattr(pipe, "named_steps") or "model" not in pipe.named_steps:
        return []
    model = pipe.named_steps["model"]
    rows = []
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        kind = "coefficient"
    else:
        return []
    for feature, value in zip(features, values):
        rows.append({"feature": feature, "value": float(value), "diagnostic": kind})
    return rows
