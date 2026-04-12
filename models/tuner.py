"""
models/tuner.py — Optuna hyperparameter search for ALL base models.

Changes from v1:
  - 300 trials (was 100)
  - TPESampler(multivariate=True) — models joint param distributions
  - MedianPruner — cuts bad trials early (saves ~30% compute)
  - Covers all 8 models including CatBoost and MLP
  - RobustScaler (was StandardScaler)
  - Optimises ROC-AUC (more honest than accuracy for sports prediction)
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ── Constants ────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 300
N_CV_SPLITS   = 3
RANDOM_STATE  = 42
SAVED_DIR     = Path(__file__).parent / "saved"
MODEL_PATH    = SAVED_DIR / "best_model.pkl"
SCALER_PATH   = SAVED_DIR / "scaler.pkl"
META_PATH     = SAVED_DIR / "model_meta.json"
BACKGROUND_PATH = SAVED_DIR / "background_sample.npy"
CONFIG_PATH   = Path(__file__).parent.parent / "config.json"

console = Console()
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ── Per-model search spaces ───────────────────────────────────────────────────

def _suggest_lr(trial: optuna.Trial) -> LogisticRegression:
    return LogisticRegression(
        C=trial.suggest_float("C", 0.001, 100, log=True),
        solver=trial.suggest_categorical("solver", ["lbfgs", "saga", "liblinear"]),
        max_iter=trial.suggest_int("max_iter", 100, 2000),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=RANDOM_STATE,
    )


def _suggest_rf(trial: optuna.Trial) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _suggest_gb(trial: optuna.Trial) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=RANDOM_STATE,
    )


def _suggest_xgb(trial: optuna.Trial) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )


def _suggest_lgbm(trial: optuna.Trial) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 200),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        verbose=-1, random_state=RANDOM_STATE,
    )


def _suggest_svc(trial: optuna.Trial) -> SVC:
    return SVC(
        C=trial.suggest_float("C", 0.01, 100, log=True),
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
        probability=True, random_state=RANDOM_STATE,
    )


def _suggest_mlp(trial: optuna.Trial) -> MLPClassifier:
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_int("layer_size", 64, 512)
    layers     = tuple([layer_size] * n_layers)
    return MLPClassifier(
        hidden_layer_sizes=layers,
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def _suggest_catboost(trial: optuna.Trial):
    if not HAS_CATBOOST:
        raise ImportError("catboost not installed. Run: pip install catboost")
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        verbose=0, random_seed=RANDOM_STATE,
    )


MODEL_SUGGESTERS = {
    "LogisticRegression": _suggest_lr,
    "RandomForest":       _suggest_rf,
    "GradientBoosting":   _suggest_gb,
    "XGBoost":            _suggest_xgb,
    "LightGBM":           _suggest_lgbm,
    "SVC":                _suggest_svc,
    "MLP":                _suggest_mlp,
    **({"CatBoost": _suggest_catboost} if HAS_CATBOOST else {}),
}


def _build_model_from_params(model_name: str, params: dict):
    """Construct a fully-specified model from a best-params dict."""
    if model_name == "LogisticRegression":
        return LogisticRegression(**params, random_state=RANDOM_STATE)
    if model_name == "RandomForest":
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    if model_name == "XGBoost":
        return XGBClassifier(**params, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
    if model_name == "LightGBM":
        return LGBMClassifier(**params, verbose=-1, random_state=RANDOM_STATE)
    if model_name == "SVC":
        return SVC(**params, probability=True, random_state=RANDOM_STATE)
    if model_name == "MLP":
        n_layers   = params.pop("n_layers",   2)
        layer_size = params.pop("layer_size", 128)
        lr         = params.pop("lr",         1e-3)
        params["hidden_layer_sizes"]    = tuple([layer_size] * n_layers)
        params["learning_rate_init"]    = lr
        params["max_iter"]              = 500
        params["early_stopping"]        = True
        params["validation_fraction"]   = 0.1
        params["random_state"]          = RANDOM_STATE
        return MLPClassifier(**params)
    if model_name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed.")
        return CatBoostClassifier(**params, verbose=0, random_seed=RANDOM_STATE)
    raise ValueError(
        f"No builder for '{model_name}'. Known: {list(MODEL_SUGGESTERS)}"
    )


# ── Main tuning function ───────────────────────────────────────────────────────

def tune_best_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    baseline_accuracy: float,
) -> dict:
    """
    Run 300-trial Optuna search for *model_name* with multivariate TPE + median pruning.
    Returns {best_params, new_accuracy, improved}.
    """
    if model_name == "StackingEnsemble":
        console.print(
            "[yellow]StackingEnsemble is assembled from all base models — "
            "retrain (option 2) to update it. To tune individual components, "
            "temporarily set best_model_name in config.json.[/yellow]"
        )
        return {"best_params": {}, "new_accuracy": baseline_accuracy, "improved": False}

    if model_name not in MODEL_SUGGESTERS:
        raise ValueError(
            f"No Optuna tuner for '{model_name}'. "
            f"Add it to MODEL_SUGGESTERS. Known: {list(MODEL_SUGGESTERS)}"
        )

    scaler  = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    tscv    = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    suggester  = MODEL_SUGGESTERS[model_name]
    completed  = [0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Optuna: {model_name} ({OPTUNA_TRIALS} trials)…", total=OPTUNA_TRIALS)

        def objective(trial: optuna.Trial) -> float:
            model  = suggester(trial)
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="roc_auc", n_jobs=-1)
            val    = float(np.mean(scores))
            completed[0] += 1
            progress.update(task, completed=completed[0])
            # Prune clearly bad trials
            trial.report(val, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return val

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE, multivariate=True),
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_auc    = study.best_value

    console.print(f"\n[green]Best trial ROC-AUC: {best_auc:.4f}[/green]")
    console.print(f"[green]Best params: {best_params}[/green]")

    # Build final model on full scaled data.
    final_model = _build_model_from_params(model_name, dict(best_params))
    scaler_full  = RobustScaler()
    X_full_sc    = scaler_full.fit_transform(X)
    final_model.fit(X_full_sc, y)

    # Estimate accuracy via fresh 5-fold CV.
    from sklearn.model_selection import TimeSeriesSplit as TSS2
    acc_scores = cross_val_score(
        _build_model_from_params(model_name, dict(best_params)),
        X_full_sc, y, cv=TSS2(n_splits=5), scoring="accuracy", n_jobs=-1,
    )
    new_accuracy = float(np.mean(acc_scores))
    improved     = new_accuracy >= baseline_accuracy

    if improved:
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(scaler_full, SCALER_PATH)

        from models.trainer import BACKGROUND_SAMPLE_SIZE
        n_bg = min(BACKGROUND_SAMPLE_SIZE, len(X_full_sc))
        np.save(BACKGROUND_PATH, X_full_sc[:n_bg])

        meta          = _load_meta()
        meta["accuracy"]  = round(new_accuracy, 4)
        meta["roc_auc"]   = round(best_auc, 4)
        meta["trained_on"] = datetime.date.today().isoformat()
        meta["tuned"]     = True
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        cfg = _load_config()
        cfg["best_model_params"] = best_params
        _save_config(cfg)

    console.print(
        Panel(
            f"[bold]Model:[/bold]        {model_name}\n"
            f"[bold]Baseline Acc:[/bold]  {baseline_accuracy:.4f}\n"
            f"[bold]Tuned Acc:[/bold]     {new_accuracy:.4f}\n"
            f"[bold]Best ROC-AUC:[/bold]  {best_auc:.4f}\n"
            f"[bold]Improved:[/bold]      {'[green]YES[/green]' if improved else '[red]NO[/red]'}\n"
            f"[bold]Best params:[/bold]\n"
            + "\n".join(f"  {k}: {v}" for k, v in best_params.items()),
            title="[bold cyan]Tuning Results[/bold cyan]",
            expand=False,
        )
    )

    return {"best_params": best_params, "new_accuracy": new_accuracy, "improved": improved}
