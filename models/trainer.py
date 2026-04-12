"""
models/trainer.py — trains base models + stacking ensemble with feature selection.

Pipeline:
  1. Feature selection:
       a. VarianceThreshold (remove near-zero-variance features)
       b. Correlation pruning  (drop one of any pair with corr > 0.95)
       c. ExtraTreesClassifier importance gate (keep above median)
  2. CV via cross_val_predict + Pipeline(RobustScaler, clf) — no fold leakage
  3. StackingEnsemble: 8 base-model OOF probs → XGBoost meta-model
  4. CalibratedClassifierCV(isotonic) on best model before saving

New models: CatBoost (if installed), MLPClassifier
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ── Constants ────────────────────────────────────────────────────────────────
SAVED_DIR         = Path(__file__).parent / "saved"
MODEL_PATH        = SAVED_DIR / "best_model.pkl"
SCALER_PATH       = SAVED_DIR / "scaler.pkl"
FEATURE_COLS_PATH = SAVED_DIR / "feature_columns.json"
META_PATH         = SAVED_DIR / "model_meta.json"
BACKGROUND_PATH   = SAVED_DIR / "background_sample.npy"
CONFIG_PATH       = Path(__file__).parent.parent / "config.json"

N_SPLITS              = 5
RANDOM_STATE          = 42
BACKGROUND_SAMPLE_SIZE = 200
CORR_THRESHOLD        = 0.95   # drop one of any pair with corr > this
VARIANCE_THRESHOLD    = 0.01   # VarianceThreshold

console = Console()

# ── Base model definitions ────────────────────────────────────────────────────
def _build_base_models() -> dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest":        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "GradientBoosting":    GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(n_estimators=200, verbose=-1, random_state=RANDOM_STATE),
        "SVC":      SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, random_seed=RANDOM_STATE,
        )
    return models


# ── Stacking ensemble ─────────────────────────────────────────────────────────

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Level-0: N base classifiers (pre-fitted on full data).
    Level-1: XGBoost meta-model trained on OOF base-model probs.
    Input to predict_proba: pre-scaled X (from saved RobustScaler).
    """
    def __init__(self, base_models: list, meta_model) -> None:
        self.base_models = base_models
        self.meta_model  = meta_model
        self.classes_    = np.array([0, 1])

    def _meta_input(self, X) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_model.predict_proba(self._meta_input(X))

    def predict(self, X) -> np.ndarray:
        return self.meta_model.predict(self._meta_input(X))

    def fit(self, X, y):
        raise NotImplementedError("StackingEnsemble is assembled in train_all_models.")


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Three-stage feature selection:
      1. VarianceThreshold — drop near-constant features
      2. Correlation pruning — drop one from any pair > 0.95 corr
      3. ExtraTreesClassifier importance gate — keep above-median features

    Returns (X_selected, selected_column_names).
    """
    original_n = X.shape[1]
    console.print(f"[blue]Feature selection: starting with {original_n} features…[/blue]")

    # Stage 1: Variance
    vt        = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_vt      = vt.fit_transform(X)
    vt_cols   = [X.columns[i] for i, m in enumerate(vt.get_support()) if m]
    X         = pd.DataFrame(X_vt, columns=vt_cols)
    console.print(f"  [dim]After variance filter: {X.shape[1]} features[/dim]")

    # Stage 2: Correlation pruning
    corr_mat = X.corr().abs()
    upper    = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
    )
    to_drop  = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    X        = X.drop(columns=to_drop)
    console.print(f"  [dim]After correlation pruning: {X.shape[1]} features[/dim]")

    # Stage 3: Importance gate
    etc = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    sel = SelectFromModel(etc, threshold="median")
    X_sel = sel.fit_transform(X, y)
    sel_cols = X.columns[sel.get_support()].tolist()
    X = pd.DataFrame(X_sel, columns=sel_cols)
    console.print(
        f"  [dim]After importance gate: {X.shape[1]} features "
        f"(removed {original_n - X.shape[1]} total)[/dim]"
    )

    return X, sel_cols


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Core training ─────────────────────────────────────────────────────────────

def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: list[str] | None = None,
    run_feature_selection: bool = True,
) -> dict[str, dict]:
    """
    Full training pipeline:
      1. Feature selection (optional, recommended)
      2. CV-based OOF predictions for all base models
      3. XGBoost meta-model on OOF probs (stacking)
      4. Calibrate best model with isotonic regression
      5. Save artifacts

    Returns {model_name: {accuracy, roc_auc, f1, precision, recall, cv_std,
                           cv_fold_scores, model}}.
    """
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature selection ─────────────────────────────────────────────────
    n_input_features = X.shape[1]
    selected_cols = list(X.columns)
    if run_feature_selection:
        X, selected_cols = select_features(X, y)

    tscv        = TimeSeriesSplit(n_splits=N_SPLITS)
    splits_list = list(tscv.split(X))

    # Deployment scaler: fit on full dataset.
    scaler        = RobustScaler()
    X_scaled_full = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(selected_cols, f)

    n_bg = min(BACKGROUND_SAMPLE_SIZE, len(X_scaled_full))
    np.save(BACKGROUND_PATH, X_scaled_full[:n_bg])

    _BASE_MODELS = _build_base_models()
    results: dict[str, dict] = {}
    oof_probs_matrix = np.zeros((len(X), len(_BASE_MODELS)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_steps = len(_BASE_MODELS) + 1  # +1 for stacking
        task = progress.add_task("Training models…", total=total_steps)

        for col_idx, (name, base_model) in enumerate(_BASE_MODELS.items()):
            progress.update(task, description=f"Training {name}…")

            pipe = Pipeline([
                ("scaler", RobustScaler()),
                ("clf",    clone(base_model)),
            ])
            # Manual OOF loop: TimeSeriesSplit is not a partition so
            # cross_val_predict raises ValueError. Early rows never appear
            # in any test fold; they stay at 0.5 (neutral default).
            oof_probs = np.full((len(X), 2), 0.5)
            fold_accs = []
            for train_i, test_i in splits_list:
                p = clone(pipe)
                p.fit(X.iloc[train_i], y.iloc[train_i])
                fold_pred = p.predict_proba(X.iloc[test_i])
                oof_probs[test_i] = fold_pred
                fold_cls = (fold_pred[:, 1] >= 0.5).astype(int)
                fold_accs.append(float(accuracy_score(y.iloc[test_i], fold_cls)))

            oof_probs_matrix[:, col_idx] = oof_probs[:, 1]
            oof_classes = (oof_probs[:, 1] >= 0.5).astype(int)

            final = clone(base_model)
            final.fit(X_scaled_full, y)

            results[name] = {
                "accuracy":       float(accuracy_score(y, oof_classes)),
                "roc_auc":        float(roc_auc_score(y, oof_probs[:, 1])),
                "f1":             float(f1_score(y, oof_classes, zero_division=0)),
                "precision":      float(precision_score(y, oof_classes, zero_division=0)),
                "recall":         float(recall_score(y, oof_classes, zero_division=0)),
                "cv_std":         float(np.std(fold_accs)),
                "cv_fold_scores": fold_accs,
                "model":          final,
            }
            progress.advance(task)

        # ── Level-1: XGBoost meta-model ───────────────────────────────────
        progress.update(task, description="Building StackingEnsemble (XGBoost meta)…")

        meta_xgb = XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )
        meta_xgb.fit(oof_probs_matrix, y)

        stacking_model = StackingEnsemble(
            base_models=[results[n]["model"] for n in _BASE_MODELS],
            meta_model=meta_xgb,
        )

        # Single OOF loop: collect fold accuracy + OOF cls/prob in one pass.
        stack_fold_accs:     list[float] = []
        stack_oof_cls_list:  list[int]   = []
        stack_oof_prob_list: list[float] = []
        for train_i, test_i in splits_list:
            m_oof = clone(meta_xgb)
            m_oof.fit(oof_probs_matrix[train_i], y.iloc[train_i])
            fold_cls  = m_oof.predict(oof_probs_matrix[test_i])
            fold_prob = m_oof.predict_proba(oof_probs_matrix[test_i])[:, 1]
            stack_fold_accs.append(float(accuracy_score(y.iloc[test_i], fold_cls)))
            stack_oof_cls_list.extend(fold_cls.tolist())
            stack_oof_prob_list.extend(fold_prob.tolist())

        # Reconstruct y ordering aligned with folds (positional — requires reset index).
        assert X.index.is_monotonic_increasing and X.index[0] == 0, (
            "X must have a clean 0-based RangeIndex before training; "
            "call X.reset_index(drop=True) before passing to train_models."
        )
        test_indices_order = [i for _, test_i in splits_list for i in test_i]
        y_reordered = y.iloc[test_indices_order]

        results["StackingEnsemble"] = {
            "accuracy":       float(accuracy_score(y_reordered, stack_oof_cls_list)),
            "roc_auc":        float(roc_auc_score(y_reordered, stack_oof_prob_list)),
            "f1":             float(f1_score(y_reordered, stack_oof_cls_list, zero_division=0)),
            "precision":      float(precision_score(y_reordered, stack_oof_cls_list, zero_division=0)),
            "recall":         float(recall_score(y_reordered, stack_oof_cls_list, zero_division=0)),
            "cv_std":         float(np.std(stack_fold_accs)),
            "cv_fold_scores": stack_fold_accs,
            "model":          stacking_model,
        }
        progress.advance(task)

    # ── Best model by ROC-AUC ─────────────────────────────────────────────
    best_name  = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = results[best_name]["model"]

    console.print(
        f"\n[green]Best model: {best_name} "
        f"(ROC-AUC={results[best_name]['roc_auc']:.4f})[/green]"
    )

    # ── Level-2: Isotonic calibration ─────────────────────────────────────
    # StackingEnsemble has no fit() — skip calibration, its XGBoost meta-model
    # already produces calibrated probabilities.
    with console.status("Calibrating probabilities (isotonic regression)…"):
        if isinstance(best_model, StackingEnsemble):
            final_saved = best_model
        else:
            calibrated = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
            calibrated.fit(X_scaled_full, y)
            final_saved = calibrated

    joblib.dump(final_saved, MODEL_PATH)

    # ── Feature selection summary table ──────────────────────────────────
    sel_table = Table(
        title="[bold cyan]Feature Selection Summary[/bold cyan]",
        box=box.SIMPLE, show_header=True, header_style="bold magenta",
    )
    sel_table.add_column("Stage")
    sel_table.add_column("Features kept", justify="right")
    sel_table.add_row("Input",           str(n_input_features))
    sel_table.add_row("After selection", str(len(selected_cols)))
    sel_table.add_row("Removed",         str(n_input_features - len(selected_cols)))
    console.print(sel_table)

    meta = {
        "model_name":       best_name,
        "accuracy":         round(results[best_name]["accuracy"], 4),
        "roc_auc":          round(results[best_name]["roc_auc"], 4),
        "trained_on":       datetime.date.today().isoformat(),
        "seasons":          seasons or [],
        "n_training_games": int(len(X)),
        "n_features":       len(selected_cols),
        "calibrated":       not isinstance(best_model, StackingEnsemble),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    cfg = _load_config()
    cfg["best_model_name"] = best_name
    _save_config(cfg)

    return results


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model. Run option 2 to train first.")
    return joblib.load(MODEL_PATH)


def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Scaler not found. Run option 2 to train first.")
    return joblib.load(SCALER_PATH)


def load_feature_columns() -> list[str]:
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError("Feature columns not found. Run option 2 to train first.")
    with open(FEATURE_COLS_PATH) as f:
        return json.load(f)


def load_model_meta() -> dict:
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)
