"""
models/evaluator.py — Rich terminal evaluation dashboard.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────
TOP_N_FEATURES = 15
N_SPLITS = 5     # must match trainer.N_SPLITS so we use the same last fold
GREEN = "bold green"
YELLOW = "bold yellow"
WHITE = "white"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _highlight_best(values: list[float], higher_is_better: bool = True) -> list[str]:
    """Return list of Rich style strings; best value gets GREEN."""
    if not values:
        return []
    best_idx = (
        max(range(len(values)), key=lambda i: values[i])
        if higher_is_better
        else min(range(len(values)), key=lambda i: values[i])
    )
    return [GREEN if i == best_idx else WHITE for i in range(len(values))]


def _fmt(v: float) -> str:
    return f"{v:.4f}"


# ── Dashboard ─────────────────────────────────────────────────────────────────

def show_evaluation_dashboard(
    results: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[list[str]] = None,
) -> None:
    """
    Render:
    1. Side-by-side model comparison table
    2. Confusion matrix for best model (evaluated on the last held-out CV fold)
    3. CV fold scores for best model
    4. Feature importance bar chart
    5. Summary panel
    """
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["roc_auc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    precs = [results[n]["precision"] for n in names]
    recs = [results[n]["recall"] for n in names]
    stds = [results[n]["cv_std"] for n in names]

    # ── 1. Comparison table ───────────────────────────────────────────────
    table = Table(
        title="[bold cyan]Model Comparison (TimeSeriesSplit CV)[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Model", style="bold", min_width=20)
    table.add_column("Accuracy", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("CV Std Dev", justify="right")

    acc_styles = _highlight_best(accs)
    auc_styles = _highlight_best(aucs)
    f1_styles = _highlight_best(f1s)
    prec_styles = _highlight_best(precs)
    rec_styles = _highlight_best(recs)
    std_styles = _highlight_best(stds, higher_is_better=False)

    for i, name in enumerate(names):
        table.add_row(
            name,
            f"[{acc_styles[i]}]{_fmt(accs[i])}[/]",
            f"[{auc_styles[i]}]{_fmt(aucs[i])}[/]",
            f"[{f1_styles[i]}]{_fmt(f1s[i])}[/]",
            f"[{prec_styles[i]}]{_fmt(precs[i])}[/]",
            f"[{rec_styles[i]}]{_fmt(recs[i])}[/]",
            f"[{std_styles[i]}]{_fmt(stds[i])}[/]",
        )

    console.print(table)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = results[best_name]
    model = best["model"]

    # ── 2. Confusion matrix on last held-out fold ─────────────────────────
    # Use the last TimeSeriesSplit fold as a held-out test set so the matrix
    # reflects true out-of-sample performance, not training-set memorisation.
    from models.trainer import load_scaler, load_feature_columns
    try:
        scaler       = load_scaler()
        feature_cols = load_feature_columns()
    except FileNotFoundError as exc:
        console.print(f"[red]Could not load scaler/feature cols: {exc}[/red]")
        return

    # Subset X to the features the scaler was fit on before splitting folds.
    missing = [c for c in feature_cols if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X_sel = X[feature_cols]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(X_sel))
    _, test_idx = splits[-1]
    X_test = X_sel.iloc[test_idx]
    y_test = y.iloc[test_idx]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    cm_text = (
        f"\n  Predicted →    [bold]Away Win[/bold]   [bold]Home Win[/bold]\n"
        f"  Actual ↓\n"
        f"  [bold]Away Win[/bold]        [green]{tn:>6}[/green]     [red]{fp:>6}[/red]\n"
        f"  [bold]Home Win[/bold]        [red]{fn:>6}[/red]     [green]{tp:>6}[/green]\n"
        f"\n  (evaluated on last CV fold — {len(y_test):,} games)"
    )
    console.print(
        Panel(
            cm_text,
            title=f"[bold cyan]Confusion Matrix — {best_name}[/bold cyan]",
            expand=False,
        )
    )

    # ── 3. CV fold scores ─────────────────────────────────────────────────
    folds = best.get("cv_fold_scores", [])
    console.print(f"\n[bold cyan]CV Fold Accuracy — {best_name}[/bold cyan]")
    for i, score in enumerate(folds, 1):
        bar = "█" * int(score * 40)
        console.print(f"  Fold {i}: [green]{bar}[/green] {score:.4f}")

    # ── 4. Feature importance ─────────────────────────────────────────────
    # Use saved feature cols (post-selection) so labels match model weights.
    feat_names = feature_cols
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None and feat_names:
        paired = sorted(
            zip(feat_names, importances), key=lambda x: x[1], reverse=True
        )[:TOP_N_FEATURES]
        max_imp = paired[0][1] if paired else 1.0

        console.print(f"\n[bold cyan]Top {TOP_N_FEATURES} Feature Importances — {best_name}[/bold cyan]")
        with Progress(
            TextColumn("{task.description}", justify="right"),
            BarColumn(bar_width=40),
            TextColumn("{task.percentage:>5.1f}%"),
            console=console,
            auto_refresh=False,
        ) as prog:
            for feat, imp in paired:
                pct = int(imp / max_imp * 100) if max_imp > 0 else 0
                prog.add_task(f"{feat:35s}", total=100, completed=pct)

    # ── 5. Summary panel ─────────────────────────────────────────────────
    summary = (
        f"[bold]Best model:[/bold]  [green]{best_name}[/green]\n"
        f"[bold]Accuracy:[/bold]    [green]{best['accuracy']:.4f}[/green]\n"
        f"[bold]ROC-AUC:[/bold]     [green]{best['roc_auc']:.4f}[/green]\n"
        f"[bold]F1 Score:[/bold]    [green]{best['f1']:.4f}[/green]\n\n"
        f"[bold]Recommendation:[/bold] Use [green]{best_name}[/green] for predictions. "
        f"Run Optuna tuning (option 3) to potentially improve further."
    )
    console.print(Panel(summary, title="[bold cyan]Training Summary[/bold cyan]", expand=False))
