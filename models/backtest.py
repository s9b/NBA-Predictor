"""
models/backtest.py — historical prediction backtesting with flat-bet simulation.

Simulates betting $100 on every prediction above a confidence threshold
and compares against three baselines:
  1. Random baseline (50% win rate)
  2. Always-pick-home baseline
  3. Predicted-favourite baseline (unconstrained, no threshold)

NOTE: If the saved model was trained on all available data (including the
test period), predictions will be optimistic due to lookahead.  For a
leak-free backtest, retrain the model on pre-test seasons (options 1→2)
using only seasons up to the start of your test window before running this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Constants ────────────────────────────────────────────────────────────────
SAVED_DIR   = Path(__file__).parent / "saved"
MODEL_PATH  = SAVED_DIR / "best_model.pkl"
SCALER_PATH = SAVED_DIR / "scaler.pkl"
FEATURE_COLS_PATH = SAVED_DIR / "feature_columns.json"
META_PATH   = SAVED_DIR / "model_meta.json"

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_artifacts():
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model. Run option 2 to train first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def _betting_stats(
    results: list[dict],
    flat_bet: float = 100.0,
    threshold: float = 0.60,
) -> dict:
    """Compute betting simulation stats from a list of prediction result dicts."""
    bets      = [r for r in results if r["prob_predicted"] >= threshold]
    n_bets    = len(bets)
    n_correct = sum(1 for r in bets if r["correct"])
    win_rate  = n_correct / n_bets if n_bets else 0.0

    # Simplified ROI: assume average payout = -110 (standard spread bet).
    # Win: profit = flat_bet * (100/110) ≈ 0.909 × flat_bet
    # Lose: loss = flat_bet
    WIN_PAYOUT  = flat_bet * (100 / 110)
    total_wagered = n_bets * flat_bet
    total_profit  = n_correct * WIN_PAYOUT - (n_bets - n_correct) * flat_bet
    roi           = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

    return {
        "n_bets":          n_bets,
        "n_correct":       n_correct,
        "win_rate":        win_rate,
        "total_wagered":   total_wagered,
        "total_profit":    total_profit,
        "roi_pct":         roi,
    }


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    matchup_df: pd.DataFrame,
    test_seasons: Optional[list[str]] = None,
    confidence_threshold: float = 0.60,
    flat_bet: float = 100.0,
) -> dict:
    """
    Run backtest simulation.

    Parameters
    ----------
    X, y         : full feature matrix and labels (must include SEASON alignment)
    matchup_df   : matchup DataFrame (must align row-for-row with X)
    test_seasons : list of season strings to evaluate on (e.g. ["2022-23", "2023-24"])
                   defaults to the last 2 seasons in matchup_df
    confidence_threshold : minimum predicted probability to "bet"
    flat_bet     : simulated bet size in dollars

    Returns full results dict.
    """
    model, scaler, feature_cols = _load_artifacts()

    meta: dict = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)

    # Align columns to trained feature set.
    missing = [c for c in feature_cols if c not in X.columns]
    for col in missing:
        X[col] = 0.0
    X = X[feature_cols]

    # Determine test window.
    if "SEASON" in matchup_df.columns:
        all_seasons = sorted(matchup_df["SEASON"].unique().tolist())
    else:
        all_seasons = []

    if test_seasons is None:
        test_seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons

    if not test_seasons:
        console.print("[red]No seasons available for backtest.[/red]")
        return {}

    # Mask to test rows.
    if "SEASON" in matchup_df.columns:
        test_mask  = matchup_df["SEASON"].isin(test_seasons)
        test_idx   = matchup_df.index[test_mask].tolist()
    else:
        # Fall back to last 20% of data.
        n = len(X)
        cutoff = int(n * 0.80)
        test_idx = list(range(cutoff, n))

    if not test_idx:
        console.print("[red]No test games found.[/red]")
        return {}

    assert len(X) == len(matchup_df), (
        f"X ({len(X)} rows) and matchup_df ({len(matchup_df)} rows) must align row-for-row."
    )
    X_test  = X.iloc[[i for i in test_idx if i < len(X)]]
    y_test  = y.iloc[[i for i in test_idx if i < len(y)]]
    n_test  = len(X_test)

    with console.status(f"Running predictions on {n_test} test games…"):
        X_scaled = scaler.transform(X_test)
        probs    = model.predict_proba(X_scaled)[:, 1]
        preds    = (probs >= 0.5).astype(int)

    actual = y_test.values

    # Build per-game results.
    results_list: list[dict] = []
    for i in range(n_test):
        prob_h = float(probs[i])
        pred   = int(preds[i])
        act    = int(actual[i])
        results_list.append({
            "prob_home":       prob_h,
            "pred_home_win":   pred,
            "actual_home_win": act,
            "correct":         pred == act,
            "prob_predicted":  max(prob_h, 1 - prob_h),  # confidence
        })

    # ── Overall accuracy ───────────────────────────────────────────────────
    overall_acc = float(np.mean([r["correct"] for r in results_list]))

    # ── Baselines ─────────────────────────────────────────────────────────
    home_baseline_acc = float(np.mean(actual))  # always-pick-home win rate

    # ── Betting simulation ─────────────────────────────────────────────────
    betting = _betting_stats(results_list, flat_bet, confidence_threshold)
    betting["flat_bet"] = flat_bet

    # ── Season breakdown ──────────────────────────────────────────────────
    test_idx_pos = {idx: pos for pos, idx in enumerate(test_idx)}  # O(1) lookup
    season_stats: list[dict] = []
    if "SEASON" in matchup_df.columns:
        for season in test_seasons:
            sm = matchup_df["SEASON"].isin([season])
            sidx = matchup_df.index[sm].tolist()
            s_results = [
                results_list[test_idx_pos[i]] for i in sidx
                if i in test_idx_pos and test_idx_pos[i] < len(results_list)
            ]
            if s_results:
                season_stats.append({
                    "season":   season,
                    "n_games":  len(s_results),
                    "accuracy": float(np.mean([r["correct"] for r in s_results])),
                    "roi_pct":  _betting_stats(s_results, flat_bet, confidence_threshold)["roi_pct"],
                })

    return {
        "meta":                meta,
        "test_seasons":        test_seasons,
        "n_test_games":        n_test,
        "overall_accuracy":    overall_acc,
        "home_baseline_acc":   home_baseline_acc,
        "confidence_threshold": confidence_threshold,
        "betting":             betting,
        "season_stats":        season_stats,
        "results":             results_list,
    }


# ── Rich output ───────────────────────────────────────────────────────────────

def show_backtest_dashboard(bt: dict) -> None:
    """Render full backtest results as a Rich terminal dashboard."""
    if not bt:
        return

    meta     = bt.get("meta", {})
    betting  = bt.get("betting", {})
    model_name = meta.get("model_name", "Unknown")

    # ── Main summary panel ─────────────────────────────────────────────────
    acc        = bt["overall_accuracy"]
    home_base  = bt["home_baseline_acc"]
    threshold  = bt["confidence_threshold"]
    roi        = betting.get("roi_pct", 0.0)
    win_rate   = betting.get("win_rate", 0.0)
    n_bets     = betting.get("n_bets", 0)
    profit     = betting.get("total_profit", 0.0)

    acc_vs_home = acc - home_base

    summary = (
        f"[bold]Model:[/bold]             [cyan]{model_name}[/cyan]\n"
        f"[bold]Test seasons:[/bold]      {', '.join(bt['test_seasons'])}\n"
        f"[bold]Games evaluated:[/bold]   {bt['n_test_games']:,}\n"
        f"\n"
        f"[bold]Overall accuracy:[/bold]  [{('bold green' if acc >= 0.55 else 'yellow')}]{acc*100:.1f}%[/]\n"
        f"[bold]Home-pick baseline:[/bold] {home_base*100:.1f}%\n"
        f"[bold]vs Home baseline:[/bold]  [{('bold green' if acc_vs_home > 0 else 'red')}]{acc_vs_home*100:+.1f}%[/]\n"
        f"\n"
        f"[bold]Threshold:[/bold]         {threshold*100:.0f}% confidence\n"
        f"[bold]Bets placed:[/bold]       {n_bets:,}\n"
        f"[bold]Bet win rate:[/bold]      [{('bold green' if win_rate >= 0.55 else 'yellow')}]{win_rate*100:.1f}%[/]\n"
        f"[bold]Simulated ROI:[/bold]     [{('bold green' if roi > 0 else 'red')}]{roi:+.1f}%[/]\n"
        f"[bold]Simulated P&L:[/bold]     [{('bold green' if profit > 0 else 'red')}]${profit:+,.0f}[/] "
        f"(flat ${betting.get('flat_bet', 100):.0f}/bet)\n"
    )
    console.print(
        Panel(summary, title="[bold cyan]Backtest Results[/bold cyan]", expand=False)
    )

    # ── Season breakdown table ─────────────────────────────────────────────
    season_stats = bt.get("season_stats", [])
    if season_stats:
        table = Table(
            title="[bold cyan]Season Breakdown[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Season",   style="bold")
        table.add_column("Games",    justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Sim ROI",  justify="right")

        for ss in season_stats:
            sa     = ss["accuracy"]
            sroi   = ss["roi_pct"]
            a_col  = "bold green" if sa >= 0.55 else "yellow"
            r_col  = "bold green" if sroi > 0 else "red"
            table.add_row(
                ss["season"],
                str(ss["n_games"]),
                f"[{a_col}]{sa*100:.1f}%[/]",
                f"[{r_col}]{sroi:+.1f}%[/]",
            )
        console.print(table)

    # ── Confidence distribution ────────────────────────────────────────────
    results = bt.get("results", [])
    if results:
        bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
        console.print("\n[bold cyan]Accuracy by Confidence Band[/bold cyan]")
        for lo, hi in bins:
            bucket = [r for r in results if lo <= r["prob_predicted"] < hi]
            if not bucket:
                continue
            band_acc = float(np.mean([r["correct"] for r in bucket]))
            bar = "█" * int(band_acc * 40)
            pct_str = f"{lo*100:.0f}-{min(hi,1.0)*100:.0f}%"
            console.print(
                f"  {pct_str:>9}  [{len(bucket):4d} games]  "
                f"[{'green' if band_acc >= 0.55 else 'yellow'}]{bar}[/] "
                f"{band_acc*100:.1f}%"
            )
