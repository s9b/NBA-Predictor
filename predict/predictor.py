"""
predict/predictor.py — loads saved model and predicts a single matchup.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Constants ────────────────────────────────────────────────────────────────
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
MODEL_PATH = SAVED_DIR / "best_model.pkl"
SCALER_PATH = SAVED_DIR / "scaler.pkl"
FEATURE_COLS_PATH = SAVED_DIR / "feature_columns.json"
META_PATH = SAVED_DIR / "model_meta.json"
BACKGROUND_PATH = SAVED_DIR / "background_sample.npy"

LOW_CONF_THRESHOLD  = 0.10   # |prob - 0.5| < 10% → Low
HIGH_CONF_THRESHOLD = 0.20   # |prob - 0.5| > 20% → High

TOP_SHAP = 5
LOG_DIR  = Path(__file__).parent.parent / "logs"
PENDING_PREDS_PATH = LOG_DIR / "pending_predictions.csv"

console = Console()
logger = logging.getLogger(__name__)

# Module-level rolling-stats cache keyed by number of rows in game_logs.
_rolling_cache: dict[int, dict] = {}


# ── Prediction logger (feed for post-game feedback loop) ─────────────────────

def _log_prediction(
    home_team: str,
    away_team: str,
    prob_home: float,
    confidence: str,
    is_final_week: int,
    end_of_season_risk: int,
) -> None:
    """Append today's prediction to pending_predictions.csv for later resolution."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{
            "date":               datetime.date.today().isoformat(),
            "home_team":          home_team,
            "away_team":          away_team,
            "prob_home":          round(prob_home, 4),
            "predicted_home_win": 1 if prob_home >= 0.5 else 0,
            "confidence":         confidence,
            "is_final_week":      is_final_week,
            "end_of_season_risk": end_of_season_risk,
            "resolved":           0,
        }])
        header = not PENDING_PREDS_PATH.exists()
        row.to_csv(PENDING_PREDS_PATH, mode="a", header=header, index=False)
    except Exception as exc:
        logger.debug("Could not log prediction: %s", exc)


# ── Schedule fetcher ──────────────────────────────────────────────────────────

def fetch_todays_schedule() -> list[dict]:
    """
    Fetch today's NBA games from ScoreboardV2.
    Returns list of {home_team, away_team, game_time, game_id}.
    Returns [] if API is unreachable or no games are scheduled.
    """
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams as nba_teams_static

    today_str = datetime.date.today().strftime("%m/%d/%Y")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        game_header = board.get_data_frames()[0]
    except Exception as exc:
        logger.warning("Could not fetch today's schedule: %s", exc)
        return []

    if game_header.empty:
        return []

    all_teams = nba_teams_static.get_teams()
    id_to_name = {t["id"]: t["full_name"] for t in all_teams}

    games: list[dict] = []
    for _, row in game_header.iterrows():
        home_id    = int(row.get("HOME_TEAM_ID",    0))
        visitor_id = int(row.get("VISITOR_TEAM_ID", 0))
        games.append({
            "home_team": id_to_name.get(home_id,    f"Team {home_id}"),
            "away_team": id_to_name.get(visitor_id, f"Team {visitor_id}"),
            "game_time": str(row.get("GAME_STATUS_TEXT", "TBD")),
            "game_id":   str(row.get("GAME_ID", "")),
        })
    return games


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_artifacts():
    """Return (model, scaler, feature_columns) or raise FileNotFoundError."""
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No saved model found. Please run option 2 (Train) first."
        )
    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def _confidence_label(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance < LOW_CONF_THRESHOLD:
        return "[yellow]Low[/yellow]"
    if distance > HIGH_CONF_THRESHOLD:
        return "[green]High[/green]"
    return "[cyan]Medium[/cyan]"


# ── Rolling stats cache ───────────────────────────────────────────────────────

def _get_rolling_stats(game_logs: pd.DataFrame) -> dict:
    from features.engineering import compute_team_rolling_stats
    key = len(game_logs)
    if key not in _rolling_cache:
        _rolling_cache[key] = compute_team_rolling_stats(game_logs)
    return _rolling_cache[key]


# ── Feature assembly for a single matchup ────────────────────────────────────

def _build_single_matchup_features(
    home_team: str,
    away_team: str,
    game_logs: pd.DataFrame,
    matchup_df: pd.DataFrame,
    feature_cols: list[str],
    injuries_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble feature vector for one hypothetical matchup using the most recent
    rolling stats for each team.  Returns a single-row DataFrame aligned to
    *feature_cols* (missing features filled with 0.0 — logged as a warning).
    """
    from features.engineering import (
        travel_distance_km,
        h2h_win_pct,
        MAX_REST_DAYS,
    )
    from features.elo import get_current_elo
    from data.injuries import get_team_injury_features

    today = pd.Timestamp.today().normalize()

    team_rolling = _get_rolling_stats(game_logs)

    def _latest_roll(team_name: str) -> dict:
        roll = team_rolling.get(team_name)
        if roll is None or roll.empty:
            return {}
        latest = roll.sort_values("GAME_DATE").iloc[-1]
        return latest.drop(["GAME_DATE", "GAME_ID", "TEAM_NAME"], errors="ignore").to_dict()

    h_stats = _latest_roll(home_team)
    a_stats = _latest_roll(away_team)

    feat: dict = {}
    for k, v in h_stats.items():
        feat[f"home_{k}"] = v
    for k, v in a_stats.items():
        feat[f"away_{k}"] = v
    for col in h_stats:
        if col in a_stats:
            feat[f"diff_{col}"] = h_stats[col] - a_stats[col]

    def _days_since_last_game(team_name: str) -> float:
        team_games = game_logs[game_logs["TEAM_NAME"] == team_name]
        if team_games.empty:
            return 3.0
        last = team_games["GAME_DATE"].max()
        return min(float((today - last).days), float(MAX_REST_DAYS))

    feat["home_days_rest"]  = _days_since_last_game(home_team)
    feat["away_days_rest"]  = _days_since_last_game(away_team)
    feat["rest_advantage"]  = feat["home_days_rest"] - feat["away_days_rest"]

    def _streak(team_name: str) -> float:
        grp = game_logs[game_logs["TEAM_NAME"] == team_name].sort_values("GAME_DATE")
        if grp.empty:
            return 0.0
        streak = 0.0
        for wl in grp["WL"].values:
            if wl == "W":
                streak = max(streak + 1, 1)
            else:
                streak = min(streak - 1, -1)
        return streak

    feat["home_win_streak"] = _streak(home_team)
    feat["away_win_streak"] = _streak(away_team)

    # H2H win pct
    h2h_mask = (
        (
            (matchup_df["HOME_TEAM_NAME"] == home_team)
            & (matchup_df["AWAY_TEAM_NAME"] == away_team)
        )
        | (
            (matchup_df["HOME_TEAM_NAME"] == away_team)
            & (matchup_df["AWAY_TEAM_NAME"] == home_team)
        )
    )
    feat["h2h_win_pct"]        = h2h_win_pct(
        matchup_df[h2h_mask].sort_values("GAME_DATE"), home_team, today
    )
    feat["travel_distance_km"] = travel_distance_km(away_team, home_team)

    # Opponent ELO strength (rolling 10-game avg of opponent pre-game ELO)
    if "home_elo" in matchup_df.columns and "away_elo" in matchup_df.columns:
        def _opp_elo_strength(team_name: str) -> float:
            team_games = matchup_df[
                (matchup_df["HOME_TEAM_NAME"] == team_name)
                | (matchup_df["AWAY_TEAM_NAME"] == team_name)
            ].sort_values("GAME_DATE")
            past = team_games[team_games["GAME_DATE"] < today].tail(10)
            if past.empty:
                return 1500.0
            opp_elos = [
                float(row["away_elo"]) if row["HOME_TEAM_NAME"] == team_name
                else float(row["home_elo"])
                for _, row in past.iterrows()
            ]
            return float(np.mean(opp_elos)) if opp_elos else 1500.0

        feat["home_opp_elo_last10"] = _opp_elo_strength(home_team)
        feat["away_opp_elo_last10"] = _opp_elo_strength(away_team)
        feat["diff_opp_elo_last10"] = feat["home_opp_elo_last10"] - feat["away_opp_elo_last10"]

    # Injuries
    if injuries_df is not None and not injuries_df.empty:
        h_inj = get_team_injury_features(home_team, injuries_df)
        a_inj = get_team_injury_features(away_team, injuries_df)
        for k, v in h_inj.items():
            feat[f"home_{k}"] = v
        for k, v in a_inj.items():
            feat[f"away_{k}"] = v
    else:
        for prefix in ("home_", "away_"):
            feat[f"{prefix}star_player_out"]       = 0
            feat[f"{prefix}num_players_out"]       = 0
            feat[f"{prefix}injury_severity_score"] = 0.0

    feat["home_elo"] = get_current_elo(home_team) + 100.0  # home advantage
    feat["away_elo"] = get_current_elo(away_team)
    feat["elo_diff"] = feat["home_elo"] - feat["away_elo"]

    missing = [col for col in feature_cols if col not in feat]
    if missing:
        console.print(
            f"[yellow]Warning: {len(missing)} features missing at inference "
            f"(filled with 0.0): {missing[:5]}{'…' if len(missing) > 5 else ''}[/yellow]"
        )
    row = {col: feat.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row])


# ── SHAP explanations ─────────────────────────────────────────────────────────

def _get_shap_contributions(
    model,
    X_row: np.ndarray,
    feature_cols: list[str],
) -> list[tuple[str, float]]:
    """
    Return top-5 (feature_name, shap_value) pairs.
    Falls back to feature_importances_ if SHAP is unavailable or model
    is a calibrated/stacking wrapper.
    """
    try:
        import shap

        # Unwrap CalibratedClassifierCV to get the base estimator.
        base = model
        if hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV stores a list of fitted calibrators.
            base = model.calibrated_classifiers_[0].estimator

        if hasattr(base, "feature_importances_"):
            explainer   = shap.TreeExplainer(base)
            shap_values = explainer.shap_values(X_row)
        elif BACKGROUND_PATH.exists() and not hasattr(base, "base_models"):
            background  = np.load(BACKGROUND_PATH)
            explainer   = shap.LinearExplainer(base, background)
            shap_values = explainer.shap_values(X_row)
        else:
            logger.debug("SHAP skipped (stacking or no background); using fallback.")
            return _importance_fallback(base, feature_cols)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        pairs = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)
        return pairs[:TOP_SHAP]

    except Exception as exc:
        logger.debug("SHAP failed (%s); falling back to feature_importances_.", exc)
        return _importance_fallback(model, feature_cols)


def _importance_fallback(model, feature_cols: list[str]) -> list[tuple[str, float]]:
    # Unwrap calibrated wrapper if needed.
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    if hasattr(base, "feature_importances_"):
        imp   = base.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: abs(x[1]), reverse=True)
        return pairs[:TOP_SHAP]
    return []


# ── Public predict function ───────────────────────────────────────────────────

def predict_game(
    home_team: str,
    away_team: str,
    game_logs: pd.DataFrame,
    matchup_df: pd.DataFrame,
    injuries_df: Optional[pd.DataFrame] = None,
    silent: bool = False,
) -> dict:
    """
    Predict a single game.
    Prints Rich output unless *silent=True*.
    Returns result dict regardless.
    """
    model, scaler, feature_cols = _load_artifacts()

    X = _build_single_matchup_features(
        home_team, away_team, game_logs, matchup_df, feature_cols, injuries_df
    )

    X_scaled  = scaler.transform(X)
    prob_home = float(model.predict_proba(X_scaled)[0][1])

    # ── Home court bias constraint ────────────────────────────────────────
    # If the model is too uncertain (45-55%), default to home team.
    # Historical NBA home win rate is ~59% — a coin-flip should lean home.
    if 0.45 <= prob_home <= 0.55:
        prob_home = 0.55

    prob_away = 1.0 - prob_home

    # ── End-of-season flags ───────────────────────────────────────────────
    is_final_week     = int(X["is_final_week"].iloc[0])     if "is_final_week"     in X.columns else 0
    end_of_season_risk = int(X["end_of_season_risk"].iloc[0]) if "end_of_season_risk" in X.columns else 0

    confidence    = _confidence_label(prob_home)
    contributions = _get_shap_contributions(model, X_scaled, feature_cols)

    # ── Log prediction for post-game feedback loop ────────────────────────
    _log_prediction(home_team, away_team, prob_home, confidence,
                    is_final_week, end_of_season_risk)

    # Injury warnings
    injury_warnings: list[str] = []
    if injuries_df is not None and not injuries_df.empty:
        from data.injuries import get_team_injury_features
        for team in (home_team, away_team):
            inj = get_team_injury_features(team, injuries_df)
            if inj["star_player_out"]:
                injury_warnings.append(f"[red]⚠  Star player OUT for {team}[/red]")
            if inj["num_players_out"] >= 3:
                injury_warnings.append(
                    f"[yellow]⚠  {inj['num_players_out']} players OUT for {team}[/yellow]"
                )

    if not silent:
        meta: dict = {}
        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)

        result_table = Table(box=box.SIMPLE_HEAD, show_header=False)
        result_table.add_column("", style="bold", min_width=25)
        result_table.add_column("Win Probability", justify="right", min_width=20)

        h_style = "bold green" if prob_home >= 0.5 else "white"
        a_style = "bold green" if prob_away > prob_home else "white"

        result_table.add_row(
            f"🏠  {home_team}",
            f"[{h_style}]{prob_home * 100:.1f}%[/{h_style}]",
        )
        result_table.add_row(
            f"✈   {away_team}",
            f"[{a_style}]{prob_away * 100:.1f}%[/{a_style}]",
        )

        console.print(
            Panel(
                result_table,
                title=f"[bold cyan]Game Prediction: {home_team} vs {away_team}[/bold cyan]",
                expand=False,
            )
        )
        console.print(
            f"  Model: [cyan]{meta.get('model_name', 'Unknown')}[/cyan]   "
            f"Confidence: {confidence}"
        )

        for w in injury_warnings:
            console.print(f"  {w}")

        if end_of_season_risk:
            console.print(
                "  [bold yellow]⚠  CAUTION: End of season — rotations unreliable. "
                "Stats may not reflect actual game plan.[/bold yellow]"
            )
        elif is_final_week:
            console.print(
                "  [yellow]ℹ  Final week of regular season — watch for rest decisions.[/yellow]"
            )

        if contributions:
            console.print("\n[bold]Key Factors (top 5 contributions):[/bold]")
            for feat, val in contributions:
                direction = "[green]▲[/green]" if val > 0 else "[red]▼[/red]"
                console.print(f"  {direction} [cyan]{feat}[/cyan]  ({val:+.4f})")

    return {
        "home_team":         home_team,
        "away_team":         away_team,
        "home_win_prob":     prob_home,
        "away_win_prob":     prob_away,
        "confidence":        confidence,
        "contributions":     contributions,
        "injury_warnings":   injury_warnings,
        "is_final_week":     is_final_week,
        "end_of_season_risk": end_of_season_risk,
    }
