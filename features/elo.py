"""
features/elo.py — CARMELO-lite ELO system.

Four ELO variants tracked per team:
  elo          — overall (all games, MOV-scaled K)
  home_elo_r   — home-game specific (updated only on home games)
  away_elo_r   — away-game specific (updated only on away games)
  elo_recent   — recent form (K = 2× base for games in last 30 days)

Season reversion between seasons:
  new_season_elo = prev_elo * 0.75 + 1505 * 0.25

All variants stored in config.json under separate keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
STARTING_ELO    = 1500.0
MEAN_REVERSION  = 1505.0   # slightly above 1500 (slight mean inflation)
REVERSION_RATE  = 0.75     # keep 75% of prior ELO across season break
K_FACTOR        = 20.0
K_RECENT_MULT   = 2.0      # recent games (last 30 days) get 2× K
RECENT_DAYS     = 30
HOME_ADVANTAGE  = 100.0    # added to home ELO in every match computation
CONFIG_PATH     = Path(__file__).parent.parent / "config.json"


# ── ELO math ─────────────────────────────────────────────────────────────────

def expected_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _k(point_diff: float, k_mult: float = 1.0) -> float:
    """K scaled with log(MOV+1) × optional recency multiplier."""
    return K_FACTOR * k_mult * math.log(abs(point_diff) + 1)


def _update(elo_w: float, elo_l: float, point_diff: float, k_mult: float = 1.0):
    """Return (new_winner_elo, new_loser_elo)."""
    exp = expected_win_prob(elo_w, elo_l)
    k   = _k(point_diff, k_mult)
    return elo_w + k * (1.0 - exp), elo_l + k * (0.0 - (1.0 - exp))


# ── State management ─────────────────────────────────────────────────────────

_STATE_KEYS = ("elo_ratings", "elo_home_ratings", "elo_away_ratings", "elo_recent_ratings")


def _load_all() -> dict[str, dict[str, float]]:
    if not CONFIG_PATH.exists():
        return {k: {} for k in _STATE_KEYS}
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return {k: cfg.get(k, {}) for k in _STATE_KEYS}


def _save_all(state: dict[str, dict[str, float]]) -> None:
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    for k, v in state.items():
        cfg[k] = v
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _get(state: dict[str, float], team: str) -> float:
    return state.get(team, STARTING_ELO)


# ── Season reversion ──────────────────────────────────────────────────────────

def _revert(ratings: dict[str, float]) -> dict[str, float]:
    """Apply mean reversion for a new season."""
    return {
        team: elo * REVERSION_RATE + MEAN_REVERSION * (1.0 - REVERSION_RATE)
        for team, elo in ratings.items()
    }


# ── Main computation ──────────────────────────────────────────────────────────

def compute_elo_features(
    matchup_df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Add 8 ELO columns to *matchup_df* (sorted chronologically):
      home_elo, away_elo, elo_diff            — overall
      home_elo_home_r, away_elo_away_r        — home/away-specific
      home_elo_recent, away_elo_recent        — recent-form
      elo_recent_diff

    Uses margin-of-victory K scaling.  Applies season reversion at
    each season boundary.  Persists all four ELO variants to config.json.

    Parameters
    ----------
    matchup_df     : game records; see column requirements below
    reference_date : anchor for the recency multiplier window.
                     - Backfill / training: pass None (default) — uses the
                       last game date in the dataset so the final RECENT_DAYS
                       games get 2× K regardless of calendar date.
                     - Live prediction: pass pd.Timestamp.today() so only
                       genuinely recent games get the boost.

    Expected columns: HOME_TEAM_NAME, AWAY_TEAM_NAME, home_win, GAME_DATE
    Optional columns: HOME_PTS, AWAY_PTS, SEASON
    """
    overall: dict[str, float]  = {}
    home_r:  dict[str, float]  = {}
    away_r:  dict[str, float]  = {}
    recent:  dict[str, float]  = {}

    home_elos_out:        list[float] = []
    away_elos_out:        list[float] = []
    home_elos_home_r_out: list[float] = []
    away_elos_away_r_out: list[float] = []
    home_elos_recent_out: list[float] = []
    away_elos_recent_out: list[float] = []

    df        = matchup_df.sort_values("GAME_DATE").reset_index(drop=True)
    has_pts   = "HOME_PTS" in df.columns and "AWAY_PTS" in df.columns
    has_season = "SEASON"   in df.columns

    prev_season: str = ""
    if reference_date is not None:
        last_date = reference_date
    else:
        last_date = pd.Timestamp(df["GAME_DATE"].max()) if len(df) else pd.Timestamp.today()

    for _, row in df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]
        game_date = pd.Timestamp(row["GAME_DATE"])

        # ── Season reversion ──────────────────────────────────────────────
        if has_season:
            curr_season = str(row["SEASON"])
            if prev_season and curr_season != prev_season:
                overall = _revert(overall)
                home_r  = _revert(home_r)
                away_r  = _revert(away_r)
                recent  = _revert(recent)
            prev_season = curr_season

        # ── Pre-game ELO snapshots ────────────────────────────────────────
        h_ov   = _get(overall, home_team)
        a_ov   = _get(overall, away_team)
        h_hr   = _get(home_r,  home_team)
        a_ar   = _get(away_r,  away_team)
        h_rec  = _get(recent,  home_team)
        a_rec  = _get(recent,  away_team)

        home_elos_out.append(h_ov + HOME_ADVANTAGE)
        away_elos_out.append(a_ov)
        home_elos_home_r_out.append(h_hr + HOME_ADVANTAGE)
        away_elos_away_r_out.append(a_ar)
        home_elos_recent_out.append(h_rec + HOME_ADVANTAGE)
        away_elos_recent_out.append(a_rec)

        # ── Point differential ────────────────────────────────────────────
        if has_pts:
            hp = float(row["HOME_PTS"]) if pd.notna(row["HOME_PTS"]) else 0.0
            ap = float(row["AWAY_PTS"]) if pd.notna(row["AWAY_PTS"]) else 0.0
            point_diff = max(abs(hp - ap), 1.0)
        else:
            point_diff = 10.0

        # ── Recency multiplier ────────────────────────────────────────────
        days_ago  = (last_date - game_date).days
        k_mult    = K_RECENT_MULT if days_ago <= RECENT_DAYS else 1.0

        # ── ELO updates ───────────────────────────────────────────────────
        home_won = int(row["home_win"]) == 1

        if home_won:
            # overall
            overall[home_team], overall[away_team] = _update(h_ov, a_ov, point_diff)
            # home/away specific — use opponent's overall ELO as reference strength
            home_r[home_team],  _                  = _update(h_hr, a_ov, point_diff)
            _,                  away_r[away_team]  = _update(h_ov, a_ar, point_diff)
            # recent
            recent[home_team],  recent[away_team]  = _update(h_rec, a_rec, point_diff, k_mult)
        else:
            overall[away_team], overall[home_team] = _update(a_ov, h_ov, point_diff)
            away_r[away_team],  _                  = _update(a_ar, h_ov, point_diff)
            _,                  home_r[home_team]  = _update(a_ov, h_hr, point_diff)
            recent[away_team],  recent[home_team]  = _update(a_rec, h_rec, point_diff, k_mult)

    df["home_elo"]          = home_elos_out
    df["away_elo"]          = away_elos_out
    df["elo_diff"]          = df["home_elo"] - df["away_elo"]
    df["home_elo_home_r"]   = home_elos_home_r_out
    df["away_elo_away_r"]   = away_elos_away_r_out
    df["home_elo_recent"]   = home_elos_recent_out
    df["away_elo_recent"]   = away_elos_recent_out
    df["elo_recent_diff"]   = df["home_elo_recent"] - df["away_elo_recent"]

    _save_all({
        "elo_ratings":        overall,
        "elo_home_ratings":   home_r,
        "elo_away_ratings":   away_r,
        "elo_recent_ratings": recent,
    })
    return df


def get_current_elo(team_name: str) -> float:
    state = _load_all()
    return state["elo_ratings"].get(team_name, STARTING_ELO)


def get_all_elo_ratings() -> dict[str, float]:
    return _load_all()["elo_ratings"]


def get_elo_variants(team_name: str) -> dict[str, float]:
    """Return all four ELO variants for one team."""
    state = _load_all()
    return {
        "elo":          state["elo_ratings"].get(team_name, STARTING_ELO),
        "home_elo_r":   state["elo_home_ratings"].get(team_name, STARTING_ELO),
        "away_elo_r":   state["elo_away_ratings"].get(team_name, STARTING_ELO),
        "elo_recent":   state["elo_recent_ratings"].get(team_name, STARTING_ELO),
    }


def update_elo_incremental(
    home_team: str,
    away_team: str,
    home_won: bool,
    point_diff: float = 10.0,
) -> tuple[float, float]:
    """
    Update all four ELO variants for a single new game.
    Returns (new_home_elo, new_away_elo) for the overall variant.
    """
    state = _load_all()
    ov    = state["elo_ratings"]
    hr    = state["elo_home_ratings"]
    ar    = state["elo_away_ratings"]
    rec   = state["elo_recent_ratings"]

    h_ov  = _get(ov,  home_team); a_ov  = _get(ov,  away_team)
    h_hr  = _get(hr,  home_team); a_ar  = _get(ar,  away_team)
    h_rec = _get(rec, home_team); a_rec = _get(rec, away_team)

    pd_capped = max(abs(point_diff), 1.0)

    if home_won:
        ov[home_team], ov[away_team]  = _update(h_ov, a_ov, pd_capped, K_RECENT_MULT)
        hr[home_team], _              = _update(h_hr, a_ar, pd_capped, K_RECENT_MULT)
        _,             ar[away_team]  = _update(h_hr, a_ar, pd_capped, K_RECENT_MULT)
        rec[home_team], rec[away_team]= _update(h_rec, a_rec, pd_capped, K_RECENT_MULT)
    else:
        ov[away_team], ov[home_team]  = _update(a_ov, h_ov, pd_capped, K_RECENT_MULT)
        ar[away_team], _              = _update(a_ar, h_hr, pd_capped, K_RECENT_MULT)
        _,             hr[home_team]  = _update(a_ar, h_hr, pd_capped, K_RECENT_MULT)
        rec[away_team], rec[home_team]= _update(a_rec, h_rec, pd_capped, K_RECENT_MULT)

    _save_all({
        "elo_ratings":        ov,
        "elo_home_ratings":   hr,
        "elo_away_ratings":   ar,
        "elo_recent_ratings": rec,
    })
    return ov[home_team], ov[away_team]
