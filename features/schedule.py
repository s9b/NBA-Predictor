"""
features/schedule.py — pre-game travel and schedule fatigue features.

Computed entirely from game_logs (no external API).

Features per (TEAM_NAME, GAME_ID):
  travel_km_last_game     — km traveled from previous game's city
  timezone_shift          — hour offset crossing (positive = West→East, harder)
  road_trip_length        — consecutive away games currently on
  days_since_home         — days elapsed since last home game
  schedule_diff_last7     — avg ELO of opponents faced in last 7 calendar days

West-to-East travel + early tip time is one of the most underrated edges
in NBA predictions. This module quantifies the first half of that edge.
"""

from __future__ import annotations

import math
import collections
import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Team coordinates (same as engineering.py, duplicated for module independence) ──
from features.engineering import TEAM_COORDS

# ── Timezone offsets (UTC, approximate for US mainland NBA cities) ────────────
# Using standard offsets (ET=-5, CT=-6, MT=-7, PT=-8).
# DST shifts both sides equally so relative shift stays consistent.
TEAM_TZ_OFFSET: dict[str, int] = {
    "Atlanta Hawks":           -5,
    "Boston Celtics":          -5,
    "Brooklyn Nets":           -5,
    "Charlotte Hornets":       -5,
    "Chicago Bulls":           -6,
    "Cleveland Cavaliers":     -5,
    "Dallas Mavericks":        -6,
    "Denver Nuggets":          -7,
    "Detroit Pistons":         -5,
    "Golden State Warriors":   -8,
    "Houston Rockets":         -6,
    "Indiana Pacers":          -5,
    "LA Clippers":             -8,
    "Los Angeles Lakers":      -8,
    "Memphis Grizzlies":       -6,
    "Miami Heat":              -5,
    "Milwaukee Bucks":         -6,
    "Minnesota Timberwolves":  -6,
    "New Orleans Pelicans":    -6,
    "New York Knicks":         -5,
    "Oklahoma City Thunder":   -6,
    "Orlando Magic":           -5,
    "Philadelphia 76ers":      -5,
    "Phoenix Suns":            -7,
    "Portland Trail Blazers":  -8,
    "Sacramento Kings":        -8,
    "San Antonio Spurs":       -6,
    "Toronto Raptors":         -5,
    "Utah Jazz":               -7,
    "Washington Wizards":      -5,
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _get_game_city(team_name: str, matchup: str, abbr_to_name: dict[str, str]) -> str:
    """
    Determine which city a team played in for a given game.
    Home game (no '@'): their own city.
    Away game ('@'): opponent's city (right side of '@').
    """
    if "@" not in matchup:
        return team_name  # home game
    parts = matchup.split("@")
    if len(parts) < 2:
        return team_name
    home_abbr = parts[1].strip().split()[0]  # first token after '@'
    return abbr_to_name.get(home_abbr, team_name)


def compute_schedule_features(
    game_logs: pd.DataFrame,
    elo_ratings: Optional[dict[str, float]] = None,
) -> dict[tuple, dict]:
    """
    Compute schedule/travel features for every (TEAM_NAME, GAME_ID) pair.

    Returns {(team_name, game_id): feature_dict}.
    Callers should use .get((team, game_id), {}) to access safely.

    Parameters
    ----------
    game_logs    : raw LeagueGameLog (TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP, TEAM_ABBREVIATION)
    elo_ratings  : {team_name: elo} for schedule_diff_last7 computation
    """
    if "MATCHUP" not in game_logs.columns:
        return {}

    # Build abbreviation → full name map from game_logs itself.
    abbr_to_name: dict[str, str] = {}
    if "TEAM_ABBREVIATION" in game_logs.columns:
        abbr_to_name = dict(
            zip(game_logs["TEAM_ABBREVIATION"], game_logs["TEAM_NAME"])
        )

    elo = elo_ratings or {}
    result: dict[tuple, dict] = {}

    for team_name, grp in game_logs.groupby("TEAM_NAME"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        n   = len(grp)

        dates    = grp["GAME_DATE"].values        # numpy datetime64
        game_ids = grp["GAME_ID"].values
        matchups = grp["MATCHUP"].values

        # Determine game city for each row.
        game_cities = [
            _get_game_city(team_name, matchups[i], abbr_to_name)
            for i in range(n)
        ]

        # Running state for O(1) road_trip and days_since_home per game.
        road_trip_running = [0]   # mutable container for closure-free update
        last_home_idx     = [-1]  # index of most recent home game

        for i in range(n):
            gid       = game_ids[i]
            curr_city = game_cities[i]
            curr_tz   = TEAM_TZ_OFFSET.get(curr_city, -5)

            # ── Travel from previous game ─────────────────────────────────
            if i == 0:
                travel_km  = 0.0
                tz_shift   = 0
            else:
                prev_city  = game_cities[i - 1]
                prev_tz    = TEAM_TZ_OFFSET.get(prev_city, -5)
                prev_coords = TEAM_COORDS.get(prev_city)
                curr_coords = TEAM_COORDS.get(curr_city)
                if prev_coords and curr_coords:
                    travel_km = _haversine_km(*prev_coords, *curr_coords)
                else:
                    travel_km = 0.0
                tz_shift = curr_tz - prev_tz  # positive = moved East (harder)

            # ── Road trip length (consecutive away games up to now) ───────
            # O(1): maintained incrementally via running counter
            road_trip = road_trip_running[0]

            # ── Days since last home game ─────────────────────────────────
            if last_home_idx[0] >= 0:
                delta = (dates[i] - dates[last_home_idx[0]]) / np.timedelta64(1, "D")
                days_since_home = int(delta)
            else:
                days_since_home = 0

            # ── Schedule difficulty last 7 days (avg opp ELO) ────────────
            if elo:
                cutoff7 = dates[i] - np.timedelta64(7, "D")
                opp_elos_7: list[float] = []
                for j in range(i - 1, -1, -1):
                    if dates[j] <= cutoff7:
                        break
                    # Extract opponent team name from matchup string.
                    # Away game: "TEAM @ OPP" → opponent is after '@'
                    # Home game: "TEAM vs. OPP" → opponent is after 'vs.'
                    m = matchups[j]
                    if "@" in m:
                        parts = m.split("@")
                        opp_name = abbr_to_name.get(parts[1].strip().split()[0], "")
                    else:
                        parts = m.split("vs.")
                        opp_name = abbr_to_name.get(parts[1].strip().split()[0], "") if len(parts) > 1 else ""
                    opp_elos_7.append(elo.get(opp_name, 1500.0))
                sched_diff_7 = float(np.mean(opp_elos_7)) if opp_elos_7 else 1500.0
            else:
                sched_diff_7 = 1500.0

            result[(team_name, gid)] = {
                "travel_km_last_game":   round(travel_km, 1),
                "timezone_shift":        tz_shift,
                "road_trip_length":      road_trip,
                "days_since_home":       days_since_home,
                "schedule_diff_last7":   round(sched_diff_7, 1),
            }

            # Update running state AFTER recording pre-game values.
            if "@" in matchups[i]:
                road_trip_running[0] += 1
            else:
                road_trip_running[0] = 0
                last_home_idx[0] = i

    return result
