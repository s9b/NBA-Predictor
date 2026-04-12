"""
features/engineering.py — all pre-game feature creation.

No data leakage: all rolling stats use .shift(1) so current game is excluded.

Rolling variants per stat:
  roll5_*   — 5-game simple rolling mean (short-form)
  roll_*    — 10-game simple rolling mean (standard)
  roll20_*  — 20-game simple rolling mean (long-form)
  ewm10_*   — span-10 EWM (recent games weighted more)
  ewm30_*   — span-30 EWM (long-form with recency weighting)

EWM gives more credit to a team on a recent 5-game tear than to one
coasting on results from 3 weeks ago.

Optional feature sources (all fail gracefully if not provided):
  - vegas_df       : odds/spread/implied probability
  - players_df     : top3_avg_pts, star_usage_rate, depth_score
  - advanced_df    : ORtg, DRtg, NRtg, Pace (season level)
  - refs_df        : ref_home_win_pct, ref_foul_rate
  - schedule_feats : travel_km, timezone_shift, road_trip_length
"""

from __future__ import annotations

import collections
import math
from typing import Optional

import numpy as np
import pandas as pd
from data.injuries import get_team_injury_features

# ── Constants ────────────────────────────────────────────────────────────────
ROLLING_WINDOW       = 10
ROLLING_WINDOW_SHORT = 5
ROLLING_WINDOW_LONG  = 20
EWM_SPAN_SHORT       = 10
EWM_SPAN_LONG        = 30
MAX_REST_DAYS        = 7
MAX_DAYS_SINCE_WIN   = 30
H2H_GAMES            = 5

# NBA team city coordinates (lat, lon)
TEAM_COORDS: dict[str, tuple[float, float]] = {
    "Atlanta Hawks":           (33.7573, -84.3963),
    "Boston Celtics":          (42.3662, -71.0621),
    "Brooklyn Nets":           (40.6826, -73.9754),
    "Charlotte Hornets":       (35.2251, -80.8392),
    "Chicago Bulls":           (41.8807, -87.6742),
    "Cleveland Cavaliers":     (41.4965, -81.6882),
    "Dallas Mavericks":        (32.7905, -96.8103),
    "Denver Nuggets":          (39.7487, -105.0077),
    "Detroit Pistons":         (42.3410, -83.0554),
    "Golden State Warriors":   (37.7680, -122.3877),
    "Houston Rockets":         (29.7508, -95.3621),
    "Indiana Pacers":          (39.7640, -86.1555),
    "LA Clippers":             (34.0430, -118.2673),
    "Los Angeles Lakers":      (34.0430, -118.2673),
    "Memphis Grizzlies":       (35.1382, -90.0505),
    "Miami Heat":              (25.7814, -80.1870),
    "Milwaukee Bucks":         (43.0436, -87.9172),
    "Minnesota Timberwolves":  (44.9795, -93.2762),
    "New Orleans Pelicans":    (29.9490, -90.0823),
    "New York Knicks":         (40.7505, -73.9934),
    "Oklahoma City Thunder":   (35.4634, -97.5151),
    "Orlando Magic":           (28.5392, -81.3839),
    "Philadelphia 76ers":      (39.9012, -75.1720),
    "Phoenix Suns":            (33.4457, -112.0712),
    "Portland Trail Blazers":  (45.5316, -122.6668),
    "Sacramento Kings":        (38.5805, -121.4993),
    "San Antonio Spurs":       (29.4270, -98.4375),
    "Toronto Raptors":         (43.6435, -79.3791),
    "Utah Jazz":               (40.7683, -111.9011),
    "Washington Wizards":      (38.8981, -77.0209),
}

ROLLING_STAT_COLS = [
    "PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "AST", "REB", "TOV", "STL", "BLK", "PLUS_MINUS",
]
ADVANCED_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT",
]


# ── Travel distance ───────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + \
        math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def travel_distance_km(team_a: str, team_b: str) -> float:
    ca, cb = TEAM_COORDS.get(team_a), TEAM_COORDS.get(team_b)
    return _haversine_km(*ca, *cb) if ca and cb else 0.0


# ── Per-team rolling features ─────────────────────────────────────────────────

def compute_team_rolling_stats(game_logs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build a per-team DataFrame of all rolling/EWM stats (pre-game, no leakage).

    Returns {team_name: DataFrame} with columns:
      roll5_{s}, roll_{s}, roll20_{s}   — 5/10/20-game simple rolling mean
      ewm10_{s}, ewm30_{s}              — span-10/30 EWM
      win_pct_5/10/20                   — rolling win percentages
      games_last_7                      — fatigue: games in previous 7 days
      days_since_last_win               — momentum metric
      clutch_win_pct                    — win% in ≤5pt games (last 20)
      GAME_DATE, GAME_ID, TEAM_NAME     — metadata
    """
    if "OPP_PTS" not in game_logs.columns:
        opp = game_logs[["GAME_ID", "TEAM_NAME", "PTS"]].rename(
            columns={"TEAM_NAME": "OPP_TEAM_NAME", "PTS": "OPP_PTS"}
        )
        game_logs = game_logs.merge(opp, on="GAME_ID", how="left")
        game_logs = game_logs[game_logs["TEAM_NAME"] != game_logs["OPP_TEAM_NAME"]]

    stat_cols = [c for c in ROLLING_STAT_COLS if c in game_logs.columns]
    adv_cols  = [c for c in ADVANCED_COLS      if c in game_logs.columns]
    all_cols  = stat_cols + adv_cols
    has_pm    = "PLUS_MINUS" in game_logs.columns

    team_rolling: dict[str, pd.DataFrame] = {}

    for team_name, grp in game_logs.groupby("TEAM_NAME"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        n   = len(grp)

        shifted = grp[all_cols].shift(1)  # exclude current game

        # ── Simple rolling means ──────────────────────────────────────────
        r5   = shifted.rolling(ROLLING_WINDOW_SHORT, min_periods=1).mean()
        r10  = shifted.rolling(ROLLING_WINDOW,       min_periods=1).mean()
        r20  = shifted.rolling(ROLLING_WINDOW_LONG,  min_periods=1).mean()
        r5.columns   = [f"roll5_{c.lower()}"  for c in all_cols]
        r10.columns  = [f"roll_{c.lower()}"   for c in all_cols]
        r20.columns  = [f"roll20_{c.lower()}" for c in all_cols]

        # ── EWM (exponentially weighted means) ───────────────────────────
        ewm10 = shifted.ewm(span=EWM_SPAN_SHORT, adjust=False).mean()
        ewm30 = shifted.ewm(span=EWM_SPAN_LONG,  adjust=False).mean()
        ewm10.columns = [f"ewm10_{c.lower()}" for c in all_cols]
        ewm30.columns = [f"ewm30_{c.lower()}" for c in all_cols]

        # ── Win percentages ───────────────────────────────────────────────
        wl_bin = (grp["WL"] == "W").astype(float)
        sh_wl  = wl_bin.shift(1)
        wp5    = sh_wl.rolling(ROLLING_WINDOW_SHORT, min_periods=1).mean()
        wp10   = sh_wl.rolling(ROLLING_WINDOW,       min_periods=1).mean()
        wp20   = sh_wl.rolling(ROLLING_WINDOW_LONG,  min_periods=1).mean()

        # ── Fatigue: games in last 7 calendar days (O(n)) ─────────────────
        dates        = grp["GAME_DATE"].values
        games_last_7 = np.zeros(n)
        dq: collections.deque = collections.deque()
        for i in range(n):
            cutoff = dates[i] - np.timedelta64(7, "D")
            while dq and dq[0] <= cutoff:
                dq.popleft()
            games_last_7[i] = float(len(dq))
            dq.append(dates[i])

        # ── Days since last win (O(n)) ────────────────────────────────────
        wl_vals             = (grp["WL"] == "W").values
        days_since_last_win = np.full(n, float(MAX_DAYS_SINCE_WIN))
        last_win_i          = -1
        for i in range(n):
            if last_win_i >= 0:
                delta = float((dates[i] - dates[last_win_i]) / np.timedelta64(1, "D"))
                days_since_last_win[i] = min(delta, float(MAX_DAYS_SINCE_WIN))
            if wl_vals[i]:
                last_win_i = i

        # ── Clutch win % (games ≤5 pts, rolling 20) ──────────────────────
        if has_pm and "PLUS_MINUS" in grp.columns:
            is_clutch   = (grp["PLUS_MINUS"].abs() <= 5).astype(float)
            won_clutch  = ((grp["WL"] == "W") & (grp["PLUS_MINUS"].abs() <= 5)).astype(float)
            c_games     = is_clutch.shift(1).fillna(0).rolling(ROLLING_WINDOW_LONG, min_periods=1).sum()
            c_wins      = won_clutch.shift(1).fillna(0).rolling(ROLLING_WINDOW_LONG, min_periods=1).sum()
            clutch_pct  = np.where(c_games == 0, 0.5, c_wins / c_games)
        else:
            clutch_pct = np.full(n, 0.5)

        combined = pd.concat([r5, r10, r20, ewm10, ewm30], axis=1)
        combined["win_pct_5"]           = wp5.values
        combined["win_pct_10"]          = wp10.values
        combined["win_pct_20"]          = wp20.values
        combined["games_last_7"]        = games_last_7
        combined["days_since_last_win"] = days_since_last_win
        combined["clutch_win_pct"]      = clutch_pct
        combined["GAME_DATE"]           = grp["GAME_DATE"].values
        combined["GAME_ID"]             = grp["GAME_ID"].values
        combined["TEAM_NAME"]           = team_name

        team_rolling[team_name] = combined

    return team_rolling


def _win_streak(results: pd.Series) -> pd.Series:
    streaks, current = [], 0
    for wl in results.shift(1).fillna("W"):
        if wl == "W":
            current = max(current + 1, 1)
        elif wl == "L":
            current = min(current - 1, -1)
        else:
            current = 0
        streaks.append(current)
    return pd.Series(streaks, index=results.index)


def _get_roll_for_game(roll_df: pd.DataFrame, game_id) -> dict:
    if roll_df.empty:
        return {}
    m = roll_df[roll_df["GAME_ID"] == game_id]
    if m.empty:
        return {}
    return m.iloc[0].drop(["GAME_DATE", "GAME_ID", "TEAM_NAME"], errors="ignore").to_dict()


def h2h_win_pct(h2h_games: pd.DataFrame, home_team: str, before_date: pd.Timestamp) -> float:
    if h2h_games.empty:
        return 0.5
    recent = h2h_games[h2h_games["GAME_DATE"] < before_date].tail(H2H_GAMES)
    if recent.empty:
        return 0.5
    wins = (
        ((recent["HOME_TEAM_NAME"] == home_team) & (recent["home_win"] == 1))
        | ((recent["AWAY_TEAM_NAME"] == home_team) & (recent["home_win"] == 0))
    ).sum()
    return wins / len(recent)


# ── Master feature builder ────────────────────────────────────────────────────

def build_feature_matrix(
    matchup_df:      pd.DataFrame,
    game_logs:       pd.DataFrame,
    injuries_df:     Optional[pd.DataFrame] = None,
    vegas_df:        Optional[pd.DataFrame] = None,
    players_df:      Optional[pd.DataFrame] = None,
    advanced_df:     Optional[pd.DataFrame] = None,
    refs_df:         Optional[pd.DataFrame] = None,
    schedule_feats:  Optional[dict]         = None,
) -> pd.DataFrame:
    """
    Build the full pre-game feature matrix from *matchup_df*.

    All optional sources are joined gracefully — missing data fills with
    zeros/defaults so the model can still train without them.
    """
    team_rolling = compute_team_rolling_stats(game_logs)

    # ── Opponent ELO map (from enriched matchup_df) ───────────────────────
    opp_elo_map: dict[tuple, float] = {}
    if "home_elo" in matchup_df.columns and "away_elo" in matchup_df.columns:
        for _, row in matchup_df.iterrows():
            gid = row["GAME_ID"]
            opp_elo_map[(row["HOME_TEAM_NAME"], gid)] = float(row["away_elo"])
            opp_elo_map[(row["AWAY_TEAM_NAME"], gid)] = float(row["home_elo"])

    # ── Per-team: rest, streak, opp-ELO rolling ───────────────────────────
    rest_lookup:        dict[tuple, float] = {}
    streak_lookup:      dict[tuple, float] = {}
    opp_elo_rolling:    dict[tuple, float] = {}

    for team_name, grp in game_logs.groupby("TEAM_NAME"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        grp["days_rest"] = (
            grp["GAME_DATE"].diff().dt.days.clip(upper=MAX_REST_DAYS).shift(1).fillna(3)
        )
        streaks = _win_streak(grp["WL"])
        for i in range(len(grp)):
            key = (team_name, grp.at[i, "GAME_ID"])
            rest_lookup[key]   = float(grp.at[i, "days_rest"])
            streak_lookup[key] = float(streaks.iat[i])

        if opp_elo_map:
            opp_elos_seq = [opp_elo_map.get((team_name, gid), 1500.0) for gid in grp["GAME_ID"]]
            opp_elo_roll = (
                pd.Series(opp_elos_seq)
                .shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
            )
            for i, gid in enumerate(grp["GAME_ID"].values):
                opp_elo_rolling[(team_name, gid)] = float(opp_elo_roll.iloc[i])

    # ── H2H index ─────────────────────────────────────────────────────────
    h2h_index: dict[frozenset, pd.DataFrame] = {}
    for (h, a), grp in matchup_df.groupby(["HOME_TEAM_NAME", "AWAY_TEAM_NAME"]):
        key = frozenset([h, a])
        gs  = grp.sort_values("GAME_DATE")
        h2h_index[key] = pd.concat([h2h_index[key], gs]).sort_values("GAME_DATE") \
            if key in h2h_index else gs

    # ── Ref lookup (keyed by GAME_ID) ─────────────────────────────────────
    ref_lookup: dict[str, dict] = {}
    if refs_df is not None and not refs_df.empty and "GAME_ID" in refs_df.columns:
        for _, row in refs_df.iterrows():
            gid_str = str(row["GAME_ID"])
            ref_lookup[gid_str] = {
                c: float(row[c]) for c in row.index
                if c != "GAME_ID" and isinstance(row[c], (int, float))
            }

    # ── Advanced stats lookup (keyed by TEAM_NAME + SEASON) ───────────────
    adv_lookup: dict[tuple, dict] = {}
    if advanced_df is not None and not advanced_df.empty:
        for _, row in advanced_df.iterrows():
            key = (str(row.get("TEAM_NAME", "")), str(row.get("SEASON", "")))
            adv_lookup[key] = {c: float(row[c]) for c in row.index
                               if c not in ("TEAM_NAME", "SEASON") and isinstance(row[c], (int, float))}

    # ── Player stats lookup (keyed by TEAM_NAME + SEASON) ─────────────────
    player_lookup: dict[tuple, dict] = {}
    if players_df is not None and not players_df.empty:
        from data.players import build_player_team_features
        if "SEASON" in players_df.columns:
            for (team, season), _ in players_df.groupby(["TEAM_NAME", "SEASON"]) \
                    if {"TEAM_NAME", "SEASON"}.issubset(players_df.columns) else []:
                player_lookup[(team, season)] = build_player_team_features(
                    players_df, team, season, injuries_df
                )

    # ── Vegas lookup (keyed by GAME_ID or home+away+date) ─────────────────
    vegas_lookup: dict[tuple, dict] = {}
    if vegas_df is not None and not vegas_df.empty:
        from data.vegas import get_vegas_features
        # For each matchup we'll call get_vegas_features lazily below.
        pass

    has_season = "SEASON" in matchup_df.columns
    rows: list[dict] = []

    for _, game in matchup_df.iterrows():
        game_id   = game["GAME_ID"]
        home      = game["HOME_TEAM_NAME"]
        away      = game["AWAY_TEAM_NAME"]
        game_date = game["GAME_DATE"]
        season    = str(game.get("SEASON", "")) if has_season else ""

        h_stats = _get_roll_for_game(team_rolling.get(home, pd.DataFrame()), game_id)
        a_stats = _get_roll_for_game(team_rolling.get(away, pd.DataFrame()), game_id)

        feat: dict = {}
        for k, v in h_stats.items():
            feat[f"home_{k}"] = v
        for k, v in a_stats.items():
            feat[f"away_{k}"] = v
        for col in h_stats:
            if col in a_stats:
                feat[f"diff_{col}"] = h_stats[col] - a_stats[col]

        feat["home_days_rest"]  = rest_lookup.get((home, game_id), 3.0)
        feat["away_days_rest"]  = rest_lookup.get((away, game_id), 3.0)
        feat["rest_advantage"]  = feat["home_days_rest"] - feat["away_days_rest"]
        feat["home_win_streak"] = streak_lookup.get((home, game_id), 0.0)
        feat["away_win_streak"] = streak_lookup.get((away, game_id), 0.0)

        # Opponent ELO strength
        feat["home_opp_elo_last10"] = opp_elo_rolling.get((home, game_id), 1500.0)
        feat["away_opp_elo_last10"] = opp_elo_rolling.get((away, game_id), 1500.0)
        feat["diff_opp_elo_last10"] = feat["home_opp_elo_last10"] - feat["away_opp_elo_last10"]

        # H2H
        pair_key = frozenset([home, away])
        feat["h2h_win_pct"]        = h2h_win_pct(
            h2h_index.get(pair_key, pd.DataFrame()), home, game_date
        )
        feat["travel_distance_km"] = travel_distance_km(away, home)

        # ELO columns (set by compute_elo_features before this call)
        feat["home_elo"]          = game.get("home_elo",         1500.0)
        feat["away_elo"]          = game.get("away_elo",         1500.0)
        feat["elo_diff"]          = game.get("elo_diff",         0.0)
        feat["home_elo_home_r"]   = game.get("home_elo_home_r",  1500.0)
        feat["away_elo_away_r"]   = game.get("away_elo_away_r",  1500.0)
        feat["home_elo_recent"]   = game.get("home_elo_recent",  1500.0)
        feat["away_elo_recent"]   = game.get("away_elo_recent",  1500.0)
        feat["elo_recent_diff"]   = game.get("elo_recent_diff",  0.0)

        # Injuries
        if injuries_df is not None and not injuries_df.empty:
            h_inj = get_team_injury_features(home, injuries_df)
            a_inj = get_team_injury_features(away, injuries_df)
            for k, v in h_inj.items():
                feat[f"home_{k}"] = v
            for k, v in a_inj.items():
                feat[f"away_{k}"] = v
        else:
            for prefix in ("home_", "away_"):
                feat[f"{prefix}star_player_out"]       = 0
                feat[f"{prefix}num_players_out"]       = 0
                feat[f"{prefix}injury_severity_score"] = 0.0

        # Schedule / travel features
        if schedule_feats:
            for prefix, key in (("home_", (home, game_id)), ("away_", (away, game_id))):
                sf = schedule_feats.get(key, {})
                for fname, fval in sf.items():
                    feat[f"{prefix}{fname}"] = fval

        # Ref features
        if ref_lookup:
            gid_str = str(game_id)
            rf      = ref_lookup.get(gid_str, {})
            for k, v in rf.items():
                feat[k] = v
        else:
            feat["ref_home_win_pct"]   = 0.0
            feat["ref_foul_rate"]      = 0.0
            feat["ref_pace_tendency"]  = 0.0
            feat["ref_home_foul_bias"] = 1.0
            feat["ref_data_available"] = 0

        # Advanced season stats
        h_adv = adv_lookup.get((home, season), {})
        a_adv = adv_lookup.get((away, season), {})
        for k, v in h_adv.items():
            feat[f"home_{k}"] = v
        for k, v in a_adv.items():
            feat[f"away_{k}"] = v
        for col in h_adv:
            if col in a_adv:
                feat[f"diff_{col}"] = h_adv[col] - a_adv[col]

        # Player impact features
        h_pl = player_lookup.get((home, season), {})
        a_pl = player_lookup.get((away, season), {})
        for k, v in h_pl.items():
            feat[f"home_{k}"] = v
        for k, v in a_pl.items():
            feat[f"away_{k}"] = v
        if h_pl and a_pl:
            feat["diff_top3_avg_pts"]   = h_pl.get("top3_avg_pts", 0.0) - a_pl.get("top3_avg_pts", 0.0)
            feat["diff_player_avail"]   = h_pl.get("player_avail_score", 0.0) - a_pl.get("player_avail_score", 0.0)

        # Vegas features
        if vegas_df is not None and not vegas_df.empty:
            from data.vegas import get_vegas_features
            vf = get_vegas_features(home, away, vegas_df)
            for k, v in vf.items():
                feat[k] = v
        else:
            feat["home_spread"]           = 0.0
            feat["total_points_line"]     = 0.0
            feat["implied_home_win_prob"] = 0.0
            feat["implied_away_win_prob"] = 0.0
            feat["vegas_data_available"]  = 0

        rows.append(feat)

    features_df = pd.DataFrame(rows)
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[num_cols] = features_df[num_cols].apply(lambda c: c.fillna(c.median()))
    return features_df


def validate_no_nan(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing  = df[num_cols].isnull().sum()
    if missing.any():
        for col in missing[missing > 0].index:
            df[col] = df[col].fillna(df[col].median())
    return df
