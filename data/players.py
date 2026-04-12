"""
data/players.py — season-level player impact features from nba_api.

Uses LeagueDashPlayerStats (Base + Advanced) to compute per-team:
  top3_avg_pts        — sum of rolling pts for top-3 players by minutes
  star_usage_rate     — usage% of primary player
  depth_score         — top-5 pts share of total (lower = deeper bench)
  player_avail_score  — injury-adjusted aggregate contribution

Season-level stats are fetched once per season and cached.
Historical matchups are joined by (TEAM_NAME, SEASON).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nba_api.stats.endpoints import leaguedashplayerstats

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR    = Path(__file__).parent / "cache"
PLAYER_CACHE = CACHE_DIR / "player_stats.csv"
SEASONS      = [
    "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]

_RETRYABLE = (requests.RequestException, ConnectionError, TimeoutError, OSError)
API_RETRIES = 4
API_BACKOFF = 2.0

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_call(fn, *args, **kwargs):
    backoff = API_BACKOFF
    for attempt in range(1, API_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE as exc:
            if attempt == API_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2.0
        except Exception:
            raise


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_player_stats(force: bool = False) -> pd.DataFrame:
    """
    Fetch season-level per-game player stats (Base + Usage%) for all seasons.

    Columns returned:
        PLAYER_NAME, TEAM_NAME, SEASON,
        MIN, PTS, AST, REB, USG_PCT (when available)

    Returns empty DataFrame on error (callers handle gracefully).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and PLAYER_CACHE.exists():
        try:
            return pd.read_csv(PLAYER_CACHE)
        except Exception:
            pass

    all_rows: list[pd.DataFrame] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        task = prog.add_task("Fetching player stats…", total=len(SEASONS) * 2)

        for season in SEASONS:
            # ── Base stats ────────────────────────────────────────────────
            prog.update(task, description=f"Player base stats {season}…")
            try:
                base_res = _api_call(
                    leaguedashplayerstats.LeagueDashPlayerStats,
                    season=season,
                    per_mode_simple="PerGame",
                    season_type_all_star="Regular Season",
                )
                base = base_res.get_data_frames()[0]
                base["SEASON"] = season
                time.sleep(0.6)
            except Exception as exc:
                console.print(f"[yellow]Player base stats failed {season}: {exc}[/yellow]")
                prog.advance(task, 2)
                continue
            prog.advance(task)

            # ── Advanced stats (usage%) ───────────────────────────────────
            prog.update(task, description=f"Player adv stats {season}…")
            try:
                adv_res = _api_call(
                    leaguedashplayerstats.LeagueDashPlayerStats,
                    season=season,
                    measure_type_simple_display="Advanced",
                    per_mode_simple="PerGame",
                    season_type_all_star="Regular Season",
                )
                adv = adv_res.get_data_frames()[0][["PLAYER_ID", "USG_PCT"]].copy()
                time.sleep(0.6)
            except Exception:
                adv = pd.DataFrame(columns=["PLAYER_ID", "USG_PCT"])
            prog.advance(task)

            # Merge base + advanced
            if not adv.empty and "PLAYER_ID" in base.columns:
                merged = base.merge(adv, on="PLAYER_ID", how="left")
            else:
                merged = base
                merged["USG_PCT"] = 0.0

            all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Standardize TEAM_NAME: nba_api returns full names.
    keep_cols = [c for c in [
        "PLAYER_NAME", "PLAYER_ID", "TEAM_NAME", "SEASON",
        "GP", "MIN", "PTS", "AST", "REB", "STL", "BLK", "USG_PCT",
    ] if c in combined.columns]
    out = combined[keep_cols].dropna(subset=["TEAM_NAME"])

    # Some players switched teams — keep the row with most games.
    if "GP" in out.columns:
        out = (
            out.sort_values("GP", ascending=False)
            .drop_duplicates(subset=["PLAYER_ID", "SEASON"])
        )

    out.to_csv(PLAYER_CACHE, index=False)
    console.print(f"[green]Player stats: saved {len(out)} player-season rows.[/green]")
    return out


# ── Feature aggregation ───────────────────────────────────────────────────────

def build_player_team_features(
    player_stats: pd.DataFrame,
    team_name: str,
    season: str,
    injuries_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Aggregate player stats to team-level features for one team-season.

    injuries_df: if provided, applies injury weights to reduce contributions.
      Out = 100%, Doubtful = 75%, Questionable = 25% reduction.
    """
    _zero = {
        "top3_avg_pts":         0.0,
        "star_usage_rate":      0.0,
        "depth_score":          0.5,
        "player_avail_score":   0.0,
        "player_data_available": 0,
    }

    if player_stats is None or player_stats.empty:
        return _zero

    mask = (player_stats["TEAM_NAME"] == team_name) & (player_stats["SEASON"] == season)
    team_players = player_stats[mask].copy()
    if team_players.empty:
        return _zero

    # Sort by minutes played (proxy for role/importance).
    if "MIN" in team_players.columns:
        team_players = team_players.sort_values("MIN", ascending=False).reset_index(drop=True)

    # Build injury penalty map: player_name → keep_fraction
    injury_penalty: dict[str, float] = {}
    if injuries_df is not None and not injuries_df.empty:
        status_col = next(
            (c for c in ["STATUS", "status", "INJURY_STATUS"] if c in injuries_df.columns), None
        )
        name_col = next(
            (c for c in ["PLAYER_NAME", "name", "NAME"] if c in injuries_df.columns), None
        )
        if status_col and name_col:
            for _, row in injuries_df.iterrows():
                status = str(row[status_col]).lower()
                name   = str(row[name_col])
                if "out" in status:
                    injury_penalty[name] = 0.0
                elif "doubtful" in status:
                    injury_penalty[name] = 0.25
                elif "questionable" in status:
                    injury_penalty[name] = 0.75
                else:
                    injury_penalty[name] = 1.0

    pts_col    = "PTS"    if "PTS"    in team_players.columns else None
    usg_col    = "USG_PCT" if "USG_PCT" in team_players.columns else None

    if pts_col is None:
        return _zero

    pts_values = team_players[pts_col].fillna(0.0).values.tolist()
    player_names = team_players["PLAYER_NAME"].values.tolist() if "PLAYER_NAME" in team_players.columns else []

    # Apply injury adjustments.
    adj_pts: list[float] = []
    for j, pts in enumerate(pts_values):
        name   = player_names[j] if j < len(player_names) else ""
        factor = injury_penalty.get(name, 1.0)
        adj_pts.append(pts * factor)

    total_pts = sum(adj_pts)

    # Top-3 players by minutes — sum of their adjusted pts.
    top3_pts = sum(adj_pts[:3])
    top5_pts = sum(adj_pts[:5])

    # Star usage rate (primary player by minutes).
    star_usage = 0.0
    if usg_col and not team_players.empty:
        star_usage = float(team_players.iloc[0][usg_col]) if pd.notna(team_players.iloc[0][usg_col]) else 0.0

    # Depth score: how concentrated is scoring in top 5? Lower = deeper.
    depth = float(top5_pts / total_pts) if total_pts > 0 else 0.5

    # Player availability score: sum of all adjusted pts (higher = more stars available).
    avail = float(total_pts)

    return {
        "top3_avg_pts":         round(top3_pts, 2),
        "star_usage_rate":      round(star_usage, 4),
        "depth_score":          round(depth, 4),
        "player_avail_score":   round(avail, 2),
        "player_data_available": 1,
    }
