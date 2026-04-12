"""
data/injuries.py — fetches NBA injury reports from ESPN's public API.
Falls back gracefully if the fetch fails so the rest of the pipeline continues.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from rich.console import Console

# ── Constants ────────────────────────────────────────────────────────────────
ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)
REQUEST_TIMEOUT = 10   # seconds

# Map ESPN status strings → canonical status
STATUS_MAP = {
    "out": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "day-to-day": "Questionable",
    "probable": "Probable",
    "active": "Active",
}

# Injury severity weights for the composite score
SEVERITY_WEIGHTS: dict[str, int] = {
    "Out": 3,
    "Doubtful": 2,
    "Questionable": 1,
    "Probable": 0,
    "Active": 0,
}

CACHE_DIR = Path(__file__).parent / "cache"
INJURIES_CACHE = CACHE_DIR / "injuries_today.csv"
CACHE_MAX_AGE_HOURS = 4   # injury reports go stale quickly

console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cache_fresh() -> bool:
    if not INJURIES_CACHE.exists():
        return False
    import time as _time
    age_hours = (_time.time() - INJURIES_CACHE.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _normalize_status(raw: str) -> str:
    return STATUS_MAP.get(raw.lower().strip(), "Unknown")


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_injuries(force: bool = False) -> pd.DataFrame:
    """
    Fetch today's injury report.
    Returns DataFrame: team_name, player_name, status, severity_score.
    Returns empty DataFrame on failure (pipeline continues without injury data).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _cache_fresh():
        return pd.read_csv(INJURIES_CACHE)

    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        console.print(
            f"[yellow]Warning: injury fetch failed ({exc}). "
            "Continuing without injury features.[/yellow]"
        )
        return pd.DataFrame(
            columns=["team_name", "player_name", "status", "severity_score"]
        )

    rows: list[dict] = []

    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("team", {}).get("displayName", "Unknown")
        for injury in team_entry.get("injuries", []):
            athlete = injury.get("athlete", {})
            player_name = athlete.get("displayName", "Unknown")
            raw_status = injury.get("status", "Unknown")
            status = _normalize_status(raw_status)
            rows.append(
                {
                    "team_name": team_name,
                    "player_name": player_name,
                    "status": status,
                    "severity_score": SEVERITY_WEIGHTS.get(status, 0),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(INJURIES_CACHE, index=False)
        console.print(
            f"[green]Injury report: {len(df)} players across "
            f"{df['team_name'].nunique()} teams.[/green]"
        )
    else:
        console.print("[yellow]Injury report returned no data.[/yellow]")

    return df


def get_team_injury_features(
    team_name: str,
    injuries_df: pd.DataFrame,
    top_player_minutes: Optional[list[str]] = None,
) -> dict:
    """
    Given a team name and the injury DataFrame, return injury feature dict:
      - star_player_out  : 1 if any top-8 minute player is Out
      - num_players_out  : count of Out players
      - injury_severity_score : weighted sum
    *top_player_minutes* is an optional list of player names ordered by minutes (most first).
    """
    if injuries_df.empty:
        return {
            "star_player_out": 0,
            "num_players_out": 0,
            "injury_severity_score": 0.0,
        }

    # Fuzzy-match team name inside the injury table
    team_mask = injuries_df["team_name"].str.lower().str.contains(
        team_name.lower().split()[-1],  # last word e.g. "Lakers"
        na=False,
    )
    team_injuries = injuries_df[team_mask]

    num_out = int((team_injuries["status"] == "Out").sum())
    severity_total = float(team_injuries["severity_score"].sum())

    star_out = 0
    if top_player_minutes:
        stars = set(p.lower() for p in top_player_minutes[:8])
        out_players = set(
            team_injuries.loc[team_injuries["status"] == "Out", "player_name"]
            .str.lower()
        )
        star_out = int(bool(stars & out_players))

    return {
        "star_player_out": star_out,
        "num_players_out": num_out,
        "injury_severity_score": severity_total,
    }
