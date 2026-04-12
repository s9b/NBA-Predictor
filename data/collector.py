"""
data/collector.py — pulls NBA game data via nba_api with caching and rate-limit retries.
"""

from __future__ import annotations

import json
import time
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as nba_teams_static

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
SEASONS = [
    "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]
GAME_LOG_CACHE = CACHE_DIR / "game_logs.csv"
CACHE_MAX_AGE_HOURS = 24
API_RETRY_ATTEMPTS = 5
API_INITIAL_BACKOFF = 2.0   # seconds
API_BACKOFF_FACTOR = 2.0
# Errors that warrant a retry (rate-limits, transient network failures).
# Programming errors (AttributeError, KeyError) are NOT retried.
_RETRYABLE = (requests.RequestException, ConnectionError, TimeoutError, OSError)

console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cache_fresh(path: Path, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    """Return True if *path* exists and was modified within *max_age_hours*."""
    if not path.exists():
        return False
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.datetime.now() - mtime
    return age.total_seconds() < max_age_hours * 3600


def _api_call_with_retry(fn, *args, **kwargs):
    """Call *fn* with exponential back-off on retryable network/rate-limit errors."""
    backoff = API_INITIAL_BACKOFF
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE as exc:
            if attempt == API_RETRY_ATTEMPTS:
                raise
            console.print(
                f"[yellow]API error (attempt {attempt}/{API_RETRY_ATTEMPTS}): "
                f"{exc}. Retrying in {backoff:.1f}s…[/yellow]"
            )
            time.sleep(backoff)
            backoff *= API_BACKOFF_FACTOR
        except Exception:
            # Non-retryable (programming error) — re-raise immediately.
            raise


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Core fetch ───────────────────────────────────────────────────────────────

def fetch_all_game_logs(force: bool = False) -> pd.DataFrame:
    """
    Pull LeagueGameLog for every season in SEASONS.
    Returns a unified DataFrame with one row per team-game.
    Caches result to GAME_LOG_CACHE.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _cache_fresh(GAME_LOG_CACHE):
        console.print("[green]Game log cache is fresh — loading from disk.[/green]")
        return pd.read_csv(GAME_LOG_CACHE, parse_dates=["GAME_DATE"])

    all_frames: list[pd.DataFrame] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching game logs…", total=len(SEASONS))

        for season in SEASONS:
            progress.update(task, description=f"Season {season}")
            result = _api_call_with_retry(
                leaguegamelog.LeagueGameLog,
                season=season,
                season_type_all_star="Regular Season",
                direction="ASC",
            )
            df = result.get_data_frames()[0]
            df["SEASON"] = season
            all_frames.append(df)
            time.sleep(0.6)   # be polite to the API
            progress.advance(task)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
    combined.sort_values("GAME_DATE", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(GAME_LOG_CACHE, index=False)
    console.print(f"[green]Saved {len(combined):,} team-game rows to cache.[/green]")

    cfg = _load_config()
    cfg["last_fetch_date"] = datetime.date.today().isoformat()
    _save_config(cfg)

    return combined


def fetch_todays_games() -> pd.DataFrame:
    """
    Incremental fetch: pull only games played since last_fetch_date.
    Appends to GAME_LOG_CACHE without re-pulling historical data.
    Returns new rows (may be empty DataFrame).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    # Use the earliest season start as a safe default so we never silently
    # skip games when the config has never been written.
    last_date_str = cfg.get("last_fetch_date") or f"{SEASONS[0][:4]}-10-01"
    last_date = datetime.date.fromisoformat(last_date_str)
    today = datetime.date.today()

    if last_date >= today:
        console.print("[yellow]Already up to date — no new games to fetch.[/yellow]")
        return pd.DataFrame()

    console.print(f"[blue]Fetching incremental games from {last_date} to {today}…[/blue]")

    def _date_to_season(d: datetime.date) -> str:
        y = d.year
        return f"{y}-{str(y + 1)[-2:]}" if d.month >= 10 else f"{y - 1}-{str(y)[-2:]}"

    seasons_needed = {_date_to_season(last_date), _date_to_season(today)}
    new_frames: list[pd.DataFrame] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Fetching new games…", total=len(seasons_needed))
        for season in seasons_needed:
            result = _api_call_with_retry(
                leaguegamelog.LeagueGameLog,
                season=season,
                season_type_all_star="Regular Season",
                direction="ASC",
            )
            df = result.get_data_frames()[0]
            df["SEASON"] = season
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            new_frames.append(df[df["GAME_DATE"].dt.date > last_date])
            time.sleep(0.6)
            p.advance(task)

    new_data = pd.concat(new_frames, ignore_index=True)

    if new_data.empty:
        console.print("[yellow]No new games found.[/yellow]")
        return new_data

    if GAME_LOG_CACHE.exists():
        existing = pd.read_csv(GAME_LOG_CACHE, parse_dates=["GAME_DATE"])
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], inplace=True)
        combined.sort_values("GAME_DATE", inplace=True)
        combined.to_csv(GAME_LOG_CACHE, index=False)
    else:
        new_data.to_csv(GAME_LOG_CACHE, index=False)

    cfg["last_fetch_date"] = today.isoformat()
    _save_config(cfg)

    console.print(f"[green]Appended {len(new_data)} new team-game rows.[/green]")
    return new_data


def build_matchup_dataframe(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-game log (one row per team per game) into matchup rows
    (one row per game: home team vs away team).
    NBA API encodes home team with no '@' in MATCHUP, away with '@'.
    """
    home = game_logs[~game_logs["MATCHUP"].str.contains("@")].copy()
    away = game_logs[game_logs["MATCHUP"].str.contains("@")].copy()

    home = home.add_prefix("HOME_")
    away = away.add_prefix("AWAY_")

    merged = pd.merge(
        home,
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        suffixes=("", "_dup"),
    )

    merged["home_win"] = (merged["HOME_WL"] == "W").astype(int)
    merged["GAME_DATE"] = merged["HOME_GAME_DATE"]
    merged["SEASON"] = merged["HOME_SEASON"]
    merged["GAME_ID"] = merged["HOME_GAME_ID"]

    return merged.sort_values("GAME_DATE").reset_index(drop=True)


def get_all_teams() -> list[dict]:
    """Return list of all 30 NBA teams from static data."""
    return nba_teams_static.get_teams()


def get_team_id(team_name: str) -> Optional[int]:
    """Fuzzy-ish lookup: return team_id for a team_name substring match."""
    teams = nba_teams_static.get_teams()
    name_lower = team_name.lower()
    for t in teams:
        if (
            name_lower in t["full_name"].lower()
            or name_lower in t["nickname"].lower()
            or name_lower in t["abbreviation"].lower()
            or name_lower in t["city"].lower()
        ):
            return t["id"]
    return None
