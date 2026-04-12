"""
data/vegas.py — NBA odds fetcher via the-odds-api.com.

Free tier: 500 requests/month (~16/day).
Get API key: https://the-odds-api.com

Set key via:
  - env var:   ODDS_API_KEY=<key>
  - config:    add "odds_api_key": "<key>" to config.json

IMPORTANT: The free tier only provides CURRENT odds, not historical.
Vegas features in the training matrix will be zeros for historical games.
The model learns to use these signals only at prediction time.
"""

from __future__ import annotations

import json
import os
import time
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from rich.console import Console

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
VEGAS_CACHE = CACHE_DIR / "vegas_lines.csv"
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CACHE_MAX_AGE_HOURS = 4   # re-fetch mid-day to capture line movement

# Some bookmakers use slightly different team names — normalise to our standard.
_NAME_OVERRIDES: dict[str, str] = {
    "Los Angeles Clippers": "LA Clippers",
}

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("odds_api_key")
    return None


def _cache_fresh(max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    if not VEGAS_CACHE.exists():
        return False
    age = (time.time() - VEGAS_CACHE.stat().st_mtime) / 3600
    return age < max_age_hours


def _american_to_prob(ml: float) -> float:
    """American moneyline → raw implied probability (before vig removal)."""
    if ml == 0:
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def _normalize(name: str) -> str:
    return _NAME_OVERRIDES.get(name, name)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_nba_odds(force: bool = False) -> pd.DataFrame:
    """
    Fetch current NBA moneyline + spread + totals.

    Columns returned:
        game_date, home_team, away_team,
        home_spread, home_moneyline, away_moneyline,
        total_points_line,
        implied_home_win_prob, implied_away_win_prob,
        vegas_data_available

    Returns empty DataFrame if no API key or on any error.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _cache_fresh():
        try:
            return pd.read_csv(VEGAS_CACHE, parse_dates=["game_date"])
        except Exception as exc:
            console.print(f"[yellow]Stale/corrupt Vegas cache ({exc}), re-fetching…[/yellow]")

    api_key = _get_api_key()
    if not api_key:
        console.print(
            "[yellow]No Odds API key — Vegas features disabled.\n"
            "Set ODDS_API_KEY env var or add 'odds_api_key' to config.json.[/yellow]"
        )
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/odds",
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h,spreads,totals",
                "oddsFormat":  "american",
                "dateFormat":  "iso",
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        console.print(f"[yellow]Odds API error: {exc}. Vegas features disabled.[/yellow]")
        return pd.DataFrame()

    remaining = resp.headers.get("x-requests-remaining", "?")
    console.print(f"[dim]Odds API: {remaining} requests remaining this month.[/dim]")

    rows: list[dict] = []
    for game in resp.json():
        home_raw  = game.get("home_team",  "")
        away_raw  = game.get("away_team",  "")
        home_team = _normalize(home_raw)
        away_team = _normalize(away_raw)

        try:
            game_date = pd.Timestamp(game.get("commence_time", "")).date()
        except Exception:
            game_date = datetime.date.today()

        # Parse markets from first bookmaker that has all three.
        h_ml = away_ml = h_spread = total = None

        for bm in game.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h" and h_ml is None:
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home_raw:
                            h_ml = float(o["price"])
                        elif o["name"] == away_raw:
                            away_ml = float(o["price"])

                elif mkt["key"] == "spreads" and h_spread is None:
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home_raw:
                            h_spread = float(o.get("point", 0.0))

                elif mkt["key"] == "totals" and total is None:
                    for o in mkt.get("outcomes", []):
                        if o.get("name") == "Over":
                            total = float(o.get("point", 220.0))

            if h_ml is not None and h_spread is not None and total is not None:
                break

        if h_ml is None and h_spread is None:
            continue  # no odds posted yet

        # Remove vig: normalise to true probabilities.
        # Only meaningful when both sides are available (denom > 1.0 due to vig).
        p_h = _american_to_prob(h_ml)    if h_ml    is not None else 0.5
        p_a = _american_to_prob(away_ml) if away_ml is not None else None
        if p_a is not None:
            denom = p_h + p_a
            p_h /= denom
            p_a /= denom
        else:
            p_a = 1.0 - p_h

        rows.append({
            "game_date":             str(game_date),
            "home_team":             home_team,
            "away_team":             away_team,
            "home_spread":           round(h_spread if h_spread is not None else 0.0, 1),
            "home_moneyline":        int(h_ml)    if h_ml    is not None else 0,
            "away_moneyline":        int(away_ml) if away_ml is not None else 0,
            "total_points_line":     round(total if total is not None else 220.0, 1),
            "implied_home_win_prob": round(p_h, 4),
            "implied_away_win_prob": round(p_a, 4),
            "vegas_data_available":  1,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df.to_csv(VEGAS_CACHE, index=False)
    console.print(f"[green]Vegas: fetched odds for {len(df)} NBA games.[/green]")
    return df


def get_vegas_features(
    home_team: str,
    away_team: str,
    odds_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Return vegas feature dict for one matchup.
    Loads from cache if *odds_df* not provided.
    Returns all-zero dict if no data available.
    """
    _empty = {
        "home_spread":           0.0,
        "home_moneyline":        0,
        "total_points_line":     0.0,
        "implied_home_win_prob": 0.0,
        "implied_away_win_prob": 0.0,
        "vegas_data_available":  0,
    }

    if odds_df is None:
        if VEGAS_CACHE.exists():
            try:
                odds_df = pd.read_csv(VEGAS_CACHE, parse_dates=["game_date"])
            except Exception:
                return _empty
        else:
            return _empty

    if odds_df.empty:
        return _empty

    # Match by either ordering of home/away.
    mask_normal = (odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)
    mask_flip   = (odds_df["home_team"] == away_team) & (odds_df["away_team"] == home_team)
    row = odds_df[mask_normal | mask_flip]
    if row.empty:
        return _empty

    row = row.iloc[0]
    flipped = bool(row["home_team"] == away_team)

    return {
        "home_spread":           float(-row["home_spread"]         if flipped else row["home_spread"]),
        "home_moneyline":        int(row.get("away_moneyline", 0)  if flipped else row.get("home_moneyline", 0)),
        "total_points_line":     float(row["total_points_line"]),
        "implied_home_win_prob": float(row["implied_away_win_prob"] if flipped else row["implied_home_win_prob"]),
        "implied_away_win_prob": float(row["implied_home_win_prob"] if flipped else row["implied_away_win_prob"]),
        "vegas_data_available":  1,
    }
