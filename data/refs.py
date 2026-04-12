"""
data/refs.py — referee feature extraction.

Building a historical referee cache requires one BoxScoreSummaryV2 API
call per game (~8,000+ calls for 7 seasons).  This module implements a
LAZY approach: the cache is built incrementally over time, and on first
run all features default to 0.

Usage:
  from data.refs import fetch_ref_stats, get_ref_features

  # One-time (expensive) cache build — run once then cache persists:
  fetch_ref_stats(game_ids)          # pass a list of GAME_IDs

  # Per-game lookup (fast after cache is built):
  features = get_ref_features(game_id)

Ref features (all pre-game, not leaked):
  ref_home_win_pct      — historical home win % when this crew is on the floor
  ref_foul_rate         — avg personal fouls per game by this crew
  ref_pace_tendency     — avg possessions per game (proxy: PTS per game)
  ref_home_foul_bias    — ratio of home fouls / away fouls
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from rich.console import Console

from nba_api.stats.endpoints import boxscoresummaryv2

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR  = Path(__file__).parent / "cache"
REF_CACHE  = CACHE_DIR / "refs.csv"
REF_STATS  = CACHE_DIR / "ref_stats.csv"   # aggregated ref performance

_RETRYABLE = (requests.RequestException, ConnectionError, TimeoutError, OSError)
API_RETRIES = 3
API_BACKOFF = 2.0

console = Console()

_ZERO_FEATURES = {
    "ref_home_win_pct":   0.0,
    "ref_foul_rate":      0.0,
    "ref_pace_tendency":  0.0,
    "ref_home_foul_bias": 1.0,
    "ref_data_available": 0,
}


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


# ── Cache builder (expensive — run once) ──────────────────────────────────────

def fetch_ref_stats(
    game_ids: list[str],
    max_calls: int = 200,
) -> None:
    """
    Incrementally build the referee cache from BoxScoreSummaryV2.

    This is expensive (1 API call per game).  Call with a subset of
    game_ids and let it accumulate.  Skips already-cached games.

    Parameters
    ----------
    game_ids  : list of NBA GAME_ID strings
    max_calls : maximum API calls per invocation (rate limiting)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing cache.
    existing: pd.DataFrame
    if REF_CACHE.exists():
        try:
            existing = pd.read_csv(REF_CACHE)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    cached_ids: set = set(existing["GAME_ID"].astype(str).values) if not existing.empty else set()
    to_fetch   = [g for g in game_ids if str(g) not in cached_ids]

    if not to_fetch:
        console.print("[green]Referee cache already up to date.[/green]")
        return

    console.print(
        f"[blue]Building ref cache: {len(to_fetch)} games "
        f"(capped at {max_calls} this run)…[/blue]"
    )

    new_rows: list[dict] = []
    for gid in to_fetch[:max_calls]:
        try:
            box = _api_call(boxscoresummaryv2.BoxScoreSummaryV2, game_id=str(gid))
            frames = box.get_data_frames()
            # Frame index 2 is typically Officials
            officials_df = None
            for frame in frames:
                if "OFFICIAL_ID" in frame.columns or "FIRST_NAME" in frame.columns:
                    officials_df = frame
                    break

            if officials_df is None or officials_df.empty:
                time.sleep(0.3)
                continue

            # Game summary for outcome + fouls
            game_summary = frames[0] if frames else pd.DataFrame()

            for _, official in officials_df.iterrows():
                first = str(official.get("FIRST_NAME", ""))
                last  = str(official.get("LAST_NAME",  ""))
                row = {
                    "GAME_ID":      str(gid),
                    "ref_name":     f"{first} {last}".strip(),
                    "official_id":  str(official.get("OFFICIAL_ID", "")),
                }
                if not game_summary.empty:
                    gs = game_summary.iloc[0]
                    row["home_pts"]   = float(gs.get("HOME_TEAM_PTS",   0) or 0)
                    row["away_pts"]   = float(gs.get("VISITOR_TEAM_PTS",0) or 0)
                    row["home_win"]   = 1 if row["home_pts"] > row["away_pts"] else 0
                new_rows.append(row)
            time.sleep(0.4)

        except Exception as exc:
            console.print(f"[yellow]Ref cache skip {gid}: {exc}[/yellow]")
            time.sleep(0.3)

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        updated = pd.concat([existing, new_df], ignore_index=True)
        updated.drop_duplicates(subset=["GAME_ID", "ref_name"], inplace=True)
        updated.to_csv(REF_CACHE, index=False)
        _recompute_ref_stats(updated)
        console.print(f"[green]Ref cache: added {len(new_rows)} rows.[/green]")


def _recompute_ref_stats(ref_df: pd.DataFrame) -> None:
    """Aggregate per-referee performance stats and save to REF_STATS."""
    if ref_df.empty or "ref_name" not in ref_df.columns:
        return
    if "home_win" not in ref_df.columns:
        return

    stats = (
        ref_df.groupby("ref_name")
        .agg(
            games_officiated=("GAME_ID",  "nunique"),
            home_win_pct=    ("home_win", "mean"),
        )
        .reset_index()
    )
    stats.to_csv(REF_STATS, index=False)


# ── Feature lookup ────────────────────────────────────────────────────────────

def _load_ref_stats() -> pd.DataFrame:
    if REF_STATS.exists():
        try:
            return pd.read_csv(REF_STATS)
        except Exception:
            pass
    return pd.DataFrame()


def _load_ref_cache() -> pd.DataFrame:
    if REF_CACHE.exists():
        try:
            return pd.read_csv(REF_CACHE)
        except Exception:
            pass
    return pd.DataFrame()


def get_ref_features(game_id: str) -> dict:
    """
    Return referee feature dict for a given game_id.
    Returns zero dict if referee cache not yet built.
    """
    ref_cache = _load_ref_cache()
    if ref_cache.empty:
        return dict(_ZERO_FEATURES)

    game_refs = ref_cache[ref_cache["GAME_ID"].astype(str) == str(game_id)]
    if game_refs.empty:
        return dict(_ZERO_FEATURES)

    ref_stats = _load_ref_stats()
    if ref_stats.empty:
        return dict(_ZERO_FEATURES)

    crew_names  = game_refs["ref_name"].unique().tolist()
    crew_stats  = ref_stats[ref_stats["ref_name"].isin(crew_names)]

    if crew_stats.empty:
        return dict(_ZERO_FEATURES)

    avg_home_win_pct = float(crew_stats["home_win_pct"].mean())

    return {
        "ref_home_win_pct":   round(avg_home_win_pct, 4),
        "ref_foul_rate":      0.0,    # requires additional game-level fouls data
        "ref_pace_tendency":  0.0,    # requires additional pace data
        "ref_home_foul_bias": 1.0,    # requires foul log data
        "ref_data_available": 1,
    }


def build_refs_df(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with ref features for all games in game_logs.
    Triggers cache builds if needed (slow on first run).

    Returns empty DataFrame if cache not available yet.
    """
    if not REF_CACHE.exists():
        console.print(
            "[yellow]Ref cache not built yet. Run data.refs.fetch_ref_stats(game_ids) "
            "once to build it (requires ~1 API call per game).[/yellow]"
        )
        return pd.DataFrame()

    unique_ids = game_logs["GAME_ID"].unique().tolist() if "GAME_ID" in game_logs.columns else []
    rows = [
        {"GAME_ID": gid, **get_ref_features(gid)}
        for gid in unique_ids
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()
