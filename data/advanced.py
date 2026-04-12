"""
data/advanced.py — season-level advanced team stats via nba_api.

Uses LeagueDashTeamStats (measure_type=Advanced) which returns:
  ORtg, DRtg, NRtg, Pace, TS%, EFG%, OFF_RATING, DEF_RATING, NET_RATING

Also computes:
  MOV (margin of victory)   — derived from game_logs directly
  SOS (strength of schedule) — avg ELO of opponents in that season

Cached to data/cache/advanced_stats.csv (one row per team per season).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nba_api.stats.endpoints import leaguedashteamstats

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(__file__).parent / "cache"
ADV_CACHE   = CACHE_DIR / "advanced_stats.csv"
CONFIG_PATH = Path(__file__).parent.parent / "config.json"

SEASONS = [
    "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]

_RETRYABLE = (requests.RequestException, ConnectionError, TimeoutError, OSError)
API_RETRIES  = 4
API_BACKOFF  = 2.0

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_call_with_retry(fn, *args, **kwargs):
    backoff = API_BACKOFF
    for attempt in range(1, API_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE as exc:
            if attempt == API_RETRIES:
                raise
            console.print(f"[yellow]Advanced stats API retry {attempt}: {exc}[/yellow]")
            time.sleep(backoff)
            backoff *= 2.0
        except Exception:
            raise


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_advanced_stats(force: bool = False) -> pd.DataFrame:
    """
    Fetch per-team advanced stats for all seasons.

    Returns DataFrame with columns:
        TEAM_NAME, SEASON,
        adv_off_rating, adv_def_rating, adv_net_rating,
        adv_pace, adv_ts_pct, adv_efg_pct,
        adv_ast_pct, adv_reb_pct, adv_tov_pct

    Returns empty DataFrame on any error (callers handle gracefully).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and ADV_CACHE.exists():
        try:
            return pd.read_csv(ADV_CACHE)
        except Exception:
            pass

    cfg = _load_config()
    seasons = cfg.get("seasons", SEASONS)

    rows: list[dict] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        task = prog.add_task("Fetching advanced team stats…", total=len(seasons))
        for season in seasons:
            prog.update(task, description=f"Advanced stats {season}…")
            try:
                result = _api_call_with_retry(
                    leaguedashteamstats.LeagueDashTeamStats,
                    season=season,
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                    season_type_all_star="Regular Season",
                )
                df = result.get_data_frames()[0]
                df["SEASON"] = season
                rows.append(df)
                time.sleep(0.6)
            except Exception as exc:
                console.print(f"[yellow]Advanced stats failed for {season}: {exc}[/yellow]")
            prog.advance(task)

    if not rows:
        console.print("[yellow]No advanced stats fetched — features will be zero.[/yellow]")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    # Rename to our convention and select relevant columns.
    col_map = {
        "TEAM_NAME":   "TEAM_NAME",
        "SEASON":      "SEASON",
        "OFF_RATING":  "adv_off_rating",
        "DEF_RATING":  "adv_def_rating",
        "NET_RATING":  "adv_net_rating",
        "PACE":        "adv_pace",
        "TS_PCT":      "adv_ts_pct",
        "EFG_PCT":     "adv_efg_pct",
        "AST_PCT":     "adv_ast_pct",
        "REB_PCT":     "adv_reb_pct",
        "TM_TOV_PCT":  "adv_tov_pct",
    }
    keep = {k: v for k, v in col_map.items() if k in combined.columns}
    out = combined.rename(columns=keep)[list(keep.values())]
    out = out.drop_duplicates(subset=["TEAM_NAME", "SEASON"])

    out.to_csv(ADV_CACHE, index=False)
    console.print(f"[green]Advanced stats: saved {len(out)} team-season rows.[/green]")
    return out


def compute_mov_sos(
    game_logs: pd.DataFrame,
    elo_ratings: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute MOV (margin of victory) and SOS (schedule strength) from game_logs.

    MOV per team per season = mean(PTS - OPP_PTS) for all games.
    SOS per team per season = mean ELO of opponents that season (proxy).

    Returns DataFrame with columns: TEAM_NAME, SEASON, mov, sos
    """
    # Self-join to get opponent name (and PTS if missing) for every game row.
    opp_cols = {"TEAM_NAME": "OPP_TEAM_NAME"}
    if "OPP_PTS" not in game_logs.columns:
        opp_cols["PTS"] = "OPP_PTS"
    opp = game_logs[["GAME_ID", "TEAM_NAME"] + (["PTS"] if "OPP_PTS" not in game_logs.columns else [])].rename(columns=opp_cols)
    gl  = game_logs.merge(opp, on="GAME_ID", how="left")
    gl  = gl[gl["TEAM_NAME"] != gl["OPP_TEAM_NAME"]]

    if "SEASON" not in gl.columns:
        return pd.DataFrame()

    gl["point_margin"] = gl["PTS"] - gl["OPP_PTS"]
    mov_df = (
        gl.groupby(["TEAM_NAME", "SEASON"])["point_margin"]
        .mean()
        .reset_index()
        .rename(columns={"point_margin": "mov"})
    )

    # SOS: avg ELO of opponents each season.
    if elo_ratings:
        gl["opp_elo"] = gl["OPP_TEAM_NAME"].map(elo_ratings).fillna(1500.0)
        sos_df = (
            gl.groupby(["TEAM_NAME", "SEASON"])["opp_elo"]
            .mean()
            .reset_index()
            .rename(columns={"opp_elo": "sos"})
        )
        result = mov_df.merge(sos_df, on=["TEAM_NAME", "SEASON"], how="left")
    else:
        mov_df["sos"] = 1500.0
        result = mov_df

    return result


def get_advanced_features(
    team_name: str,
    season: str,
    advanced_df: pd.DataFrame,
    mov_sos_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Return advanced feature dict for one team-season. Zeros if missing."""
    _zero = {
        "adv_off_rating": 0.0, "adv_def_rating": 0.0, "adv_net_rating": 0.0,
        "adv_pace": 0.0, "adv_ts_pct": 0.0, "adv_efg_pct": 0.0,
        "adv_ast_pct": 0.0, "adv_reb_pct": 0.0, "adv_tov_pct": 0.0,
        "mov": 0.0, "sos": 1500.0,
    }

    if advanced_df is None or advanced_df.empty:
        return _zero

    mask = (advanced_df["TEAM_NAME"] == team_name) & (advanced_df["SEASON"] == season)
    row  = advanced_df[mask]
    result = dict(_zero)
    if not row.empty:
        for col in _zero:
            if col in row.columns:
                result[col] = float(row.iloc[0][col])

    if mov_sos_df is not None and not mov_sos_df.empty:
        ms = mov_sos_df[
            (mov_sos_df["TEAM_NAME"] == team_name) & (mov_sos_df["SEASON"] == season)
        ]
        if not ms.empty:
            result["mov"] = float(ms.iloc[0].get("mov", 0.0))
            result["sos"] = float(ms.iloc[0].get("sos", 1500.0))

    return result
