"""
scheduler.py — daily auto-retrain at midnight ET.
Run: python scheduler.py            → Rich daemon UI
Run: python scheduler.py --headless → plain-log mode for GitHub Actions / CI
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import sys
from pathlib import Path

import pytz

# ── Constants ────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "retrain.log"
META_PATH = Path(__file__).parent / "models" / "saved" / "model_meta.json"
MODEL_PATH = Path(__file__).parent / "models" / "saved" / "best_model.pkl"
MODEL_BACKUP_PATH = MODEL_PATH.with_suffix(".pkl.bak")
CONFIG_PATH = Path(__file__).parent / "config.json"

ET_ZONE = pytz.timezone("America/New_York")

# NBA offseason: mid-June → mid-October
OFFSEASON_START_MONTH = 6
OFFSEASON_START_DAY = 20
OFFSEASON_END_MONTH = 10
OFFSEASON_END_DAY = 15

MIN_NEW_GAMES = 1


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(headless: bool) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nba_retrain")
    logger.setLevel(logging.INFO)

    # Guard against adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if headless:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ── Offseason check ───────────────────────────────────────────────────────────

def _is_offseason(today: datetime.date) -> bool:
    start = datetime.date(today.year, OFFSEASON_START_MONTH, OFFSEASON_START_DAY)
    end = datetime.date(today.year, OFFSEASON_END_MONTH, OFFSEASON_END_DAY)
    return start <= today <= end


# ── Meta helpers ──────────────────────────────────────────────────────────────

def _load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


# ── ET midnight helpers ───────────────────────────────────────────────────────

def _seconds_until_midnight_et() -> float:
    """Return seconds until next midnight ET, handling DST correctly."""
    now_et = datetime.datetime.now(ET_ZONE)
    next_midnight_et = (now_et + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (next_midnight_et - now_et).total_seconds()


# ── Core retrain logic ────────────────────────────────────────────────────────

def run_retrain_headless(headless: bool = True) -> int:
    """
    Run one retrain cycle.
    Returns 0 on success, 1 on failure.

    Model rollback: a backup of best_model.pkl is made before training.
    If the new model is worse, the backup is restored so we never silently
    persist a degraded model.
    """
    logger = _setup_logging(headless)
    today = datetime.date.today()

    def _log(msg: str, level: str = "info") -> None:
        getattr(logger, level)(msg)
        if not headless:
            from rich.console import Console
            Console().print(msg)

    _log(f"=== Retrain cycle started: {today.isoformat()} ===")

    if _is_offseason(today):
        _log("Offseason — no new games expected. Skipping retrain.", "warning")
        return 0

    try:
        # 1. Incremental fetch
        from data.collector import fetch_todays_games
        new_data = fetch_todays_games()

        n_new_games = len(new_data) // 2 if not new_data.empty else 0
        _log(f"New games fetched: {n_new_games}")

        if n_new_games < MIN_NEW_GAMES:
            _log("Not enough new games to warrant retraining. Skipping.", "warning")
            return 0

        # 2. Reload full cache
        import pandas as pd
        from data.collector import GAME_LOG_CACHE, build_matchup_dataframe
        game_logs = pd.read_csv(GAME_LOG_CACHE, parse_dates=["GAME_DATE"])
        matchup_df = build_matchup_dataframe(game_logs)

        # 3. Incremental ELO updates for new games only
        from features.elo import update_elo_incremental
        if not new_data.empty:
            home_new = new_data[~new_data["MATCHUP"].str.contains("@")]
            away_new = new_data[new_data["MATCHUP"].str.contains("@")]
            merged_new = pd.merge(
                home_new[["GAME_ID", "TEAM_NAME", "WL"]].add_prefix("H_"),
                away_new[["GAME_ID", "TEAM_NAME"]].add_prefix("A_"),
                left_on="H_GAME_ID", right_on="A_GAME_ID",
            )
            for _, row in merged_new.iterrows():
                update_elo_incremental(
                    row["H_TEAM_NAME"],
                    row["A_TEAM_NAME"],
                    home_won=(row["H_WL"] == "W"),
                )
            _log(f"ELO updated for {len(merged_new)} new games.")

        # 4. Full retrain
        from features.engineering import build_feature_matrix, validate_no_nan
        from features.elo import compute_elo_features
        from data.injuries import fetch_injuries
        from models.trainer import train_all_models

        matchup_df = compute_elo_features(matchup_df)
        injuries_df = fetch_injuries()

        X = build_feature_matrix(matchup_df, game_logs, injuries_df)
        X = validate_no_nan(X)
        y = matchup_df["home_win"].reset_index(drop=True)
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]

        cfg = _load_config()
        seasons = cfg.get("seasons", [])

        old_meta = _load_meta()
        old_accuracy = old_meta.get("accuracy", 0.0)

        # Back up current model before overwriting.
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, MODEL_BACKUP_PATH)

        # Headless: monkey-patch Console to suppress Rich output.
        # Use try/finally so the patch is always restored even on error.
        if headless:
            import io
            import rich.console as _rc
            _orig_console = _rc.Console

            def _silent_console(*a, **kw):
                return _orig_console(file=io.StringIO(), **(
                    {k: v for k, v in kw.items() if k != "file"}
                ))

            _rc.Console = _silent_console  # type: ignore[assignment]

        try:
            results = train_all_models(X, y, seasons=seasons)
        finally:
            if headless:
                _rc.Console = _orig_console  # type: ignore[assignment]

        best_name = max(results, key=lambda n: results[n]["roc_auc"])
        new_accuracy = results[best_name]["accuracy"]

        # 5. Keep or discard new model
        if new_accuracy >= old_accuracy:
            _log(
                f"Model improved or equal: {old_accuracy:.4f} → {new_accuracy:.4f}. "
                f"Saved {best_name}."
            )
            # Remove stale backup
            if MODEL_BACKUP_PATH.exists():
                MODEL_BACKUP_PATH.unlink()
        else:
            _log(
                f"New model WORSE: {new_accuracy:.4f} < {old_accuracy:.4f}. "
                "Restoring previous model.",
                "warning",
            )
            if MODEL_BACKUP_PATH.exists():
                shutil.copy2(MODEL_BACKUP_PATH, MODEL_PATH)
                MODEL_BACKUP_PATH.unlink()
                _log("Previous model restored from backup.")
            else:
                _log("No backup found — cannot restore previous model.", "error")

        _log(
            f"Retrain complete. Games added: {n_new_games}, "
            f"Old acc: {old_accuracy:.4f}, New acc: {new_accuracy:.4f}"
        )
        return 0

    except Exception as exc:
        _log(f"FATAL error during retrain: {exc}", "error")
        import traceback
        logger.error(traceback.format_exc())
        return 1


# ── Schedule daemon ───────────────────────────────────────────────────────────

def run_daemon() -> None:
    """
    Run the retrain scheduler until interrupted.

    Sleeps until the next midnight ET (DST-aware) before each run, which is
    more reliable than the `schedule` library's wall-clock-based approach.
    """
    import time
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print("[green]Retrain daemon running. Ctrl-C to stop.[/green]")

    try:
        while True:
            wait_secs = _seconds_until_midnight_et()
            next_et = datetime.datetime.now(ET_ZONE) + datetime.timedelta(seconds=wait_secs)
            console.print(
                f"[blue]Next retrain at {next_et.strftime('%Y-%m-%d %I:%M %p ET')} "
                f"(in {wait_secs / 3600:.1f}h)[/blue]"
            )
            time.sleep(wait_secs)
            run_retrain_headless(headless=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped.[/yellow]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Predictor auto-retrain scheduler")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run one retrain cycle with plain-text logging (for CI/GitHub Actions).",
    )
    args = parser.parse_args()

    if args.headless:
        sys.exit(run_retrain_headless(headless=True))
    else:
        run_daemon()


if __name__ == "__main__":
    main()
