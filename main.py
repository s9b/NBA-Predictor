"""
main.py — Rich-powered interactive terminal menu for the NBA Predictor.
"""

from __future__ import annotations

import json
import sys
import threading
import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box
from thefuzz import process as fuzzy_process

# ── Constants ────────────────────────────────────────────────────────────────
SAVED_DIR = Path(__file__).parent / "models" / "saved"
MODEL_PATH = SAVED_DIR / "best_model.pkl"
META_PATH = SAVED_DIR / "model_meta.json"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
GAME_LOG_CACHE = CACHE_DIR / "game_logs.csv"
CONFIG_PATH = Path(__file__).parent / "config.json"

console = Console()

ALL_TEAM_NAMES = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
    "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
    "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]


# ── Thread-safe data cache ────────────────────────────────────────────────────
# The daemon thread and main thread both read this cache; protect with a lock.

class _DataCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game_logs = None
        self._matchup_df = None

    def get(self):
        with self._lock:
            return self._game_logs, self._matchup_df

    def set(self, game_logs, matchup_df) -> None:
        with self._lock:
            self._game_logs = game_logs
            self._matchup_df = matchup_df

    def invalidate(self) -> None:
        with self._lock:
            self._game_logs = None
            self._matchup_df = None


_data_cache = _DataCache()


# ── Status helpers ────────────────────────────────────────────────────────────

def _model_status() -> str:
    if not MODEL_PATH.exists():
        return "[red]NOT TRAINED[/red]"
    meta = _load_meta()
    name = meta.get("model_name", "Unknown")
    acc = meta.get("accuracy", 0)
    trained = meta.get("trained_on", "unknown date")
    return f"[green]LOADED[/green] ({name}, {acc*100:.1f}% accuracy, trained {trained})"


def _cache_status() -> str:
    if not GAME_LOG_CACHE.exists():
        return "[red]NOT FOUND[/red]"
    import time as _time
    age_hours = (_time.time() - GAME_LOG_CACHE.stat().st_mtime) / 3600
    if age_hours < 1:
        return f"[green]FRESH[/green] (last updated {int(age_hours * 60)} minutes ago)"
    elif age_hours < 24:
        return f"[green]FRESH[/green] (last updated {age_hours:.1f} hours ago)"
    else:
        return f"[yellow]STALE[/yellow] (last updated {age_hours/24:.1f} days ago)"


def _predict_status() -> str:
    return "[green]READY[/green]" if MODEL_PATH.exists() else "[red]NEEDS TRAINING[/red]"


def _load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ── Fuzzy team matching ───────────────────────────────────────────────────────

def resolve_team_name(user_input: str) -> Optional[str]:
    """Return the best-matching full NBA team name, or None if score is too low."""
    match, score = fuzzy_process.extractOne(user_input, ALL_TEAM_NAMES)
    return match if score >= 50 else None


# ── Data loading ──────────────────────────────────────────────────────────────

def _ensure_data_loaded():
    """Load data from cache into the thread-safe store if not already loaded."""
    game_logs, matchup_df = _data_cache.get()
    if game_logs is not None:
        return game_logs, matchup_df

    if not GAME_LOG_CACHE.exists():
        console.print(
            "[red]No data cache found. Please run option 1 to fetch data first.[/red]"
        )
        return None, None

    import pandas as pd
    from data.collector import build_matchup_dataframe

    with console.status("Loading data from cache…"):
        game_logs = pd.read_csv(GAME_LOG_CACHE, parse_dates=["GAME_DATE"])
        matchup_df = build_matchup_dataframe(game_logs)

    _data_cache.set(game_logs, matchup_df)
    return game_logs, matchup_df


# ── Menu actions ──────────────────────────────────────────────────────────────

def action_fetch_data() -> None:
    from data.collector import fetch_all_game_logs
    force = Prompt.ask(
        "Force re-fetch even if cache is fresh?", choices=["y", "n"], default="n"
    ) == "y"
    fetch_all_game_logs(force=force)
    _data_cache.invalidate()


def action_train_models() -> None:
    import pandas as pd
    from features.engineering import build_feature_matrix, validate_no_nan
    from features.elo import compute_elo_features
    from features.schedule import compute_schedule_features
    from data.injuries import fetch_injuries
    from data.advanced import fetch_advanced_stats, compute_mov_sos
    from data.vegas import fetch_nba_odds
    from models.trainer import train_all_models
    from models.evaluator import show_evaluation_dashboard

    game_logs, matchup_df = _ensure_data_loaded()
    if game_logs is None:
        return

    with console.status("Computing ELO ratings (CARMELO-lite)…"):
        matchup_df = compute_elo_features(matchup_df)

    _data_cache.set(game_logs, matchup_df)

    injuries_df = fetch_injuries()

    with console.status("Fetching advanced team stats…"):
        advanced_df = fetch_advanced_stats()

    with console.status("Computing MOV / SOS…"):
        from features.elo import get_all_elo_ratings
        mov_sos_df = compute_mov_sos(game_logs, get_all_elo_ratings())

    with console.status("Computing schedule/travel features…"):
        from features.elo import get_all_elo_ratings
        sched_feats = compute_schedule_features(game_logs, get_all_elo_ratings())

    # Vegas lines for current odds (zero for historical training rows)
    vegas_df = fetch_nba_odds()

    with console.status("Engineering features…"):
        X = build_feature_matrix(
            matchup_df, game_logs, injuries_df,
            vegas_df=vegas_df,
            advanced_df=advanced_df,
            schedule_feats=sched_feats,
        )
        X = validate_no_nan(X)
        y = matchup_df["home_win"].reset_index(drop=True)
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]

    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)

    results = train_all_models(X, y, seasons=cfg.get("seasons", []))
    show_evaluation_dashboard(results, X, y, feature_names=list(X.columns))


def action_tune_model() -> None:
    from features.engineering import build_feature_matrix, validate_no_nan
    from features.elo import compute_elo_features
    from features.schedule import compute_schedule_features
    from data.injuries import fetch_injuries
    from data.advanced import fetch_advanced_stats
    from data.vegas import fetch_nba_odds
    from models.trainer import load_model_meta, load_feature_columns
    from models.tuner import tune_best_model

    game_logs, matchup_df = _ensure_data_loaded()
    if game_logs is None:
        return

    meta = load_model_meta()
    if not meta:
        console.print("[red]No trained model found. Run option 2 first.[/red]")
        return

    best_name    = meta.get("model_name", "LightGBM")
    baseline_acc = meta.get("accuracy", 0.0)

    with console.status("Computing ELO ratings…"):
        matchup_df = compute_elo_features(matchup_df)

    _data_cache.set(game_logs, matchup_df)

    injuries_df = fetch_injuries()

    with console.status("Fetching advanced stats…"):
        advanced_df = fetch_advanced_stats()

    with console.status("Computing schedule features…"):
        from features.elo import get_all_elo_ratings
        sched_feats = compute_schedule_features(game_logs, get_all_elo_ratings())

    vegas_df = fetch_nba_odds()

    with console.status("Engineering features…"):
        X = build_feature_matrix(
            matchup_df, game_logs, injuries_df,
            vegas_df=vegas_df, advanced_df=advanced_df, schedule_feats=sched_feats,
        )
        # Align to saved feature columns so tuner uses same feature set.
        try:
            saved_cols = load_feature_columns()
            missing = [c for c in saved_cols if c not in X.columns]
            for col in missing:
                X[col] = 0.0
            X = X[saved_cols]
        except FileNotFoundError:
            pass
        X = validate_no_nan(X)
        y = matchup_df["home_win"].reset_index(drop=True)
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]

    tune_best_model(best_name, X, y, baseline_accuracy=baseline_acc)


def action_predict_game() -> None:
    from data.injuries import fetch_injuries
    from predict.predictor import predict_game, fetch_todays_schedule

    game_logs, matchup_df = _ensure_data_loaded()
    if game_logs is None:
        return

    if not MODEL_PATH.exists():
        console.print("[red]No saved model. Run option 2 first.[/red]")
        return

    injuries_df = fetch_injuries()

    # Fetch today's schedule
    with console.status("Fetching today's schedule…"):
        games = fetch_todays_schedule()

    if not games:
        console.print(
            "[yellow]No games found for today (API unavailable or no games scheduled).\n"
            "Falling back to manual entry.[/yellow]"
        )
        home_input = Prompt.ask("Enter [bold]home team[/bold] name")
        away_input = Prompt.ask("Enter [bold]away team[/bold] name")
        home_team = resolve_team_name(home_input)
        away_team = resolve_team_name(away_input)
        if home_team is None:
            console.print(f"[red]Could not match '{home_input}' to an NBA team.[/red]")
            return
        if away_team is None:
            console.print(f"[red]Could not match '{away_input}' to an NBA team.[/red]")
            return
        predict_game(home_team, away_team, game_logs, matchup_df, injuries_df)
        return

    while True:
        # Display today's games table
        today_str = datetime.date.today().strftime("%B %d %Y")
        table = Table(
            title=f"[bold cyan]Today's Games — {today_str}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#",         justify="right",  style="bold cyan", min_width=3)
        table.add_column("Home Team", style="bold",     min_width=24)
        table.add_column("Away Team", min_width=24)
        table.add_column("Time (ET)", justify="center", min_width=12)

        for i, g in enumerate(games, 1):
            table.add_row(str(i), g["home_team"], g["away_team"], g["game_time"])

        console.print(table)
        console.print("[dim]Enter a game number to predict, or 0 to predict ALL games.[/dim]")

        valid_choices = ["0"] + [str(i) for i in range(1, len(games) + 1)]
        choice = Prompt.ask("Select", choices=valid_choices)

        if choice == "0":
            # Predict ALL games and show summary table
            all_results: list[tuple[dict, dict]] = []
            for g in games:
                with console.status(
                    f"Predicting {g['home_team']} vs {g['away_team']}…"
                ):
                    result = predict_game(
                        g["home_team"], g["away_team"],
                        game_logs, matchup_df, injuries_df,
                        silent=True,
                    )
                all_results.append((g, result))

            summary = Table(
                title=f"[bold cyan]Full Day Predictions — {today_str}[/bold cyan]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            summary.add_column("Home Team", style="bold",      min_width=24)
            summary.add_column("Away Team", min_width=24)
            summary.add_column("Pick",      justify="center",  min_width=20)
            summary.add_column("Confidence", justify="center", min_width=14)

            for g, r in all_results:
                prob_home = r["home_win_prob"]
                prob_away = r["away_win_prob"]
                if prob_home >= prob_away:
                    pick      = g["home_team"].split()[-1].upper()
                    pick_prob = prob_home
                else:
                    pick      = g["away_team"].split()[-1].upper()
                    pick_prob = prob_away

                dist = abs(pick_prob - 0.5)
                if dist > 0.20:
                    conf_label = "High"
                    style = "bold green"
                elif dist > 0.10:
                    conf_label = "Medium"
                    style = "yellow"
                else:
                    conf_label = "Low"
                    style = "white"

                summary.add_row(
                    g["home_team"],
                    g["away_team"],
                    f"[{style}]{pick}[/{style}]",
                    f"{conf_label}  {pick_prob * 100:.0f}%",
                )
            console.print(summary)

        else:
            idx  = int(choice) - 1
            g    = games[idx]
            console.print(
                f"\nMatchup: [bold cyan]{g['home_team']}[/bold cyan] (home) vs "
                f"[bold cyan]{g['away_team']}[/bold cyan] (away)\n"
            )
            predict_game(
                g["home_team"], g["away_team"],
                game_logs, matchup_df, injuries_df,
            )

        again = Prompt.ask("\nPredict another game?", choices=["y", "n"], default="n")
        if again == "n":
            break


def action_backtest() -> None:
    from features.engineering import build_feature_matrix, validate_no_nan
    from features.elo import compute_elo_features
    from features.schedule import compute_schedule_features
    from data.injuries import fetch_injuries
    from data.advanced import fetch_advanced_stats
    from data.vegas import fetch_nba_odds
    from models.trainer import load_feature_columns
    from models.backtest import run_backtest, show_backtest_dashboard
    from rich.prompt import FloatPrompt

    game_logs, matchup_df = _ensure_data_loaded()
    if game_logs is None:
        return

    if not MODEL_PATH.exists():
        console.print("[red]No saved model. Run option 2 first.[/red]")
        return

    # Season selection
    if "SEASON" in matchup_df.columns:
        all_seasons = sorted(matchup_df["SEASON"].unique().tolist())
        console.print(f"Available seasons: {', '.join(all_seasons)}")
        seasons_input = Prompt.ask(
            "Seasons to backtest (comma-separated, or press Enter for last 2)",
            default="",
        )
        if seasons_input.strip():
            test_seasons = [s.strip() for s in seasons_input.split(",")]
        else:
            test_seasons = all_seasons[-2:]
    else:
        test_seasons = None

    threshold = 0.60
    try:
        threshold_input = Prompt.ask("Confidence threshold for betting (e.g. 0.60)", default="0.60")
        threshold = float(threshold_input)
    except ValueError:
        pass

    with console.status("Computing ELO ratings…"):
        matchup_df = compute_elo_features(matchup_df)

    _data_cache.set(game_logs, matchup_df)
    injuries_df = fetch_injuries()
    advanced_df = fetch_advanced_stats()
    vegas_df    = fetch_nba_odds()

    with console.status("Computing schedule features…"):
        from features.elo import get_all_elo_ratings
        sched_feats = compute_schedule_features(game_logs, get_all_elo_ratings())

    with console.status("Engineering feature matrix…"):
        X = build_feature_matrix(
            matchup_df, game_logs, injuries_df,
            vegas_df=vegas_df, advanced_df=advanced_df, schedule_feats=sched_feats,
        )
        try:
            saved_cols = load_feature_columns()
            missing = [c for c in saved_cols if c not in X.columns]
            for col in missing:
                X[col] = 0.0
            X = X[saved_cols]
        except FileNotFoundError:
            pass
        X = validate_no_nan(X)
        y = matchup_df["home_win"].reset_index(drop=True)
        min_len = min(len(X), len(y))
        X, y = X.iloc[:min_len], y.iloc[:min_len]
        matchup_df_aligned = matchup_df.iloc[:min_len].reset_index(drop=True)

    bt = run_backtest(
        X, y, matchup_df_aligned,
        test_seasons=test_seasons,
        confidence_threshold=threshold,
    )
    show_backtest_dashboard(bt)


def action_show_elo() -> None:
    from features.elo import get_all_elo_ratings

    ratings = get_all_elo_ratings()
    if not ratings:
        console.print("[yellow]No ELO ratings computed yet. Run option 2 to train.[/yellow]")
        return

    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    table = Table(
        title="[bold cyan]Current ELO Rankings — All Teams[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("ELO Rating", justify="right")

    for rank, (team, elo) in enumerate(sorted_teams, 1):
        style = "bold green" if rank <= 5 else ("yellow" if rank <= 15 else "red")
        table.add_row(str(rank), team, f"[{style}]{elo:.1f}[/{style}]")

    console.print(table)


def action_start_daemon() -> None:
    """Start the auto-retrain daemon; blocks until Ctrl-C."""
    import time
    import pytz
    from scheduler import run_retrain_headless, _seconds_until_midnight_et, ET_ZONE

    stop_event = threading.Event()
    last_retrain_str = _load_meta().get("trained_on", "Never")

    def _daemon_loop() -> None:
        nonlocal last_retrain_str
        while not stop_event.is_set():
            wait = _seconds_until_midnight_et()
            # Sleep in 30-second chunks so we can check stop_event promptly
            elapsed = 0.0
            while elapsed < wait and not stop_event.is_set():
                time.sleep(min(30, wait - elapsed))
                elapsed += 30
            if stop_event.is_set():
                break
            run_retrain_headless(headless=False)
            last_retrain_str = datetime.date.today().isoformat()

    daemon = threading.Thread(target=_daemon_loop, daemon=True)
    daemon.start()

    console.print("[green]Daemon started. Press Ctrl-C to stop.[/green]")
    try:
        while True:
            next_et = datetime.datetime.now(ET_ZONE) + datetime.timedelta(
                seconds=_seconds_until_midnight_et()
            )
            now_et = datetime.datetime.now(ET_ZONE)
            delta = next_et - now_et
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            mins, secs = divmod(rem, 60)

            status_line = (
                f"[bold cyan]NBA Auto-Retrain Daemon[/bold cyan]\n"
                f"  Next retrain:  [green]{next_et.strftime('%I:%M %p ET')}[/green] "
                f"(in {hours:02d}:{mins:02d}:{secs:02d})\n"
                f"  Last retrain:  [cyan]{last_retrain_str}[/cyan]\n"
                f"  Status:        [green]WATCHING[/green]\n\n"
                f"  [dim]Press Ctrl-C to stop[/dim]"
            )
            console.clear()
            console.print(Panel(status_line, title="[bold]Daemon Status[/bold]", expand=False))
            time.sleep(10)
    except KeyboardInterrupt:
        stop_event.set()
        console.print("\n[yellow]Daemon stopped.[/yellow]")


# ── Startup banner ────────────────────────────────────────────────────────────

def show_startup_banner() -> None:
    status_content = (
        f"[bold]Model status:[/bold]  {_model_status()}\n"
        f"[bold]Data cache:[/bold]    {_cache_status()}\n"
        f"[bold]Prediction:[/bold]    {_predict_status()}"
    )
    console.print(
        Panel(
            status_content,
            title="[bold cyan]🏀  NBA Game Outcome Predictor[/bold cyan]",
            expand=False,
        )
    )


# ── Option 8: Weekly accuracy trend ──────────────────────────────────────────

def action_show_accuracy_trend() -> None:
    from rich.table import Table
    from rich import box as rbox

    acc_path = Path(__file__).parent / "logs" / "prediction_accuracy.csv"
    if not acc_path.exists():
        console.print(
            "[yellow]No prediction accuracy data yet.\n"
            "Make predictions (option 4), then run the retrain daemon (option 6) "
            "after games complete — it will resolve and log results automatically.[/yellow]"
        )
        return

    try:
        import pandas as pd
        acc_df = pd.read_csv(acc_path, parse_dates=["date"])
    except Exception as exc:
        console.print(f"[red]Could not read accuracy log: {exc}[/red]")
        return

    if acc_df.empty:
        console.print("[yellow]Accuracy log is empty — no resolved predictions yet.[/yellow]")
        return

    acc_df["week"] = acc_df["date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")

    table = Table(
        title="[bold cyan]Weekly Prediction Accuracy[/bold cyan]",
        box=rbox.ROUNDED, show_header=True, header_style="bold magenta",
    )
    table.add_column("Week of",     style="bold")
    table.add_column("Games",       justify="right")
    table.add_column("Correct",     justify="right")
    table.add_column("Accuracy",    justify="right")
    table.add_column("High-conf",   justify="right")
    table.add_column("Trend",       justify="left")

    weekly = acc_df.groupby("week")
    all_accs = []
    for week, grp in sorted(weekly, key=lambda x: x[0]):
        n       = len(grp)
        correct = int(grp["correct"].sum())
        acc     = correct / n
        all_accs.append(acc)
        high    = grp[grp["confidence"] == "High"]
        h_str   = f"{int(high['correct'].sum())}/{len(high)}" if len(high) else "—"
        bar     = "█" * int(acc * 20)
        col     = "bold green" if acc >= 0.60 else ("yellow" if acc >= 0.55 else "red")
        # rolling arrow vs previous week
        arrow   = ""
        if len(all_accs) >= 2:
            arrow = "[green]▲[/green]" if all_accs[-1] > all_accs[-2] else "[red]▼[/red]"
        table.add_row(
            week, str(n), str(correct),
            f"[{col}]{acc*100:.1f}%[/]",
            h_str,
            f"{arrow} [{col}]{bar}[/]",
        )

    console.print(table)

    # Summary line
    overall_acc = float(acc_df["correct"].mean())
    n_total     = len(acc_df)
    high_conf   = acc_df[acc_df["confidence"] == "High"]
    h_acc       = float(high_conf["correct"].mean()) if len(high_conf) else 0.0
    console.print(
        f"\n  [bold]Overall:[/bold] {overall_acc*100:.1f}% on {n_total} predictions   "
        f"[bold]High-confidence:[/bold] {h_acc*100:.1f}% on {len(high_conf)} predictions"
    )

    # End-of-season vs normal breakdown
    if "end_of_season_risk" in acc_df.columns:
        risky   = acc_df[acc_df["end_of_season_risk"] == 1]
        normal  = acc_df[acc_df["end_of_season_risk"] == 0]
        if len(risky):
            r_acc = float(risky["correct"].mean())
            n_acc = float(normal["correct"].mean()) if len(normal) else 0.0
            console.print(
                f"  [bold]Normal games:[/bold] {n_acc*100:.1f}%   "
                f"[bold yellow]End-of-season flagged:[/bold yellow] {r_acc*100:.1f}%"
            )


# ── Main menu ─────────────────────────────────────────────────────────────────

def show_menu() -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Option", style="bold cyan", justify="right")
    table.add_column("Action")
    table.add_row("1", "Fetch / refresh data")
    table.add_row("2", "Train all models + show evaluation dashboard")
    table.add_row("3", "Tune best model (Optuna, 100 trials)")
    table.add_row("4", "Predict a game")
    table.add_row("5", "Show current ELO rankings")
    table.add_row("6", "Start auto-retrain daemon (midnight ET)")
    table.add_row("7", "Backtest predictions (flat-bet simulation)")
    table.add_row("8", "Weekly accuracy trend (post-game feedback)")
    table.add_row("9", "Exit")
    console.print(table)


def main() -> None:
    console.clear()
    show_startup_banner()

    while True:
        console.print()
        show_menu()
        choice = Prompt.ask("\nSelect option", choices=["1","2","3","4","5","6","7","8","9"])
        console.print()

        if choice == "1":
            action_fetch_data()
        elif choice == "2":
            action_train_models()
        elif choice == "3":
            action_tune_model()
        elif choice == "4":
            action_predict_game()
        elif choice == "5":
            action_show_elo()
        elif choice == "6":
            action_start_daemon()
        elif choice == "7":
            action_backtest()
        elif choice == "8":
            action_show_accuracy_trend()
        elif choice == "9":
            console.print("[cyan]Goodbye.[/cyan]")
            sys.exit(0)


if __name__ == "__main__":
    main()
