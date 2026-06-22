"""
predict_mlb.py -- Daily MLB totals prediction script.

Runs at 10am UTC on VPS via cron after the overnight data collection and
feature engineering crons have populated today's game_features.

Pipeline:
    1. Load ensemble pkl
    2. Load today's game_features + current market lines (ET date, not UTC)
    3. prepare_X() using the model's training-time feature column list
    4. run_monte_carlo()     -> save to monte_carlo_results
    5. run_shap_explain()    -> save to shap_results  (skip with --skip-shap)
    6. apply_edge_filters()  -> save to edge_bets
    7. print_bet_slip()
    8. send_edge_sms() per EdgeBet (skip with --no-notify)

Exit codes:
    0  Completed normally (0 or more edges found is not an error)
    1  Fatal error (missing pkl, DB unavailable, game_features absent)
    2  No games scheduled today (not an error -- cron should treat as success)

Usage:
    python predict_mlb.py                        # today's games (ET)
    python predict_mlb.py --date 2024-09-15      # backfill a specific date
    python predict_mlb.py --no-notify            # run pipeline, skip SMS
    python predict_mlb.py --no-save              # print only, no DB writes
    python predict_mlb.py --skip-shap            # omit SHAP (faster reruns)
    python predict_mlb.py --skip-shap --no-notify  # fastest possible rerun
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9 fallback

import pandas as pd

from models.ensemble import EnsembleTotalsModel
from models.monte_carlo import run_monte_carlo, save_mc_results
from models.shap_explain import format_sms_line, run_shap_explain, save_shap_results
from models.edge_detection import apply_edge_filters, print_bet_slip, save_edge_bets
from utils.features import prepare_X
from utils.config import cfg_get, load_config

# notify.py is Step 12 -- graceful degradation if not yet deployed
try:
    from notify import send_edge_sms, send_summary_sms, send_test_sms
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# If 8 or more edges qualify today, send a count summary SMS before the individual alerts.
SUMMARY_THRESHOLD = 8

# Columns added by the prediction JOIN that were not in game_features at training time.
# Must be stripped before feeding feat_df into the model.
_JOIN_EXTRAS = frozenset({
    "game_date", "market_line", "juice", "opening_line", "current_line",
    "doubleheader",
})


# ---------------------------------------------------------------------------
# ET date helpers
# ---------------------------------------------------------------------------

def get_et_today() -> str:
    """Return today's date in America/New_York as YYYY-MM-DD."""
    return datetime.now(ET).date().isoformat()


def snapshot_cutoff_for(date_str: str) -> str:
    """
    Return the snapshot cutoff timestamp for a given date.
    For today: now() in UTC (captures all snapshots up to this moment).
    For a past date: end-of-day so all that day's snapshots are included.
    """
    today = datetime.now(ET).date().isoformat()
    if date_str == today:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"{date_str} 23:59:59"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_todays_games(
    con: sqlite3.Connection,
    today: str,
    feature_version: str,
    cutoff: str,
) -> pd.DataFrame:
    """
    Load game_features for today joined with the best available market line.

    Market line resolution (first non-NULL wins):
        1. Latest odds_snapshots row before cutoff  (most current live line)
        2. game_features.current_total              (as of feature engineering run)
        3. game_features.opening_total              (opening line fallback)

    Games without any market line are included with market_line = NULL.
    The downstream edge-detection pass uses has_line = market_line IS NOT NULL
    and silently skips no-line games for bet-flagging — but the model still
    produces a predicted residual for every game.

    Returns an empty DataFrame only if no game_features rows exist for today.
    """
    return pd.read_sql_query(
        """
        SELECT  f.*,
                g.game_date,
                g.doubleheader,
                ht.team_code                        AS home_team,
                at.team_code                        AS away_team,
                COALESCE(snap.total_line,
                         f.current_total,
                         f.opening_total)           AS market_line,
                COALESCE(snap.over_juice,
                         f.over_juice)              AS juice,
                f.opening_total                     AS opening_line,
                COALESCE(snap.total_line,
                         f.current_total)           AS current_line
        FROM    game_features f
        JOIN    games g  ON f.game_id        = g.game_id
        JOIN    teams ht ON g.home_team_id   = ht.team_id
        JOIN    teams at ON g.away_team_id   = at.team_id
        LEFT JOIN (
            SELECT  os.game_id,
                    os.total_line,
                    os.over_juice
            FROM    odds_snapshots os
            INNER JOIN (
                SELECT  game_id, MAX(snapshot_time) AS latest
                FROM    odds_snapshots
                WHERE   snapshot_time <= ?
                GROUP BY game_id
            ) ls ON os.game_id = ls.game_id
               AND os.snapshot_time = ls.latest
        ) snap ON f.game_id = snap.game_id
        WHERE   g.game_date       = ?
          AND   f.feature_version = ?
        ORDER BY f.game_id
        """,
        con,
        params=(cutoff, today, feature_version),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Daily MLB totals prediction -- full pipeline with edge alerts."
    )
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Predict for a specific ET date (default: today in ET). Use for backfill.",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Run full pipeline but skip SMS notifications",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Print results only; do not write to DB",
    )
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help=(
            "Skip SHAP computation (faster reruns; SMS sends without "
            "driver explanation line)"
        ),
    )
    p.add_argument(
        "--ensemble-pkl",
        default=None,
        help="Override ensemble pkl path (default: from config.json)",
    )
    p.add_argument(
        "--test-sms",
        action="store_true",
        help="Send a test SMS to verify credentials and gateway, then exit",
    )
    p.add_argument("--db",     default=None,
                   help="SQLite DB path (default: MLB_DB_PATH env var, "
                        "then mlb_data.db)")
    p.add_argument("--config", default=None)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    load_dotenv(_DIR / ".env")
    args = _build_parser().parse_args(argv)
    cfg  = load_config(args.config)
    db   = args.db or os.environ.get("MLB_DB_PATH") or "mlb_data.db"

    # -- Test SMS (credential check only -- no pipeline) --------------------
    if args.test_sms:
        if not HAS_NOTIFY:
            log.error("notify.py not found -- cannot send test SMS.")
            sys.exit(1)
        success = send_test_sms(cfg)
        sys.exit(0 if success else 1)

    # -- Date (ET) -----------------------------------------------------------
    today  = args.date or get_et_today()
    cutoff = snapshot_cutoff_for(today)
    log.info("=== MLB predictions for %s (snapshot cutoff %s UTC) ===", today, cutoff)

    # -- Config --------------------------------------------------------------
    ver        = cfg_get(cfg, "model",   "feature_version",         default="v1.0")
    pkl_path   = args.ensemble_pkl or cfg_get(
                     cfg, "model", "ensemble_pkl",
                     default="models/saved/ensemble_v1.0.pkl")
    bankroll   = cfg_get(cfg, "betting", "bankroll_units",           default=100.0)
    n_sims     = cfg_get(cfg, "betting", "n_sims",                   default=10_000)
    min_edge   = cfg_get(cfg, "betting", "min_edge_runs",            default=1.5)
    min_conf   = cfg_get(cfg, "betting", "min_confidence",           default=0.55)
    div_thresh    = cfg_get(cfg, "betting",    "divergence_flag_threshold", default=0.05)
    skip_shap_cfg = cfg_get(cfg, "prediction", "skip_shap_on_low_memory",   default=False)
    # Paper/monitor mode: still compute + persist predictions and edge_bets (for
    # CLV tracking) but send NO outbound alerts. Guarantees "no betting" regardless
    # of whether SMS credentials happen to be configured on the host.
    paper_mode    = cfg_get(cfg, "prediction", "paper_mode",                default=False)

    # -- Load ensemble -------------------------------------------------------
    if not Path(pkl_path).exists():
        log.error("Ensemble pkl not found: %s", pkl_path)
        sys.exit(1)

    log.info("Loading ensemble from %s", pkl_path)
    try:
        ensemble = EnsembleTotalsModel.load(pkl_path)
    except Exception as exc:
        log.error("Failed to load ensemble pkl: %s", exc)
        sys.exit(1)

    # -- Connect to DB -------------------------------------------------------
    try:
        con = sqlite3.connect(db)
    except Exception as exc:
        log.error("Cannot open DB %s: %s", db, exc)
        sys.exit(1)

    # -- Load today's games --------------------------------------------------
    log.info("Loading game_features (feature_version=%s) ...", ver)
    feat_df = load_todays_games(con, today, ver, cutoff)

    if feat_df.empty:
        n_sched = con.execute(
            "SELECT COUNT(*) FROM games WHERE game_date = ?", (today,)
        ).fetchone()[0]
        n_features = con.execute(
            """SELECT COUNT(*) FROM game_features f
               JOIN games g ON g.game_id = f.game_id
               WHERE g.game_date = ? AND f.feature_version = ?""",
            (today, ver),
        ).fetchone()[0]
        con.close()
        if n_sched == 0:
            log.info("No games scheduled for %s.", today)
            sys.exit(2)
        if n_features == 0:
            log.error(
                "%d game(s) scheduled for %s but game_features are missing "
                "(feature_version=%s) -- run engineer_features.py first.",
                n_sched, today, ver,
            )
        else:
            # Features exist but no market lines — model can still predict,
            # but edge-flagging won't fire.  This is normal before odds open.
            log.warning(
                "%d game(s) have features for %s but no market lines yet "
                "(odds_snapshots empty for today). Predictions will run "
                "without edge detection until odds are collected.",
                n_features, today,
            )
        sys.exit(1)

    n_games      = len(feat_df)
    game_ids     = feat_df["game_id"].tolist()
    market_lines = feat_df["market_line"].to_numpy(dtype=float)
    juices       = feat_df["juice"].fillna(-110).astype(int).to_numpy()
    # games.doubleheader (0=normal, 1/2=DH halves) — carried into MonteCarloResult
    # so edge_detection drops 7-inning DH games the model can't handle.
    doubleheaders = feat_df["doubleheader"].fillna(0).astype(int).to_numpy()

    log.info("Loaded %d game(s).", n_games)

    # -- Market-feature imputation (Fix 4) -----------------------------------
    # Defensive backstop to XGBoost's NaN default branch: a missing line_movement
    # routes to a +7-run OVER leaf (2026-06-04 audit). Games with BOTH
    # line_movement and current_total NULL are hard-blocked downstream; this
    # cleans the PARTIAL-NULL cases that still get scored. Order matters:
    #   opening_total -> market_line   (so current_total's fallback is non-null)
    #   current_total -> opening_total
    #   line_movement -> 0.0           (treat unknown movement as flat, not NaN)
    # IMPORTANT: capture hard-block set BEFORE imputation — after fillna(0.0),
    # line_movement is never NULL and the condition below always yields empty set.
    null_market_game_ids = {
        str(row["game_id"])
        for _, row in feat_df.iterrows()
        if pd.isna(row.get("line_movement")) and pd.isna(row.get("current_total"))
    }
    market_series = pd.Series(market_lines, index=feat_df.index)
    n_imp_open = int(feat_df["opening_total"].isna().sum())
    n_imp_curr = int(feat_df["current_total"].isna().sum())
    n_imp_move = int(feat_df["line_movement"].isna().sum())
    feat_df["opening_total"] = feat_df["opening_total"].fillna(market_series)
    feat_df["current_total"] = feat_df["current_total"].fillna(feat_df["opening_total"])
    feat_df["line_movement"] = feat_df["line_movement"].fillna(0.0)
    if n_imp_open or n_imp_curr or n_imp_move:
        log.info(
            "Market imputation: opening_total<-market_line x%d, "
            "current_total<-opening_total x%d, line_movement<-0.0 x%d",
            n_imp_open, n_imp_curr, n_imp_move,
        )

    # -- Feature matrix ------------------------------------------------------
    # Use the model's training-time feature_cols list as the authoritative set.
    # This ensures predict_mlb.py never silently includes JOIN-added columns
    # (game_date, market_line, etc.) that were absent at training time.
    feature_cols = ensemble.lgbm_model.feature_cols
    X = prepare_X(feat_df, feature_cols)

    # -- Monte Carlo ---------------------------------------------------------
    log.info("Running Monte Carlo (%d sims x %d games) ...", n_sims, n_games)
    mc_results = run_monte_carlo(
        ensemble       = ensemble,
        X              = X,
        market_lines   = market_lines,
        juices         = juices,
        game_ids       = game_ids,
        bankroll_units = bankroll,
        n_sims         = n_sims,
        doubleheaders  = doubleheaders,
    )
    if not args.no_save:
        save_mc_results(con, mc_results, run_date=today)

    # -- SHAP ----------------------------------------------------------------
    shap_results = []
    shap_by_game = {}

    if args.skip_shap:
        log.info("SHAP skipped (--skip-shap).")
    elif skip_shap_cfg:
        log.info("SHAP skipped (prediction.skip_shap_on_low_memory=true in config.json).")
    else:
        log.info("Computing SHAP explanations ...")
        try:
            shap_results = run_shap_explain(
                ensemble     = ensemble,
                X            = X,
                game_ids     = game_ids,
                market_lines = market_lines,
            )
            shap_by_game = {r.game_id: r for r in shap_results}
            if not args.no_save:
                save_shap_results(con, shap_results, run_date=today)
        except ImportError:
            log.warning("shap package not installed -- SHAP skipped. pip install shap")

    # -- Edge detection ------------------------------------------------------
    if null_market_game_ids:
        log.warning(
            "%d game(s) have NULL market features and will be skipped: %s",
            len(null_market_game_ids), sorted(null_market_game_ids),
        )

    edge_bets = apply_edge_filters(
        mc_results,
        min_edge_runs        = min_edge,
        min_confidence       = min_conf,
        bankroll_units       = bankroll,
        divergence_threshold = div_thresh,
        null_market_game_ids = null_market_game_ids,
    )

    # Populate team names from feat_df (games JOIN teams added to load_todays_games)
    _team_lookup = {
        str(row["game_id"]): (str(row.get("away_team", "")), str(row.get("home_team", "")))
        for _, row in feat_df[["game_id", "away_team", "home_team"]].iterrows()
    }
    for bet in edge_bets:
        bet.away_team, bet.home_team = _team_lookup.get(str(bet.game_id), ("", ""))

    if not args.no_save:
        save_edge_bets(con, edge_bets, run_date=today)

    con.close()

    # -- Report --------------------------------------------------------------
    print_bet_slip(edge_bets, run_date=today)
    log.info(
        "Done: %d game(s) processed, %d qualifying edge(s).",
        n_games, len(edge_bets),
    )

    # -- Notify --------------------------------------------------------------
    if paper_mode:
        log.info(
            "PAPER MODE (prediction.paper_mode=true): %d edge(s) computed and saved "
            "to edge_bets for CLV monitoring; NO alerts sent, NO wagers.",
            len(edge_bets),
        )
    elif args.no_notify:
        log.info("Notifications skipped (--no-notify).")
    elif not HAS_NOTIFY:
        log.warning(
            "notify.py not available -- SMS skipped "
            "(deploy Step 12 to enable notifications)."
        )
    elif not edge_bets:
        log.info("No qualifying edges -- no SMS sent.")
    else:
        if len(edge_bets) >= SUMMARY_THRESHOLD:
            send_summary_sms(len(edge_bets), cfg)

        for bet in edge_bets:
            shap = shap_by_game.get(bet.game_id)
            send_edge_sms(bet, shap, cfg)

    sys.exit(0)


if __name__ == "__main__":
    main()
