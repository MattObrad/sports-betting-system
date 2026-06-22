"""
models/edge_detection.py -- Final bet/no-bet filter for MLB totals edges.

Applies multi-criteria filtering to Monte Carlo results and produces the
daily bet slip.  This is the only module in the pipeline that makes the
final bet/no-bet decision.

Filter criteria (both must pass):
  1. |predicted_total - market_line| >= min_edge_runs   (run gap)
  2. p_win >= min_confidence                             (meta-learner confidence)

Additional automatic exclusions:
  - kelly_half <= 0  (no mathematical edge against the juice)
  - direction_conflict: regression and meta-learner disagree on OVER/UNDER

Flags (included in bet slip but do NOT auto-exclude):
  - divergence_flag: |sim_over_prob - ensemble_over_prob| > divergence_threshold

Threshold defaults come from config.json; CLI args override when provided.
Secrets never live in config.json -- Twilio auth token is an env variable.

Usage:
    python models/edge_detection.py --run-date 2024-09-15
    python models/edge_detection.py --run-date 2024-09-15 --min-edge 2.0
    python models/edge_detection.py --run-date 2024-09-15 --no-save
    python models/edge_detection.py --run-date 2024-09-15 --config /path/to/config.json
"""

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.monte_carlo import MonteCarloResult, american_to_payout, kelly_fraction  # noqa: E402
from utils.config import cfg_get, load_config  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (used when key is absent from config.json)
# ---------------------------------------------------------------------------

_DEFAULT_MIN_EDGE_RUNS          = 1.5
_DEFAULT_MIN_CONFIDENCE         = 0.55
_DEFAULT_BANKROLL_UNITS         = 100.0
_DEFAULT_DIVERGENCE_THRESHOLD   = 0.05
# |predicted_total - market_line| above this => !EXTREME flag (model likely
# extrapolating on thin/early-season data). Flagged, NOT excluded.
_DEFAULT_EXTREME_RUN_EDGE       = 3.0
# Market lines above this are excluded from alerts. A 9-inning game effectively
# never opens above ~13; higher values are data errors (e.g. a stray 26.5) or a
# line mismatched onto the wrong half of a doubleheader. NOTE: this does NOT
# catch ordinary 7-inning doubleheader games — those carry LOWER totals (~7-8)
# and are identified by games.doubleheader, not by the line. See module notes.
_DEFAULT_MAX_TOTAL_LINE         = 13.0

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_EDGE_DDL = """
CREATE TABLE IF NOT EXISTS edge_bets (
    run_date           TEXT NOT NULL,
    game_id            TEXT NOT NULL,
    bet_direction      TEXT,
    market_line        REAL,
    predicted_total    REAL,
    raw_edge_runs      REAL,
    p_win              REAL,
    implied_prob       REAL,
    prob_edge          REAL,
    juice              INTEGER,
    kelly_half         REAL,
    recommended_units  REAL,
    prob_divergence    REAL,
    divergence_flag    INTEGER,
    direction_conflict INTEGER,
    PRIMARY KEY (run_date, game_id)
);
"""

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EdgeBet:
    game_id:            str
    bet_direction:      str    # "OVER" | "UNDER"
    market_line:        float
    predicted_total:    float
    raw_edge_runs:      float  # predicted_total - market_line (signed)
    p_win:              float  # meta-learner prob for the bet direction
    implied_prob:       float  # market-implied prob for the bet direction
    prob_edge:          float  # p_win - implied_prob
    juice:              int    # American odds (over-side; assumed symmetric for under Kelly)
    kelly_half:         float  # half Kelly fraction for the bet direction
    recommended_units:  float  # kelly_half * bankroll_units
    prob_divergence:    float  # |sim_over_prob - ensemble_over_prob|
    divergence_flag:    bool   # True if prob_divergence > divergence_threshold
    direction_conflict: bool   # True if regression and meta-learner disagree on direction
    extreme_flag:       bool = False  # True if |raw_edge_runs| > extreme_run_edge
    away_team:          str = ""   # populated by predict_mlb.py from games JOIN teams
    home_team:          str = ""   # populated by predict_mlb.py from games JOIN teams


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def apply_edge_filters(
    results: list,
    min_edge_runs: float       = _DEFAULT_MIN_EDGE_RUNS,
    min_confidence: float      = _DEFAULT_MIN_CONFIDENCE,
    bankroll_units: float      = _DEFAULT_BANKROLL_UNITS,
    divergence_threshold: float = _DEFAULT_DIVERGENCE_THRESHOLD,
    extreme_run_edge: float    = _DEFAULT_EXTREME_RUN_EDGE,
    max_total_line: float      = _DEFAULT_MAX_TOTAL_LINE,
    exclude_doubleheaders: bool = True,
    null_market_game_ids: set  = None,
) -> list:
    """
    Apply multi-criteria edge filter to a list of MonteCarloResult objects.

    Parameters
    ----------
    results               : MonteCarloResult objects from run_monte_carlo()
    min_edge_runs         : minimum |predicted_total - market_line| to qualify
    min_confidence        : minimum p_win (meta-learner) to qualify
    bankroll_units        : scales recommended_units output
    divergence_threshold  : prob_divergence above this sets divergence_flag
    extreme_run_edge      : |raw_edge| above this sets the !EXTREME flag
    max_total_line        : market lines above this are excluded (data errors)
    exclude_doubleheaders : drop games where MonteCarloResult.doubleheader != 0.
                            The model is trained on 9-inning games only; 7-inning
                            doubleheader halves have different scoring dynamics it
                            cannot handle, so they are dropped from alerts entirely.
    null_market_game_ids  : HARD BLOCK. Set of game_id strings whose game_features
                            market columns are NULL (line_movement IS NULL AND
                            current_total IS NULL). These games are skipped entirely
                            before any bet is flagged. A model that leans heavily on
                            line_movement must never score a game where it is missing
                            — XGBoost routes NaN line_movement to a +7-run OVER leaf
                            (see 2026-06-04 audit, game 822727). Build this set with
                            null_market_game_ids_for_run().

    Returns
    -------
    list of EdgeBet, sorted by prob_edge descending (strongest edges first).
    Direction-conflicted games are excluded silently (logged at DEBUG level).
    """
    bets = []
    null_market_game_ids = null_market_game_ids or set()

    for r in results:
        # HARD BLOCK: market features NULL -> skip entirely, before anything else.
        # Stops the model from betting on the catastrophic NaN-line_movement path.
        if str(r.game_id) in null_market_game_ids:
            log.warning(
                "Game %s skipped -- market features NULL", r.game_id
            )
            continue

        # Exclude doubleheaders entirely: the model is trained on 9-inning games;
        # 7-inning DH halves (games.doubleheader != 0) have different run dynamics.
        if exclude_doubleheaders and getattr(r, "doubleheader", 0):
            log.info(
                "%s: doubleheader=%s -- excluded from alerts "
                "(model not trained on 7-inning games)",
                r.game_id, getattr(r, "doubleheader", 0),
            )
            continue

        # Exclude implausibly high lines: data errors (e.g. a stray 26.5) or a
        # line mismatched onto the wrong half of a doubleheader. A real 9-inning
        # game effectively never opens above ~13. (Ordinary 7-inning DH games
        # have LOW totals and are not caught here — see _DEFAULT_MAX_TOTAL_LINE.)
        if r.market_line > max_total_line:
            log.info(
                "%s: market_line=%.1f > %.1f -- excluded from alerts "
                "(likely data error or doubleheader line mismatch)",
                r.game_id, r.market_line, max_total_line,
            )
            continue

        raw_edge = r.predicted_total - r.market_line
        if raw_edge == 0.0:
            continue

        # Extrapolation guardrail: a huge gap from the line on thin/early-season
        # data is almost always the model extrapolating, not a real edge. We
        # FLAG (still alert, with a warning) rather than exclude.
        extreme_flag = abs(raw_edge) > extreme_run_edge

        reg_direction  = "OVER" if raw_edge > 0 else "UNDER"
        meta_direction = "OVER" if r.ensemble_over_prob > 0.5 else "UNDER"
        direction_conflict = reg_direction != meta_direction

        # Auto-exclude direction conflicts: regression and meta-learner disagree
        if direction_conflict:
            log.debug(
                "%s: direction conflict (reg=%s, meta=%s) -- excluded",
                r.game_id, reg_direction, meta_direction,
            )
            continue

        # p_win and prob_edge for the bet direction
        if reg_direction == "OVER":
            p_win     = r.ensemble_over_prob
            prob_edge = r.edge                          # ensemble_over_prob - implied_over_prob
        else:
            p_win     = 1.0 - r.ensemble_over_prob
            # implied_under_prob = 1 - implied_over_prob (symmetric juice assumption)
            prob_edge = r.implied_prob - r.ensemble_over_prob  # == -r.edge

        # Kelly for the bet direction
        # Under Kelly uses over juice (totals markets are nearly always symmetric;
        # if a line is -115/+105 the Kelly difference is under 0.2 units on a 100-unit bank)
        payout = american_to_payout(r.juice)
        kf     = kelly_fraction(p_win, payout)

        # Filter: must have positive Kelly (edge vs the juice)
        if kf <= 0:
            continue

        # Filter 1: run edge
        if abs(raw_edge) < min_edge_runs:
            continue

        # Filter 2: confidence
        if p_win < min_confidence:
            continue

        kh    = kf / 2.0
        units = kh * bankroll_units

        bets.append(EdgeBet(
            game_id            = r.game_id,
            bet_direction      = reg_direction,
            market_line        = r.market_line,
            predicted_total    = r.predicted_total,
            raw_edge_runs      = round(raw_edge, 3),
            p_win              = round(p_win, 4),
            implied_prob       = round(r.implied_prob, 4),
            prob_edge          = round(prob_edge, 4),
            juice              = r.juice,
            kelly_half         = round(kh, 4),
            recommended_units  = round(units, 2),
            prob_divergence    = r.prob_divergence,
            divergence_flag    = r.prob_divergence > divergence_threshold,
            direction_conflict = False,
            extreme_flag       = extreme_flag,
        ))

    bets.sort(key=lambda b: b.prob_edge, reverse=True)
    return bets


# ---------------------------------------------------------------------------
# HARD BLOCK helper: which games have NULL market features?
# ---------------------------------------------------------------------------

def null_market_game_ids_for_run(
    con: sqlite3.Connection,
    run_date: str,
    feature_version: str = "v1.0",
) -> set:
    """
    Return the set of game_id strings (games on `run_date`) whose game_features
    market columns are NULL: line_movement IS NULL AND current_total IS NULL.

    These games must be skipped by apply_edge_filters -- scoring them feeds NaN
    line_movement into the base models, which XGBoost treats as a strong OVER
    signal (2026-06-04 audit). Matches predict_mlb.py's ET-date convention by
    keying on games.game_date = run_date.
    """
    rows = con.execute(
        """
        SELECT f.game_id
        FROM   game_features f
        JOIN   games g ON f.game_id = g.game_id
        WHERE  g.game_date       = ?
          AND  f.feature_version = ?
          AND  f.line_movement IS NULL
          AND  f.current_total  IS NULL
        """,
        (run_date, feature_version),
    ).fetchall()
    return {str(r[0]) for r in rows}


# ---------------------------------------------------------------------------
# DB path: read MC results, reconstruct MonteCarloResult objects, filter
# ---------------------------------------------------------------------------

def load_and_filter(
    con: sqlite3.Connection,
    run_date: str,
    min_edge_runs: float       = _DEFAULT_MIN_EDGE_RUNS,
    min_confidence: float      = _DEFAULT_MIN_CONFIDENCE,
    bankroll_units: float      = _DEFAULT_BANKROLL_UNITS,
    divergence_threshold: float = _DEFAULT_DIVERGENCE_THRESHOLD,
    exclude_doubleheaders: bool = True,
    feature_version: str       = "v1.0",
) -> list:
    """
    Load monte_carlo_results for run_date from DB, reconstruct MonteCarloResult
    objects, and apply apply_edge_filters.  Allows re-running the filter with
    different thresholds against already-computed MC results.
    """
    df = pd.read_sql_query(
        "SELECT * FROM monte_carlo_results WHERE run_date = ?",
        con,
        params=(run_date,),
    )

    if df.empty:
        log.warning("No monte_carlo_results found for run_date=%s", run_date)
        return []

    log.info("Loaded %d MC results for %s", len(df), run_date)

    # doubleheader column may be absent in rows written before the migration.
    has_dh = "doubleheader" in df.columns

    results = [
        MonteCarloResult(
            game_id            = row.game_id,
            predicted_total    = float(row.predicted_total),
            market_line        = float(row.market_line),
            sigma_effective    = float(row.sigma_effective),
            sim_over_prob      = float(row.sim_over_prob),
            ensemble_over_prob = float(row.ensemble_over_prob),
            prob_divergence    = float(row.prob_divergence),
            juice              = int(row.juice),
            implied_prob       = float(row.implied_prob),
            edge               = float(row.edge),
            kelly_full         = float(row.kelly_full),
            kelly_half         = float(row.kelly_half),
            recommended_units  = float(row.recommended_units),
            doubleheader       = int(row.doubleheader) if has_dh and pd.notna(row.doubleheader) else 0,
        )
        for _, row in df.iterrows()
    ]

    return apply_edge_filters(
        results,
        min_edge_runs        = min_edge_runs,
        min_confidence       = min_confidence,
        bankroll_units       = bankroll_units,
        divergence_threshold = divergence_threshold,
        exclude_doubleheaders = exclude_doubleheaders,
        null_market_game_ids = null_market_game_ids_for_run(con, run_date, feature_version),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_bet_slip(bets: list, run_date: str = None) -> None:
    """Print the daily bet slip sorted by prob_edge descending."""
    header_date = run_date or str(date.today())
    print(f"\nBet Slip -- {header_date}")

    if not bets:
        print("No qualifying edges found.\n")
        return

    divider = "-" * 88
    print(divider)
    print(
        f"{'#':<3} {'Matchup':<12} {'Dir':<5} {'Line':>5} {'Pred':>5} "
        f"{'RunEdge':>7} {'P(win)':>6} {'Juice':>6} {'Kelly%':>6} {'Units':>5}  Flags"
    )
    print(divider)

    for i, b in enumerate(bets, start=1):
        flags = []
        if b.extreme_flag:
            flags.append("!EXTREME")
        if b.divergence_flag:
            flags.append("!DIV")
        flag_str = " ".join(flags)

        matchup = (
            f"{b.away_team}@{b.home_team}"
            if b.away_team and b.home_team
            else str(b.game_id)
        )
        edge_sign = "+" if b.raw_edge_runs > 0 else ""
        print(
            f"{i:<3} {matchup:<12} {b.bet_direction:<5} "
            f"{b.market_line:>5.1f} {b.predicted_total:>5.2f} "
            f"{edge_sign}{b.raw_edge_runs:>6.2f} "
            f"{b.p_win*100:>5.1f}% "
            f"{b.juice:>6} "
            f"{b.kelly_half*100:>5.2f}% "
            f"{b.recommended_units:>5.2f} "
            f" {flag_str}"
        )

    n_div     = sum(1 for b in bets if b.divergence_flag)
    n_extreme = sum(1 for b in bets if b.extreme_flag)
    print(divider)
    summary = f"{len(bets)} bet(s)"
    if n_extreme:
        summary += f"  |  {n_extreme} EXTREME (likely extrapolating) -- do not trust"
    if n_div:
        summary += f"  |  {n_div} divergence flag(s) -- review before placing"
    print(f"{summary}\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_edge_bets(
    con: sqlite3.Connection,
    bets: list,
    run_date: str,
) -> None:
    con.execute(_EDGE_DDL)
    rows = [
        {
            "run_date": run_date,
            **{
                k: (1 if v is True else (0 if v is False else v))
                for k, v in asdict(b).items()
            },
        }
        for b in bets
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO edge_bets
            (run_date, game_id, bet_direction, market_line, predicted_total,
             raw_edge_runs, p_win, implied_prob, prob_edge, juice,
             kelly_half, recommended_units, prob_divergence,
             divergence_flag, direction_conflict)
        VALUES
            (:run_date, :game_id, :bet_direction, :market_line, :predicted_total,
             :raw_edge_runs, :p_win, :implied_prob, :prob_edge, :juice,
             :kelly_half, :recommended_units, :prob_divergence,
             :divergence_flag, :direction_conflict)
        """,
        rows,
    )
    con.commit()
    log.info("Saved %d edge bet(s) for run_date=%s", len(bets), run_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Apply edge filter to monte_carlo_results and produce the daily bet slip."
        )
    )
    p.add_argument(
        "--run-date",
        default=str(date.today()),
        help="Date of MC results to filter (YYYY-MM-DD, default: today)",
    )
    p.add_argument(
        "--min-edge",
        type=float,
        default=None,
        metavar="RUNS",
        help="Min |predicted - line| to qualify (overrides config.json)",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        metavar="PROB",
        help="Min meta-learner p_win to qualify (overrides config.json)",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Bankroll in units for recommended_units (overrides config.json)",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.json (default: project root config.json)",
    )
    p.add_argument("--db",      default="mlb_data.db")
    p.add_argument("--no-save", action="store_true",
                   help="Print bet slip only; do not write to DB")
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    cfg = load_config(args.config)

    # CLI args override config; config overrides built-in defaults
    min_edge   = args.min_edge       if args.min_edge       is not None else cfg_get(cfg, "betting", "min_edge_runs",          default=_DEFAULT_MIN_EDGE_RUNS)
    min_conf   = args.min_confidence if args.min_confidence is not None else cfg_get(cfg, "betting", "min_confidence",         default=_DEFAULT_MIN_CONFIDENCE)
    bankroll   = args.bankroll       if args.bankroll       is not None else cfg_get(cfg, "betting", "bankroll_units",         default=_DEFAULT_BANKROLL_UNITS)
    div_thresh =                                                              cfg_get(cfg, "betting", "divergence_flag_threshold", default=_DEFAULT_DIVERGENCE_THRESHOLD)

    log.info(
        "Edge filter: min_edge=%.1f runs  min_confidence=%.0f%%  bankroll=%.0f units",
        min_edge, min_conf * 100, bankroll,
    )

    con  = sqlite3.connect(args.db)
    bets = load_and_filter(
        con,
        run_date             = args.run_date,
        min_edge_runs        = min_edge,
        min_confidence       = min_conf,
        bankroll_units       = bankroll,
        divergence_threshold = div_thresh,
    )

    print_bet_slip(bets, run_date=args.run_date)

    if not args.no_save and bets:
        save_edge_bets(con, bets, run_date=args.run_date)

    con.close()


if __name__ == "__main__":
    main()
