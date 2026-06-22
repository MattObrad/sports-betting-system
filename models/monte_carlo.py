"""
models/monte_carlo.py -- Monte Carlo simulation and Kelly criterion sizing.

For each game: draws 10,000 simulated totals from Normal(predicted, sigma_effective),
computes sim_over_prob as a raw sanity check, then computes Kelly-optimal bet fractions
using the calibrated ensemble_over_prob from the logistic meta-learner.

Sigma is the inverse-RMSE-weighted blend of base model residual stds -- internally
consistent with how predicted_total is computed.

bet/no-bet decision (is_bet) is NOT made here.  Step 9 (edge_detection.py) applies
the full filter: min edge, min confidence, divergence flags, and any other criteria.
Monte Carlo writes raw numbers only.

Usage:
    python models/monte_carlo.py --season 2024
    python models/monte_carlo.py --date 2024-09-15
    python models/monte_carlo.py --season 2024 --no-save        # dry run
    python models/monte_carlo.py --season 2024 --n-sims 50000   # higher fidelity
"""

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.ensemble import EnsembleTotalsModel  # noqa: E402
from utils.features import get_feature_columns, prepare_X  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_MC_DDL = """
CREATE TABLE IF NOT EXISTS monte_carlo_results (
    run_date           TEXT NOT NULL,
    game_id            TEXT NOT NULL,
    predicted_total    REAL,
    market_line        REAL,
    sigma_effective    REAL,
    sim_over_prob      REAL,
    ensemble_over_prob REAL,
    prob_divergence    REAL,
    juice              INTEGER,
    implied_prob       REAL,
    edge               REAL,
    kelly_full         REAL,
    kelly_half         REAL,
    recommended_units  REAL,
    doubleheader       INTEGER DEFAULT 0,
    PRIMARY KEY (run_date, game_id)
);
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    game_id:            str
    predicted_total:    float
    market_line:        float
    sigma_effective:    float
    sim_over_prob:      float    # raw Monte Carlo estimate (sanity check only)
    ensemble_over_prob: float    # logistic meta-learner output (drives Kelly)
    prob_divergence:    float    # abs(sim - ensemble); flag if > 0.05
    juice:              int      # American odds for the over side
    implied_prob:       float    # market's implied win probability
    edge:               float    # ensemble_over_prob - implied_prob
    kelly_full:         float    # full Kelly fraction, clamped [0, 0.25]
    kelly_half:         float    # half Kelly fraction, clamped [0, 0.125]
    recommended_units:  float    # kelly_half * bankroll_units
    doubleheader:       int = 0   # games.doubleheader (0=normal, 1/2=DH halves)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def simulate_game(
    predicted_total: float,
    sigma: float,
    market_line: float,
    n_sims: int = 10_000,
    seed: Optional[int] = None,
) -> float:
    """Return fraction of simulated totals exceeding market_line."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(predicted_total, max(sigma, 0.5), n_sims)
    return float((draws > market_line).mean())


def american_to_implied_prob(juice: int) -> float:
    """Convert American odds to market-implied win probability."""
    if juice > 0:
        return 100.0 / (juice + 100.0)
    return abs(juice) / (abs(juice) + 100.0)


def american_to_payout(juice: int) -> float:
    """Return profit per $1 wagered for American odds (b in Kelly formula)."""
    if juice > 0:
        return juice / 100.0
    return 100.0 / abs(juice)


def kelly_fraction(p_win: float, payout: float) -> float:
    """
    Full Kelly fraction: (b*p - (1-p)) / b, clamped to [0.0, 0.25].
    Negative Kelly (no edge) returns 0.0.
    """
    raw = (payout * p_win - (1.0 - p_win)) / payout
    return float(np.clip(raw, 0.0, 0.25))


def _blended_sigma(ensemble: EnsembleTotalsModel) -> float:
    """
    Weighted average of base-model residual stds using the same inverse-RMSE
    blend_weights that produce predicted_total.  Floor at 0.5 runs.
    """
    w_l, w_x = ensemble.blend_weights
    sigma_l = getattr(ensemble.lgbm_model, "residual_std", None)
    sigma_x = getattr(ensemble.xgb_model,  "residual_std", None)

    if sigma_l is None or sigma_x is None:
        log.warning("Base model residual_std not available -- using fallback sigma=3.0")
        return 3.0

    blended = w_l * sigma_l + w_x * sigma_x
    return float(max(blended, 0.5))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_monte_carlo(
    ensemble: EnsembleTotalsModel,
    X: pd.DataFrame,
    market_lines: np.ndarray,
    juices: np.ndarray,
    game_ids: list,
    bankroll_units: float = 100.0,
    n_sims: int = 10_000,
    doubleheaders: np.ndarray = None,
) -> list:
    """
    Run Monte Carlo simulation and Kelly sizing for every game in X.

    Parameters
    ----------
    ensemble       : fitted EnsembleTotalsModel
    X              : feature DataFrame aligned to ensemble's feature_cols
    market_lines   : over/under line per game
    juices         : American odds for the over side per game (e.g., -110)
    game_ids       : game_id strings in same row order as X
    bankroll_units : total bankroll in units; scales recommended_units output
    n_sims         : Monte Carlo draws per game
    doubleheaders  : optional per-game games.doubleheader flag (0=normal, 1/2=DH
                     halves), same order as game_ids. Carried through to the
                     MonteCarloResult so edge_detection can exclude 7-inning DH
                     games. Defaults to all-0 (no game treated as a DH).

    Returns
    -------
    list of MonteCarloResult, one per game, in input order
    """
    market_lines = np.asarray(market_lines, dtype=float)
    juices       = np.asarray(juices, dtype=int)
    if doubleheaders is None:
        doubleheaders = np.zeros(len(game_ids), dtype=int)
    else:
        doubleheaders = np.asarray(doubleheaders, dtype=int)

    # The base/ensemble regressors are trained on the RESIDUAL target
    # (actual_total - market_line), so ensemble.predict_total() returns a
    # predicted residual (~0), NOT an absolute total (~9).  Reconstruct the
    # absolute predicted total by adding the market line back.  This is the value
    # the Normal distribution must be centred on for simulate_game(), and the value
    # edge_detection compares to the line — where
    #     raw_edge = predicted_total - market_line = predicted_residual  (correct).
    predicted_residuals, ensemble_over_probs = ensemble.predict(X, market_lines)
    predicted_totals = np.asarray(predicted_residuals, dtype=float) + market_lines
    sigma = _blended_sigma(ensemble)

    log.info(
        "Running %d-sim Monte Carlo for %d games  sigma_effective=%.3f",
        n_sims, len(game_ids), sigma,
    )

    results = []
    for i, gid in enumerate(game_ids):
        mu         = float(predicted_totals[i])
        line       = float(market_lines[i])
        p_ensemble = float(ensemble_over_probs[i])
        juice      = int(juices[i])

        sim_prob   = simulate_game(mu, sigma, line, n_sims=n_sims)
        divergence = abs(sim_prob - p_ensemble)

        implied = american_to_implied_prob(juice)
        payout  = american_to_payout(juice)
        edge    = p_ensemble - implied

        kf    = kelly_fraction(p_ensemble, payout)
        kh    = kf / 2.0
        units = kh * bankroll_units

        results.append(MonteCarloResult(
            game_id            = gid,
            predicted_total    = round(mu, 3),
            market_line        = line,
            sigma_effective    = round(sigma, 4),
            sim_over_prob      = round(sim_prob,   4),
            ensemble_over_prob = round(p_ensemble, 4),
            prob_divergence    = round(divergence, 4),
            juice              = juice,
            implied_prob       = round(implied, 4),
            edge               = round(edge, 4),
            kelly_full         = round(kf, 4),
            kelly_half         = round(kh, 4),
            recommended_units  = round(units, 2),
            doubleheader       = int(doubleheaders[i]),
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_mc_report(results: list, top_n: int = 10) -> None:
    """Print top-N positive-edge games sorted by edge descending."""
    positive = sorted(
        (r for r in results if r.edge > 0),
        key=lambda r: r.edge,
        reverse=True,
    )
    shown = positive[:top_n]

    if not shown:
        print("\nNo positive-edge games found.\n")
        return

    divider = "-" * 80
    print(f"\n{'Game ID':<16} {'Pred':>6} {'Line':>6} {'SimP':>6} "
          f"{'EnsP':>6} {'Edge':>6} {'Kelly%':>7} {'Units':>6}  Flags")
    print(divider)

    for r in shown:
        flags = []
        if r.prob_divergence > 0.05:
            flags.append(f"DIV={r.prob_divergence:.3f}")

        print(
            f"{r.game_id:<16} "
            f"{r.predicted_total:>6.2f} "
            f"{r.market_line:>6.1f} "
            f"{r.sim_over_prob:>6.3f} "
            f"{r.ensemble_over_prob:>6.3f} "
            f"{r.edge:>+6.3f} "
            f"{r.kelly_half * 100:>6.2f}% "
            f"{r.recommended_units:>6.2f} "
            f" {'  '.join(flags)}"
        )

    n_flagged = sum(1 for r in shown if r.prob_divergence > 0.05)
    print(divider)
    print(
        f"{len(positive)} positive-edge games of {len(results)} total"
        + (f"  |  {n_flagged} high-divergence flag(s)" if n_flagged else "")
        + "\n"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_mc_results(
    con: sqlite3.Connection,
    results: list,
    run_date: str,
) -> None:
    con.execute(_MC_DDL)
    # Migrate pre-existing tables that predate the doubleheader column.
    try:
        con.execute("ALTER TABLE monte_carlo_results ADD COLUMN doubleheader INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    rows = [{"run_date": run_date, **asdict(r)} for r in results]
    con.executemany(
        """
        INSERT OR REPLACE INTO monte_carlo_results
            (run_date, game_id, predicted_total, market_line, sigma_effective,
             sim_over_prob, ensemble_over_prob, prob_divergence, juice,
             implied_prob, edge, kelly_full, kelly_half, recommended_units,
             doubleheader)
        VALUES
            (:run_date, :game_id, :predicted_total, :market_line, :sigma_effective,
             :sim_over_prob, :ensemble_over_prob, :prob_divergence, :juice,
             :implied_prob, :edge, :kelly_full, :kelly_half, :recommended_units,
             :doubleheader)
        """,
        rows,
    )
    con.commit()
    log.info("Saved %d Monte Carlo results for run_date=%s", len(results), run_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monte Carlo simulation + Kelly sizing for MLB totals."
    )
    p.add_argument(
        "--ensemble-pkl",
        default="models/saved/ensemble_v1.0.pkl",
        help="Path to fitted EnsembleTotalsModel pkl (default: models/saved/ensemble_v1.0.pkl)",
    )
    p.add_argument("--feature-version", default="v1.0")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="Run for all games in YYYY season")
    g.add_argument("--date", help="Run for a single date (YYYY-MM-DD)")
    p.add_argument(
        "--bankroll",
        type=float,
        default=100.0,
        help="Total bankroll in units; scales recommended_units column (default 100)",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Monte Carlo draws per game (default 10000)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Games to show in report (default 10)",
    )
    p.add_argument("--db", default="mlb_data.db")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Print report only; do not write to DB",
    )
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    ensemble = EnsembleTotalsModel.load(args.ensemble_pkl)
    log.info("Loaded ensemble from %s", args.ensemble_pkl)

    con = sqlite3.connect(args.db)

    if args.date:
        date_filter = "AND g.game_date = ?"
        params      = [args.feature_version, args.date]
    else:
        date_filter = "AND strftime('%Y', g.game_date) = ?"
        params      = [args.feature_version, str(args.season)]

    feat_df = pd.read_sql_query(
        f"""
        SELECT f.*, g.game_date,
               o.opening_total       AS market_line,
               o.opening_over_juice  AS juice
        FROM   game_features f
        JOIN   games g  ON f.game_id = g.game_id
        LEFT JOIN odds o ON f.game_id = o.game_id
        WHERE  f.feature_version = ?
          {date_filter}
          AND  o.opening_total IS NOT NULL
        ORDER  BY g.game_date
        """,
        con,
        params=params,
    )

    if feat_df.empty:
        log.warning("No games with market lines found for the requested scope.")
        con.close()
        sys.exit(0)

    log.info("Processing %d games.", len(feat_df))

    feature_cols = get_feature_columns(feat_df)
    X            = prepare_X(feat_df, feature_cols)
    market_lines = feat_df["market_line"].to_numpy(dtype=float)
    juices       = feat_df["juice"].fillna(-110).astype(int).to_numpy()
    game_ids     = feat_df["game_id"].tolist()

    results = run_monte_carlo(
        ensemble       = ensemble,
        X              = X,
        market_lines   = market_lines,
        juices         = juices,
        game_ids       = game_ids,
        bankroll_units = args.bankroll,
        n_sims         = args.n_sims,
    )

    print_mc_report(results, top_n=args.top_n)

    if not args.no_save:
        run_date = str(date.today())
        save_mc_results(con, results, run_date)

    con.close()


if __name__ == "__main__":
    main()
