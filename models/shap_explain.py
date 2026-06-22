"""
models/shap_explain.py -- SHAP explainability for MLB totals predictions.

Explains base model (LGBM + XGB) predictions at the raw feature level using
TreeExplainer.  SHAP values from both models are blended with the same
inverse-RMSE weights used for predicted_total, so feature attributions are
internally consistent with the prediction itself.

The meta-learner (11 meta-features) is not explained here; its coefficients
are already surfaced by ensemble.py's print_coefficient_report().

Outputs per game:
  - top-5 OVER drivers  (largest positive blended SHAP values)
  - top-5 UNDER drivers (largest negative blended SHAP values)
  - full {feature: shap} dict stored as JSON in shap_results table

format_sms_line() uses top-3 drivers for Twilio SMS (Step 12).

Usage:
    python models/shap_explain.py --season 2024
    python models/shap_explain.py --date 2024-09-15
    python models/shap_explain.py --season 2024 --global-summary
    python models/shap_explain.py --season 2024 --no-save   # dry run
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.ensemble import EnsembleTotalsModel  # noqa: E402
from utils.features import get_feature_columns, prepare_X  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature display-name map
# Anything not in this map falls back to the raw column name.
# ---------------------------------------------------------------------------

FEATURE_DISPLAY = {
    # Starting pitcher -- ERA / FIP / peripherals
    "home_sp_era_l3":            "Home SP ERA (L3)",
    "home_sp_era_l5":            "Home SP ERA (L5)",
    "home_sp_era_szn":           "Home SP ERA (Season)",
    "away_sp_era_l3":            "Away SP ERA (L3)",
    "away_sp_era_l5":            "Away SP ERA (L5)",
    "away_sp_era_szn":           "Away SP ERA (Season)",
    "home_sp_fip_l5":            "Home SP FIP (L5)",
    "away_sp_fip_l5":            "Away SP FIP (L5)",
    "home_sp_xfip_l5":           "Home SP xFIP (L5)",
    "away_sp_xfip_l5":           "Away SP xFIP (L5)",
    "home_sp_siera_szn":         "Home SP SIERA (Season)",
    "away_sp_siera_szn":         "Away SP SIERA (Season)",
    "home_sp_k9_l5":             "Home SP K/9 (L5)",
    "away_sp_k9_l5":             "Away SP K/9 (L5)",
    "home_sp_bb9_l5":            "Home SP BB/9 (L5)",
    "away_sp_bb9_l5":            "Away SP BB/9 (L5)",
    "home_sp_hr9_l5":            "Home SP HR/9 (L5)",
    "away_sp_hr9_l5":            "Away SP HR/9 (L5)",
    "home_sp_career_vs_opp_era": "Home SP vs Opp ERA",
    "away_sp_career_vs_opp_era": "Away SP vs Opp ERA",
    "home_sp_career_vs_opp_pa":  "Home SP vs Opp PA",
    "away_sp_career_vs_opp_pa":  "Away SP vs Opp PA",
    "home_sp_throws_R":          "Home SP Throws (R)",
    "away_sp_throws_R":          "Away SP Throws (R)",
    # Team offense
    "home_runs_per_game_5":      "Home Runs/G (L5)",
    "home_runs_per_game_10":     "Home Runs/G (L10)",
    "home_runs_per_game_30":     "Home Runs/G (L30)",
    "away_runs_per_game_5":      "Away Runs/G (L5)",
    "away_runs_per_game_10":     "Away Runs/G (L10)",
    "away_runs_per_game_30":     "Away Runs/G (L30)",
    "home_runs_vs_rhp_30":       "Home Runs vs RHP",
    "home_runs_vs_lhp_30":       "Home Runs vs LHP",
    "away_runs_vs_rhp_30":       "Away Runs vs RHP",
    "away_runs_vs_lhp_30":       "Away Runs vs LHP",
    "home_wrc_plus_szn":         "Home wRC+",
    "away_wrc_plus_szn":         "Away wRC+",
    # Bullpen
    "home_bullpen_era_10":       "Home Bullpen ERA (L10)",
    "away_bullpen_era_10":       "Away Bullpen ERA (L10)",
    "home_bullpen_fip_10":       "Home Bullpen FIP (L10)",
    "away_bullpen_fip_10":       "Away Bullpen FIP (L10)",
    "home_bullpen_ip_3days":     "Home Bullpen Load (3d)",
    "away_bullpen_ip_3days":     "Away Bullpen Load (3d)",
    # Weather / park
    "park_factor_runs":          "Park Factor",
    "wind_speed_mph":            "Wind Speed (mph)",
    "wind_to_cf":                "Wind to CF",
    "temperature_f":             "Temperature (F)",
    "humidity_pct":              "Humidity (%)",
    "precipitation_prob":        "Precip Prob",
    # Umpire
    "ump_career_runs_per_game":  "Umpire Runs/G",
    "ump_career_over_rate":      "Umpire Over Rate",
    "ump_zone_size_score":       "Umpire Zone Score",
    "ump_games_sampled":         "Umpire Sample (G)",
    # Market
    "opening_total":             "Opening Line",
    "current_total":             "Current Line",
    "line_movement":             "Line Movement",
    "over_juice":                "Over Juice",
    "implied_over_prob":         "Implied Over Prob",
    "hours_since_open":          "Hours Since Open",
    # Schedule
    "home_days_rest":            "Home Days Rest",
    "away_days_rest":            "Away Days Rest",
    "home_travel_dist_miles":    "Home Travel Miles",
    "away_travel_dist_miles":    "Away Travel Miles",
}


def display_name(raw: str) -> str:
    """Return human-readable feature name, falling back to raw column name."""
    return FEATURE_DISPLAY.get(raw, raw)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SHAP_DDL = """
CREATE TABLE IF NOT EXISTS shap_results (
    run_date          TEXT NOT NULL,
    game_id           TEXT NOT NULL,
    predicted_total   REAL,
    market_line       REAL,
    base_value        REAL,
    top_over_drivers  TEXT,   -- JSON list of {feature, shap, display}
    top_under_drivers TEXT,   -- JSON list of {feature, shap, display}
    all_shap          TEXT,   -- JSON {feature: shap_value}
    PRIMARY KEY (run_date, game_id)
);
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ShapResult:
    game_id:           str
    predicted_total:   float
    market_line:       float
    base_value:        float                  # blended SHAP base value
    top_over_drivers:  List[Dict]             # top-5 positive SHAP features
    top_under_drivers: List[Dict]             # top-5 negative SHAP features
    all_shap:          Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def build_explainer(model):
    """
    Build a SHAP TreeExplainer from a LGBMTotalsModel or XGBTotalsModel.
    Both classes store the fitted sklearn-compatible estimator as .model.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed: pip install shap")

    return shap.TreeExplainer(model.model)


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Return SHAP value matrix of shape (n_games, n_features).
    For regression tree models, shap_values() returns a 2D array directly.
    """
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        # Some versions wrap regression output in a single-element list
        return np.array(raw[0])
    return np.array(raw)


def blend_shap_values(
    shap_lgbm: np.ndarray,
    shap_xgb: np.ndarray,
    base_lgbm: float,
    base_xgb: float,
    blend_weights: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    """
    Inverse-RMSE weighted blend of SHAP arrays and base values.
    Uses the same weights as predicted_total so attributions are consistent.
    Returns (blended_shap_matrix, blended_base_value).
    """
    w_l, w_x = blend_weights
    blended      = w_l * shap_lgbm + w_x * shap_xgb
    blended_base = w_l * float(base_lgbm) + w_x * float(base_xgb)
    return blended, blended_base


def top_drivers(
    shap_row: np.ndarray,
    feature_names: List[str],
    n: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return top-N over drivers (positive SHAP) and top-N under drivers
    (negative SHAP), each sorted by |shap| descending.
    """
    pairs = list(zip(feature_names, shap_row.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    over_drivers  = [
        {"feature": f, "shap": round(v, 4), "display": display_name(f)}
        for f, v in pairs if v > 0
    ][:n]
    under_drivers = [
        {"feature": f, "shap": round(v, 4), "display": display_name(f)}
        for f, v in pairs if v < 0
    ][:n]

    return over_drivers, under_drivers


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_shap_explain(
    ensemble: EnsembleTotalsModel,
    X: pd.DataFrame,
    game_ids: List[str],
    market_lines: np.ndarray,
    n_top: int = 5,
) -> List[ShapResult]:
    """
    Compute blended SHAP explanations for every game in X.

    Parameters
    ----------
    ensemble     : fitted EnsembleTotalsModel (holds both base models)
    X            : feature DataFrame (output of prepare_X)
    game_ids     : game_id strings in same row order as X
    market_lines : over/under line per game
    n_top        : top-N drivers to store (5 over + 5 under)

    Returns
    -------
    list of ShapResult, one per game
    """
    lgbm_model = ensemble.lgbm_model
    xgb_model  = ensemble.xgb_model
    w_l, w_x   = ensemble.blend_weights

    # Align X to each model's training-time feature columns
    X_lgbm = X.reindex(columns=lgbm_model.feature_cols, fill_value=np.nan).astype(float)
    X_xgb  = X.reindex(columns=xgb_model.feature_cols,  fill_value=np.nan).astype(float)

    log.info("Building SHAP explainers ...")
    exp_lgbm = build_explainer(lgbm_model)
    exp_xgb  = build_explainer(xgb_model)

    log.info("Computing SHAP values for %d games ...", len(game_ids))
    sv_lgbm = compute_shap_values(exp_lgbm, X_lgbm)
    sv_xgb  = compute_shap_values(exp_xgb,  X_xgb)

    # Both models must share the same feature_cols order for the blend to be valid.
    # If they differ (shouldn't happen with shared training data), align xgb to lgbm.
    feat_names = lgbm_model.feature_cols
    if xgb_model.feature_cols != feat_names:
        log.warning(
            "LGBM and XGB feature_cols differ -- re-aligning XGB SHAP values. "
            "Check that both models were trained on the same feature set."
        )
        xgb_col_idx = {c: i for i, c in enumerate(xgb_model.feature_cols)}
        sv_xgb_aligned = np.zeros_like(sv_lgbm)
        for j, col in enumerate(feat_names):
            if col in xgb_col_idx:
                sv_xgb_aligned[:, j] = sv_xgb[:, xgb_col_idx[col]]
        sv_xgb = sv_xgb_aligned

    blended_sv, blended_base = blend_shap_values(
        sv_lgbm, sv_xgb,
        float(exp_lgbm.expected_value),
        float(exp_xgb.expected_value),
        (w_l, w_x),
    )

    # Get ensemble predicted totals for display.
    # ensemble.predict_total() returns a RESIDUAL (model is trained on
    # actual_total - market_line), so add the line back to get the absolute
    # total used for the "OVER/UNDER line->total" display string.
    market_lines = np.asarray(market_lines, dtype=float)
    predicted_residuals, _ = ensemble.predict(X, market_lines)
    predicted_totals = np.asarray(predicted_residuals, dtype=float) + market_lines

    results = []
    for i, gid in enumerate(game_ids):
        over_d, under_d = top_drivers(blended_sv[i], feat_names, n=n_top)
        all_shap = {
            feat_names[j]: round(float(blended_sv[i, j]), 5)
            for j in range(len(feat_names))
        }
        results.append(ShapResult(
            game_id          = gid,
            predicted_total  = round(float(predicted_totals[i]), 3),
            market_line      = float(market_lines[i]),
            base_value       = round(blended_base, 3),
            top_over_drivers  = over_d,
            top_under_drivers = under_d,
            all_shap         = all_shap,
        ))

    return results


# ---------------------------------------------------------------------------
# SMS formatting
# ---------------------------------------------------------------------------

def format_sms_line(result: ShapResult, n: int = 3) -> str:
    """
    Return a compact SMS-ready explanation for one game.

    Format:
        OVER 9.5->10.3  | Home SP ERA (L5) +1.3, Wind to CF +0.9, Park Factor -0.7
        UNDER 9.5->8.8  | Away SP ERA (L5) -1.2, Home wRC+ -0.8, Wind to CF +0.4

    Top-n drivers are chosen by largest |shap| regardless of sign, so the
    reader sees what most influenced the prediction in either direction.
    """
    direction = "OVER" if result.predicted_total > result.market_line else "UNDER"
    header = f"{direction} {result.market_line:.1f}->{result.predicted_total:.1f}"

    all_drivers = result.top_over_drivers + result.top_under_drivers
    all_drivers.sort(key=lambda d: abs(d["shap"]), reverse=True)
    top = all_drivers[:n]

    parts = [
        f"{d['display']} {'+' if d['shap'] >= 0 else ''}{d['shap']:.2f}"
        for d in top
    ]
    return f"{header}  |  {', '.join(parts)}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_shap_report(results: List[ShapResult], top_n: int = 10) -> None:
    """Print top-N games by largest single driver |shap|, showing top-3 drivers."""
    def max_abs_shap(r: ShapResult) -> float:
        drivers = r.top_over_drivers + r.top_under_drivers
        return max((abs(d["shap"]) for d in drivers), default=0.0)

    ranked = sorted(results, key=max_abs_shap, reverse=True)[:top_n]

    print(f"\n{'Game ID':<16} {'Pred':>6} {'Line':>6}  Top drivers")
    print("-" * 78)
    for r in ranked:
        all_d = r.top_over_drivers + r.top_under_drivers
        all_d.sort(key=lambda d: abs(d["shap"]), reverse=True)
        driver_str = "  |  ".join(
            f"{d['display']} {'+' if d['shap'] >= 0 else ''}{d['shap']:.2f}"
            for d in all_d[:3]
        )
        print(
            f"{r.game_id:<16} "
            f"{r.predicted_total:>6.2f} "
            f"{r.market_line:>6.1f}  "
            f"{driver_str}"
        )
    print()


def print_global_summary(results: List[ShapResult], top_n: int = 20) -> None:
    """
    Print mean |SHAP| per feature across all games -- global feature importance.
    Useful for understanding what the model is actually learning.
    """
    if not results:
        print("No results to summarize.")
        return

    all_features = list(results[0].all_shap.keys())
    abs_matrix   = np.array([
        [abs(r.all_shap.get(f, 0.0)) for f in all_features]
        for r in results
    ])
    mean_abs = abs_matrix.mean(axis=0)

    ranked = sorted(zip(all_features, mean_abs), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Feature':<35} {'Display Name':<30} {'Mean |SHAP|':>11}")
    print("-" * 82)
    for rank, (feat, importance) in enumerate(ranked[:top_n], start=1):
        print(
            f"{rank:<5} {feat:<35} {display_name(feat):<30} {importance:>11.4f}"
        )
    print(f"\n(across {len(results)} games)\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_shap_results(
    con: sqlite3.Connection,
    results: List[ShapResult],
    run_date: str,
) -> None:
    con.execute(_SHAP_DDL)
    rows = [
        {
            "run_date":          run_date,
            "game_id":           r.game_id,
            "predicted_total":   r.predicted_total,
            "market_line":       r.market_line,
            "base_value":        r.base_value,
            "top_over_drivers":  json.dumps(r.top_over_drivers),
            "top_under_drivers": json.dumps(r.top_under_drivers),
            "all_shap":          json.dumps(r.all_shap),
        }
        for r in results
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO shap_results
            (run_date, game_id, predicted_total, market_line, base_value,
             top_over_drivers, top_under_drivers, all_shap)
        VALUES
            (:run_date, :game_id, :predicted_total, :market_line, :base_value,
             :top_over_drivers, :top_under_drivers, :all_shap)
        """,
        rows,
    )
    con.commit()
    log.info("Saved %d SHAP results for run_date=%s", len(results), run_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SHAP explainability for MLB totals predictions."
    )
    p.add_argument(
        "--ensemble-pkl",
        default="models/saved/ensemble_v1.0.pkl",
        help="Path to fitted EnsembleTotalsModel pkl",
    )
    p.add_argument("--feature-version", default="v1.0")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="Explain all games in YYYY season")
    g.add_argument("--date",   help="Explain games on a single date (YYYY-MM-DD)")
    p.add_argument(
        "--n-top",
        type=int,
        default=5,
        help="Top-N drivers to store per game (default 5 each direction)",
    )
    p.add_argument(
        "--global-summary",
        action="store_true",
        help="Print mean |SHAP| per feature across all games after per-game report",
    )
    p.add_argument("--db",      default="mlb_data.db")
    p.add_argument("--no-save", action="store_true",
                   help="Print report only; do not write to DB")
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
               o.opening_total      AS market_line
        FROM   game_features f
        JOIN   games g ON f.game_id = g.game_id
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

    log.info("Explaining %d games.", len(feat_df))

    feature_cols = get_feature_columns(feat_df)
    X            = prepare_X(feat_df, feature_cols)
    market_lines = feat_df["market_line"].to_numpy(dtype=float)
    game_ids     = feat_df["game_id"].tolist()

    results = run_shap_explain(
        ensemble     = ensemble,
        X            = X,
        game_ids     = game_ids,
        market_lines = market_lines,
        n_top        = args.n_top,
    )

    print_shap_report(results)

    if args.global_summary:
        print_global_summary(results)

    if not args.no_save:
        run_date = str(date.today())
        save_shap_results(con, results, run_date)

    con.close()


if __name__ == "__main__":
    main()
