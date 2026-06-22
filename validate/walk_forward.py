"""
walk_forward.py — Expanding-window walk-forward cross-validation for MLB totals model.

Trains on all seasons up to a cutoff, tests on the next season, rolls forward.
Every test-set game is predicted using only data that existed before it was played.

Model swap: default placeholder is LGBMRegressor. To use the real ensemble after
Step 5-6, pass --model-module and --model-class, or import and call run_fold()
directly with a custom model_cls argument.

Usage:
    python validate/walk_forward.py                              # all folds, defaults
    python validate/walk_forward.py --min-train-seasons 4       # first test=2021
    python validate/walk_forward.py --include-covid             # include 2020
    python validate/walk_forward.py --fold 2024                 # single fold only
    python validate/walk_forward.py --feature-version v1.1
    python validate/walk_forward.py --edge-threshold 1.5
    python validate/walk_forward.py --juice -110
"""

import os
import sys
import json
import math
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3

# Project root on path so sibling packages resolve correctly
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.features import (          # noqa: E402
    EXCLUDE_COLS as _EXCLUDE_COLS,
    get_feature_columns,
    prepare_X,
    regression_to_proba,
)

try:
    from sklearn.metrics import log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
DB_PATH    = os.path.normpath(os.path.join(_DIR, "..", "mlb_data.db"))
OOS_DIR    = os.path.join(_DIR, "oos_predictions")

# ── Constants ─────────────────────────────────────────────────────────────────
COVID_SEASON        = 2020
DEFAULT_FEATURE_VER = "v1.0"
DEFAULT_MIN_TRAIN   = 4        # position-based: first test = 2021
DEFAULT_EDGE_THRESH = 1.5      # runs away from market line to flag an edge
DEFAULT_JUICE       = -110     # American odds for ROI simulation

# LightGBM placeholder defaults (swapped for real ensemble in Step 5-6)
_LGBM_DEFAULTS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "colsample_bytree":  0.8,
    "subsample":         0.8,
    "subsample_freq":    1,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# SQLite DDL for the OOS predictions table (created at runtime, not in setup_db.py)
_OOS_DDL = """
CREATE TABLE IF NOT EXISTS oos_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date         TEXT    NOT NULL,
    feature_version  TEXT    NOT NULL,
    fold_season      INTEGER NOT NULL,
    game_id          INTEGER NOT NULL,
    game_date        TEXT,
    season           INTEGER,
    actual_total     INTEGER,
    predicted_total  REAL,
    over_prob        REAL,
    market_line      REAL,
    edge_value       REAL,
    direction        TEXT,         -- 'O' | 'U' | NULL (no market line)
    actual_direction TEXT,         -- 'O' | 'U' | 'P' | NULL
    is_edge          INTEGER DEFAULT 0,
    UNIQUE (run_date, game_id, feature_version)
);
"""

# =============================================================================
# FOLD GENERATION
# =============================================================================

def get_available_seasons(con: sqlite3.Connection) -> list:
    """Seasons present in both game_features and completed games."""
    rows = con.execute("""
        SELECT DISTINCT g.season
        FROM game_features gf
        JOIN games g ON g.game_id = gf.game_id
        WHERE g.status IN ('Final', 'Completed Early')
          AND g.total_runs IS NOT NULL
          AND g.season IS NOT NULL
        ORDER BY g.season
    """).fetchall()
    return [r[0] for r in rows]


def get_season_folds(
    all_seasons: list,
    min_train_seasons: int,
    include_covid: bool,
) -> list:
    """
    Expanding-window folds. COVID position counts toward min_train_seasons
    but COVID rows are excluded from training data unless --include-covid is set.

    Example (default, min_train_seasons=4, include_covid=False):
      all_seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
      Fold 1: train=[2017,2018,2019]  test=2021  (2020 excluded from train)
      Fold 2: train=[2017..2021]      test=2022
      ...
    """
    folds = []
    for i, test_s in enumerate(all_seasons):
        if i < min_train_seasons:
            continue                             # not enough history yet
        if test_s == COVID_SEASON and not include_covid:
            continue                             # 2020 cannot be a test fold

        train_s = all_seasons[:i]
        if not include_covid:
            train_s = [s for s in train_s if s != COVID_SEASON]

        if not train_s:
            continue

        folds.append({"train_seasons": train_s, "test_season": test_s})

    return folds

# =============================================================================
# DATA LOADING
# =============================================================================

def _in_clause(seasons: list) -> str:
    return ",".join(str(s) for s in seasons)


def load_fold_data(
    con: sqlite3.Connection,
    seasons: list,
    feature_version: str,
) -> tuple:
    """
    Returns (feat_df, meta_df) for the given seasons.

    Only games with an opening market line are included (INNER JOIN on
    odds_snapshots).  Games without odds are uninformative for residual
    training and would dilute the bet-sizing signal.

    feat_df : game_id + all game_features columns
    meta_df : game_id, game_date, season, actual_total, market_line,
              actual_residual, actual_direction

    actual_residual = actual_total - market_line
        > 0  → game went OVER the opening line
        < 0  → game went UNDER
        = 0  → push (excluded from O/U accuracy but tracked)
    """
    s_in = _in_clause(seasons)

    meta = pd.read_sql(f"""
        SELECT g.game_id, g.game_date, g.season,
               g.total_runs              AS actual_total,
               o.total_line              AS market_line
        FROM games g
        INNER JOIN (
            -- True opening line: total_line at the EARLIEST is_opening snapshot
            -- per game (ties broken deterministically by total_line).
            -- MIN(total_line) was WRONG: when a game accumulated multiple
            -- is_opening snapshots across daily syncs, it picked the lowest line,
            -- biasing the opening total downward and inflating actual_residual.
            SELECT game_id, total_line FROM (
                SELECT game_id, total_line,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_id
                           ORDER BY snapshot_time ASC, total_line ASC
                       ) AS rn
                FROM odds_snapshots
                WHERE is_opening = 1 AND total_line IS NOT NULL
            ) WHERE rn = 1
        ) o USING (game_id)
        WHERE g.season IN ({s_in})
          AND g.status IN ('Final', 'Completed Early')
          AND g.total_runs IS NOT NULL
        ORDER BY g.game_date
    """, con, parse_dates=["game_date"])

    feat = pd.read_sql(f"""
        SELECT gf.*
        FROM game_features gf
        JOIN games g ON g.game_id = gf.game_id
        INNER JOIN (
            SELECT DISTINCT game_id FROM odds_snapshots WHERE is_opening = 1
        ) o ON o.game_id = gf.game_id
        WHERE g.season IN ({s_in})
          AND gf.feature_version = '{feature_version}'
          AND g.status IN ('Final', 'Completed Early')
          AND g.total_runs IS NOT NULL
    """, con)

    combined = meta.merge(feat, on="game_id", how="inner")

    meta_out = combined[["game_id", "game_date", "season",
                          "actual_total", "market_line"]].copy()
    mkt = meta_out["market_line"].astype(float)
    tot = meta_out["actual_total"].astype(float)

    # actual_direction: ground-truth for O/U evaluation
    meta_out["actual_direction"] = np.where(
        mkt.isna(), None,
        np.where(tot > mkt, "O",
        np.where(tot < mkt, "U", "P")),
    )

    # actual_residual: the training TARGET for the residual model.
    # Positive = game went over the line; negative = went under.
    meta_out["actual_residual"] = (tot - mkt).where(mkt.notna(), other=np.nan)

    # feat_df: drop meta columns merged in above; keep game_id for alignment
    drop = [c for c in ["game_date", "season", "actual_total", "market_line"]
            if c in combined.columns]
    feat_out = combined.drop(columns=drop, errors="ignore")

    return feat_out, meta_out

# prepare_X, get_feature_columns, regression_to_proba imported from utils.features

# =============================================================================
# MODEL FACTORY
# =============================================================================

def make_model(model_cls=None, model_params: dict = None):
    """
    Return an unfitted sklearn-compatible regressor.

    Default: LightGBM placeholder (fast, handles NaN, good baseline).
    To swap in the real ensemble after Step 5-6:
        make_model(model_cls=MyEnsemble, model_params={...})
    or pass --model-module / --model-class via CLI.
    """
    if model_cls is not None:
        return model_cls(**(model_params or {}))

    if not HAS_LGBM:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    params = {**_LGBM_DEFAULTS, **(model_params or {})}
    return lgb.LGBMRegressor(**params)

# regression_to_proba imported from utils.features

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    over_prob: np.ndarray,
    meta_df: pd.DataFrame,
    edge_threshold: float,
    juice: int,
) -> dict:
    """
    Compute all validation metrics for one residual-model fold.

    y_true = actual_residual  (actual_total - market_line)
    y_pred = predicted_residual

    Direction logic:
      predicted_residual > 0 → model predicts OVER (expects total above the line)
      predicted_residual < 0 → model predicts UNDER
      edge = |predicted_residual| >= edge_threshold (model's confidence gap vs line)
    """
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))

    # All games already have a line (load_fold_data uses INNER JOIN on odds).
    # Keep has_line mask for robustness in case any NaN slips through.
    has_line = meta_df["market_line"].notna().values
    act_dir  = meta_df["actual_direction"].values

    # Over/under accuracy: direction of predicted residual vs actual direction
    if has_line.any():
        # Residual > 0 → predict OVER; < 0 → predict UNDER
        pred_dir = np.where(y_pred > 0, "O", "U")
        acc = float(np.mean(pred_dir[has_line] == act_dir[has_line]))

        no_push = has_line & (act_dir != "P")
        if HAS_SKLEARN and no_push.any():
            y_bin  = (act_dir[no_push] == "O").astype(int)
            p_clip = np.clip(over_prob[no_push], 1e-6, 1 - 1e-6)
            ll = float(log_loss(y_bin, p_clip))
        else:
            ll = None
    else:
        acc = None
        ll  = None

    # Edge detection: confidence = |predicted_residual| (how far model deviates from line)
    edge_val  = np.where(has_line, np.abs(y_pred), np.nan)
    is_edge   = has_line & ~np.isnan(edge_val) & (edge_val >= edge_threshold)
    n_edges   = int(is_edge.sum())
    edge_rate = n_edges / max(len(y_true), 1) * 100.0

    if n_edges:
        pred_dir_all = np.where(y_pred >= 0, "O", "U")
        wins   = int(np.sum(pred_dir_all[is_edge] == act_dir[is_edge]))
        losses = n_edges - wins
        win_payout = 100.0 / abs(juice) if juice < 0 else float(juice) / 100.0
        roi = (wins * win_payout - losses) / n_edges * 100.0
    else:
        wins = losses = 0
        roi = 0.0

    return {
        "n_games":     len(y_true),
        "rmse":        round(rmse, 4),
        "mae":         round(mae,  4),
        "uo_acc":      round(acc,  4) if acc is not None else None,
        "log_loss":    round(ll,   4) if ll  is not None else None,
        "n_edges":     n_edges,
        "edge_rate":   round(edge_rate, 2),
        "edge_wins":   wins,
        "edge_losses": losses,
        "sim_roi":     round(roi, 2),
    }


def calibration_table(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Win rate by model confidence bucket (over-side only, no pushes)."""
    df = oos_df.dropna(subset=["market_line", "over_prob"]).copy()
    df = df[df["actual_direction"] != "P"]
    df["actual_over"] = (df["actual_direction"] == "O").astype(int)

    bins   = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 1.01]
    labels = ["50-52.5%", "52.5-55%", "55-57.5%", "57.5-60%", "60-65%", "65-70%", "70%+"]
    df["bucket"] = pd.cut(
        df["over_prob"].clip(0.5, 1.0), bins=bins, labels=labels, right=False
    )
    return (
        df.groupby("bucket", observed=True)
        .agg(n=("actual_over", "count"), win_rate=("actual_over", "mean"))
        .reset_index()
        .assign(win_rate=lambda x: x["win_rate"].round(3))
    )

# =============================================================================
# FOLD RUNNER
# =============================================================================

def run_fold(
    con: sqlite3.Connection,
    fold: dict,
    feature_version: str,
    feature_cols: list,
    edge_threshold: float,
    juice: int,
    run_date: str,
    model_cls=None,
    model_params: dict = None,
) -> tuple:
    """
    Train one fold, predict on test season, return (oos_df, metrics_dict).
    oos_df has one row per test-set game.
    """
    test_s  = fold["test_season"]
    train_s = fold["train_seasons"]
    log.info("── Fold %d  |  train=%s  |  test=%d ──", test_s, train_s, test_s)

    # ── Load ───────────────────────────────────────────────────────────────
    train_feat, train_meta = load_fold_data(con, train_s, feature_version)
    test_feat,  test_meta  = load_fold_data(con, [test_s], feature_version)

    if train_feat.empty:
        log.warning("Fold %d: no training data — skipping", test_s)
        return pd.DataFrame(), {}
    if test_feat.empty:
        log.warning("Fold %d: no test data — skipping", test_s)
        return pd.DataFrame(), {}

    # ── Prepare features ───────────────────────────────────────────────────
    X_train = prepare_X(train_feat, feature_cols)
    X_test  = prepare_X(test_feat,  feature_cols)

    # TARGET: actual_residual = actual_total - market_line.
    # The model learns "how far off is the market line?" not "what is the total?".
    # Positive residual → game went OVER; negative → UNDER.
    y_train = train_meta["actual_residual"].values.astype(float)
    y_test  = test_meta["actual_residual"].values.astype(float)

    # Align columns: add any train column missing from test (filled NaN)
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[X_train.columns]

    log.info("  Train: %d games  |  Test: %d games  |  Features: %d",
             len(X_train), len(X_test), X_train.shape[1])

    # ── Train ──────────────────────────────────────────────────────────────
    model = make_model(model_cls, model_params)
    model.fit(X_train, y_train)

    # residual_std: used by regression_to_proba to scale P(over).
    # We take max(in-sample prediction error std, historical residual std * 0.8).
    # Rationale: MLB game total residuals (actual - market line) have a natural
    # spread of ~4.0-4.5 runs.  If the model overfits training (in-sample std → 0),
    # the raw in-sample std gives delusional confidence (e.g., Φ(0.3/0.15) ≈ 97%).
    # Flooring at 80% of the training-set actual residual std ensures probabilities
    # remain plausible even when the model memorises training data in small folds.
    train_pred        = model.predict(X_train)
    insample_err_std  = float(np.std(y_train - train_pred))
    actual_resid_std  = float(np.std(y_train))          # spread of actual residuals
    residual_std      = max(insample_err_std, actual_resid_std * 0.80)
    log.info(
        "  In-sample residual RMSE=%.3f  in-sample-std=%.3f  "
        "actual-resid-std=%.3f  prob-std=%.3f",
        float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        insample_err_std, actual_resid_std, residual_std,
    )

    # ── Predict ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test).astype(float)   # predicted residual from market line

    has_line = test_meta["market_line"].notna().values   # always True after INNER JOIN filter

    # P(over) = P(actual_residual > 0 | predicted_residual, residual_std)
    # = Φ(predicted_residual / residual_std)   [Normal CDF at 0]
    # regression_to_proba(predicted, market_lines=0, std) achieves this.
    over_prob = np.where(
        has_line,
        regression_to_proba(y_pred, np.zeros(len(y_pred)), residual_std),
        np.nan,
    )

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred, over_prob, test_meta, edge_threshold, juice)
    metrics["fold_season"]   = test_s
    metrics["train_seasons"] = train_s
    metrics["residual_std"]  = round(residual_std, 4)

    # Top-20 feature importances for drift detection
    if hasattr(model, "feature_importances_"):
        imp = dict(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))
        metrics["top_features"] = dict(
            sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:20]
        )

    _log_fold_metrics(metrics)

    # ── Build OOS rows ─────────────────────────────────────────────────────
    # predicted_total column now stores the predicted residual (signed deviation from line).
    # direction: sign of predicted residual (O if positive, U if negative).
    # edge_value: |predicted_residual| (model's confidence gap relative to market).
    edge_val  = np.where(has_line, np.abs(y_pred), np.nan)
    direction = np.where(has_line, np.where(y_pred >= 0, "O", "U"), None)
    is_edge   = has_line & ~np.isnan(np.where(has_line, edge_val, 0.0)) & (edge_val >= edge_threshold)

    oos = test_meta.copy().reset_index(drop=True)
    oos["fold_season"]     = test_s
    oos["run_date"]        = run_date
    oos["feature_version"] = feature_version
    oos["predicted_total"] = np.round(y_pred, 3)   # stores predicted residual
    oos["over_prob"]       = np.round(over_prob, 4)
    oos["edge_value"]      = np.round(edge_val, 3)
    oos["direction"]       = direction
    oos["is_edge"]         = is_edge.astype(int)

    return oos, metrics


def _log_fold_metrics(m: dict) -> None:
    log.info(
        "  RMSE=%.3f  MAE=%.3f  O/U-acc=%.1f%%  log-loss=%s  "
        "edges=%d (%.1f%%)  sim-ROI=%.1f%%",
        m["rmse"],
        m["mae"],
        (m["uo_acc"] or 0.0) * 100,
        f"{m['log_loss']:.4f}" if m["log_loss"] is not None else "n/a",
        m["n_edges"],
        m["edge_rate"],
        m["sim_roi"],
    )

# =============================================================================
# OUTPUT
# =============================================================================

def write_oos_results(
    con: sqlite3.Connection,
    oos_df: pd.DataFrame,
    output_dir: str,
    run_date: str,
) -> None:
    """Write OOS predictions to SQLite table + dated CSV."""
    if oos_df.empty:
        log.warning("No OOS predictions to write")
        return

    # ── SQLite ──────────────────────────────────────────────────────────────
    con.executescript(_OOS_DDL)

    db_cols = [
        "run_date", "feature_version", "fold_season", "game_id",
        "game_date", "season", "actual_total", "predicted_total",
        "over_prob", "market_line", "edge_value", "direction",
        "actual_direction", "is_edge",
    ]
    df_db = oos_df[[c for c in db_cols if c in oos_df.columns]].copy()
    if "game_date" in df_db.columns and pd.api.types.is_datetime64_any_dtype(df_db["game_date"]):
        df_db["game_date"] = df_db["game_date"].dt.strftime("%Y-%m-%d")

    con.execute("BEGIN")
    try:
        cols_str  = ", ".join(df_db.columns)
        ph_str    = ", ".join("?" * len(df_db.columns))
        sql       = f"INSERT OR REPLACE INTO oos_predictions ({cols_str}) VALUES ({ph_str})"
        rows      = [tuple(r) for r in df_db.itertuples(index=False)]
        con.executemany(sql, rows)
        con.execute("COMMIT")
        log.info("Wrote %d rows -> oos_predictions (SQLite)", len(rows))
    except Exception:
        con.execute("ROLLBACK")
        raise

    # ── CSV ────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_name = f"oos_predictions_{run_date}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    oos_df.to_csv(csv_path, index=False)
    log.info("Wrote CSV  -> %s", csv_path)


def save_fold_metrics(all_metrics: list, output_dir: str, run_date: str) -> None:
    """Serialize per-fold metrics to JSON (includes feature importances)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"fold_metrics_{run_date}.json")

    # numpy types aren't JSON-serialisable; convert
    def _clean(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj

    clean = [{k: _clean(v) for k, v in m.items()} for m in all_metrics]
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)
    log.info("Fold metrics -> %s", path)


def print_summary(all_metrics: list) -> None:
    """Formatted fold-by-fold summary table with cross-fold averages."""
    if not all_metrics:
        return

    header = (
        f"\n  {'Season':>6}  {'Games':>6}  {'RMSE':>6}  {'MAE':>6}  "
        f"{'O/U%':>6}  {'LogLoss':>8}  {'Edges':>7}  {'ROI%':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for m in all_metrics:
        acc_str = f"{(m['uo_acc'] or 0)*100:5.1f}%" if m["uo_acc"] is not None else "   n/a"
        ll_str  = f"{m['log_loss']:.4f}"             if m["log_loss"] is not None else "    n/a"
        print(
            f"  {m['fold_season']:>6}  {m['n_games']:>6}  {m['rmse']:>6.3f}  {m['mae']:>6.3f}  "
            f"{acc_str:>6}  {ll_str:>8}  "
            f"{m['n_edges']:>4} ({m['edge_rate']:.1f}%)  {m['sim_roi']:>6.1f}%"
        )

    print(sep)
    valid_acc = [m["uo_acc"]   for m in all_metrics if m["uo_acc"]   is not None]
    valid_ll  = [m["log_loss"] for m in all_metrics if m["log_loss"] is not None]
    avg_rmse  = np.mean([m["rmse"]    for m in all_metrics])
    avg_mae   = np.mean([m["mae"]     for m in all_metrics])
    avg_roi   = np.mean([m["sim_roi"] for m in all_metrics])
    total_edg = sum(m["n_edges"] for m in all_metrics)
    print(
        f"  {'AVG':>6}  {'':>6}  {avg_rmse:>6.3f}  {avg_mae:>6.3f}  "
        f"  {np.mean(valid_acc)*100:4.1f}%  "
        f"{np.mean(valid_ll):>8.4f}  "
        f"{'total:':>4} {total_edg:>4}     {avg_roi:>6.1f}%\n"
    )

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward cross-validation for MLB totals model"
    )
    ap.add_argument(
        "--min-train-seasons", type=int, default=DEFAULT_MIN_TRAIN,
        metavar="N",
        help=f"Seasons before first test fold, by position (default {DEFAULT_MIN_TRAIN} -> first test=2021)",
    )
    ap.add_argument(
        "--include-covid", action="store_true",
        help="Include 2020 (60-game COVID season) in training and testing. Excluded by default.",
    )
    ap.add_argument(
        "--fold", type=int, metavar="YYYY",
        help="Run only this single test season (e.g. --fold 2024)",
    )
    ap.add_argument(
        "--feature-version", type=str, default=DEFAULT_FEATURE_VER,
        help=f"game_features version tag (default {DEFAULT_FEATURE_VER})",
    )
    ap.add_argument(
        "--edge-threshold", type=float, default=DEFAULT_EDGE_THRESH,
        help=f"Runs from market line to flag an edge (default {DEFAULT_EDGE_THRESH})",
    )
    ap.add_argument(
        "--juice", type=int, default=DEFAULT_JUICE,
        help=f"American odds for flat-bet ROI simulation (default {DEFAULT_JUICE})",
    )
    ap.add_argument("--db",         type=str, default=DB_PATH)
    ap.add_argument("--output-dir", type=str, default=OOS_DIR,
                    help="Directory for OOS CSV outputs")
    args = ap.parse_args()

    if not HAS_LGBM:
        log.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Determine available data ───────────────────────────────────────────
    all_seasons = get_available_seasons(con)
    if not all_seasons:
        log.error(
            "No seasons in game_features. Run engineer_features.py first."
        )
        con.close()
        sys.exit(1)

    if not args.include_covid and COVID_SEASON in all_seasons:
        log.info("COVID season (%d) excluded from training and testing. Use --include-covid to override.", COVID_SEASON)

    log.info("Seasons with features: %s", all_seasons)

    # ── Build folds ────────────────────────────────────────────────────────
    folds = get_season_folds(all_seasons, args.min_train_seasons, args.include_covid)

    if args.fold:
        folds = [f for f in folds if f["test_season"] == args.fold]
        if not folds:
            log.error(
                "--fold %d not valid (no fold found — insufficient training data or season unavailable)",
                args.fold,
            )
            con.close()
            sys.exit(1)

    if not folds:
        log.error(
            "No valid folds. Available seasons=%s, min_train_seasons=%d",
            all_seasons, args.min_train_seasons,
        )
        con.close()
        sys.exit(1)

    log.info(
        "%d fold(s) to run — test seasons: %s",
        len(folds), [f["test_season"] for f in folds],
    )

    # ── Determine feature column list once from schema ─────────────────────
    sample = pd.read_sql(
        f"SELECT * FROM game_features WHERE feature_version = ? LIMIT 1",
        con,
        params=(args.feature_version,),
    )
    if sample.empty:
        log.error(
            "No rows in game_features for version='%s'. Run engineer_features.py first.",
            args.feature_version,
        )
        con.close()
        sys.exit(1)

    feature_cols = get_feature_columns(sample)
    log.info("%d base feature columns (+ 2 encoded throws columns)", len(feature_cols))

    # ── Run all folds ──────────────────────────────────────────────────────
    all_oos: list     = []
    all_metrics: list = []

    for fold in folds:
        oos_df, metrics = run_fold(
            con           = con,
            fold          = fold,
            feature_version = args.feature_version,
            feature_cols  = feature_cols,
            edge_threshold = args.edge_threshold,
            juice         = args.juice,
            run_date      = run_date,
        )
        if not oos_df.empty:
            all_oos.append(oos_df)
        if metrics:
            all_metrics.append(metrics)

    if not all_oos:
        log.error("No OOS predictions generated. Check feature availability.")
        con.close()
        sys.exit(1)

    combined = pd.concat(all_oos, ignore_index=True)

    # ── Write results ──────────────────────────────────────────────────────
    write_oos_results(con, combined, args.output_dir, run_date)
    save_fold_metrics(all_metrics, args.output_dir, run_date)

    # ── Calibration table ─────────────────────────────────────────────────
    cal = calibration_table(combined)
    if not cal.empty:
        print("\nCalibration — over-side win rate by model confidence:")
        print(cal.to_string(index=False))

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary(all_metrics)

    con.close()
    log.info("Walk-forward validation complete.")


if __name__ == "__main__":
    main()
