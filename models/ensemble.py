"""
ensemble.py -- Stacking meta-learner for MLB totals prediction.

Combines LightGBM and XGBoost base model outputs via:
  - Logistic regression meta-learner -> calibrated over/under probability
  - Inverse-RMSE weighted blend     -> ensemble predicted total

Inputs at training time: OOF predictions from both base models (oof_predictions table).
Inputs at prediction time: raw game_features X + market_lines array.

The EnsembleTotalsModel is the single object loaded by predict_mlb.py (Step 11).
It holds both base models internally so predict_mlb.py loads one pkl and calls one method.

Usage:
    python models/ensemble.py                              # train, print report, save
    python models/ensemble.py --feature-version v1.1
    python models/ensemble.py --lgbm-pkl models/saved/lgbm_v1.0.pkl
    python models/ensemble.py --xgb-pkl  models/saved/xgb_v1.0.pkl
    python models/ensemble.py --calibrate                  # isotonic calibration layer
    python models/ensemble.py --no-save                    # dry run
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3
import joblib

# -- Project root on sys.path --------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.features import prepare_X, get_feature_columns, regression_to_proba  # noqa: E402
DEFAULT_FEATURE_VER = "v1.0"  # validate/ is a dev tool; keep it off the production import path
# Must be imported before any joblib.load() call so pickle can resolve the class
# even when base_models.py was originally run as __main__.
from models.base_models import LGBMTotalsModel, XGBTotalsModel                  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
_DIR      = os.path.dirname(__file__)
DB_PATH   = os.path.normpath(os.path.join(_DIR, "..", "mlb_data.db"))
SAVED_DIR = os.path.join(_DIR, "saved")

# -- Meta-feature schema (order is fixed; used for coefficient report) ---------
META_FEATURE_NAMES = [
    "lgbm_predicted_total",
    "xgb_predicted_total",
    "avg_predicted_total",
    "lgbm_over_prob",
    "xgb_over_prob",
    "avg_over_prob",
    "lgbm_edge",
    "xgb_edge",
    "avg_edge",
    "model_agreement",
    "market_line",
]

# =============================================================================
# META-FEATURE BUILDER
# =============================================================================

def build_meta_features(
    lgbm_pred:   np.ndarray,
    xgb_pred:    np.ndarray,
    lgbm_prob:   np.ndarray,
    xgb_prob:    np.ndarray,
    market_lines: np.ndarray,
) -> pd.DataFrame:
    """
    Build the 11-column meta-feature matrix fed to the logistic regression.
    Called at both train time (from OOF data) and predict time (from live outputs).
    Column order matches META_FEATURE_NAMES exactly.
    """
    avg_pred = (lgbm_pred + xgb_pred) / 2.0
    avg_prob = (lgbm_prob + xgb_prob) / 2.0

    # Residual model: predictions ARE deviations from the market line.
    # Edge = predicted residual (positive → expects over, negative → under).
    lgbm_edge = lgbm_pred
    xgb_edge  = xgb_pred
    avg_edge  = avg_pred

    # 1.0 = both models predict same direction (both +ve or both -ve); 0.0 = split
    model_agreement = (np.sign(lgbm_pred) == np.sign(xgb_pred)).astype(float)

    return pd.DataFrame({
        "lgbm_predicted_total": lgbm_pred,
        "xgb_predicted_total":  xgb_pred,
        "avg_predicted_total":  avg_pred,
        "lgbm_over_prob":       lgbm_prob,
        "xgb_over_prob":        xgb_prob,
        "avg_over_prob":        avg_prob,
        "lgbm_edge":            lgbm_edge,
        "xgb_edge":             xgb_edge,
        "avg_edge":             avg_edge,
        "model_agreement":      model_agreement,
        "market_line":          market_lines,
    }, columns=META_FEATURE_NAMES)


# =============================================================================
# BLEND WEIGHTS
# =============================================================================

def compute_blend_weights(oof_df: pd.DataFrame) -> tuple:
    """
    Inverse-RMSE weights for the regression blend of predicted totals.
    Better OOF RMSE -> higher weight in the blended prediction.
    Returns (w_lgbm, w_xgb) normalized to sum = 1.
    """
    # Residual model: predicted_total stores the predicted residual.
    # Compute actual_residual = actual_total - market_line for RMSE comparison.
    actual_residual = (
        oof_df["actual_total"].astype(float) - oof_df["market_line"].astype(float)
    ).values

    lgbm_rmse = float(np.sqrt(np.mean((oof_df["lgbm_predicted_total"].values - actual_residual) ** 2)))
    xgb_rmse  = float(np.sqrt(np.mean((oof_df["xgb_predicted_total"].values  - actual_residual) ** 2)))

    w_l = 1.0 / max(lgbm_rmse, 1e-6)
    w_x = 1.0 / max(xgb_rmse,  1e-6)
    total = w_l + w_x

    log.info(
        "OOF RMSE: lgbm=%.4f  xgb=%.4f  ->  blend weights: lgbm=%.1f%%  xgb=%.1f%%",
        lgbm_rmse, xgb_rmse, w_l / total * 100, w_x / total * 100,
    )
    return (w_l / total, w_x / total)


# =============================================================================
# OOF DATA LOADING
# =============================================================================

def load_oof_data(con: sqlite3.Connection, feature_version: str) -> pd.DataFrame:
    """
    Load and pivot OOF predictions from both base models into one row per game.

    Uses the most recent run_date per model independently, then inner-joins on
    game_id so only games with predictions from BOTH models are kept.

    Excluded:
      - Games where market_line is NULL (no line to bet against)
      - Pushes (actual_total == market_line, no binary label possible)
      - Games with NULL over_prob from either model (data gap)
    """
    # Most recent OOF run per model
    recent = pd.read_sql("""
        SELECT model, MAX(run_date) AS run_date
        FROM oof_predictions
        WHERE feature_version = ?
        GROUP BY model
    """, con, params=(feature_version,))

    if recent.empty:
        log.error(
            "oof_predictions table is empty for version='%s'. "
            "Run base_models.py first.", feature_version,
        )
        return pd.DataFrame()

    models_found = set(recent["model"].tolist())
    if not {"lgbm", "xgb"}.issubset(models_found):
        log.error(
            "OOF predictions missing for model(s): %s  (found: %s)",
            {"lgbm", "xgb"} - models_found, models_found,
        )
        return pd.DataFrame()

    lgbm_run = recent.loc[recent["model"] == "lgbm", "run_date"].iloc[0]
    xgb_run  = recent.loc[recent["model"] == "xgb",  "run_date"].iloc[0]

    lgbm_oof = pd.read_sql("""
        SELECT game_id, actual_total, market_line, actual_direction, fold_season,
               predicted_total  AS lgbm_predicted_total,
               over_prob        AS lgbm_over_prob
        FROM oof_predictions
        WHERE feature_version = ? AND model = 'lgbm' AND run_date = ?
    """, con, params=(feature_version, lgbm_run))

    xgb_oof = pd.read_sql("""
        SELECT game_id,
               predicted_total  AS xgb_predicted_total,
               over_prob        AS xgb_over_prob
        FROM oof_predictions
        WHERE feature_version = ? AND model = 'xgb' AND run_date = ?
    """, con, params=(feature_version, xgb_run))

    merged = lgbm_oof.merge(xgb_oof, on="game_id", how="inner")

    n_before = len(merged)
    merged = merged.dropna(
        subset=["market_line", "lgbm_over_prob", "xgb_over_prob"]
    )
    merged = merged[merged["actual_direction"] != "P"]
    merged = merged.reset_index(drop=True)

    log.info(
        "OOF data: %d games total, %d usable (lgbm_run=%s, xgb_run=%s)",
        n_before, len(merged), lgbm_run, xgb_run,
    )
    return merged


# =============================================================================
# ENSEMBLE MODEL CLASS
# =============================================================================

class EnsembleTotalsModel:
    """
    Stacking ensemble for MLB totals.

    Holds:
      lgbm_model     -- fitted LGBMTotalsModel
      xgb_model      -- fitted XGBTotalsModel
      meta_clf       -- fitted LogisticRegression (or CalibratedClassifierCV)
      blend_weights  -- (w_lgbm, w_xgb) for regression predicted_total
      oof_metrics    -- dict of in-sample validation metrics
    """

    MODEL_NAME = "ensemble"

    def __init__(self):
        self.lgbm_model      = None
        self.xgb_model       = None
        self.meta_clf        = None
        self.blend_weights   = None
        self.feature_version = None
        self.trained_at      = None
        self.calibrated      = False
        self.oof_metrics     = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        lgbm_model,
        xgb_model,
        oof_df: pd.DataFrame,
        calibrate: bool = False,
        feature_version: str = DEFAULT_FEATURE_VER,
    ) -> "EnsembleTotalsModel":
        """
        Train the meta-learner on OOF predictions from both base models.

        Steps:
          1. Compute inverse-RMSE blend weights for predicted_total.
          2. Build meta-feature matrix from OOF predictions.
          3. Fit logistic regression on binary over/under label.
          4. Optionally wrap with isotonic regression calibration.
          5. Store in-sample metrics and both base models.
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed: pip install scikit-learn")

        self.lgbm_model      = lgbm_model
        self.xgb_model       = xgb_model
        self.feature_version = feature_version
        self.trained_at      = datetime.utcnow().isoformat()
        self.calibrated      = calibrate

        # -- 1. Blend weights -----------------------------------------------
        self.blend_weights = compute_blend_weights(oof_df)

        # -- 2. Meta-features -----------------------------------------------
        meta_X = build_meta_features(
            lgbm_pred    = oof_df["lgbm_predicted_total"].values,
            xgb_pred     = oof_df["xgb_predicted_total"].values,
            lgbm_prob    = oof_df["lgbm_over_prob"].values,
            xgb_prob     = oof_df["xgb_over_prob"].values,
            market_lines = oof_df["market_line"].values,
        )

        # -- 3. Binary label: 1 = over ---------------------------------------
        y = (oof_df["actual_total"].values > oof_df["market_line"].values).astype(int)
        log.info(
            "Meta-learner training set: %d games  over_rate=%.1f%%",
            len(y), y.mean() * 100,
        )

        # -- 4. Fit logistic regression --------------------------------------
        base_lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        )
        if calibrate:
            log.info("Applying isotonic regression calibration (cv=5) ...")
            self.meta_clf = CalibratedClassifierCV(
                base_lr, method="isotonic", cv=5
            )
        else:
            self.meta_clf = base_lr

        self.meta_clf.fit(meta_X.values, y)

        # -- 5. In-sample metrics -------------------------------------------
        self.oof_metrics = _compute_oof_metrics(
            meta_X, y, self.meta_clf, oof_df, self.blend_weights
        )
        log.info(
            "Ensemble in-sample (optimistic): RMSE=%.3f  O/U=%.1f%%  log-loss=%.4f",
            self.oof_metrics["blend_rmse"],
            self.oof_metrics["uo_accuracy"] * 100,
            self.oof_metrics["log_loss"],
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame, market_lines: np.ndarray) -> tuple:
        """
        Full ensemble prediction from raw game_features.

        Returns:
            predicted_total  -- inverse-RMSE blended regression output (float array)
            over_prob        -- meta-learner calibrated P(over) (float array)
        """
        return self.predict_total(X), self.predict_proba(X, market_lines)

    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        """Inverse-RMSE weighted blend of base model regression outputs."""
        lgbm_pred = self.lgbm_model.predict(X)
        xgb_pred  = self.xgb_model.predict(X)
        w_l, w_x  = self.blend_weights
        return w_l * lgbm_pred + w_x * xgb_pred

    def predict_proba(self, X: pd.DataFrame, market_lines: np.ndarray) -> np.ndarray:
        """
        Meta-learner over probability from raw game_features + market lines.

        Runs both base models internally; caller only needs to provide the
        game_features matrix and the current market line per game.
        """
        market_lines = np.asarray(market_lines, dtype=float)
        lgbm_pred = self.lgbm_model.predict(X)
        xgb_pred  = self.xgb_model.predict(X)

        # Base-model over_prob MUST be computed with market_lines=0 so it means
        # P(residual > 0) = P(actual_total > market_line) — identical to how the
        # OOF over_prob columns were generated at training time (base_models.py
        # passes np.zeros).  The base models are residual regressors; passing the
        # real line here would yield P(residual > ~8.5) ≈ 0.04, a train/serve skew
        # the meta-learner never saw, which inflates the ensemble toward OVER.
        zeros = np.zeros(len(lgbm_pred), dtype=float)
        lgbm_prob = self.lgbm_model.predict_proba(X, zeros)
        xgb_prob  = self.xgb_model.predict_proba(X, zeros)

        # The real market line is still passed through as the `market_line`
        # meta-feature (the meta-learner was trained on the real opening line).
        meta_X = build_meta_features(
            lgbm_pred, xgb_pred, lgbm_prob, xgb_prob, market_lines
        )
        # LogisticRegression cannot handle NaN.  Games with no market line
        # (odds not yet available) produce NaN in the edge and market_line
        # meta-features.  Replace with 0 — a neutral value that degrades
        # gracefully: the meta-learner will output ~50% probability, and
        # edge detection will not fire for no-line games anyway.
        meta_vals = np.nan_to_num(meta_X.values, nan=0.0)
        return self.meta_clf.predict_proba(meta_vals)[:, 1]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info("[ensemble] Saved -> %s  (%.1f MB)",
                 path, os.path.getsize(path) / 1_048_576)

    @classmethod
    def load(cls, path: str) -> "EnsembleTotalsModel":
        return joblib.load(path)


# =============================================================================
# METRICS + REPORTING
# =============================================================================

def _compute_oof_metrics(
    meta_X: pd.DataFrame,
    y: np.ndarray,
    meta_clf,
    oof_df: pd.DataFrame,
    blend_weights: tuple,
) -> dict:
    """In-sample metrics on the OOF data (optimistic — same data used for training)."""
    over_prob = meta_clf.predict_proba(meta_X.values)[:, 1]
    p_clip    = np.clip(over_prob, 1e-6, 1 - 1e-6)

    w_l, w_x = blend_weights
    mkt             = oof_df["market_line"].values.astype(float)
    actual_residual = (oof_df["actual_total"].values.astype(float) - mkt)
    blend           = w_l * oof_df["lgbm_predicted_total"].values + w_x * oof_df["xgb_predicted_total"].values

    # blend and actual_residual are both in residual space (~0).
    # Previously this compared blend (residual) to actual_total (absolute),
    # giving blend_rmse ~9 and uo_acc = "how often did under hit" — both wrong.
    blend_rmse = float(np.sqrt(np.mean((blend - actual_residual) ** 2)))
    blend_mae  = float(np.mean(np.abs(blend - actual_residual)))
    pred_dir   = np.where(blend > 0, "O", "U")   # residual > 0 → predicted over
    act_dir    = oof_df["actual_direction"].values
    uo_acc     = float(np.mean(pred_dir == act_dir))

    ensemble_ll = float(log_loss(y, p_clip))
    lgbm_ll = float(log_loss(y, np.clip(oof_df["lgbm_over_prob"].values, 1e-6, 1 - 1e-6)))
    xgb_ll  = float(log_loss(y, np.clip(oof_df["xgb_over_prob"].values,  1e-6, 1 - 1e-6)))

    return {
        "n_games":              len(y),
        "blend_rmse":           round(blend_rmse, 4),
        "blend_mae":            round(blend_mae,  4),
        "uo_accuracy":          round(uo_acc,     4),
        "log_loss":             round(ensemble_ll, 4),
        "lgbm_log_loss":        round(lgbm_ll,    4),
        "xgb_log_loss":         round(xgb_ll,     4),
        "ll_vs_lgbm":           round(lgbm_ll - ensemble_ll, 4),   # positive = improvement
        "ll_vs_xgb":            round(xgb_ll  - ensemble_ll, 4),
    }


def print_coefficient_report(ensemble: EnsembleTotalsModel) -> None:
    """Print blend weights and meta-learner coefficients (by absolute magnitude)."""
    w_l, w_x = ensemble.blend_weights
    print(f"\n  Regression blend weights:")
    print(f"    LightGBM  {w_l * 100:5.1f}%")
    print(f"    XGBoost   {w_x * 100:5.1f}%")

    clf = ensemble.meta_clf
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        inter = float(clf.intercept_[0])
        print(f"\n  Logistic meta-learner coefficients (|coef| descending):")
        print(f"    {'Feature':<30} {'Coef':>9}")
        print("    " + "-" * 42)
        for feat, coef in sorted(
            zip(META_FEATURE_NAMES, coefs), key=lambda t: abs(t[1]), reverse=True
        ):
            print(f"    {feat:<30} {coef:+9.4f}")
        print(f"    {'(intercept)':<30} {inter:+9.4f}")
    else:
        print("\n  Meta-learner: isotonic-calibrated")
        print("  (Coefficients not directly accessible from calibrated wrapper)")


def print_calibration_table(
    oof_df: pd.DataFrame, meta_clf, meta_X: pd.DataFrame
) -> None:
    """
    Win rate by predicted probability bucket — always printed regardless of --calibrate.
    Values < 50% are folded to the under side and counted symmetrically.
    """
    over_prob   = meta_clf.predict_proba(meta_X.values)[:, 1]
    actual_over = (oof_df["actual_total"].values > oof_df["market_line"].values).astype(int)

    df = pd.DataFrame({"over_prob": over_prob, "actual_over": actual_over})
    bins   = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 1.01]
    labels = ["50-52.5%", "52.5-55%", "55-57.5%", "57.5-60%",
              "60-65%",   "65-70%",   "70%+"]
    df["bucket"] = pd.cut(
        df["over_prob"].clip(0.5, 1.0), bins=bins, labels=labels, right=False
    )
    cal = (
        df.groupby("bucket", observed=True)
        .agg(n=("actual_over", "count"), win_rate=("actual_over", "mean"))
        .reset_index()
        .assign(win_rate=lambda x: (x["win_rate"] * 100).round(1))
    )

    print("\n  Calibration (in-sample -- optimistic; use walk_forward for true OOS):")
    print(f"    {'Bucket':<12} {'N':>6} {'Win%':>7}")
    print("    " + "-" * 28)
    for _, row in cal.iterrows():
        print(f"    {str(row['bucket']):<12} {int(row['n']):>6} {row['win_rate']:>6.1f}%")


def print_metrics_report(metrics: dict) -> None:
    """Print in-sample summary with base model comparison."""
    print(f"\n  In-sample OOF metrics (optimistic):")
    print(f"    Games:              {metrics['n_games']}")
    print(f"    Blend RMSE:         {metrics['blend_rmse']:.4f}")
    print(f"    Blend MAE:          {metrics['blend_mae']:.4f}")
    print(f"    O/U accuracy:       {metrics['uo_accuracy']*100:.1f}%")
    print(f"    Ensemble log-loss:  {metrics['log_loss']:.4f}")
    print(f"    LGBM log-loss:      {metrics['lgbm_log_loss']:.4f}  "
          f"(delta: {metrics['ll_vs_lgbm']:+.4f})")
    print(f"    XGB  log-loss:      {metrics['xgb_log_loss']:.4f}  "
          f"(delta: {metrics['ll_vs_xgb']:+.4f})")


# =============================================================================
# MODEL VERSION LOGGING
# =============================================================================

def log_ensemble_version(
    con: sqlite3.Connection,
    ensemble: EnsembleTotalsModel,
    pkl_path: str,
) -> None:
    """Insert or replace a row in model_versions for this ensemble."""
    version_name = f"ensemble_{ensemble.feature_version}"
    m = ensemble.oof_metrics

    con.execute("""
        INSERT OR REPLACE INTO model_versions
            (version_name, trained_at, val_accuracy, val_roi,
             feature_version, file_path, notes)
        VALUES (?, ?, ?, NULL, ?, ?, ?)
    """, (
        version_name,
        ensemble.trained_at,
        m.get("uo_accuracy"),
        ensemble.feature_version,
        pkl_path,
        json.dumps({
            "blend_weights":  ensemble.blend_weights,
            "log_loss":       m.get("log_loss"),
            "blend_rmse":     m.get("blend_rmse"),
            "calibrated":     ensemble.calibrated,
        }),
    ))
    con.commit()
    log.info("Logged model_versions: %s", version_name)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train stacking ensemble (logistic meta-learner) for MLB totals"
    )
    ap.add_argument(
        "--feature-version", type=str, default=DEFAULT_FEATURE_VER,
        help=f"Feature version tag (default: {DEFAULT_FEATURE_VER})",
    )
    ap.add_argument(
        "--lgbm-pkl", type=str,
        help="Path to fitted LGBMTotalsModel pkl (default: models/saved/lgbm_<ver>.pkl)",
    )
    ap.add_argument(
        "--xgb-pkl", type=str,
        help="Path to fitted XGBTotalsModel pkl (default: models/saved/xgb_<ver>.pkl)",
    )
    ap.add_argument(
        "--calibrate", action="store_true",
        help="Wrap meta-learner with isotonic regression calibration (cv=5)",
    )
    ap.add_argument(
        "--no-save", action="store_true",
        help="Dry run: train and report but do not write pkl or log model_versions",
    )
    ap.add_argument("--db",         type=str, default=DB_PATH)
    ap.add_argument("--saved-dir",  type=str, default=SAVED_DIR)
    args = ap.parse_args()

    if not HAS_SKLEARN:
        log.error("scikit-learn not installed: pip install scikit-learn")
        sys.exit(1)

    feature_ver = args.feature_version

    # -- Resolve pkl paths ----------------------------------------------------
    lgbm_pkl = args.lgbm_pkl or os.path.join(args.saved_dir, f"lgbm_{feature_ver}.pkl")
    xgb_pkl  = args.xgb_pkl  or os.path.join(args.saved_dir, f"xgb_{feature_ver}.pkl")

    for path, name in [(lgbm_pkl, "lgbm"), (xgb_pkl, "xgb")]:
        if not os.path.exists(path):
            log.error(
                "%s pkl not found: %s\n"
                "Run 'python models/base_models.py --model %s' first.",
                name, path, name,
            )
            sys.exit(1)

    # -- Load base models -----------------------------------------------------
    log.info("Loading LightGBM model from %s ...", lgbm_pkl)
    lgbm_model = joblib.load(lgbm_pkl)

    log.info("Loading XGBoost model from %s ...", xgb_pkl)
    xgb_model = joblib.load(xgb_pkl)

    # -- Load OOF data --------------------------------------------------------
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")

    log.info("Loading OOF predictions (feature_version=%s) ...", feature_ver)
    oof_df = load_oof_data(con, feature_ver)

    if oof_df.empty:
        log.error("No usable OOF data found. Aborting.")
        con.close()
        sys.exit(1)

    # -- Train ensemble -------------------------------------------------------
    log.info(
        "Training ensemble meta-learner on %d OOF games%s ...",
        len(oof_df), " (+ isotonic calibration)" if args.calibrate else "",
    )
    ensemble = EnsembleTotalsModel()
    ensemble.fit(
        lgbm_model  = lgbm_model,
        xgb_model   = xgb_model,
        oof_df      = oof_df,
        calibrate   = args.calibrate,
        feature_version = feature_ver,
    )

    # -- Reports --------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Ensemble v{feature_ver}  {'(calibrated)' if args.calibrate else ''}")
    print("=" * 60)

    print_coefficient_report(ensemble)

    # Build meta_X once for report functions
    meta_X_report = build_meta_features(
        lgbm_pred    = oof_df["lgbm_predicted_total"].values,
        xgb_pred     = oof_df["xgb_predicted_total"].values,
        lgbm_prob    = oof_df["lgbm_over_prob"].values,
        xgb_prob     = oof_df["xgb_over_prob"].values,
        market_lines = oof_df["market_line"].values,
    )
    print_calibration_table(oof_df, ensemble.meta_clf, meta_X_report)
    print_metrics_report(ensemble.oof_metrics)
    print()

    # -- Save -----------------------------------------------------------------
    if not args.no_save:
        pkl_name = f"ensemble_{feature_ver}.pkl"
        pkl_path = os.path.join(args.saved_dir, pkl_name)
        ensemble.save(pkl_path)
        log_ensemble_version(con, ensemble, pkl_path)
    else:
        log.info("--no-save: pkl and model_versions entry skipped")

    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
