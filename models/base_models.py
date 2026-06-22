"""
base_models.py -- LightGBM and XGBoost base regressors for MLB totals prediction.

Each model predicts total_runs (regression). Over/under probability is derived via
Normal CDF using the in-sample residual std (regression_to_proba in utils/features.py).
The stacking meta-learner (Step 6) replaces this with calibrated logistic probs.

Training uses two passes per model:
  Pass 1: fit with last-season hold-out to find best_iteration via early stopping.
  Pass 2: re-fit on the FULL training window using n_estimators = best_iteration.
          This is the model saved to disk and used in production.

Running this script:
  1. Generates OOF predictions for both models (needed by Step 6 meta-learner).
  2. Trains final models on all available data.
  3. Saves fitted models to models/saved/*.pkl.
  4. Logs metadata to model_versions table.

Usage:
    python models/base_models.py                       # train both, full pipeline
    python models/base_models.py --model lgbm          # LightGBM only
    python models/base_models.py --model xgb           # XGBoost only
    python models/base_models.py --no-oof              # skip OOF generation
    python models/base_models.py --feature-version v1.1
    python models/base_models.py --include-covid
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

# -- Project root on sys.path so sibling packages resolve ----------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.features import prepare_X, get_feature_columns, regression_to_proba
# validate/ is a dev tool; hardcode shared constants here so it is not a
# production import dependency.  Walk-forward *functions* are lazy-imported
# inside the two call sites below (generate_oof_predictions / main).
COVID_SEASON        = 2020
DEFAULT_FEATURE_VER = "v1.0"
DEFAULT_MIN_TRAIN   = 4      # first test fold = season index 4 (2021)
DEFAULT_EDGE_THRESH = 1.5    # runs away from market line to flag an edge
DEFAULT_JUICE       = -110   # American odds used for ROI simulation

# Walk-forward helpers — needed by generate_oof_predictions(), train_final_model(),
# and main().  Imported at module level so all three callers share one definition.
# The except branch provides functional stubs for environments where validate/ is
# absent; the SQL mirrors the real implementations in walk_forward.py exactly.
try:
    from validate.walk_forward import (
        get_season_folds,
        load_fold_data,
        compute_metrics,
        get_available_seasons,
        print_summary,
    )
except ModuleNotFoundError:
    def get_available_seasons(con, min_train=3):
        df = pd.read_sql(
            "SELECT DISTINCT g.season FROM game_features gf "
            "JOIN games g ON gf.game_id = g.game_id "
            "WHERE g.status IN ('Final','Completed Early') ORDER BY g.season",
            con,
        )
        return df["season"].tolist()

    def get_season_folds(all_seasons, min_train_seasons, include_covid):
        folds = []
        for i, test_s in enumerate(all_seasons):
            if i < min_train_seasons:
                continue
            if test_s == COVID_SEASON and not include_covid:
                continue
            train_s = [s for s in all_seasons[:i]
                       if include_covid or s != COVID_SEASON]
            if train_s:
                folds.append({"train_seasons": train_s, "test_season": test_s})
        return folds

    def load_fold_data(con, seasons, feature_version):
        s_in = ",".join(str(s) for s in seasons)
        meta = pd.read_sql(f"""
            SELECT g.game_id, g.game_date, g.season,
                   g.total_runs AS actual_total, o.total_line AS market_line
            FROM games g
            INNER JOIN (
                -- Earliest is_opening snapshot per game (NOT MIN(total_line),
                -- which biases the opening line downward — see walk_forward.py).
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
            SELECT gf.* FROM game_features gf
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
        meta_out["actual_direction"] = np.where(
            mkt.isna(), None,
            np.where(tot > mkt, "O", np.where(tot < mkt, "U", "P")),
        )
        meta_out["actual_residual"] = (tot - mkt).where(mkt.notna(), other=np.nan)
        drop = [c for c in ["game_date", "season", "actual_total", "market_line"]
                if c in combined.columns]
        feat_out = combined.drop(columns=drop, errors="ignore")
        return feat_out, meta_out

    def compute_metrics(y_true, y_pred, over_prob, meta_df, edge_threshold, juice):
        residuals = np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)
        return {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae":  float(np.mean(np.abs(residuals))),
        }

    def print_summary(fold_results):
        for r in fold_results:
            print(f"  {r}")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
OOF_DIR   = os.path.normpath(os.path.join(_DIR, "..", "validate", "oos_predictions"))

# -- OOF predictions table DDL (model-specific; separate from walk_forward's table)
_OOF_DDL = """
CREATE TABLE IF NOT EXISTS oof_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date         TEXT    NOT NULL,
    feature_version  TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    fold_season      INTEGER NOT NULL,
    game_id          INTEGER NOT NULL,
    game_date        TEXT,
    season           INTEGER,
    actual_total     INTEGER,
    predicted_total  REAL,
    over_prob        REAL,
    market_line      REAL,
    edge_value       REAL,
    direction        TEXT,
    actual_direction TEXT,
    is_edge          INTEGER DEFAULT 0,
    UNIQUE (run_date, model, game_id, feature_version)
);
"""

# =============================================================================
# LGBM MODEL WRAPPER
# =============================================================================

class LGBMTotalsModel:
    """
    LightGBM regressor wrapper with two-pass training and probability output.

    Exposes the same interface as XGBTotalsModel so the ensemble (Step 6)
    and prediction pipeline (Step 11) can treat both identically.
    """

    MODEL_NAME = "lgbm"

    # Hyperparameters chosen to complement XGBoost (different inductive bias,
    # leaf-wise vs level-wise growth, different regularization profile).
    PARAMS = {
        "learning_rate":     0.03,
        "num_leaves":        127,
        "min_child_samples": 30,
        "colsample_bytree":  0.7,
        "subsample":         0.8,
        "subsample_freq":    1,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    def __init__(self, params: dict = None):
        self.params        = {**self.PARAMS, **(params or {})}
        self.model         = None
        self.feature_cols  = None    # column list at training time (for alignment)
        self.residual_std  = None    # in-sample residual std (for P(over) via Normal CDF)
        self.feature_version = None
        self.trained_at    = None
        self.best_iteration = None
        self.train_seasons = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        train_dates: pd.Series = None,
        feature_version: str = DEFAULT_FEATURE_VER,
        train_seasons: list = None,
    ) -> "LGBMTotalsModel":
        """
        Two-pass training.

        Pass 1 (if train_dates provided):
            Hold out the last year's games chronologically.
            Fit with early stopping (patience=50) to find best_iteration.

        Pass 2:
            Re-fit on the FULL X/y with n_estimators = best_iteration.
            This is the model that gets saved and deployed.
        """
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed: pip install lightgbm")

        self.feature_cols    = X.columns.tolist()
        self.feature_version = feature_version
        self.trained_at      = datetime.utcnow().isoformat()
        self.train_seasons   = train_seasons or []

        # -- Pass 1: early stopping to find best_iteration -------------------
        # After the odds filter, some folds have all training games in a single
        # season (e.g. 2021 fold: train=[2017,2018,2019] but only 2019 has odds).
        # Holding out last_year would empty X_tr → "Found array with 0 sample(s)".
        # Guard: if the non-holdout split has < 100 rows, skip early stopping
        # and train directly on the full set with a conservative fixed budget.
        _do_two_pass = False
        if train_dates is not None:
            last_year = pd.to_datetime(train_dates).dt.year.max()
            val_mask  = pd.to_datetime(train_dates).dt.year == last_year
            X_es, y_es = X[val_mask].values, y[val_mask]
            X_tr, y_tr = X[~val_mask].values, y[~val_mask]
            _do_two_pass = len(X_tr) >= 100 and len(X_es) >= 20

        if _do_two_pass:
            log.info(
                "[lgbm] Pass 1 -- early stopping holdout=%d  "
                "train_n=%d  val_n=%d",
                last_year, len(X_tr), len(X_es),
            )
            m1 = lgb.LGBMRegressor(**self.params, n_estimators=2000)
            m1.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            self.best_iteration = m1.best_iteration_
            log.info("[lgbm] Best iteration: %d", self.best_iteration)

            # -- Pass 2: re-fit on full training data with best_iteration ------
            log.info(
                "[lgbm] Pass 2 -- fitting on full set  n=%d  n_estimators=%d",
                len(X), self.best_iteration,
            )
            self.model = lgb.LGBMRegressor(**self.params, n_estimators=self.best_iteration)
            self.model.fit(X.values, y)
        else:
            # Single-pass fallback: holdout unavailable (thin data after odds filter)
            # or no dates provided. Train directly on full X with fixed budget.
            self.best_iteration = 500
            if train_dates is not None:
                log.warning(
                    "[lgbm] Two-pass skipped: holdout=%d would leave only %d train rows "
                    "(need >=100). Single-pass on full set, n_estimators=%d.",
                    last_year if train_dates is not None else -1,
                    len(X_tr) if train_dates is not None else len(X),
                    self.best_iteration,
                )
            log.info(
                "[lgbm] Single-pass -- fitting on full set  n=%d  n_estimators=%d",
                len(X), self.best_iteration,
            )
            self.model = lgb.LGBMRegressor(**self.params, n_estimators=self.best_iteration)
            self.model.fit(X.values, y)

        train_pred        = self.model.predict(X.values)
        insample_err_std  = float(np.std(y - train_pred))
        actual_resid_std  = float(np.std(y))
        # Floor prob-sigma at 0.8 x actual-residual spread (matches walk_forward.py).
        # Without this, an overfit model drives in-sample std toward 0 and gives
        # delusional confidence; the DEPLOYED model must use the same sigma as the
        # backtest or live P(over) won't match the validated calibration.
        self.residual_std = max(insample_err_std, actual_resid_std * 0.80)
        log.info(
            "[lgbm] In-sample RMSE=%.3f  insample-std=%.3f  actual-resid-std=%.3f  prob-std=%.3f",
            float(np.sqrt(np.mean((train_pred - y) ** 2))),
            insample_err_std, actual_resid_std, self.residual_std,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted total runs. Aligns columns to training set."""
        return self.model.predict(self._align(X).values)

    def predict_proba(self, X: pd.DataFrame, market_lines: np.ndarray) -> np.ndarray:
        """P(actual_total > market_line) via Normal CDF."""
        return regression_to_proba(self.predict(X), market_lines, self.residual_std)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Stamp the canonical module so joblib.load() resolves the class
        # correctly regardless of how this script was invoked (__main__ vs import).
        self.__class__.__module__ = "models.base_models"
        joblib.dump(self, path)
        log.info("[lgbm] Saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "LGBMTotalsModel":
        return joblib.load(path)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder/fill columns to match the training feature set."""
        dupes = X.columns[X.columns.duplicated()].tolist()
        if dupes:
            log.warning("[lgbm] duplicate columns before reindex (will be dropped by prepare_X): %s", dupes)
            X = X.loc[:, ~X.columns.duplicated()]
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            log.warning("[lgbm] %d feature(s) missing at predict time: %s",
                        len(missing), missing[:5])
        return X.reindex(columns=self.feature_cols, fill_value=np.nan).astype(float)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_ if self.model else None


# =============================================================================
# XGB MODEL WRAPPER
# =============================================================================

class XGBTotalsModel:
    """
    XGBoost regressor wrapper with two-pass training and probability output.

    Level-wise growth (vs LightGBM's leaf-wise) produces different tree structures,
    giving the ensemble meaningful diversity to exploit.
    """

    MODEL_NAME = "xgb"

    PARAMS = {
        "learning_rate":   0.03,
        "max_depth":       6,
        "min_child_weight": 5,
        "colsample_bytree": 0.7,
        "subsample":       0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      2.0,
        "tree_method":     "hist",
        "random_state":    42,
        "n_jobs":         -1,
        "verbosity":       0,
    }

    def __init__(self, params: dict = None):
        self.params          = {**self.PARAMS, **(params or {})}
        self.model           = None
        self.feature_cols    = None
        self.residual_std    = None
        self.feature_version = None
        self.trained_at      = None
        self.best_iteration  = None
        self.train_seasons   = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        train_dates: pd.Series = None,
        feature_version: str = DEFAULT_FEATURE_VER,
        train_seasons: list = None,
    ) -> "XGBTotalsModel":
        """Two-pass training identical in structure to LGBMTotalsModel.fit()."""
        if not HAS_XGB:
            raise ImportError("xgboost not installed: pip install xgboost")

        self.feature_cols    = X.columns.tolist()
        self.feature_version = feature_version
        self.trained_at      = datetime.utcnow().isoformat()
        self.train_seasons   = train_seasons or []

        # -- Pass 1 (with same thin-data guard as LGBMTotalsModel) -------------
        _do_two_pass = False
        if train_dates is not None:
            last_year = pd.to_datetime(train_dates).dt.year.max()
            val_mask  = pd.to_datetime(train_dates).dt.year == last_year
            X_es, y_es = X[val_mask].values, y[val_mask]
            X_tr, y_tr = X[~val_mask].values, y[~val_mask]
            _do_two_pass = len(X_tr) >= 100 and len(X_es) >= 20

        if _do_two_pass:
            log.info(
                "[xgb] Pass 1 -- early stopping holdout=%d  "
                "train_n=%d  val_n=%d",
                last_year, len(X_tr), len(X_es),
            )
            m1 = xgb.XGBRegressor(
                **self.params,
                n_estimators=2000,
                early_stopping_rounds=50,
            )
            m1.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                verbose=100,
            )
            self.best_iteration = int(m1.best_iteration)
            log.info("[xgb] Best iteration: %d", self.best_iteration)

            # -- Pass 2 --------------------------------------------------------
            log.info(
                "[xgb] Pass 2 -- fitting on full set  n=%d  n_estimators=%d",
                len(X), self.best_iteration,
            )
            self.model = xgb.XGBRegressor(**self.params, n_estimators=self.best_iteration)
            self.model.fit(X.values, y)
        else:
            self.best_iteration = 500
            if train_dates is not None:
                log.warning(
                    "[xgb] Two-pass skipped: holdout=%d would leave only %d train rows "
                    "(need >=100). Single-pass on full set, n_estimators=%d.",
                    last_year if train_dates is not None else -1,
                    len(X_tr) if train_dates is not None else len(X),
                    self.best_iteration,
                )
            log.info(
                "[xgb] Single-pass -- fitting on full set  n=%d  n_estimators=%d",
                len(X), self.best_iteration,
            )
            self.model = xgb.XGBRegressor(**self.params, n_estimators=self.best_iteration)
            self.model.fit(X.values, y)

        train_pred        = self.model.predict(X.values)
        insample_err_std  = float(np.std(y - train_pred))
        actual_resid_std  = float(np.std(y))
        # Floor prob-sigma at 0.8 x actual-residual spread (matches walk_forward.py).
        # See LGBMTotalsModel.fit() for rationale.
        self.residual_std = max(insample_err_std, actual_resid_std * 0.80)
        log.info(
            "[xgb] In-sample RMSE=%.3f  insample-std=%.3f  actual-resid-std=%.3f  prob-std=%.3f",
            float(np.sqrt(np.mean((train_pred - y) ** 2))),
            insample_err_std, actual_resid_std, self.residual_std,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._align(X).values)

    def predict_proba(self, X: pd.DataFrame, market_lines: np.ndarray) -> np.ndarray:
        return regression_to_proba(self.predict(X), market_lines, self.residual_std)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.__class__.__module__ = "models.base_models"
        joblib.dump(self, path)
        log.info("[xgb] Saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBTotalsModel":
        return joblib.load(path)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        dupes = X.columns[X.columns.duplicated()].tolist()
        if dupes:
            log.warning("[xgb] duplicate columns before reindex (will be dropped by prepare_X): %s", dupes)
            X = X.loc[:, ~X.columns.duplicated()]
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            log.warning("[xgb] %d feature(s) missing at predict time: %s",
                        len(missing), missing[:5])
        return X.reindex(columns=self.feature_cols, fill_value=np.nan).astype(float)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_ if self.model else None


# =============================================================================
# OOF GENERATION
# =============================================================================

def generate_oof_predictions(
    con: sqlite3.Connection,
    model_cls,
    feature_version: str,
    min_train_seasons: int,
    include_covid: bool,
    edge_threshold: float,
    juice: int,
    run_date: str,
) -> tuple:
    """
    Walk-forward OOF predictions for model_cls with two-pass training per fold.

    Returns (oof_df, metrics_list).
    oof_df has one row per test-set game across all folds.
    metrics_list has one dict per fold with RMSE, MAE, ROI, etc.
    """
    all_seasons  = get_available_seasons(con)
    folds        = get_season_folds(all_seasons, min_train_seasons, include_covid)

    # Canonical feature column list from schema
    sample = pd.read_sql(
        "SELECT * FROM game_features WHERE feature_version = ? LIMIT 1",
        con, params=(feature_version,),
    )
    if sample.empty:
        log.error("No game_features rows for version=%s", feature_version)
        return pd.DataFrame(), []

    feature_cols = get_feature_columns(sample)

    all_oof: list     = []
    all_metrics: list = []
    name = model_cls.MODEL_NAME

    for fold in folds:
        test_s  = fold["test_season"]
        train_s = fold["train_seasons"]
        log.info("[%s] OOF fold %d | train=%s", name, test_s, train_s)

        train_feat, train_meta = load_fold_data(con, train_s, feature_version)
        test_feat,  test_meta  = load_fold_data(con, [test_s], feature_version)

        if train_feat.empty or test_feat.empty:
            log.warning("[%s] Fold %d: insufficient data -- skipped", name, test_s)
            continue

        X_train = prepare_X(train_feat, feature_cols)
        X_test  = prepare_X(test_feat,  feature_cols)

        # Minimum fold size guard: after the odds filter, early folds may have
        # very few training games (e.g. 2021 fold has only 2019-May→Sep ≈ 1,600).
        # Thin folds produce unreliable OOF predictions that pollute the meta-learner.
        MIN_TRAIN_GAMES = 500
        if len(X_train) < MIN_TRAIN_GAMES:
            log.warning(
                "[%s] Fold %d: train_n=%d < %d after odds filter — skipping "
                "(insufficient market-line data in training seasons %s)",
                name, test_s, len(X_train), MIN_TRAIN_GAMES, train_s,
            )
            continue
        # Residual model: target = actual_total - market_line
        y_train = train_meta["actual_residual"].values.astype(float)
        y_test  = test_meta["actual_residual"].values.astype(float)

        # Align test columns to training set
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = np.nan
        X_test = X_test[X_train.columns]

        log.info("[%s] Fold %d: train=%d  test=%d  features=%d",
                 name, test_s, len(X_train), len(X_test), X_train.shape[1])

        # Two-pass training for this fold
        instance = model_cls()
        instance.fit(
            X_train, y_train,
            train_dates    = train_meta["game_date"],
            feature_version= feature_version,
            train_seasons  = train_s,
        )

        # Predictions — y_pred is predicted residual from market line
        y_pred   = instance.predict(X_test)
        has_line = test_meta["market_line"].notna().values
        # P(over) = P(residual > 0) = Φ(y_pred / residual_std)
        over_prob = np.where(
            has_line,
            instance.predict_proba(X_test, np.zeros(len(X_test))),
            np.nan,
        )

        # Metrics
        metrics = compute_metrics(
            y_test, y_pred, over_prob, test_meta, edge_threshold, juice
        )
        metrics["fold_season"]   = test_s
        metrics["train_seasons"] = train_s
        metrics["model"]         = name
        metrics["best_iteration"] = instance.best_iteration
        metrics["residual_std"]   = round(instance.residual_std, 4)

        if hasattr(instance, "feature_importances_") and instance.feature_importances_ is not None:
            imp = dict(zip(X_train.columns.tolist(),
                           instance.feature_importances_.tolist()))
            metrics["top_features"] = dict(
                sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:20]
            )

        log.info(
            "[%s] Fold %d: RMSE=%.3f  MAE=%.3f  O/U=%.1f%%  edges=%d  ROI=%.1f%%",
            name, test_s,
            metrics["rmse"], metrics["mae"],
            (metrics["uo_acc"] or 0) * 100,
            metrics["n_edges"], metrics["sim_roi"],
        )
        all_metrics.append(metrics)

        # Build OOF rows — edge = |predicted residual|, direction = sign(residual)
        edge_val  = np.where(has_line, np.abs(y_pred), np.nan)
        direction = np.where(has_line, np.where(y_pred >= 0, "O", "U"), None)
        is_edge   = (
            has_line
            & ~np.isnan(np.where(has_line, edge_val, 0.0))
            & (edge_val >= edge_threshold)
        )

        oof = test_meta.copy().reset_index(drop=True)
        oof["fold_season"]     = test_s
        oof["run_date"]        = run_date
        oof["feature_version"] = feature_version
        oof["model"]           = name
        oof["predicted_total"] = np.round(y_pred, 3)
        oof["over_prob"]       = np.round(over_prob, 4)
        oof["edge_value"]      = np.round(edge_val, 3)
        oof["direction"]       = direction
        oof["is_edge"]         = is_edge.astype(int)
        all_oof.append(oof)

    combined = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    return combined, all_metrics


# =============================================================================
# OOF PERSISTENCE
# =============================================================================

def write_oof_to_db(con: sqlite3.Connection, oof_df: pd.DataFrame) -> None:
    """Write OOF predictions to the oof_predictions SQLite table."""
    if oof_df.empty:
        return
    con.executescript(_OOF_DDL)

    db_cols = [
        "run_date", "feature_version", "model", "fold_season", "game_id",
        "game_date", "season", "actual_total", "predicted_total",
        "over_prob", "market_line", "edge_value", "direction",
        "actual_direction", "is_edge",
    ]
    df_out = oof_df[[c for c in db_cols if c in oof_df.columns]].copy()
    if "game_date" in df_out.columns and pd.api.types.is_datetime64_any_dtype(df_out["game_date"]):
        df_out["game_date"] = df_out["game_date"].dt.strftime("%Y-%m-%d")

    con.execute("BEGIN")
    try:
        cols_str = ", ".join(df_out.columns)
        ph_str   = ", ".join("?" * len(df_out.columns))
        sql      = f"INSERT OR REPLACE INTO oof_predictions ({cols_str}) VALUES ({ph_str})"
        rows     = [tuple(r) for r in df_out.itertuples(index=False)]
        con.executemany(sql, rows)
        con.execute("COMMIT")
        log.info("Wrote %d rows -> oof_predictions (SQLite)", len(rows))
    except Exception:
        con.execute("ROLLBACK")
        raise


def write_oof_to_csv(oof_df: pd.DataFrame, output_dir: str, model_name: str,
                     feature_version: str, run_date: str) -> str:
    """Write OOF predictions to a per-model dated CSV. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"oof_{model_name}_{feature_version}_{run_date}.csv"
    path  = os.path.join(output_dir, fname)
    oof_df.to_csv(path, index=False)
    log.info("Wrote CSV -> %s", path)
    return path


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(
    con: sqlite3.Connection,
    model_cls,
    all_seasons: list,
    include_covid: bool,
    feature_version: str,
    feature_cols: list,
) -> object:
    """
    Train model_cls on the full available dataset (no test holdout).
    Two-pass: last season = early stopping holdout; then re-fit on all data.
    Returns the fitted model instance.
    """
    train_seasons = [s for s in all_seasons
                     if include_covid or s != COVID_SEASON]

    log.info(
        "[%s] Final training on seasons %s ...",
        model_cls.MODEL_NAME, train_seasons,
    )
    feat_df, meta_df = load_fold_data(con, train_seasons, feature_version)

    if feat_df.empty:
        raise RuntimeError(
            f"No game_features rows for seasons={train_seasons}, version={feature_version}"
        )

    X = prepare_X(feat_df, feature_cols)
    # Residual model: target = actual_total - market_line
    y = meta_df["actual_residual"].values.astype(float)

    log.info("[%s] Final: n=%d  features=%d", model_cls.MODEL_NAME, len(X), X.shape[1])

    instance = model_cls()
    instance.fit(
        X, y,
        train_dates    = meta_df["game_date"],
        feature_version= feature_version,
        train_seasons  = train_seasons,
    )
    return instance


# =============================================================================
# MODEL VERSION LOGGING
# =============================================================================

def log_model_version(
    con: sqlite3.Connection,
    model: object,
    val_metrics: list,
    pkl_path: str,
) -> None:
    """Insert or replace a row in model_versions for this trained model."""
    version_name = f"{model.MODEL_NAME}_{model.feature_version}"

    # Aggregate OOF metrics across folds for summary
    acc_vals = [m["uo_acc"]   for m in val_metrics if m.get("uo_acc")   is not None]
    roi_vals = [m["sim_roi"]  for m in val_metrics if m.get("sim_roi")  is not None]
    val_accuracy = float(np.mean(acc_vals)) if acc_vals else None
    val_roi      = float(np.mean(roi_vals)) if roi_vals else None

    train_start = min(model.train_seasons) if model.train_seasons else None
    train_end   = max(model.train_seasons) if model.train_seasons else None

    con.execute("""
        INSERT OR REPLACE INTO model_versions
            (version_name, trained_at, train_start_date, train_end_date,
             val_accuracy, val_roi, feature_version, file_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_name,
        model.trained_at,
        str(train_start),
        str(train_end),
        val_accuracy,
        val_roi,
        model.feature_version,
        pkl_path,
        json.dumps({"best_iteration": model.best_iteration,
                    "residual_std":   model.residual_std}),
    ))
    con.commit()
    log.info("Logged model_versions: %s", version_name)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train LightGBM and XGBoost base models for MLB totals"
    )
    ap.add_argument(
        "--model", choices=["lgbm", "xgb", "both"], default="both",
        help="Which model(s) to train (default: both)",
    )
    ap.add_argument(
        "--no-oof", action="store_true",
        help="Skip OOF generation; train final models only",
    )
    ap.add_argument(
        "--feature-version", type=str, default=DEFAULT_FEATURE_VER,
    )
    ap.add_argument(
        "--min-train-seasons", type=int, default=DEFAULT_MIN_TRAIN,
    )
    ap.add_argument(
        "--include-covid", action="store_true",
        help="Include 2020 season in training (excluded by default)",
    )
    ap.add_argument(
        "--edge-threshold", type=float, default=DEFAULT_EDGE_THRESH,
    )
    ap.add_argument(
        "--juice", type=int, default=DEFAULT_JUICE,
    )
    ap.add_argument("--db",         type=str, default=DB_PATH)
    ap.add_argument("--output-dir", type=str, default=OOF_DIR,
                    help="Directory for OOF CSV files")
    ap.add_argument("--saved-dir",  type=str, default=SAVED_DIR,
                    help="Directory for fitted .pkl files")
    args = ap.parse_args()

    # -- Validate library availability ----------------------------------------
    model_map = {"lgbm": LGBMTotalsModel, "xgb": XGBTotalsModel}
    if args.model in ("lgbm", "both") and not HAS_LGBM:
        log.error("lightgbm not installed: pip install lightgbm")
        sys.exit(1)
    if args.model in ("xgb", "both") and not HAS_XGB:
        log.error("xgboost not installed: pip install xgboost")
        sys.exit(1)

    models_to_train = (
        [LGBMTotalsModel, XGBTotalsModel] if args.model == "both"
        else [model_map[args.model]]
    )

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    run_date      = datetime.now().strftime("%Y-%m-%d")
    feature_ver   = args.feature_version

    # -- Determine canonical feature columns once -----------------------------
    sample = pd.read_sql(
        "SELECT * FROM game_features WHERE feature_version = ? LIMIT 1",
        con, params=(feature_ver,),
    )
    if sample.empty:
        log.error(
            "No rows in game_features for version='%s'. "
            "Run engineer_features.py first.", feature_ver,
        )
        con.close()
        sys.exit(1)

    feature_cols = get_feature_columns(sample)
    log.info("%d base feature columns (+2 encoded throws)", len(feature_cols))

    all_seasons = get_available_seasons(con)
    log.info("Seasons with features: %s", all_seasons)

    os.makedirs(args.saved_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    all_fold_metrics: list = []

    for model_cls in models_to_train:
        name = model_cls.MODEL_NAME
        log.info("=" * 60)
        log.info("Model: %s", name.upper())
        log.info("=" * 60)

        fold_metrics: list = []

        # -- OOF predictions --------------------------------------------------
        if not args.no_oof:
            log.info("[%s] Generating OOF predictions ...", name)
            oof_df, fold_metrics = generate_oof_predictions(
                con            = con,
                model_cls      = model_cls,
                feature_version= feature_ver,
                min_train_seasons = args.min_train_seasons,
                include_covid  = args.include_covid,
                edge_threshold = args.edge_threshold,
                juice          = args.juice,
                run_date       = run_date,
            )
            if not oof_df.empty:
                write_oof_to_db(con, oof_df)
                write_oof_to_csv(oof_df, args.output_dir, name, feature_ver, run_date)
                print_summary(fold_metrics)
            else:
                log.warning("[%s] No OOF predictions generated", name)

            all_fold_metrics.extend(fold_metrics)

        # -- Final model (all available data) ---------------------------------
        log.info("[%s] Training final model on all seasons ...", name)
        final_model = train_final_model(
            con           = con,
            model_cls     = model_cls,
            all_seasons   = all_seasons,
            include_covid = args.include_covid,
            feature_version = feature_ver,
            feature_cols  = feature_cols,
        )
        final_model.train_seasons = [
            s for s in all_seasons
            if args.include_covid or s != COVID_SEASON
        ]

        pkl_name = f"{name}_{feature_ver}.pkl"
        pkl_path = os.path.join(args.saved_dir, pkl_name)
        final_model.save(pkl_path)

        # -- Log to model_versions --------------------------------------------
        log_model_version(con, final_model, fold_metrics, pkl_path)

    log.info("Done. Models saved to %s", args.saved_dir)
    con.close()


if __name__ == "__main__":
    # Pickle requires that the class being serialised is reachable via the
    # module path stored in __module__.  When this script runs as __main__,
    # Python's module registry has the class under '__main__', not under
    # 'models.base_models'.  Registering __main__ under the canonical name
    # makes the identity check in pickle.save_global() pass:
    #   sys.modules['models.base_models'].LGBMTotalsModel
    #       is sys.modules['__main__'].LGBMTotalsModel   → True
    sys.modules.setdefault("models.base_models", sys.modules["__main__"])
    main()
