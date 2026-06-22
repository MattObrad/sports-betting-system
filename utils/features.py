"""
utils/features.py -- Shared feature preparation utilities.

Single source of truth imported by:
    walk_forward.py, base_models.py, ensemble.py, predict_mlb.py

Never duplicate these functions. All four callers must encode features
identically or train/predict divergence will silently corrupt predictions.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

# Columns present in game_features that are metadata, not model inputs.
# TEXT categoricals (home/away_sp_throws) are excluded here and encoded
# as binary _R columns inside prepare_X().
EXCLUDE_COLS = frozenset({
    "game_id",
    "feature_version",
    "computed_at",
    "home_sp_throws",
    "away_sp_throws",
})


def get_feature_columns(sample_df: pd.DataFrame) -> list:
    """
    Return the ordered list of model input column names from a game_features
    sample row (or the full DataFrame).  Excludes metadata and TEXT categoricals.
    The returned list is used as the canonical column order for every prepare_X call
    in a given training run.
    """
    return [c for c in sample_df.columns if c not in EXCLUDE_COLS and c != "game_id"]


def prepare_X(feat_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Encode TEXT categoricals and return a float feature matrix aligned to feature_cols.

    Encoding applied:
        home_sp_throws / away_sp_throws  ->  *_R binary column:
            1.0 = right-handed   0.0 = left-handed   NaN = unknown / missing

    The encoded columns are appended after feature_cols so the column set is
    deterministic across train and predict calls.

    Missing data policy:
        LightGBM and XGBoost handle NaN natively -- no imputation is performed.
        Games with >30% NULL features are flagged (likely early-season thin windows)
        but are retained; removing them would create train/predict asymmetry.
    """
    df = feat_df.copy()

    # Drop duplicate columns before anything else.
    # load_todays_games() joins game_features + games + odds_snapshots; column
    # names like game_id or game_date can appear in more than one source table.
    # Keeping duplicates causes pandas.DataFrame.reindex to raise
    # "cannot reindex on an axis with duplicate labels".
    duped = df.columns[df.columns.duplicated()].tolist()
    if duped:
        log.warning(
            "prepare_X: dropping %d duplicate column(s): %s",
            len(duped), duped,
        )
        df = df.loc[:, ~df.columns.duplicated()]

    for src, dst in [
        ("home_sp_throws", "home_sp_throws_R"),
        ("away_sp_throws", "away_sp_throws_R"),
    ]:
        if src in df.columns:
            df[dst] = np.where(
                df[src] == "R", 1.0,
                np.where(df[src] == "L", 0.0, np.nan),
            )

    # Only append encoded columns if they aren't already listed in feature_cols.
    # When feature_cols was built at training time, it already included
    # home_sp_throws_R / away_sp_throws_R; appending them again creates
    # duplicate column names that crash reindex in _align().
    feature_cols_set = set(feature_cols)
    encoded_extras = [
        c for c in ["home_sp_throws_R", "away_sp_throws_R"]
        if c in df.columns and c not in feature_cols_set
    ]
    cols = [c for c in feature_cols if c in df.columns] + encoded_extras

    X = df[cols].copy().astype(float)

    high_null = X.isnull().mean(axis=1) > 0.30
    if high_null.any():
        log.warning(
            "%d games have >30%% NULL features (data gaps -- retained, may degrade fit)",
            high_null.sum(),
        )

    return X


def regression_to_proba(
    predicted: np.ndarray,
    market_lines: np.ndarray,
    residual_std: float,
) -> np.ndarray:
    """
    P(actual_total > market_line) modelling prediction errors as Normal(0, residual_std).

    This is the shared probability estimator used by:
      - walk_forward.py  (standalone validation placeholder)
      - LGBMTotalsModel.predict_proba()
      - XGBTotalsModel.predict_proba()

    The stacking meta-learner (Step 6) replaces this with a logistic regression
    trained on the base-model OOF outputs for better-calibrated probabilities.

    residual_std floor of 0.5 runs prevents probabilities collapsing to 0 or 1
    when the model is highly confident (e.g. early-season few-game windows).
    """
    std = max(residual_std, 0.5)
    return 1.0 - norm.cdf(market_lines, loc=predicted, scale=std)
