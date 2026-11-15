"""Loading and validating the heart failure clinical records data."""
import pandas as pd

from .config import DATA_PATH, FEATURES, LEAKY, TARGET


def load(path=DATA_PATH, include_leaky=False):
    """Return features X and the binary target y (1 = died during follow-up).

    `time` (follow-up days) is excluded unless `include_leaky=True`, which is only
    used to demonstrate how much it would inflate the scores.
    """
    df = pd.read_csv(path)
    missing = set(FEATURES + LEAKY + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Data contains missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must be 0/1")
    cols = FEATURES + LEAKY if include_leaky else FEATURES
    return df[cols], df[TARGET].astype(int)


def check_features(X):
    """Validate input columns and put them in training order, so column order never matters.

    Extra columns (an ID, the target, `time`) are dropped.
    """
    missing = set(FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return X[FEATURES]
