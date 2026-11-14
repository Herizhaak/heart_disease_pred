"""Loading and validating the Statlog heart disease data."""
import pandas as pd

from .config import DATA_PATH, FEATURES, POSITIVE_LABEL, TARGET


def load(path=DATA_PATH):
    """Return features X and a binary target y (1 = heart disease present)."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df[FEATURES], y


def check_features(X):
    """Validate input columns and put them in training order, so column order never matters."""
    missing = set(FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return X[FEATURES]
