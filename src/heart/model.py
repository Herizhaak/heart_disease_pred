"""Model definitions.

Preprocessing sits inside each Pipeline, so it is fit on the training folds
only and never sees validation or test rows (no leakage).
"""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import BINARY, CATEGORICAL, NUMERIC, SEED


def preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidates():
    """Models compared in cross-validation, simplest first."""
    return {
        "baseline (majority class)": Pipeline(
            [("prep", preprocessor()), ("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic regression": Pipeline(
            [("prep", preprocessor()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "random forest": Pipeline(
            [
                ("prep", preprocessor()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=3, random_state=SEED)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", preprocessor()),
                ("clf", XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, eval_metric="logloss")),
            ]
        ),
    }
