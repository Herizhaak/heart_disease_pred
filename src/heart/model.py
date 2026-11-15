"""Model definitions.

Preprocessing sits inside each Pipeline, so it is fit on the training folds
only and never sees validation rows (no leakage through scaling).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from .config import BINARY, CLINICAL_BASELINE, FEATURES, LEAKY, NUMERIC, SEED, SKEWED

# Real models, simplest first (the order used by the one-standard-error rule).
SIMPLEST_FIRST = ["logistic regression", "random forest", "gradient boosting"]


def preprocessor(features=FEATURES):
    """Log + standardise skewed labs, standardise other numerics, pass binaries through.

    Only the groups present in `features` are used, so the same function builds the
    two-feature clinical baseline and the leaky demonstration model.
    """
    numeric = [c for c in NUMERIC + LEAKY if c in features]
    skewed = [c for c in SKEWED if c in features]
    binary = [c for c in BINARY if c in features]
    parts = [
        ("num", StandardScaler(), numeric),
        ("log", make_pipeline(FunctionTransformer(np.log, feature_names_out="one-to-one"),
                              StandardScaler()), skewed),
        ("bin", "passthrough", binary),
    ]
    return ColumnTransformer([p for p in parts if p[2]], verbose_feature_names_out=False)


def logistic(features=FEATURES):
    return Pipeline([("prep", preprocessor(features)),
                     ("clf", LogisticRegression(max_iter=1000))])


def random_forest(features=FEATURES):
    return Pipeline([
        ("prep", preprocessor(features)),
        ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt",
            random_state=SEED, n_jobs=1)),
    ])


def gradient_boosting(features=FEATURES):
    return Pipeline([
        ("prep", preprocessor(features)),
        ("clf", XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=SEED, eval_metric="logloss", n_jobs=1)),
    ])


def candidates():
    """Baselines and the three real models compared in cross-validation."""
    return {
        "baseline (majority class)": Pipeline(
            [("prep", preprocessor()), ("clf", DummyClassifier(strategy="prior"))]),
        "baseline (EF + creatinine, logistic)": logistic(CLINICAL_BASELINE),
        "logistic regression": logistic(),
        "random forest": random_forest(),
        "gradient boosting": gradient_boosting(),
    }


def leaky_candidates():
    """The same models given the follow-up `time` column (demonstration only)."""
    with_time = FEATURES + LEAKY
    return {
        "logistic regression + time": logistic(with_time),
        "gradient boosting + time": gradient_boosting(with_time),
    }
