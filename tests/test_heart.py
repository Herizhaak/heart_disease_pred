import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.config import DATA_PATH, FEATURES, TARGET  # noqa: E402
from heart.data import check_features, load  # noqa: E402
from heart.evaluate import bootstrap_ci, calibration_fit, fast_scores, scores  # noqa: E402
from heart.model import candidates, leaky_candidates  # noqa: E402
from heart.predict import predict  # noqa: E402
from heart.train import choose  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    X, y = data
    model = candidates()["logistic regression"].fit(X, y)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return X, path


# --- data -----------------------------------------------------------------

def test_schema_and_size(data):
    X, y = data
    assert X.shape == (299, 11)
    assert list(X.columns) == FEATURES
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 96
    assert not X.isna().any().any()


def test_time_is_never_a_feature(data, fitted):
    X, path = fitted
    assert "time" not in FEATURES
    assert "time" not in X.columns
    assert "time" not in joblib.load(path).feature_names_in_
    for pipe in candidates().values():
        assert "time" not in pipe.fit(*data).feature_names_in_


def test_time_only_loaded_on_request():
    X, _ = load(include_leaky=True)
    assert "time" in X.columns
    assert all("time" in p.named_steps["prep"].fit(X).feature_names_in_
               for p in leaky_candidates().values())


def test_missing_raw_column_is_rejected(tmp_path):
    df = pd.read_csv(DATA_PATH).drop(columns=["serum_creatinine"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load(path)


def test_non_binary_target_is_rejected(tmp_path):
    df = pd.read_csv(DATA_PATH)
    df.loc[0, TARGET] = 2
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="0/1"):
        load(path)


# --- pipelines -------------------------------------------------------------

@pytest.mark.parametrize("name", list(candidates()))
def test_every_pipeline_fits_and_predicts(data, name):
    X, y = data
    prob = candidates()[name].fit(X.iloc[:200], y.iloc[:200]).predict_proba(X.iloc[200:])[:, 1]
    assert prob.shape == (99,)
    assert np.all((prob >= 0) & (prob <= 1))


# --- prediction CLI --------------------------------------------------------

def test_column_order_does_not_change_predictions(fitted):
    X, path = fitted
    shuffled = X[list(reversed(FEATURES))]
    a = predict(X.head(20), model_path=path)["risk_probability"]
    b = predict(shuffled.head(20), model_path=path)["risk_probability"]
    pd.testing.assert_series_equal(a, b)


def test_extra_columns_including_time_are_ignored(fitted):
    X, path = fitted
    raw = pd.read_csv(DATA_PATH).head(20)  # has time and DEATH_EVENT
    a = predict(X.head(20), model_path=path)["risk_probability"]
    b = predict(raw, model_path=path)["risk_probability"]
    pd.testing.assert_series_equal(a, b)


def test_missing_column_is_rejected(data):
    X, _ = data
    with pytest.raises(ValueError, match="missing"):
        check_features(X.drop(columns=["ejection_fraction"]))


def test_threshold_controls_labels(fitted):
    X, path = fitted
    out = predict(X, threshold=0.3, model_path=path)
    assert out["risk_probability"].between(0, 1).all()
    assert ((out["risk_probability"] >= 0.3) == (out["prediction"] == "High risk")).all()


# --- metrics ---------------------------------------------------------------

def test_scores_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    perfect = scores(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert perfect["roc_auc"] == 1 and perfect["pr_auc"] == 1
    assert perfect["recall"] == 1 and perfect["precision"] == 1 and perfect["specificity"] == 1
    assert perfect["brier"] == pytest.approx(0.025)
    flat = scores(y, np.full(4, 0.5), threshold=0.6)
    assert flat["roc_auc"] == 0.5 and flat["recall"] == 0 and flat["precision"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_scores_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 120)
    prob = np.round(rng.uniform(size=120), 1)  # rounding creates ties
    fast, ref = fast_scores(y, prob), scores(y, prob)
    for k in ref:
        assert fast[k] == pytest.approx(ref[k]), k


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    prob = np.clip(0.3 * y + rng.uniform(0, 0.7, 300), 0, 1)
    ci = bootstrap_ci(y, prob, reference_prob=rng.uniform(size=300), n=300)
    point = scores(y, prob)
    for k in ["roc_auc", "pr_auc", "brier", "recall", "precision"]:
        assert ci[k][0] <= point[k] <= ci[k][1]
    assert ci["roc_auc_difference_vs_reference"][0] > 0


def test_calibration_fit_recovers_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    y = rng.binomial(1, p)
    cal = calibration_fit(y, p)
    assert cal["slope"] == pytest.approx(1, abs=0.15)
    assert cal["intercept"] == pytest.approx(0, abs=0.15)


def test_one_se_rule_prefers_simplest():
    table = pd.DataFrame({
        "model": ["logistic regression", "random forest", "gradient boosting"],
        "roc_auc_mean": [0.74, 0.75, 0.70],
        "roc_auc_sd": [0.05, 0.05, 0.05],
    })
    assert choose(table, n_folds=4)[0] == "logistic regression"  # SE 0.025
    assert choose(table, n_folds=100)[0] == "random forest"  # SE 0.005
