import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.config import FEATURES  # noqa: E402
from heart.data import check_features, load  # noqa: E402
from heart.model import candidates  # noqa: E402
from heart.predict import predict  # noqa: E402


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    X, y = load()
    model = candidates()["logistic regression"].fit(X, y)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return X, path


def test_target_is_binary():
    _, y = load()
    assert set(y.unique()) == {0, 1}


def test_column_order_does_not_change_predictions(fitted):
    # Regression test for the bug in the original predict notebook, which
    # reordered the columns before scaling.
    X, path = fitted
    shuffled = X[list(reversed(FEATURES))]
    a = predict(X.head(20), model_path=path)["risk_probability"]
    b = predict(shuffled.head(20), model_path=path)["risk_probability"]
    pd.testing.assert_series_equal(a, b)


def test_missing_column_is_rejected():
    X, _ = load()
    with pytest.raises(ValueError, match="missing"):
        check_features(X.drop(columns=["Age"]))


def test_probabilities_in_range(fitted):
    X, path = fitted
    assert predict(X, model_path=path)["risk_probability"].between(0, 1).all()
