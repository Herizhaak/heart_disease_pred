"""Score new patients with the saved pipeline.

Run from the repo root:  python -m heart.predict path/to/patients.csv [--threshold 0.5]

The CSV needs the same feature columns as data/heart_disease.csv, in any
order. A target column, if present, is ignored.
"""
import argparse

import joblib
import pandas as pd

from .config import MODEL_PATH
from .data import check_features


def predict(df, threshold=0.5, model_path=MODEL_PATH):
    model = joblib.load(model_path)
    prob = model.predict_proba(check_features(df))[:, 1]
    out = df.copy()
    out["risk_probability"] = prob.round(3)
    out["prediction"] = ["Disease" if p >= threshold else "No disease" for p in prob]
    return out


def main():
    parser = argparse.ArgumentParser(description="Score patients with the heart disease model.")
    parser.add_argument("csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = predict(pd.read_csv(args.csv), args.threshold)
    print(result[["risk_probability", "prediction"]].to_string())


if __name__ == "__main__":
    main()
