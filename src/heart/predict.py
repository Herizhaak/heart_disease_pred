"""Score new patients with the saved pipeline.

Run from the repo root:  python -m heart.predict path/to/patients.csv [--threshold 0.3]

The CSV needs the 11 baseline feature columns of
data/heart_failure_clinical_records_dataset.csv, in any order. Extra columns
(`time`, `DEATH_EVENT`, an ID) are ignored: the model never uses follow-up time.
"""
import argparse

import joblib
import pandas as pd

from .config import MODEL_PATH, THRESHOLD
from .data import check_features


def predict(df, threshold=THRESHOLD, model_path=MODEL_PATH):
    model = joblib.load(model_path)
    # Label from the rounded value so the printed probability and label never disagree.
    prob = model.predict_proba(check_features(df))[:, 1].round(3)
    out = df.copy()
    out["risk_probability"] = prob
    out["prediction"] = ["High risk" if p >= threshold else "Lower risk" for p in prob]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Score heart failure patients with the mortality risk model.")
    parser.add_argument("csv")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    result = predict(pd.read_csv(args.csv), args.threshold)
    print(result[["risk_probability", "prediction"]].to_string())


if __name__ == "__main__":
    main()
