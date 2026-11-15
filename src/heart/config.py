"""Shared constants for the heart failure mortality project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "heart_failure_clinical_records_dataset.csv"
MODEL_PATH = ROOT / "models" / "heart_model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "DEATH_EVENT"
SEED = 42

# Follow-up period in days. It is only known once follow-up has ended, and patients
# who die have short follow-up by construction, so using it as a predictor is target
# leakage. It is loaded for the leakage demonstration only and never used as a feature.
LEAKY = ["time"]

# Right-skewed lab values: log-transform, then standardise.
SKEWED = ["creatinine_phosphokinase", "serum_creatinine", "platelets"]
# Roughly symmetric measurements: standardise.
NUMERIC = ["age", "ejection_fraction", "serum_sodium"]
# Already 0/1.
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
FEATURES = NUMERIC + SKEWED + BINARY

# The two predictors the original study found most informative.
CLINICAL_BASELINE = ["ejection_fraction", "serum_creatinine"]

# Decision threshold for recall/precision. Chosen before looking at results: roughly the
# base rate (32% died), i.e. flag patients whose predicted risk is above average.
THRESHOLD = 0.3
