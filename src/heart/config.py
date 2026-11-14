"""Shared constants for the heart disease project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "heart_model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "Heart Disease"
POSITIVE_LABEL = "Presence"
SEED = 42

# Nominal codes: one-hot encode them (the numbers are labels, not quantities).
CATEGORICAL = ["Chest pain type", "EKG results", "Slope of ST", "Thallium"]
# Continuous measurements and ordinal counts: standardise them.
NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
# Already 0/1.
BINARY = ["Sex", "FBS over 120", "Exercise angina"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
