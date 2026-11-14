"""Compare models with repeated cross-validation on the training set, then
evaluate the chosen model once on a held-out test set.

Run from the repo root:  python -m heart.train   (with src/ on PYTHONPATH)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)

from .config import FIGURES, MODEL_PATH, REPORTS, SEED  # noqa: E402
from .data import load  # noqa: E402
from .model import candidates  # noqa: E402

SCORING = ["roc_auc", "accuracy", "recall", "f1"]
SIMPLEST_FIRST = ["logistic regression", "random forest", "xgboost"]


def bootstrap_ci(y_true, y_pred, y_prob, n=2000, seed=SEED):
    """95% percentile bootstrap intervals for test-set metrics."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_prob = map(np.asarray, (y_true, y_pred, y_prob))
    stats = {"accuracy": [], "roc_auc": [], "recall": [], "precision": []}
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[i])) < 2:
            continue
        stats["accuracy"].append(accuracy_score(y_true[i], y_pred[i]))
        stats["roc_auc"].append(roc_auc_score(y_true[i], y_prob[i]))
        stats["recall"].append(recall_score(y_true[i], y_pred[i]))
        stats["precision"].append(precision_score(y_true[i], y_pred[i], zero_division=0))
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            for k, v in stats.items()}


def compare_models(X_train, y_train):
    """5-fold x 10-repeat stratified CV for every candidate (training data only)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows, fold_auc = [], {}
    for name, pipe in candidates().items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        fold_auc[name] = res["test_roc_auc"]
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = round(float(res[f"test_{m}"].mean()), 3)
            row[f"{m}_std"] = round(float(res[f"test_{m}"].std()), 3)
        rows.append(row)
    return pd.DataFrame(rows), fold_auc


def choose(cv_table, n_folds):
    """Simplest model whose mean CV ROC-AUC is within one standard error of the best."""
    real = cv_table[cv_table.model.isin(SIMPLEST_FIRST)].set_index("model")
    best = real.roc_auc_mean.idxmax()
    se = real.loc[best, "roc_auc_std"] / np.sqrt(n_folds)
    return next(m for m in SIMPLEST_FIRST
                if real.loc[m, "roc_auc_mean"] >= real.loc[best, "roc_auc_mean"] - se), best


def save_figures(fold_auc, model, chosen, X_test, y_test, prob, pred):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot(list(fold_auc.values()), orientation="horizontal")
    ax.set_yticks(range(1, len(fold_auc) + 1), list(fold_auc.keys()))
    ax.set_xlabel("ROC-AUC across 50 CV folds (training set)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "cv_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax, name=chosen)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC curve, held-out test set")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_test.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No disease", "Disease"], ax=ax, colorbar=False)
    ax.set_title("Confusion matrix, test set")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_test.png", dpi=150)
    plt.close(fig)

    if chosen == "logistic regression":
        names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
        odds = pd.Series(np.exp(model.named_steps["clf"].coef_[0]), index=names).sort_values()
        odds.rename("odds_ratio").round(3).to_csv(REPORTS / "odds_ratios.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(odds.index, odds.values)
        ax.axvline(1, color="grey", ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (numeric features: per 1 SD; log scale)")
        ax.set_title("Logistic regression: what moves the risk")
        fig.tight_layout()
        fig.savefig(FIGURES / "odds_ratios.png", dpi=150)
        plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv_table, fold_auc = compare_models(X_train, y_train)
    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)
    print(cv_table.to_string(index=False))
    chosen, best = choose(cv_table, n_folds=50)
    print(f"\nHighest CV ROC-AUC: {best}. Chosen (simplest within 1 SE): {chosen}")

    model = candidates()[chosen].fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "chosen_model": chosen,
        "highest_cv_roc_auc_model": best,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_prevalence": round(float(y_test.mean()), 3),
        "threshold": 0.5,
        "test": {
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "roc_auc": round(roc_auc_score(y_test, prob), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_test, pred).tolist(),
        },
        "test_95ci_bootstrap": bootstrap_ci(y_test, pred, prob),
        "versions": {"scikit-learn": sklearn.__version__},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    joblib.dump(model, MODEL_PATH)
    save_figures(fold_auc, model, chosen, X_test, y_test, prob, pred)


if __name__ == "__main__":
    main()
