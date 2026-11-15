"""Compare models with repeated stratified cross-validation, check the selection step
with nested CV, evaluate the chosen model's out-of-fold predictions, then fit it on
all 299 patients.

With 299 patients and 96 deaths, a 20% hold-out would hold ~19 deaths, so its AUC
would swing by roughly +-0.07 depending on the split. Repeated CV uses every patient
for evaluation and averages over many splits instead.

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
import xgboost  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402

from .config import FEATURES, FIGURES, MODEL_PATH, REPORTS, SEED, THRESHOLD  # noqa: E402
from .data import load  # noqa: E402
from .evaluate import bootstrap_ci, calibration_fit, scores  # noqa: E402
from .model import SIMPLEST_FIRST, candidates, leaky_candidates, logistic  # noqa: E402

N_SPLITS, N_REPEATS = 5, 10
CLINICAL = "baseline (EF + creatinine, logistic)"
METRICS = ["roc_auc", "pr_auc", "brier", "recall", "precision", "specificity"]


def _parallel(pipe):
    """Worker processes only pay off for the tree ensembles; linear and dummy models
    fit in milliseconds, so they run serially to skip process start-up."""
    heavy = not isinstance(pipe.steps[-1][1], (LogisticRegression, DummyClassifier))
    return Parallel(n_jobs=-1 if heavy else 1)


def _fit_score(pipe, X, y, train, test):
    model = clone(pipe).fit(X.iloc[train], y.iloc[train])
    prob = model.predict_proba(X.iloc[test])[:, 1]
    return test, prob, scores(y.iloc[test], prob)


def _fold_importance(pipe, X, y, train, test):
    model = clone(pipe).fit(X.iloc[train], y.iloc[train])
    r = permutation_importance(model, X.iloc[test], y.iloc[test], scoring="roc_auc",
                               n_repeats=10, random_state=SEED)
    return r.importances_mean


def cross_validate_oof(models, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Per-fold scores and out-of-fold probabilities (one column per repeat) for each model."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(cv.split(X, y))
    fold_scores, oof = {}, {}
    for name, pipe in models.items():
        out = _parallel(pipe)(delayed(_fit_score)(pipe, X, y, tr, te) for tr, te in splits)
        probs = np.zeros((n_repeats, len(y)))
        for k, (test, prob, _) in enumerate(out):
            probs[k // n_splits, test] = prob
        fold_scores[name] = pd.DataFrame([s for _, _, s in out])
        oof[name] = probs
    return fold_scores, oof


def summarise(fold_scores):
    rows = []
    for name, df in fold_scores.items():
        row = {"model": name}
        for m in METRICS:
            row[f"{m}_mean"] = round(float(df[m].mean()), 3)
            row[f"{m}_sd"] = round(float(df[m].std()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def choose(cv_table, n_folds):
    """Simplest model whose mean CV ROC-AUC is within one standard error of the best."""
    real = cv_table[cv_table.model.isin(SIMPLEST_FIRST)].set_index("model")
    best = real.roc_auc_mean.idxmax()
    se = real.loc[best, "roc_auc_sd"] / np.sqrt(n_folds)
    chosen = next(m for m in SIMPLEST_FIRST
                  if real.loc[m, "roc_auc_mean"] >= real.loc[best, "roc_auc_mean"] - se)
    return chosen, best, float(se)


def _outer_fold(X, y, train, test, seed):
    """Run the whole selection procedure on the outer-training part, score on outer-test."""
    real = {k: v for k, v in candidates().items() if k in SIMPLEST_FIRST}
    Xtr, ytr = X.iloc[train], y.iloc[train]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, pipe in real.items():
        aucs = [scores(ytr.iloc[te], clone(pipe).fit(Xtr.iloc[tr], ytr.iloc[tr])
                       .predict_proba(Xtr.iloc[te])[:, 1])["roc_auc"]
                for tr, te in cv.split(Xtr, ytr)]
        rows.append({"model": name, "roc_auc_mean": np.mean(aucs), "roc_auc_sd": np.std(aucs)})
    chosen, _, _ = choose(pd.DataFrame(rows), n_folds=5)
    prob = clone(real[chosen]).fit(Xtr, ytr).predict_proba(X.iloc[test])[:, 1]
    return chosen, scores(y.iloc[test], prob)["roc_auc"]


def nested_cv(X, y):
    """Estimate of the full procedure (model selection included): 5-fold x 2 outer,
    5-fold inner. Guards against optimism from picking the best of several models."""
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED + 1)
    out = Parallel(n_jobs=-1)(delayed(_outer_fold)(X, y, tr, te, SEED + k)
                              for k, (tr, te) in enumerate(outer.split(X, y)))
    aucs = np.array([a for _, a in out])
    picks = pd.Series([c for c, _ in out]).value_counts().to_dict()
    return {"roc_auc_mean": round(float(aucs.mean()), 3), "roc_auc_sd": round(float(aucs.std()), 3),
            "n_outer_folds": len(aucs), "times_each_model_chosen": picks}


def odds_ratios(X, y, n_boot=300):
    """Odds ratios from the full-feature logistic pipeline with bootstrap 95% CIs.

    Numeric features are per 1 SD (of the log value for the three skewed labs);
    binary features are yes vs no.
    """
    def fit_or(Xb, yb):
        m = logistic().fit(Xb, yb)
        return pd.Series(np.exp(m.named_steps["clf"].coef_[0]),
                         index=m.named_steps["prep"].get_feature_names_out())

    point = fit_or(X, y)
    rng = np.random.default_rng(SEED)
    boots = pd.DataFrame([fit_or(X.iloc[i], y.iloc[i])
                          for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    table = pd.DataFrame({"odds_ratio": point,
                          "ci_low": boots.quantile(0.025), "ci_high": boots.quantile(0.975)})
    return table.sort_values("odds_ratio").round(3)


def cv_permutation_importance(pipe, X, y):
    """Permutation importance (drop in ROC-AUC) measured on held-out folds, not training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    drops = np.array(_parallel(pipe)(delayed(_fold_importance)(pipe, X, y, tr, te)
                                     for tr, te in cv.split(X, y)))
    return (pd.DataFrame({"auc_drop_mean": drops.mean(0), "auc_drop_sd": drops.std(0)},
                         index=X.columns)
            .sort_values("auc_drop_mean", ascending=False).round(4))


def threshold_table(y, prob):
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        s = scores(y, prob, t)
        pred = prob >= t
        rows.append({"threshold": t, "flagged_share": round(float(pred.mean()), 3),
                     **{k: round(float(s[k]), 3) for k in ["recall", "precision", "specificity"]}})
    return pd.DataFrame(rows)


def save_figures(fold_scores, y, prob, base_prob, chosen, ors, perm, leak_scores, time, n_folds):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = list(fold_scores)
    ax.boxplot([fold_scores[n]["roc_auc"] for n in names], orientation="horizontal")
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.axvline(0.5, color="grey", ls=":")
    ax.set_xlabel(f"ROC-AUC across {n_folds} CV folds (time excluded)")
    ax.set_title("Model comparison, 5-fold x 10-repeat stratified CV")
    fig.tight_layout()
    fig.savefig(FIGURES / "cv_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for p, label in [(prob, chosen), (base_prob, CLINICAL)]:
        RocCurveDisplay.from_predictions(y, p, ax=axes[0], name=label)
        PrecisionRecallDisplay.from_predictions(y, p, ax=axes[1], name=label)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].axhline(y.mean(), ls="--", color="grey", label=f"prevalence ({y.mean():.2f})")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("ROC, out-of-fold predictions")
    axes[1].set_title("Precision-recall, out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_pr_oof.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for p, label in [(prob, chosen), (base_prob, CLINICAL)]:
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.set_xlabel("Mean predicted risk (out-of-fold, octiles)")
    ax.set_ylabel("Observed death rate")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, (prob >= THRESHOLD).astype(int), display_labels=["Survived", "Died"],
        ax=ax, colorbar=False)
    ax.set_title(f"Out-of-fold, threshold {THRESHOLD}")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_oof.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    yy = np.arange(len(ors))
    ax.errorbar(ors.odds_ratio, yy, fmt="o",
                xerr=[ors.odds_ratio - ors.ci_low, ors.ci_high - ors.odds_ratio], capsize=3)
    ax.set_yticks(yy, ors.index)
    ax.axvline(1, color="grey", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio with bootstrap 95% CI (numeric: per 1 SD; log scale)")
    ax.set_title("Logistic regression: what moves the risk")
    fig.tight_layout()
    fig.savefig(FIGURES / "odds_ratios.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    p = perm.iloc[::-1]
    ax.barh(p.index, p.auc_drop_mean, xerr=p.auc_drop_sd, capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Drop in held-out ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance ({chosen})")
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_importance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([time[y == 0], time[y == 1]], bins=20, label=["Survived", "Died"], stacked=True)
    axes[0].set_xlabel("Follow-up time (days)")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("Deaths end follow-up early")
    axes[0].legend()
    names = list(leak_scores)
    axes[1].boxplot([leak_scores[n]["roc_auc"] for n in names], orientation="horizontal")
    axes[1].set_yticks(range(1, len(names) + 1), names)
    axes[1].set_xlabel("CV ROC-AUC")
    axes[1].set_title("Adding `time` inflates AUC (leakage)")
    fig.tight_layout()
    fig.savefig(FIGURES / "leakage_time.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    X, y = load()
    n_folds = N_SPLITS * N_REPEATS

    fold_scores, oof = cross_validate_oof(candidates(), X, y)
    cv_table = summarise(fold_scores)
    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)
    print(cv_table.to_string(index=False))
    chosen, best, se = choose(cv_table, n_folds=n_folds)
    print(f"\nHighest CV ROC-AUC: {best}. Chosen (simplest within 1 SE = {se:.3f}): {chosen}")

    # Leakage demonstration: same folds, with `time` added.
    X_leak, _ = load(include_leaky=True)
    leak_scores, _ = cross_validate_oof(leaky_candidates(), X_leak, y)
    leak_table = summarise(leak_scores)
    leak_scores = {**{k: fold_scores[k] for k in ["logistic regression", "gradient boosting"]},
                   **leak_scores}

    nested = nested_cv(X, y)
    print("Nested CV:", nested)

    # Out-of-fold probabilities averaged over the 10 repeats: every patient's score
    # comes only from models that never saw that patient.
    prob, base_prob = oof[chosen].mean(0), oof[CLINICAL].mean(0)
    point = {k: round(float(v), 3) for k, v in scores(y, prob).items()}
    base_point = {k: round(float(v), 3) for k, v in scores(y, base_prob).items()}
    thresholds = threshold_table(y, prob)
    thresholds.to_csv(REPORTS / "threshold_table.csv", index=False)

    ors = odds_ratios(X, y)
    ors.to_csv(REPORTS / "odds_ratios.csv", index_label="feature")
    perm = cv_permutation_importance(candidates()[chosen], X, y)
    perm.to_csv(REPORTS / "permutation_importance.csv", index_label="feature")

    metrics = {
        "chosen_model": chosen,
        "highest_cv_roc_auc_model": best,
        "one_se_roc_auc": round(se, 4),
        "n_patients": int(len(y)),
        "n_deaths": int(y.sum()),
        "prevalence": round(float(y.mean()), 3),
        "features": FEATURES,
        "excluded_as_leakage": ["time"],
        "evaluation": f"RepeatedStratifiedKFold({N_SPLITS} folds x {N_REPEATS} repeats)",
        "threshold": THRESHOLD,
        "cv_mean_sd": cv_table.set_index("model").to_dict(orient="index"),
        "nested_cv_selection_procedure": nested,
        "oof_chosen": {
            **point,
            "confusion_matrix [[TN, FP], [FN, TP]]":
                confusion_matrix(y, (prob >= THRESHOLD).astype(int)).tolist(),
            "calibration": {k: round(v, 3) for k, v in calibration_fit(y, prob).items()},
        },
        "oof_chosen_95ci_bootstrap": bootstrap_ci(y, prob, reference_prob=base_prob),
        "oof_clinical_baseline": base_point,
        "oof_clinical_baseline_95ci_bootstrap": bootstrap_ci(y, base_prob),
        "leakage_demo_with_time": leak_table.set_index("model")[
            ["roc_auc_mean", "roc_auc_sd", "pr_auc_mean"]].to_dict(orient="index"),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in
                      ["oof_chosen", "oof_chosen_95ci_bootstrap", "leakage_demo_with_time"]},
                     indent=2))

    model = candidates()[chosen].fit(X, y)
    joblib.dump(model, MODEL_PATH)
    time = load(include_leaky=True)[0]["time"]
    save_figures(fold_scores, y, prob, base_prob, chosen, ors, perm, leak_scores, time, n_folds)


if __name__ == "__main__":
    main()
