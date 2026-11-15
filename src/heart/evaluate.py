"""Evaluation metrics, bootstrap confidence intervals and calibration summaries."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import SEED, THRESHOLD


def scores(y_true, prob, threshold=THRESHOLD):
    """Ranking, probability and threshold metrics for one set of predictions."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def calibration_fit(y_true, prob, eps=1e-6):
    """Calibration intercept and slope: logistic regression of y on logit(p).

    A well-calibrated model has slope ~1 and intercept ~0; slope < 1 means the
    predictions are too extreme (overfitting), slope > 1 too timid.
    """
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=np.inf).fit(logit, y_true)
    return {"intercept": float(fit.intercept_[0]), "slope": float(fit.coef_[0, 0])}


def bootstrap_ci(y_true, prob, reference_prob=None, threshold=THRESHOLD, n=2000, seed=SEED):
    """95% percentile bootstrap intervals, resampling patients.

    If `reference_prob` is given, also returns the CI of the paired ROC-AUC
    difference (prob minus reference) on the same resamples.
    """
    rng = np.random.default_rng(seed)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    draws, diffs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        s = fast_scores(y_true[i], prob[i], threshold)
        draws.append(s)
        if reference_prob is not None:
            ref = fast_scores(y_true[i], np.asarray(reference_prob)[i], threshold)
            diffs.append(s["roc_auc"] - ref["roc_auc"])
    ci = {k: _interval([d[k] for d in draws]) for k in draws[0]}
    if reference_prob is not None:
        ci["roc_auc_difference_vs_reference"] = _interval(diffs)
    return ci


def fast_scores(y_true, prob, threshold=THRESHOLD):
    """Same numbers as `scores` (ties handled the same way as scikit-learn), computed
    with one sort in plain numpy so thousands of bootstrap resamples stay fast."""
    y, p = np.asarray(y_true).astype(float), np.asarray(prob, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]  # end of each tied block
    tp, fp = np.cumsum(ys)[last], np.cumsum(1 - ys)[last]
    pos, neg = tp[-1], fp[-1]
    tpr, fpr = np.r_[0, tp / pos], np.r_[0, fp / neg]
    pred = p >= threshold
    tp_t, fp_t = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    return {
        "roc_auc": float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)),
        "pr_auc": float(np.sum(np.diff(tpr) * tp / (tp + fp))),
        "brier": float(np.mean((p - y) ** 2)),
        "recall": float(tp_t / pos),
        "precision": float(tp_t / (tp_t + fp_t)) if tp_t + fp_t else 0.0,
        "specificity": float((neg - fp_t) / neg),
    }


def _interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 3), round(float(high), 3)]
