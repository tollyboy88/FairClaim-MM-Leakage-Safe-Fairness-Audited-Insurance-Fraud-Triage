from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class DecisionThresholds:
    approve_below: float
    investigate_at_or_above: float
    target_negative_predictive_value: float


def _negative_predictive_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn)) if (tn + fn) else 0.0


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_npv: float = 0.99,
) -> DecisionThresholds:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), probabilities])
    # For each threshold, searchsorted gives the number of observations with
    # p < threshold. Cumulative labels then yield every confusion count at
    # once. This is algebraically identical to calling fbeta_score for every
    # candidate, but is fast enough for bootstrap threshold stability.
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probability = probabilities[order]
    sorted_label = y_true[order].astype(np.int64)
    cumulative_positive = np.r_[0, np.cumsum(sorted_label)]
    below = np.searchsorted(sorted_probability, candidates, side="left")
    false_negative = cumulative_positive[below]
    true_negative = below - false_negative
    total_positive = int(sorted_label.sum())
    true_positive = total_positive - false_negative
    false_positive = (len(y_true) - below) - true_positive
    denominator = 5 * true_positive + 4 * false_negative + false_positive
    scores = np.divide(
        5 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    investigate = float(candidates[int(np.argmax(scores))])

    approval_npv = np.divide(
        true_negative,
        below,
        out=np.zeros_like(candidates, dtype=float),
        where=below > 0,
    )
    eligible = (candidates < investigate) & (below > 0) & (approval_npv >= target_npv)
    approve = (
        float(candidates[eligible].max())
        if eligible.any()
        else max(0.01, investigate * 0.5)
    )
    if approve >= investigate:
        approve = max(0.0, investigate - 0.01)
    return DecisionThresholds(approve, investigate, target_npv)


def assign_decision(probability: float, thresholds: DecisionThresholds | dict) -> str:
    if isinstance(thresholds, dict):
        approve = float(thresholds["approve_below"])
        investigate = float(thresholds["investigate_at_or_above"])
    else:
        approve = thresholds.approve_below
        investigate = thresholds.investigate_at_or_above
    if probability < approve:
        return "approve"
    if probability >= investigate:
        return "investigate"
    return "abstain"
