from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score


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
    scores = [fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0) for threshold in candidates]
    investigate = float(candidates[int(np.argmax(scores))])

    approve_candidates: list[float] = []
    for threshold in candidates[candidates < investigate]:
        approved = probabilities < threshold
        if approved.sum() == 0:
            continue
        npv = float((y_true[approved] == 0).mean())
        if npv >= target_npv:
            approve_candidates.append(float(threshold))

    approve = max(approve_candidates, default=max(0.01, investigate * 0.5))
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

