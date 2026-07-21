from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_prediction_rate": float(pred.mean()),
    }


def make_age_band(age: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(age, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 0, 24, 34, 44, 54, np.inf],
        labels=["Unknown", "Under 25", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)


def fairness_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    groups = {
        "Sex": audit["Sex"].astype(str),
        "AgeBand": make_age_band(audit["Age"]),
        "MaritalStatus": audit["MaritalStatus"].astype(str),
    }
    groups["Sex_x_AgeBand"] = groups["Sex"] + " × " + groups["AgeBand"]

    rows: list[dict] = []
    for attribute, values in groups.items():
        for group in sorted(values.unique()):
            mask = (values == group).to_numpy()
            if mask.sum() < 20:
                continue
            yt, pp, yp = y[mask], p[mask], pred[mask]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n": int(mask.sum()),
                    "fraud_prevalence": float(yt.mean()),
                    "selection_rate": float(yp.mean()),
                    "recall_tpr": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                    "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                    "pr_auc": float(average_precision_score(yt, pp)) if len(np.unique(yt)) > 1 else np.nan,
                    "roc_auc": float(roc_auc_score(yt, pp)) if len(np.unique(yt)) > 1 else np.nan,
                    "mean_score": float(pp.mean()),
                }
            )
    detail = pd.DataFrame(rows)

    gaps: list[dict] = []
    for attribute, part in detail.groupby("attribute"):
        selection_max = part["selection_rate"].max()
        selection_min = part["selection_rate"].min()
        gaps.append(
            {
                "attribute": attribute,
                "groups_included": int(len(part)),
                "demographic_parity_difference": float(selection_max - selection_min),
                "demographic_parity_ratio": float(selection_min / selection_max) if selection_max else np.nan,
                "equal_opportunity_difference": float(part["recall_tpr"].max() - part["recall_tpr"].min()),
                "false_positive_rate_difference": float(
                    part["false_positive_rate"].max() - part["false_positive_rate"].min()
                ),
                "predictive_parity_difference": float(part["precision"].max() - part["precision"].min()),
                "mean_score_difference": float(part["mean_score"].max() - part["mean_score"].min()),
            }
        )
    return detail, pd.DataFrame(gaps)


def bootstrap_fairness_gaps(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    audit: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    audit = audit.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    samples: list[pd.DataFrame] = []
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        _, gaps = fairness_table(y[indices], p[indices], threshold, audit.iloc[indices].reset_index(drop=True))
        if gaps.empty:
            continue
        gaps["iteration"] = iteration
        samples.append(gaps)
    boot = pd.concat(samples, ignore_index=True)
    metrics = [
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equal_opportunity_difference",
        "false_positive_rate_difference",
        "predictive_parity_difference",
        "mean_score_difference",
    ]
    rows: list[dict] = []
    for attribute, part in boot.groupby("attribute"):
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "attribute": attribute,
                    "metric": metric,
                    "bootstrap_samples": int(len(values)),
                    "ci_2_5": float(values.quantile(0.025)),
                    "median": float(values.quantile(0.5)),
                    "ci_97_5": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)
