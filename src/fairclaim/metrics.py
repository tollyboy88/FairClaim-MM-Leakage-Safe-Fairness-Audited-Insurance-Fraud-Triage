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

from fairclaim.decision import DecisionThresholds, assign_decision, select_thresholds


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + (z**2 / total)
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * np.sqrt(
        (proportion * (1 - proportion) / total) + (z**2 / (4 * total**2))
    )
    return float(centre - margin), float(centre + margin)


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
            tpr_low, tpr_high = _wilson_interval(int(tp), int(tp + fn))
            fnr_low, fnr_high = _wilson_interval(int(fn), int(tp + fn))
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n": int(mask.sum()),
                    "fraud_prevalence": float(yt.mean()),
                    "selection_rate": float(yp.mean()),
                    "recall_tpr": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "recall_tpr_ci_95_low": tpr_low,
                    "recall_tpr_ci_95_high": tpr_high,
                    "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else np.nan,
                    "false_negative_rate_ci_95_low": fnr_low,
                    "false_negative_rate_ci_95_high": fnr_high,
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
                "false_negative_rate_difference": float(
                    part["false_negative_rate"].max() - part["false_negative_rate"].min()
                ),
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
        "false_negative_rate_difference",
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


def triage_policy_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: DecisionThresholds | dict,
) -> dict[str, float | int]:
    """Operational metrics for a three-way risk triage policy.

    The middle region is a risk-based defer region. These metrics deliberately do
    not interpret it as a direct estimate of predictive uncertainty.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    decisions = np.asarray([assign_decision(probability, thresholds) for probability in p])
    approved = decisions == "approve"
    deferred = decisions == "abstain"
    investigated = decisions == "investigate"
    approved_nonfraud = int((y[approved] == 0).sum())
    investigation_fraud = int((y[investigated] == 1).sum())
    total_fraud = int((y == 1).sum())
    return {
        "approval_npv": float(approved_nonfraud / approved.sum()) if approved.any() else np.nan,
        "fraud_capture": float(investigation_fraud / total_fraud) if total_fraud else np.nan,
        "investigation_precision": float(investigation_fraud / investigated.sum()) if investigated.any() else np.nan,
        "abstention_coverage": float(deferred.mean()),
        "investigation_coverage": float(investigated.mean()),
        "approval_coverage": float(approved.mean()),
        "approved_claims": int(approved.sum()),
        "abstained_claims": int(deferred.sum()),
        "investigated_claims": int(investigated.sum()),
    }


def bootstrap_threshold_policy_metrics(
    y_validation: Iterable[int],
    validation_probabilities: Iterable[float],
    y_test: Iterable[int],
    test_probabilities: Iterable[float],
    *,
    target_npv: float = 0.99,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Propagate validation threshold-selection variation to a fixed test set.

    This quantifies threshold-selection uncertainty conditional on the fitted
    model and untouched test probabilities; it is not a refit-the-model interval.
    """
    y_val = np.asarray(y_validation, dtype=int)
    p_val = np.asarray(validation_probabilities, dtype=float)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(y_val), size=len(y_val))
        thresholds = select_thresholds(y_val[indices], p_val[indices], target_npv=target_npv)
        row: dict[str, float | int] = {
            "iteration": iteration,
            "approve_below": thresholds.approve_below,
            "investigate_at_or_above": thresholds.investigate_at_or_above,
        }
        row.update(triage_policy_metrics(y_test, test_probabilities, thresholds))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_bootstrap_distribution(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarise bootstrap draws with the medians and 95% percentile intervals."""
    excluded = {"iteration"}
    rows: list[dict[str, float | int | str]] = []
    for metric in [column for column in samples.columns if column not in excluded]:
        values = pd.to_numeric(samples[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "bootstrap_samples": int(len(values)),
                "ci_2_5": float(values.quantile(0.025)),
                "median": float(values.quantile(0.5)),
                "ci_97_5": float(values.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def triage_action_fairness_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: DecisionThresholds | dict,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report group action rates and fraud-case TPR/FNR for the three-way policy."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    decisions = np.asarray([assign_decision(probability, thresholds) for probability in p])
    groups = {
        "Sex": audit["Sex"].astype(str),
        "AgeBand": make_age_band(audit["Age"]),
        "MaritalStatus": audit["MaritalStatus"].astype(str),
    }
    rows: list[dict[str, float | int | str]] = []
    for attribute, values in groups.items():
        for group in sorted(values.unique()):
            mask = (values == group).to_numpy()
            if mask.sum() < 20:
                continue
            yt = y[mask]
            decisions_group = decisions[mask]
            positives = int(yt.sum())
            captured = int(((yt == 1) & (decisions_group == "investigate")).sum())
            missed = positives - captured
            row: dict[str, float | int | str] = {
                "attribute": attribute,
                "group": group,
                "n": int(mask.sum()),
                "fraud_prevalence": float(yt.mean()),
                "true_positive_rate": float(captured / positives) if positives else np.nan,
                "false_negative_rate": float(missed / positives) if positives else np.nan,
            }
            tpr_low, tpr_high = _wilson_interval(captured, positives)
            fnr_low, fnr_high = _wilson_interval(missed, positives)
            row.update(
                {
                    "true_positive_rate_ci_95_low": tpr_low,
                    "true_positive_rate_ci_95_high": tpr_high,
                    "false_negative_rate_ci_95_low": fnr_low,
                    "false_negative_rate_ci_95_high": fnr_high,
                }
            )
            for action in ("approve", "abstain", "investigate"):
                count = int((decisions_group == action).sum())
                low, high = _wilson_interval(count, int(mask.sum()))
                row[f"{action}_rate"] = float(count / mask.sum())
                row[f"{action}_rate_ci_95_low"] = low
                row[f"{action}_rate_ci_95_high"] = high
            rows.append(row)
    detail = pd.DataFrame(rows)
    gap_metrics = [
        "approve_rate",
        "abstain_rate",
        "investigate_rate",
        "true_positive_rate",
        "false_negative_rate",
    ]
    gaps = [
        {
            "attribute": attribute,
            "groups_included": int(len(part)),
            **{f"{metric}_difference": float(part[metric].max() - part[metric].min()) for metric in gap_metrics},
        }
        for attribute, part in detail.groupby("attribute")
    ]
    return detail, pd.DataFrame(gaps)


def bootstrap_triage_action_gaps(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: DecisionThresholds | dict,
    audit: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap max-min subgroup gaps for triage actions and TPR/FNR."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    audit = audit.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    samples: list[pd.DataFrame] = []
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        _, gaps = triage_action_fairness_table(
            y[indices], p[indices], thresholds, audit.iloc[indices].reset_index(drop=True)
        )
        if not gaps.empty:
            gaps["iteration"] = iteration
            samples.append(gaps)
    boot = pd.concat(samples, ignore_index=True)
    metrics = [column for column in boot.columns if column.endswith("_difference")]
    rows: list[dict[str, float | int | str]] = []
    for attribute, part in boot.groupby("attribute"):
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if not values.empty:
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


def compare_binary_and_triage_costs(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: DecisionThresholds | dict,
    scenarios: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compare binary and triage policies at exactly the same investigation cutoff.

    Both policies investigate every claim at or above ``investigate_at_or_above``.
    The binary policy approves the remaining claims; the triage policy sends only
    the intermediate-risk group for a priced evidence/human-review step. Costs
    are therefore transparent illustrative inputs, not observed financial loss.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    investigate_threshold = (
        float(thresholds["investigate_at_or_above"])
        if isinstance(thresholds, dict)
        else thresholds.investigate_at_or_above
    )
    binary = np.where(p >= investigate_threshold, "investigate", "approve")
    triage = np.asarray([assign_decision(probability, thresholds) for probability in p])
    if int((binary == "investigate").sum()) != int((triage == "investigate").sum()):
        raise RuntimeError("Policies must use the same investigation capacity.")
    rows: list[dict[str, float | int | str]] = []
    for scenario, costs in scenarios.items():
        required = {"false_approval_cost", "investigation_cost", "defer_cost"}
        missing = required.difference(costs)
        if missing:
            raise ValueError(f"Scenario {scenario!r} is missing costs for {sorted(missing)}")
        for policy, decisions in (("binary", binary), ("three_way_triage", triage)):
            false_approvals = int(((decisions == "approve") & (y == 1)).sum())
            investigations = int((decisions == "investigate").sum())
            deferrals = int((decisions == "abstain").sum())
            total = (
                false_approvals * float(costs["false_approval_cost"])
                + investigations * float(costs["investigation_cost"])
                + deferrals * float(costs["defer_cost"])
            )
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "investigation_threshold": investigate_threshold,
                    "investigations": investigations,
                    "deferrals": deferrals,
                    "false_approvals": false_approvals,
                    "total_illustrative_cost": float(total),
                    "false_approval_cost": float(costs["false_approval_cost"]),
                    "investigation_cost": float(costs["investigation_cost"]),
                    "defer_cost": float(costs["defer_cost"]),
                }
            )
    return pd.DataFrame(rows)
