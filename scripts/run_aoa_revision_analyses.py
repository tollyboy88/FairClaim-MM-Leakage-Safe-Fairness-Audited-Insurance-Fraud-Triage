from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairclaim.data import load_primary_dataset, split_dataset  # noqa: E402
from fairclaim.decision import assign_decision, select_thresholds  # noqa: E402
from fairclaim.metrics import classification_metrics, make_age_band  # noqa: E402

DATA = PROJECT_ROOT / "fraud_oracle.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "aoa_revision"
PUB_DIR = PROJECT_ROOT / "publication" / "aoa"
TABLE_DIR = PUB_DIR / "tables"
FIGURE_DIR = PUB_DIR / "figures"
SEED = 42
TARGET = "FraudFound_P"
AUDIT = ["Sex", "Age", "MaritalStatus"]
IDS = ["PolicyNumber"]


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def hgb(seed: int = SEED, class_weight=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        class_weight=class_weight,
        random_state=seed,
    )


def pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([("preprocess", preprocessor(X)), ("classifier", estimator)])


def model_candidates(seed: int = SEED) -> dict[str, object]:
    # Deliberately no class weights: this is a pure model-family comparison.
    return {
        "Dummy prior": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradientBoosting": hgb(seed),
    }


def run_model_comparison(split) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    for name, estimator in model_candidates().items():
        model = pipeline(split.X_train, estimator)
        scores = cross_validate(
            model,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"},
            n_jobs=1,
            error_score="raise",
        )
        model.fit(split.X_train, split.y_train)
        probability = model.predict_proba(split.X_validation)[:, 1]
        rows.append(
            {
                "model": name,
                "imbalance_treatment": "none",
                "cv_pr_auc_mean": np.mean(scores["test_pr_auc"]),
                "cv_pr_auc_sd": np.std(scores["test_pr_auc"], ddof=1),
                "cv_roc_auc_mean": np.mean(scores["test_roc_auc"]),
                "cv_brier_mean": -np.mean(scores["test_brier"]),
                "validation_pr_auc": average_precision_score(split.y_validation, probability),
                "validation_roc_auc": roc_auc_score(split.y_validation, probability),
                "validation_brier": brier_score_loss(split.y_validation, probability),
            }
        )
        print(f"model family: {name}")
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def smotenc_model(X: pd.DataFrame) -> ImbalancedPipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    encoded = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    final_encoding = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(range(len(numeric)))),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_indices,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return ImbalancedPipeline(
        [
            ("encode", encoded),
            (
                "resample",
                SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=SEED,
                    k_neighbors=5,
                ),
            ),
            ("final_encoding", final_encoding),
            ("classifier", hgb()),
        ]
    )


def imbalance_models(X: pd.DataFrame) -> dict[str, object]:
    return {
        "None": pipeline(X, hgb()),
        "Class weight": pipeline(X, hgb(class_weight="balanced")),
        "Random oversampling": ImbalancedPipeline(
            [
                ("resample", RandomOverSampler(random_state=SEED)),
                ("preprocess", preprocessor(X)),
                ("classifier", hgb()),
            ]
        ),
        "Random undersampling": ImbalancedPipeline(
            [
                ("resample", RandomUnderSampler(random_state=SEED)),
                ("preprocess", preprocessor(X)),
                ("classifier", hgb()),
            ]
        ),
        "SMOTENC": smotenc_model(X),
    }


def run_imbalance_comparison(split) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    for name, model in imbalance_models(split.X_train).items():
        scores = cross_validate(
            model,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
            error_score="raise",
        )
        model.fit(split.X_train, split.y_train)
        probability = model.predict_proba(split.X_validation)[:, 1]
        rows.append(
            {
                "model": "HistGradientBoosting",
                "imbalance_strategy": name,
                "cv_pr_auc_mean": np.mean(scores["test_pr_auc"]),
                "cv_pr_auc_sd": np.std(scores["test_pr_auc"], ddof=1),
                "validation_pr_auc": average_precision_score(split.y_validation, probability),
                "validation_roc_auc": roc_auc_score(split.y_validation, probability),
                "validation_brier": brier_score_loss(split.y_validation, probability),
                "resampling_scope": "inside training folds" if name not in {"None", "Class weight"} else "not applicable",
            }
        )
        print(f"imbalance: {name}")
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def calibrated_model(X: pd.DataFrame, *, method: str = "sigmoid", seed: int = SEED, folds: int = 5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        estimator=pipeline(X, hgb(seed)),
        method=method,
        cv=cv,
    )


def policy_evaluation(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    validation_probability = model.predict_proba(X_validation)[:, 1]
    thresholds = select_thresholds(y_validation.to_numpy(), validation_probability, target_npv=0.99)
    threshold_dict = {
        "approve_below": thresholds.approve_below,
        "investigate_at_or_above": thresholds.investigate_at_or_above,
        "target_negative_predictive_value": 0.99,
    }
    test_probability = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, thresholds.investigate_at_or_above)
    return validation_probability, test_probability, threshold_dict, metrics


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def decision_summary(y: pd.Series, p: np.ndarray, thresholds: dict) -> pd.DataFrame:
    decisions = np.array([assign_decision(value, thresholds) for value in p])
    rows = []
    for action in ["approve", "abstain", "investigate"]:
        mask = decisions == action
        cases = int(mask.sum())
        fraud = int(y.to_numpy()[mask].sum())
        low, high = wilson_interval(fraud, cases)
        rows.append(
            {
                "action": action,
                "claims": cases,
                "coverage_share": cases / len(y),
                "fraud_cases": fraud,
                "fraud_rate": fraud / cases if cases else np.nan,
                "fraud_rate_ci_low": low,
                "fraud_rate_ci_high": high,
                "npv_if_approved": 1 - fraud / cases if action == "approve" and cases else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(y: pd.Series, p: np.ndarray, threshold: float, n_bootstrap: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y_array = y.to_numpy()
    rows = []
    for iteration in range(n_bootstrap):
        idx = rng.integers(0, len(y_array), len(y_array))
        if len(np.unique(y_array[idx])) < 2:
            continue
        result = classification_metrics(y_array[idx], p[idx], threshold)
        result["iteration"] = iteration
        rows.append(result)
    frame = pd.DataFrame(rows)
    metrics = ["pr_auc", "roc_auc", "brier_score", "precision", "recall", "f2"]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": classification_metrics(y_array, p, threshold)[metric],
                "ci_2_5": frame[metric].quantile(0.025),
                "ci_97_5": frame[metric].quantile(0.975),
                "bootstrap_samples": len(frame),
            }
            for metric in metrics
        ]
    )


def threshold_bootstrap(y: pd.Series, p: np.ndarray, n_bootstrap: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y_array = y.to_numpy()
    rows = []
    for iteration in range(n_bootstrap):
        idx = rng.integers(0, len(y_array), len(y_array))
        thresholds = select_thresholds(y_array[idx], p[idx], target_npv=0.99)
        rows.append(
            {
                "iteration": iteration,
                "approve_below": thresholds.approve_below,
                "investigate_at_or_above": thresholds.investigate_at_or_above,
            }
        )
    return pd.DataFrame(rows)


def action_fairness(
    y: pd.Series,
    p: np.ndarray,
    thresholds: dict,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    decisions = np.array([assign_decision(value, thresholds) for value in p])
    group_map = {
        "Sex": audit["Sex"].astype(str),
        "AgeBand": make_age_band(audit["Age"]),
        "MaritalStatus": audit["MaritalStatus"].astype(str),
    }
    group_map["Sex_x_AgeBand"] = group_map["Sex"] + " x " + group_map["AgeBand"]
    y_array = y.to_numpy()
    rows = []
    for attribute, groups in group_map.items():
        for group in sorted(groups.unique()):
            mask = (groups == group).to_numpy()
            if mask.sum() < 20:
                continue
            yg = y_array[mask]
            pg = p[mask]
            dg = decisions[mask]
            approval = dg == "approve"
            abstention = dg == "abstain"
            investigation = dg == "investigate"
            approved_n = int(approval.sum())
            approved_legitimate = int(((yg == 0) & approval).sum())
            fraud_n = int((yg == 1).sum())
            approve_low, approve_high = wilson_interval(int(approval.sum()), len(yg))
            abstain_low, abstain_high = wilson_interval(int(abstention.sum()), len(yg))
            investigate_low, investigate_high = wilson_interval(int(investigation.sum()), len(yg))
            missed_low, missed_high = wilson_interval(
                int(((yg == 1) & approval).sum()), fraud_n
            )
            npv_low, npv_high = wilson_interval(approved_legitimate, approved_n)
            ece = 0.0
            for bin_indices in np.array_split(np.argsort(pg), min(10, len(pg))):
                if len(bin_indices):
                    ece += len(bin_indices) / len(pg) * abs(pg[bin_indices].mean() - yg[bin_indices].mean())
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n": int(mask.sum()),
                    "fraud_prevalence": yg.mean(),
                    "approval_rate": approval.mean(),
                    "approval_rate_ci_low": approve_low,
                    "approval_rate_ci_high": approve_high,
                    "abstention_rate": abstention.mean(),
                    "abstention_rate_ci_low": abstain_low,
                    "abstention_rate_ci_high": abstain_high,
                    "investigation_rate": investigation.mean(),
                    "investigation_rate_ci_low": investigate_low,
                    "investigation_rate_ci_high": investigate_high,
                    "fraud_missed_in_approval_n": int(((yg == 1) & approval).sum()),
                    "fraud_missed_in_approval_rate": ((yg == 1) & approval).sum() / max(fraud_n, 1),
                    "fraud_missed_in_approval_rate_ci_low": missed_low,
                    "fraud_missed_in_approval_rate_ci_high": missed_high,
                    "approval_region_npv": approved_legitimate / approved_n if approved_n else np.nan,
                    "approval_region_npv_ci_low": npv_low,
                    "approval_region_npv_ci_high": npv_high,
                    "mean_probability": pg.mean(),
                    "expected_calibration_error": ece,
                }
            )
    return pd.DataFrame(rows)


def capacity_analysis(y: pd.Series, p: np.ndarray) -> pd.DataFrame:
    y_array = y.to_numpy()
    order = np.argsort(-p)
    total_fraud = int(y_array.sum())
    rows = []
    for share in [0.05, 0.10, 0.20, 0.30]:
        n = int(np.ceil(len(y_array) * share))
        selected = order[:n]
        captured = int(y_array[selected].sum())
        rows.append(
            {
                "investigation_capacity": share,
                "claims_investigated": n,
                "fraud_captured": captured,
                "fraud_capture_rate": captured / total_fraud,
                "investigation_precision": captured / n,
                "legitimate_claims_investigated": n - captured,
            }
        )
    return pd.DataFrame(rows)


def cost_analysis(y: pd.Series, p: np.ndarray, thresholds: dict) -> pd.DataFrame:
    decisions = np.array([assign_decision(value, thresholds) for value in p])
    y_array = y.to_numpy()
    scenarios = {
        "Low-cost": {"investigation": 100, "legitimate_investigation": 50, "missed_fraud": 5000, "delay": 40},
        "Base": {"investigation": 250, "legitimate_investigation": 150, "missed_fraud": 10000, "delay": 100},
        "High-loss": {"investigation": 500, "legitimate_investigation": 300, "missed_fraud": 25000, "delay": 250},
    }
    policies = {"Binary investigate": np.where(p >= thresholds["investigate_at_or_above"], "investigate", "approve"),
                "Three-way selective": decisions}
    rows = []
    for scenario, costs in scenarios.items():
        for policy, action in policies.items():
            investigate = action == "investigate"
            abstain = action == "abstain"
            approve = action == "approve"
            missed_approve = int(((y_array == 1) & approve).sum())
            missed_abstain_equivalent = 0.25 * int(((y_array == 1) & abstain).sum())
            total = (
                investigate.sum() * costs["investigation"]
                + ((y_array == 0) & investigate).sum() * costs["legitimate_investigation"]
                + (missed_approve + missed_abstain_equivalent) * costs["missed_fraud"]
                + ((y_array == 0) & abstain).sum() * costs["delay"]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    **{f"cost_{key}": value for key, value in costs.items()},
                    "investigations": int(investigate.sum()),
                    "abstentions": int(abstain.sum()),
                    "missed_fraud_equivalent": missed_approve + missed_abstain_equivalent,
                    "expected_total_cost": total,
                    "expected_cost_per_claim": total / len(y_array),
                }
            )
    return pd.DataFrame(rows)


def selective_ablation(
    y_validation: pd.Series,
    raw_validation: np.ndarray,
    calibrated_validation: np.ndarray,
    y_test: pd.Series,
    raw_test: np.ndarray,
    calibrated_test: np.ndarray,
) -> pd.DataFrame:
    def f2_threshold(y, p):
        candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), p])
        scores = [fbeta_score(y, p >= threshold, beta=2, zero_division=0) for threshold in candidates]
        return float(candidates[int(np.argmax(scores))])

    raw_binary_threshold = f2_threshold(y_validation, raw_validation)
    calibrated_binary_threshold = f2_threshold(y_validation, calibrated_validation)
    raw_selective = select_thresholds(y_validation.to_numpy(), raw_validation, 0.99)
    calibrated_selective = select_thresholds(y_validation.to_numpy(), calibrated_validation, 0.99)
    configurations = [
        ("Uncalibrated binary", raw_test, {"approve_below": raw_binary_threshold, "investigate_at_or_above": raw_binary_threshold}, False),
        ("Calibrated binary", calibrated_test, {"approve_below": calibrated_binary_threshold, "investigate_at_or_above": calibrated_binary_threshold}, False),
        ("Uncalibrated selective", raw_test, {"approve_below": raw_selective.approve_below, "investigate_at_or_above": raw_selective.investigate_at_or_above}, False),
        ("Calibrated selective, no fairness audit", calibrated_test, {"approve_below": calibrated_selective.approve_below, "investigate_at_or_above": calibrated_selective.investigate_at_or_above}, False),
        ("Complete framework", calibrated_test, {"approve_below": calibrated_selective.approve_below, "investigate_at_or_above": calibrated_selective.investigate_at_or_above}, True),
    ]
    rows = []
    y_array = y_test.to_numpy()
    for name, p, thresholds, audit_present in configurations:
        decisions = np.array([assign_decision(value, thresholds) for value in p])
        covered = decisions != "abstain"
        predicted = decisions == "investigate"
        approved = decisions == "approve"
        investigated = decisions == "investigate"
        rows.append(
            {
                "configuration": name,
                "calibrated": "Calibrated" in name or name == "Complete framework",
                "abstention_enabled": bool((decisions == "abstain").any()),
                "fairness_audit_present": audit_present,
                "pr_auc": average_precision_score(y_array, p),
                "brier_score": brier_score_loss(y_array, p),
                "coverage": covered.mean(),
                "selective_error_risk": (predicted[covered] != y_array[covered]).mean() if covered.any() else np.nan,
                "approval_npv": (y_array[approved] == 0).mean() if approved.any() else np.nan,
                "investigation_precision": y_array[investigated].mean() if investigated.any() else np.nan,
                "abstention_rate": (decisions == "abstain").mean(),
            }
        )
    return pd.DataFrame(rows)


def coverage_risk_curve(y: pd.Series, p: np.ndarray) -> pd.DataFrame:
    y_array = y.to_numpy()
    # Confidence is distance from the binary operating threshold chosen for F2.
    precision, recall, thresholds = precision_recall_curve(y_array, p)
    f2 = 5 * precision * recall / np.maximum(4 * precision + recall, 1e-12)
    threshold = thresholds[min(int(np.nanargmax(f2)), len(thresholds) - 1)]
    confidence = np.abs(p - threshold)
    order = np.argsort(-confidence)
    rows = []
    for coverage in np.linspace(0.50, 1.00, 26):
        n = max(1, int(np.floor(len(y_array) * coverage)))
        idx = order[:n]
        pred = p[idx] >= threshold
        rows.append(
            {
                "coverage": n / len(y_array),
                "selective_error_risk": (pred != y_array[idx]).mean(),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def run_age_proxy_sensitivity(split) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, dict]]]:
    rows = []
    outputs: dict[str, tuple[np.ndarray, dict]] = {}
    for specification, drop in [
        ("Retain AgeOfPolicyHolder", []),
        ("Exclude all direct age-derived fields", ["AgeOfPolicyHolder"]),
    ]:
        X_train = split.X_train.drop(columns=drop)
        X_validation = split.X_validation.drop(columns=drop)
        X_test = split.X_test.drop(columns=drop)
        model = calibrated_model(X_train)
        model.fit(X_train, split.y_train)
        validation_p, test_p, thresholds, metrics = policy_evaluation(
            model, X_validation, split.y_validation, X_test, split.y_test
        )
        summary = decision_summary(split.y_test, test_p, thresholds)
        age_groups = make_age_band(split.audit_test["Age"])
        action = np.array([assign_decision(value, thresholds) for value in test_p])
        investigation_rates = pd.Series(action == "investigate").groupby(age_groups).mean()
        rows.append(
            {
                "specification": specification,
                "feature_count": X_train.shape[1],
                "validation_pr_auc": average_precision_score(split.y_validation, validation_p),
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_brier": metrics["brier_score"],
                "approve_threshold": thresholds["approve_below"],
                "investigate_threshold": thresholds["investigate_at_or_above"],
                "approval_npv": float(summary.loc[summary.action == "approve", "npv_if_approved"].iloc[0]),
                "age_investigation_rate_gap": investigation_rates.max() - investigation_rates.min(),
            }
        )
        outputs[specification] = (test_p, thresholds)
        print(f"age sensitivity: {specification}")
    return pd.DataFrame(rows), outputs


def repeated_seed_analysis(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in [7, 21, 42, 84, 168]:
        indices = np.arange(len(data))
        train_validation_idx, test_idx = train_test_split(
            indices, test_size=0.20, stratify=data[TARGET], random_state=seed
        )
        train_idx, validation_idx = train_test_split(
            train_validation_idx,
            test_size=0.25,
            stratify=data.iloc[train_validation_idx][TARGET],
            random_state=seed,
        )
        features = [c for c in data.columns if c not in {TARGET, *AUDIT, *IDS}]
        X_train = data.iloc[train_idx][features].reset_index(drop=True)
        y_train = data.iloc[train_idx][TARGET].reset_index(drop=True)
        X_validation = data.iloc[validation_idx][features].reset_index(drop=True)
        y_validation = data.iloc[validation_idx][TARGET].reset_index(drop=True)
        X_test = data.iloc[test_idx][features].reset_index(drop=True)
        y_test = data.iloc[test_idx][TARGET].reset_index(drop=True)
        model = calibrated_model(X_train, seed=seed, folds=3)
        model.fit(X_train, y_train)
        validation_p, test_p, thresholds, metrics = policy_evaluation(
            model, X_validation, y_validation, X_test, y_test
        )
        rows.append(
            {
                "seed": seed,
                "validation_pr_auc": average_precision_score(y_validation, validation_p),
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_brier": metrics["brier_score"],
                "approve_threshold": thresholds["approve_below"],
                "investigate_threshold": thresholds["investigate_at_or_above"],
            }
        )
        print(f"repeated seed: {seed}")
    return pd.DataFrame(rows)


def temporal_validation(data: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in data.columns if c not in {TARGET, *AUDIT, *IDS, "Year"}]
    train = data[data.Year == 1994].reset_index(drop=True)
    validation = data[data.Year == 1995].reset_index(drop=True)
    test = data[data.Year == 1996].reset_index(drop=True)
    model = calibrated_model(train[features], folds=5)
    model.fit(train[features], train[TARGET])
    validation_p, test_p, thresholds, metrics = policy_evaluation(
        model,
        validation[features],
        validation[TARGET],
        test[features],
        test[TARGET],
    )
    return pd.DataFrame(
        [
            {
                "training_year": 1994,
                "validation_year": 1995,
                "test_year": 1996,
                "train_n": len(train),
                "validation_n": len(validation),
                "test_n": len(test),
                "validation_pr_auc": average_precision_score(validation[TARGET], validation_p),
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_brier": metrics["brier_score"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "approve_threshold": thresholds["approve_below"],
                "investigate_threshold": thresholds["investigate_at_or_above"],
            }
        ]
    )


def calibration_comparison(split) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    probabilities = {}
    for method in ["sigmoid", "isotonic"]:
        model = calibrated_model(split.X_train, method=method)
        model.fit(split.X_train, split.y_train)
        validation_p, test_p, thresholds, metrics = policy_evaluation(
            model,
            split.X_validation,
            split.y_validation,
            split.X_test,
            split.y_test,
        )
        rows.append(
            {
                "method": method,
                "validation_pr_auc": average_precision_score(split.y_validation, validation_p),
                "validation_brier": brier_score_loss(split.y_validation, validation_p),
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_brier": metrics["brier_score"],
                "approve_threshold": thresholds["approve_below"],
                "investigate_threshold": thresholds["investigate_at_or_above"],
            }
        )
        probabilities[method] = test_p
        probabilities[f"{method}_validation"] = validation_p
        probabilities[f"{method}_thresholds"] = thresholds
        print(f"calibration: {method}")
    return pd.DataFrame(rows), probabilities


def plot_outputs(
    model_comparison: pd.DataFrame,
    imbalance: pd.DataFrame,
    age_sensitivity: pd.DataFrame,
    capacity: pd.DataFrame,
    costs: pd.DataFrame,
    coverage: pd.DataFrame,
    action_fair: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    temporal: pd.DataFrame,
    seeds: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    blue, gold, red, green = "#285781", "#D99A2B", "#B94F45", "#4D8C57"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].barh(model_comparison["model"][::-1], model_comparison["validation_pr_auc"][::-1], color=blue)
    axes[0].set_title("Model families: no imbalance treatment")
    axes[0].set_xlabel("Validation PR-AUC")
    axes[1].barh(imbalance["imbalance_strategy"][::-1], imbalance["validation_pr_auc"][::-1], color=gold)
    axes[1].set_title("HistGradientBoosting imbalance strategies")
    axes[1].set_xlabel("Validation PR-AUC")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_01_consistent_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metrics = ["test_pr_auc", "test_roc_auc", "approval_npv", "age_investigation_rate_gap"]
    x = np.arange(len(metrics))
    width = 0.36
    for i, (_, row) in enumerate(age_sensitivity.iterrows()):
        ax.bar(x + (i - 0.5) * width, [row[m] for m in metrics], width, label=row["specification"])
    ax.set_xticks(x, ["PR-AUC", "ROC-AUC", "Approval NPV", "Age investigation-rate gap"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Proxy-feature sensitivity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_02_age_proxy_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(capacity["investigation_capacity"] * 100, capacity["fraud_capture_rate"] * 100, marker="o", color=blue)
    axes[0].plot(capacity["investigation_capacity"] * 100, capacity["investigation_precision"] * 100, marker="s", color=red)
    axes[0].set_xlabel("Claims investigated (%)")
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Capacity-constrained investigation")
    axes[0].legend(["Fraud captured", "Investigation precision"], frameon=False)
    pivot = costs.pivot(index="scenario", columns="policy", values="expected_cost_per_claim")
    pivot.plot(kind="bar", ax=axes[1], color=[gold, green])
    axes[1].set_ylabel("Expected cost per claim (£)")
    axes[1].set_title("Illustrative operational cost")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_03_operational_value.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(coverage["coverage"], coverage["selective_error_risk"], color=blue, linewidth=2)
    ax.set_xlabel("Coverage (fraction receiving approve/investigate)")
    ax.set_ylabel("Selective classification error")
    ax.set_title("Coverage-risk curve")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_04_coverage_risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = action_fair[action_fair.attribute.isin(["Sex", "AgeBand"])].copy()
    selected["audit_group"] = selected["attribute"] + ": " + selected["group"]
    chart = selected.pivot_table(
        index="audit_group",
        values=["approval_rate", "abstention_rate", "investigation_rate"],
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    chart.plot(kind="bar", stacked=True, ax=ax, color=[green, gold, red])
    ax.set_ylabel("Action rate")
    ax.set_xlabel("Audited group")
    ax.set_title("Three-action fairness audit")
    ax.legend(["Abstention", "Approval", "Investigation"], frameon=False)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_05_three_action_fairness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ci = bootstrap_ci.set_index("metric")
    estimates = ci["estimate"]
    lower = estimates - ci["ci_2_5"]
    upper = ci["ci_97_5"] - estimates
    axes[0].errorbar(np.arange(len(ci)), estimates, yerr=[lower, upper], fmt="o", color=blue, capsize=4)
    axes[0].set_xticks(np.arange(len(ci)), ci.index, rotation=35, ha="right")
    axes[0].set_title("Held-out metrics with bootstrap 95% CIs")
    axes[1].scatter(seeds["test_pr_auc"], seeds["test_brier"], color=gold, s=55, label="Random split seeds")
    axes[1].scatter(temporal["test_pr_auc"], temporal["test_brier"], color=red, s=80, marker="X", label="1994→1996 temporal")
    axes[1].set_xlabel("Test PR-AUC")
    axes[1].set_ylabel("Test Brier score")
    axes[1].set_title("Random-split and temporal robustness")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_06_uncertainty_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [REPORT_DIR, PUB_DIR, TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_primary_dataset(str(DATA))
    split = split_dataset(data)

    model_comparison = run_model_comparison(split)
    imbalance = run_imbalance_comparison(split)
    age_sensitivity, age_outputs = run_age_proxy_sensitivity(split)

    raw_model = pipeline(split.X_train, hgb())
    raw_model.fit(split.X_train, split.y_train)
    raw_validation = raw_model.predict_proba(split.X_validation)[:, 1]
    raw_test = raw_model.predict_proba(split.X_test)[:, 1]

    calibration, calibration_outputs = calibration_comparison(split)
    sigmoid_validation = calibration_outputs["sigmoid_validation"]
    sigmoid_test = calibration_outputs["sigmoid"]
    sigmoid_thresholds = calibration_outputs["sigmoid_thresholds"]

    decision = decision_summary(split.y_test, sigmoid_test, sigmoid_thresholds)
    bootstrap_ci = bootstrap_metrics(
        split.y_test, sigmoid_test, sigmoid_thresholds["investigate_at_or_above"]
    )
    threshold_samples = threshold_bootstrap(split.y_validation, sigmoid_validation)
    action_fair = action_fairness(
        split.y_test, sigmoid_test, sigmoid_thresholds, split.audit_test
    )
    capacity = capacity_analysis(split.y_test, sigmoid_test)
    costs = cost_analysis(split.y_test, sigmoid_test, sigmoid_thresholds)
    ablations = selective_ablation(
        split.y_validation,
        raw_validation,
        sigmoid_validation,
        split.y_test,
        raw_test,
        sigmoid_test,
    )
    coverage = coverage_risk_curve(split.y_test, sigmoid_test)
    seeds = repeated_seed_analysis(data)
    temporal = temporal_validation(data)

    outputs = {
        "table_01_model_family_no_balancing.csv": model_comparison,
        "table_02_hgb_imbalance_strategies.csv": imbalance,
        "table_03_age_proxy_sensitivity.csv": age_sensitivity,
        "table_04_calibration_comparison.csv": calibration,
        "table_05_decision_regions_with_ci.csv": decision,
        "table_06_metric_bootstrap_ci.csv": bootstrap_ci,
        "table_07_threshold_bootstrap_samples.csv": threshold_samples,
        "table_08_three_action_fairness.csv": action_fair,
        "table_09_fixed_capacity.csv": capacity,
        "table_10_operational_cost_sensitivity.csv": costs,
        "table_11_framework_ablations.csv": ablations,
        "table_12_coverage_risk.csv": coverage,
        "table_13_repeated_seeds.csv": seeds,
        "table_14_temporal_validation.csv": temporal,
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLE_DIR / filename, index=False)
        frame.to_csv(REPORT_DIR / filename, index=False)

    summary = {
        "target_journal": "Applied Operations and Analytics",
        "title": "Calibrated Selective Prediction for Leakage-Safe and Fairness-Audited Motor Insurance Fraud Triage",
        "rows": len(data),
        "fraud_prevalence": float(data[TARGET].mean()),
        "primary_model": "HistGradientBoosting without balancing",
        "thresholds": sigmoid_thresholds,
        "held_out_metrics": classification_metrics(
            split.y_test, sigmoid_test, sigmoid_thresholds["investigate_at_or_above"]
        ),
        "temporal_validation": temporal.iloc[0].to_dict(),
        "repeated_seed_test_pr_auc_mean": seeds.test_pr_auc.mean(),
        "repeated_seed_test_pr_auc_sd": seeds.test_pr_auc.std(ddof=1),
        "threshold_bootstrap": {
            "approve_ci_2_5": threshold_samples.approve_below.quantile(0.025),
            "approve_median": threshold_samples.approve_below.median(),
            "approve_ci_97_5": threshold_samples.approve_below.quantile(0.975),
            "investigate_ci_2_5": threshold_samples.investigate_at_or_above.quantile(0.025),
            "investigate_median": threshold_samples.investigate_at_or_above.median(),
            "investigate_ci_97_5": threshold_samples.investigate_at_or_above.quantile(0.975),
        },
        "cost_assumption_note": "Illustrative GBP scenarios, not observed claim amounts. Abstained fraud is assigned 25% of missed-fraud cost, representing 75% recovery through manual review.",
    }
    save_json(REPORT_DIR / "analysis_summary.json", summary)

    plot_outputs(
        model_comparison,
        imbalance,
        age_sensitivity,
        capacity,
        costs,
        coverage,
        action_fair,
        bootstrap_ci,
        temporal,
        seeds,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
