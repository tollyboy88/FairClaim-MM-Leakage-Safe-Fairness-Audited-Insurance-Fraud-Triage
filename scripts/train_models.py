from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairclaim.config import (  # noqa: E402
    ARTIFACT_DIR,
    DATA_PATH,
    MODEL_BUNDLE_PATH,
    RANDOM_STATE,
    REPORT_DIR,
)
from fairclaim.data import (  # noqa: E402
    build_input_schema,
    load_primary_dataset,
    reference_values,
    split_dataset,
)
from fairclaim.decision import assign_decision, select_thresholds  # noqa: E402
from fairclaim.metrics import (  # noqa: E402
    bootstrap_fairness_gaps,
    bootstrap_threshold_policy_metrics,
    bootstrap_triage_action_gaps,
    classification_metrics,
    compare_binary_and_triage_costs,
    fairness_table,
    summarise_bootstrap_distribution,
    triage_action_fairness_table,
    triage_policy_metrics,
)


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
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


def candidates() -> dict[str, object]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight=None,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "gaussian_naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=3,
            class_weight=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450,
            min_samples_leaf=3,
            class_weight=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            class_weight=None,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_primary_dataset(str(DATA_PATH))
    split = split_dataset(data)
    preprocessor = make_preprocessor(split.X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name, classifier in candidates().items():
        pipeline = Pipeline(
            [("preprocess", clone(preprocessor)), ("classifier", classifier)]
        )
        scores = cross_validate(
            pipeline,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(split.X_train, split.y_train)
        validation_probability = pipeline.predict_proba(split.X_validation)[:, 1]
        row = {
            "model": name,
            "cv_pr_auc_mean": float(np.mean(scores["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(scores["test_pr_auc"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "validation_pr_auc": float(
                average_precision_score(split.y_validation, validation_probability)
            ),
            "validation_roc_auc": float(
                roc_auc_score(split.y_validation, validation_probability)
            ),
        }
        comparison.append(row)
        fitted[name] = pipeline
        print(f"{name}: validation PR-AUC={row['validation_pr_auc']:.4f}")

    comparison_frame = pd.DataFrame(comparison).sort_values(
        ["cv_pr_auc_mean", "validation_pr_auc"], ascending=False
    )
    comparison_frame.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    selected_name = str(comparison_frame.iloc[0]["model"])
    print(f"Selected model: {selected_name}")

    calibrated_model = CalibratedClassifierCV(
        estimator=clone(fitted[selected_name]),
        method="sigmoid",
        cv=cv,
    )
    calibrated_model.fit(split.X_train, split.y_train)
    validation_probability = calibrated_model.predict_proba(split.X_validation)[:, 1]
    thresholds = select_thresholds(
        split.y_validation.to_numpy(), validation_probability, target_npv=0.99
    )
    threshold_dict = {
        "approve_below": thresholds.approve_below,
        "investigate_at_or_above": thresholds.investigate_at_or_above,
        "target_negative_predictive_value": thresholds.target_negative_predictive_value,
        "selection_dataset": "validation",
    }

    test_probability = calibrated_model.predict_proba(split.X_test)[:, 1]
    test_metrics = classification_metrics(
        split.y_test,
        test_probability,
        thresholds.investigate_at_or_above,
    )
    validation_metrics = classification_metrics(
        split.y_validation,
        validation_probability,
        thresholds.investigate_at_or_above,
    )
    test_decisions = [assign_decision(p, threshold_dict) for p in test_probability]

    fairness_detail, fairness_gaps = fairness_table(
        split.y_test,
        test_probability,
        thresholds.investigate_at_or_above,
        split.audit_test,
    )
    fairness_detail.to_csv(REPORT_DIR / "fairness_by_group.csv", index=False)
    fairness_gaps.to_csv(REPORT_DIR / "fairness_gaps.csv", index=False)
    bootstrap_fairness_gaps(
        split.y_test,
        test_probability,
        thresholds.investigate_at_or_above,
        split.audit_test,
        n_bootstrap=500,
        random_state=RANDOM_STATE,
    ).to_csv(REPORT_DIR / "fairness_gap_confidence_intervals.csv", index=False)

    triage_fairness_detail, triage_fairness_gaps = triage_action_fairness_table(
        split.y_test, test_probability, threshold_dict, split.audit_test
    )
    triage_fairness_detail.to_csv(REPORT_DIR / "triage_action_fairness_by_group.csv", index=False)
    triage_fairness_gaps.to_csv(REPORT_DIR / "triage_action_fairness_gaps.csv", index=False)
    bootstrap_triage_action_gaps(
        split.y_test,
        test_probability,
        threshold_dict,
        split.audit_test,
        n_bootstrap=500,
        random_state=RANDOM_STATE,
    ).to_csv(REPORT_DIR / "triage_action_gap_confidence_intervals.csv", index=False)

    threshold_bootstrap = bootstrap_threshold_policy_metrics(
        split.y_validation,
        validation_probability,
        split.y_test,
        test_probability,
        target_npv=thresholds.target_negative_predictive_value,
        n_bootstrap=500,
        random_state=RANDOM_STATE,
    )
    threshold_bootstrap.to_csv(REPORT_DIR / "threshold_policy_bootstrap_draws.csv", index=False)
    summarise_bootstrap_distribution(threshold_bootstrap).to_csv(
        REPORT_DIR / "threshold_policy_bootstrap_summary.csv", index=False
    )

    cost_scenarios = {
        "low_fraud_loss": {"false_approval_cost": 5_000.0, "investigation_cost": 250.0, "defer_cost": 100.0},
        "moderate_fraud_loss": {"false_approval_cost": 15_000.0, "investigation_cost": 250.0, "defer_cost": 100.0},
        "high_fraud_loss": {"false_approval_cost": 30_000.0, "investigation_cost": 250.0, "defer_cost": 100.0},
    }
    compare_binary_and_triage_costs(
        split.y_test, test_probability, threshold_dict, cost_scenarios
    ).to_csv(REPORT_DIR / "binary_vs_triage_cost_comparison.csv", index=False)

    importance = permutation_importance(
        calibrated_model,
        split.X_test,
        split.y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": split.X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)

    fraction_positive, mean_predicted = calibration_curve(
        split.y_test, test_probability, n_bins=10, strategy="quantile"
    )
    pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "observed_fraud_rate": fraction_positive}
    ).to_csv(REPORT_DIR / "calibration_curve.csv", index=False)

    predictions = pd.concat(
        [
            split.ids_test,
            split.audit_test,
            pd.DataFrame(
                {
                    "actual_fraud": split.y_test,
                    "fraud_probability": test_probability,
                    "decision": test_decisions,
                }
            ),
        ],
        axis=1,
    )
    predictions.to_csv(REPORT_DIR / "held_out_test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "actual_fraud": split.y_validation,
            "fraud_probability": validation_probability,
        }
    ).to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    predictions.groupby("decision", as_index=False).agg(
        claims=("actual_fraud", "size"),
        fraud_cases=("actual_fraud", "sum"),
        observed_fraud_rate=("actual_fraud", "mean"),
        mean_score=("fraud_probability", "mean"),
    ).to_csv(REPORT_DIR / "decision_policy_test_summary.csv", index=False)

    model_version = datetime.now(timezone.utc).strftime("fairclaim-%Y%m%dT%H%M%SZ")
    schema = build_input_schema(data, split.X_train.columns.tolist())
    bundle = {
        "model": calibrated_model,
        "selected_model": selected_name,
        "selected_imbalance_strategy": "no_balancing",
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_features": split.X_train.columns.tolist(),
        "protected_attributes": ["Sex", "Age", "MaritalStatus"],
        "input_schema": schema,
        "reference_values": reference_values(split.X_train),
        "thresholds": threshold_dict,
        "test_metrics": test_metrics,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    json_dump(REPORT_DIR / "test_metrics.json", test_metrics)
    json_dump(REPORT_DIR / "validation_metrics.json", validation_metrics)
    json_dump(
        REPORT_DIR / "triage_policy_test_metrics.json",
        triage_policy_metrics(split.y_test, test_probability, threshold_dict),
    )
    json_dump(REPORT_DIR / "decision_thresholds.json", threshold_dict)
    json_dump(
        REPORT_DIR / "training_summary.json",
        {
            "model_version": model_version,
            "selected_model": selected_name,
            "selected_imbalance_strategy": "no_balancing",
            "data_rows": len(data),
            "train_rows": len(split.X_train),
            "validation_rows": len(split.X_validation),
            "test_rows": len(split.X_test),
            "protected_attributes_excluded_from_score": ["Sex", "Age", "MaritalStatus"],
            "id_columns_excluded_from_score": ["PolicyNumber"],
            "test_decision_counts": pd.Series(test_decisions).value_counts().to_dict(),
        },
    )
    print(json.dumps({"selected_model": selected_name, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
