from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairclaim.config import DATA_PATH, RANDOM_STATE, REPORT_DIR  # noqa: E402
from fairclaim.data import load_primary_dataset, split_dataset  # noqa: E402


def classifier(class_weight=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def standard_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def smotenc_pipeline(X: pd.DataFrame) -> ImbalancedPipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    before = ColumnTransformer(
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
    after = ColumnTransformer(
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
            ("encode_for_smotenc", before),
            (
                "resample",
                SMOTENC(
                    categorical_features=categorical_indices,
                    sampling_strategy="auto",
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            ),
            ("model_encoding", after),
            ("classifier", classifier()),
        ]
    )


def strategies(X: pd.DataFrame) -> dict[str, object]:
    prep = standard_preprocessor(X)
    return {
        "no_balancing": Pipeline([("preprocess", prep), ("classifier", classifier())]),
        "class_weight_balanced": Pipeline(
            [("preprocess", standard_preprocessor(X)), ("classifier", classifier("balanced"))]
        ),
        "random_undersampling": ImbalancedPipeline(
            [
                ("resample", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("preprocess", standard_preprocessor(X)),
                ("classifier", classifier()),
            ]
        ),
        "random_oversampling": ImbalancedPipeline(
            [
                ("resample", RandomOverSampler(random_state=RANDOM_STATE)),
                ("preprocess", standard_preprocessor(X)),
                ("classifier", classifier()),
            ]
        ),
        "smotenc_training_folds_only": smotenc_pipeline(X),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_dataset(load_primary_dataset(str(DATA_PATH)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in strategies(split.X_train).items():
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
        row = {
            "strategy": name,
            "cv_pr_auc_mean": float(np.mean(scores["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(scores["test_pr_auc"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "validation_pr_auc": float(average_precision_score(split.y_validation, probability)),
            "validation_roc_auc": float(roc_auc_score(split.y_validation, probability)),
            "validation_brier": float(brier_score_loss(split.y_validation, probability)),
            "resampling_scope": "training folds only" if name not in {"no_balancing", "class_weight_balanced"} else "not applicable",
        }
        rows.append(row)
        print(name, row)
    result = pd.DataFrame(rows).sort_values(["validation_pr_auc", "cv_pr_auc_mean"], ascending=False)
    result.to_csv(REPORT_DIR / "imbalance_strategy_comparison.csv", index=False)
    print("\nRanked strategies:\n", result.to_string(index=False))


if __name__ == "__main__":
    main()
