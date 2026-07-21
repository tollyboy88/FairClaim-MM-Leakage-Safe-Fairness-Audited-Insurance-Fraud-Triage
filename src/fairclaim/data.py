from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AUDIT_ONLY_COLUMNS, ID_COLUMNS, RANDOM_STATE, TARGET


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    audit_train: pd.DataFrame
    audit_validation: pd.DataFrame
    audit_test: pd.DataFrame
    ids_train: pd.DataFrame
    ids_validation: pd.DataFrame
    ids_test: pd.DataFrame


def load_primary_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1 values")
    return data


def split_dataset(data: pd.DataFrame) -> DatasetSplit:
    indices = np.arange(len(data))
    train_validation_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    train_idx, validation_idx = train_test_split(
        train_validation_idx,
        test_size=0.25,
        stratify=data.iloc[train_validation_idx][TARGET],
        random_state=RANDOM_STATE,
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in {TARGET, *ID_COLUMNS, *AUDIT_ONLY_COLUMNS}
    ]

    def frame(rows: np.ndarray, columns: list[str]) -> pd.DataFrame:
        return data.iloc[rows][columns].reset_index(drop=True)

    def target(rows: np.ndarray) -> pd.Series:
        return data.iloc[rows][TARGET].astype(int).reset_index(drop=True)

    return DatasetSplit(
        X_train=frame(train_idx, feature_columns),
        X_validation=frame(validation_idx, feature_columns),
        X_test=frame(test_idx, feature_columns),
        y_train=target(train_idx),
        y_validation=target(validation_idx),
        y_test=target(test_idx),
        audit_train=frame(train_idx, AUDIT_ONLY_COLUMNS),
        audit_validation=frame(validation_idx, AUDIT_ONLY_COLUMNS),
        audit_test=frame(test_idx, AUDIT_ONLY_COLUMNS),
        ids_train=frame(train_idx, ID_COLUMNS),
        ids_validation=frame(validation_idx, ID_COLUMNS),
        ids_test=frame(test_idx, ID_COLUMNS),
    )


def build_input_schema(data: pd.DataFrame, model_features: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {"model_features": {}, "audit_fields": {}, "id_fields": {}}
    for column in model_features:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            schema["model_features"][column] = {
                "type": "number",
                "required": True,
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "default": float(series.median()),
            }
        else:
            counts = series.astype(str).value_counts()
            schema["model_features"][column] = {
                "type": "category",
                "required": True,
                "options": counts.index.tolist(),
                "default": str(counts.index[0]),
            }

    for column in AUDIT_ONLY_COLUMNS:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            schema["audit_fields"][column] = {
                "type": "number",
                "required": False,
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "default": float(series.median()),
                "purpose": "fairness_audit_only",
            }
        else:
            counts = series.astype(str).value_counts()
            schema["audit_fields"][column] = {
                "type": "category",
                "required": False,
                "options": counts.index.tolist(),
                "default": str(counts.index[0]),
                "purpose": "fairness_audit_only",
            }

    for column in ID_COLUMNS:
        schema["id_fields"][column] = {
            "type": "identifier",
            "required": False,
            "purpose": "traceability_only",
        }
    return schema


def reference_values(data: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in data.columns:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            values[column] = float(series.median())
        else:
            values[column] = str(series.astype(str).mode().iloc[0])
    return values

