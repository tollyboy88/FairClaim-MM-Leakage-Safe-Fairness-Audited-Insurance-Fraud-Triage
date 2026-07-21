from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .decision import assign_decision
from .explain import local_sensitivity_explanation


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model", "model_features", "thresholds", "reference_values", "model_version"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def validate_input(frame: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    required = list(bundle["model_features"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return frame, [f"Missing required column: {column}" for column in missing]
    prepared = frame[required].copy()
    schema = bundle["input_schema"]["model_features"]
    errors: list[str] = []
    for column in required:
        if schema[column]["type"] == "number":
            converted = pd.to_numeric(prepared[column], errors="coerce")
            invalid = converted.isna() & prepared[column].notna()
            if invalid.any():
                errors.append(f"{column}: {int(invalid.sum())} value(s) are not numeric")
            prepared[column] = converted
        else:
            prepared[column] = prepared[column].astype(str)
    return prepared, errors


def score_frame(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    prepared, errors = validate_input(frame, bundle)
    if errors:
        raise ValueError("; ".join(errors))
    probabilities = bundle["model"].predict_proba(prepared)[:, 1]
    result = frame.copy()
    result["fraud_probability"] = probabilities
    result["decision"] = [assign_decision(value, bundle["thresholds"]) for value in probabilities]
    return result


def score_single(
    values: dict[str, Any], bundle: dict[str, Any], explain: bool = True
) -> dict[str, Any]:
    frame = pd.DataFrame([{column: values[column] for column in bundle["model_features"]}])
    prepared, errors = validate_input(frame, bundle)
    if errors:
        raise ValueError("; ".join(errors))
    probability = float(bundle["model"].predict_proba(prepared)[:, 1][0])
    output = {
        "fraud_probability": probability,
        "decision": assign_decision(probability, bundle["thresholds"]),
    }
    if explain:
        output["explanation"] = local_sensitivity_explanation(
            bundle["model"], prepared, bundle["reference_values"]
        )
    return output

