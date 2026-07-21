from __future__ import annotations

from typing import Any

import pandas as pd


def local_sensitivity_explanation(
    model: Any,
    row: pd.DataFrame,
    reference_values: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Explain a score using one-feature-at-a-time reference substitutions.

    This is a model-sensitivity explanation, not a causal explanation.
    """
    probability = float(model.predict_proba(row)[:, 1][0])
    contributions: list[dict[str, Any]] = []
    for column in row.columns:
        comparison = row.copy()
        comparison.loc[comparison.index[0], column] = reference_values[column]
        reference_probability = float(model.predict_proba(comparison)[:, 1][0])
        contributions.append(
            {
                "feature": column,
                "value": row.iloc[0][column],
                "reference": reference_values[column],
                "score_change": probability - reference_probability,
                "direction": "raises risk" if probability > reference_probability else "lowers risk",
            }
        )
    contributions.sort(key=lambda item: abs(float(item["score_change"])), reverse=True)
    return contributions[:limit]

