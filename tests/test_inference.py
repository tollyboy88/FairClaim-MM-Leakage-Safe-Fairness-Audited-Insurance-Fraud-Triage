import pandas as pd

from fairclaim.config import MODEL_BUNDLE_PATH
from fairclaim.inference import load_bundle, score_frame, score_single


def test_bundle_scores_single_and_batch():
    bundle = load_bundle(MODEL_BUNDLE_PATH)
    values = bundle["reference_values"].copy()
    single = score_single(values, bundle)
    assert 0 <= single["fraud_probability"] <= 1
    assert single["decision"] in {"approve", "defer", "investigate"}
    assert single["explanation"]

    batch = score_frame(pd.DataFrame([values, values]), bundle)
    assert len(batch) == 2
    assert batch["fraud_probability"].between(0, 1).all()


def test_missing_batch_column_is_reported():
    bundle = load_bundle(MODEL_BUNDLE_PATH)
    values = bundle["reference_values"].copy()
    values.pop(bundle["model_features"][0])
    try:
        score_frame(pd.DataFrame([values]), bundle)
    except ValueError as exc:
        assert "Missing required column" in str(exc)
    else:
        raise AssertionError("Missing required input column was accepted")

