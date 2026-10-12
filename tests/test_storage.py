from fairclaim.storage import load_history, save_prediction


def test_prediction_and_evidence_are_saved(tmp_path):
    database = tmp_path / "test.db"
    prediction_id = save_prediction(
        database,
        source="single",
        model_version="test-model",
        probability=0.42,
        decision="defer",
        inputs={"field": "value"},
        evidence=[("evidence.txt", "text/plain", b"example")],
    )
    history = load_history(database)
    assert len(history) == 1
    assert history.iloc[0]["prediction_id"] == prediction_id
    assert history.iloc[0]["decision"] == "defer"

