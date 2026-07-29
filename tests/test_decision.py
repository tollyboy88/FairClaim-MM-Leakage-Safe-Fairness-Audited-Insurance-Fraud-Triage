import numpy as np
from sklearn.metrics import fbeta_score

from fairclaim.decision import assign_decision, select_thresholds


def test_three_way_decision_boundaries():
    thresholds = {"approve_below": 0.2, "investigate_at_or_above": 0.7}
    assert assign_decision(0.1, thresholds) == "approve"
    assert assign_decision(0.2, thresholds) == "abstain"
    assert assign_decision(0.69, thresholds) == "abstain"
    assert assign_decision(0.7, thresholds) == "investigate"


def test_selected_thresholds_are_ordered():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.01, 0.03, 0.08, 0.2, 0.6, 0.9])
    thresholds = select_thresholds(y, p, target_npv=0.95)
    assert 0 <= thresholds.approve_below < thresholds.investigate_at_or_above <= 1


def test_vectorised_investigation_threshold_matches_f2_definition():
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.02, 0.71, 0.13, 0.42, 0.83, 0.18, 0.37, 0.04])
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), p])
    reference_scores = [
        fbeta_score(y, p >= threshold, beta=2, zero_division=0)
        for threshold in candidates
    ]
    expected = float(candidates[int(np.argmax(reference_scores))])
    actual = select_thresholds(y, p, target_npv=0.95)
    assert actual.investigate_at_or_above == expected
