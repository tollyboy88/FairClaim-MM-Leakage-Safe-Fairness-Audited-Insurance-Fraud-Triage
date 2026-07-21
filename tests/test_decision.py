import numpy as np

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

