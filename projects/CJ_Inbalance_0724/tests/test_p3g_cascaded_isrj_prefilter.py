import numpy as np

from cj0724.p3g_cascaded_isrj_prefilter import select_gate_threshold


def test_gate_threshold_minimizes_coverage_with_recall_constraint():
    changed = np.zeros(600, dtype=bool); changed[::10] = True
    score = np.where(changed, 0.95, 0.05)
    regime = np.repeat(np.arange(5), 120)
    power = np.tile(np.repeat(np.array([25, 35, 50]), 40), 5)
    selected = select_gate_threshold(score, changed, regime, power)
    assert selected["feasible"]
    assert selected["changed_recall"] >= 0.98
    assert selected["coverage"] <= 0.11
