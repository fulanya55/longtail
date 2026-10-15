import numpy as np

from cj0724.p3f_shared_isrj_router import choose_threshold, cp_upper


def test_clopper_pearson_upper_is_bounded():
    assert 0.0 < cp_upper(0, 100) < 0.1
    assert cp_upper(100, 100) == 1.0


def test_threshold_selection_contract():
    labels = np.tile(np.tile(np.array([0, 1], dtype=np.uint8), 20), 15)
    probability = np.tile(np.tile(np.array([0.01, 0.99]), 20), 15)
    regime = np.repeat(np.arange(5), 3 * 40)
    power = np.tile(np.repeat(np.array([25, 35, 50]), 40), 5)
    selected = choose_threshold(probability, labels, regime, power)
    assert selected["feasible"]
    assert 0.01 < selected["threshold"] <= 0.99
