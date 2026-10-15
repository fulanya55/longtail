import numpy as np

from experiments.analyze_route_a_fullalpha_periodicity_v7_stratified import _groups, _positive_bin_mask


def test_positive_bin_keeps_all_negatives_and_selected_positives():
    target = np.array([0, 1, 1, 0], dtype=np.uint8)
    inside = np.array([False, True, False, False])
    assert _positive_bin_mask(target, inside).tolist() == [True, True, False, True]


def test_strata_include_required_axes():
    labels = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]], dtype=np.uint8)
    powers = np.zeros((2, 6), dtype=np.float32)
    values = np.zeros((2, 79), dtype=np.float32)
    mask = np.ones((2, 79), dtype=bool)
    receiver = np.array([10, 11])
    groups = _groups(labels, 3, powers, values, mask, receiver)
    assert "relative_gap_db/le_m15" in groups
    assert "jnr_db/10_15" in groups
    assert "num_targets/3_4" in groups
    assert "composition_order/1" in groups
    assert "co_label/SawFM_present" in groups
    assert "receiver_cohort/0" in groups
