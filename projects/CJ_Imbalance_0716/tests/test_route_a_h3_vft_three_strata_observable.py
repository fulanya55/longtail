import numpy as np

from experiments.run_route_a_h3_vft_three_strata_observable import _binary_metrics, _triplets


def test_triplets_sort_by_arm_and_keep_identity_alignment():
    true = np.array([0.1, 0.4, 0.8, 0.2, 0.5, 0.9])
    permuted = np.array([0.1, 0.3, 0.6, 0.2, 0.4, 0.7])
    pair = np.array([11, 11, 11, 12, 12, 12], dtype=np.uint64)
    arm = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)
    values = _triplets(true, permuted, pair, arm)
    assert np.array_equal(values["absent_true"], [0.1, 0.2])
    assert np.array_equal(values["weak_true"], [0.4, 0.5])
    assert np.array_equal(values["strong_permuted"], [0.6, 0.7])


def test_binary_metrics_reports_real_control_difference():
    metrics = _binary_metrics(np.array([0.1, 0.2]), np.array([0.7, 0.8]), np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    assert metrics["true_auroc"] == 1.0
    assert metrics["true_minus_permuted_auroc"] == 0.0
    assert metrics["paired_present_gt_absent_fraction"] == 1.0
