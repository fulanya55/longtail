import numpy as np

from experiments.analyze_route_a_isrj_f3_stratified_robustness_v3 import (
    _cohen_dz,
    _positive_bin_mask,
    _subset_metrics,
)


def test_positive_bin_mask_keeps_all_negatives_and_selected_positives():
    target = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    positive_bin = np.asarray([True, False, False, False])
    assert _positive_bin_mask(target, positive_bin).tolist() == [True, False, True, True]


def test_subset_metrics_counts_weak_and_strong_positives():
    target = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    weak = np.asarray([True, False, False, False])
    probability = np.asarray([0.9, 0.8, 0.2, 0.1])
    prediction = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    result = _subset_metrics(target, weak, probability, prediction, np.ones(4, dtype=bool),
                             min_positive=1, min_negative=1, min_weak_positive=1)
    assert result["counts"]["weak_positive"] == 1
    assert result["counts"]["strong_positive"] == 1
    assert result["weak_recall"] == 1.0
    assert result["negative_fpr"] == 0.0
    assert result["weak_auroc"] == 1.0


def test_cohen_dz_is_positive_for_consistently_positive_differences():
    assert _cohen_dz([0.1, 0.2, 0.15, 0.12, 0.18]) > 1.0

