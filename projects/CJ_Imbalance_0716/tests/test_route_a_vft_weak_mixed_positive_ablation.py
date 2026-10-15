import numpy as np

from experiments.run_route_a_vft_weak_mixed_positive_ablation import _all_positive_matched_view


def test_mixed_view_keeps_only_compositions_available_on_both_sides():
    targets = np.array([
        [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)
    selected, labels, receipt = _all_positive_matched_view(targets, target_index=4)
    # Only the [1, 0, 0, 0, 0] co-label composition occurs on both sides.
    assert selected.tolist() == [True, True, False, False, True]
    assert labels.tolist() == [1, 1, 0]
    assert receipt["positive_count"] == 2
    assert receipt["matched_negative_count"] == 1
