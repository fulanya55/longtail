import numpy as np

from experiments.run_route_a_h1j_vft_dechirped_composite import _matched_view


def test_matched_view_keeps_only_shared_other_label_contexts():
    labels = np.asarray([
        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ], dtype=np.uint8)
    weak = np.zeros_like(labels, dtype=bool); weak[[0, 2], 4] = True
    keep, y, receipt = _matched_view(labels, weak, evaluation=True)
    assert keep.sum() == 4
    assert y.sum() == 2
    assert receipt['matched_context_count'] == 2
