import numpy as np
import pytest

from experiments.run_route_a_v18_bounded_predicted_width_residual import _select_c


def test_fit_grid_selects_smallest_safe_maximum_fpr_reduction():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[:2, 5] = 1
    labels[2:, 4] = 1
    weak = np.zeros_like(labels, dtype=bool)
    labels[0, 1] = 1
    weak[0, 1] = True
    weak[2:, 4] = True
    logits = np.zeros((4, 6), dtype=np.float32)
    logits[:, 4] = [.2, .3, .8, .8]
    calibration = {"parameters": [{"coefficient": 1., "intercept": 0.} for _ in range(6)]}
    evidence = np.array([.5, .5, 0., 0.])
    selected, baseline, grid = _select_c(labels, weak, logits, calibration, np.full(6, .5), evidence, np.arange(4))
    assert baseline["hard_fpr"] == pytest.approx(1.0)
    # At c=.6, the second hard negative is exactly at the >= threshold;
    # c=.7 is therefore the first grid point that suppresses both.
    assert selected["c"] == pytest.approx(.7)
    assert selected["fpr_reduction"] == pytest.approx(1.0)
    assert selected["positive_recall_loss"] == pytest.approx(0.0)
    assert len(grid) == 11
