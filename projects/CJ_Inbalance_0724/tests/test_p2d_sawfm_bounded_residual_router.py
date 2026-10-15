import numpy as np
import pytest

from cj0724.p2d_sawfm_bounded_residual_router import (
    relative_reduction,
    row_metrics,
    select_threshold,
)


def test_row_metrics_and_reduction():
    target = np.asarray([1.0, 2.0])
    baseline = row_metrics(np.asarray([0.0, 3.0]), target)
    candidate = row_metrics(target, target)
    assert relative_reduction(candidate, baseline) == pytest.approx(1.0)


def test_threshold_selection_can_choose_beneficial_half():
    predicted_gain = np.asarray([1.0, 0.9, -0.2, -0.3])
    raw = np.asarray([0.0, 0.0, 0.0, 0.0])
    residual = np.asarray([1.0, 1.0, -1.0, -1.0])
    target = np.asarray([1.0, 1.0, 0.5, 0.5])
    no_variable = np.asarray([False, False, True, True])
    selected = select_threshold(predicted_gain, raw, residual, target, no_variable)
    assert selected["eligible"]
    assert selected["coverage"] == pytest.approx(0.5)
    assert selected["overall_reduction"] == pytest.approx(2.0 / 3.0)
