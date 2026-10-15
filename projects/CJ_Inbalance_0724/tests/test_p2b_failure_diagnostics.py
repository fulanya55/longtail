import numpy as np
import pytest

from cj0724.p2b_failure_diagnostics import combination_name, group_metrics, relative


def test_combination_name_uses_global_class_order():
    assert combination_name(np.asarray([1, 1, 0, 0, 1, 0])) == "FMB+SawFM+VFT"


def test_group_metrics_uses_high_power_columns():
    target = np.asarray([1.0, 2.0])
    prediction = np.tile(target[:, None], (1, 9))
    metrics = group_metrics(prediction, target, np.asarray([0, 1]))
    assert metrics["mae_s"] == 0.0
    assert metrics["pearson"] == pytest.approx(1.0)


def test_relative_reduction():
    assert relative(0.8, 1.0) == 0.19999999999999996
