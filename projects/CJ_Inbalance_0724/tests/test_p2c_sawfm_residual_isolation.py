import numpy as np
import pytest
import torch

from cj0724.p2c_sawfm_residual_isolation import (
    reduction,
    rms_normalize_batch,
    select_alpha,
    subset_metrics,
)


def test_rms_normalize_batch_has_unit_complex_power():
    iq = torch.randn(3, 2, 100)
    normalized = rms_normalize_batch(iq)
    power = normalized.square().sum(1).mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1.0e-5)


def test_select_alpha_uses_mae_then_smaller_alpha():
    assert select_alpha({0.25: 2.0, 0.5: 1.0, 0.75: 1.0}) == 0.5


def test_subset_metrics_and_reduction():
    target = np.asarray([1.0, 2.0])
    baseline_prediction = np.tile(np.asarray([[0.0], [3.0]]), (1, 5))
    candidate_prediction = np.tile(target[:, None], (1, 5))
    baseline = subset_metrics(baseline_prediction, target)
    candidate = subset_metrics(candidate_prediction, target)
    assert candidate["high_35_50"]["mae"] == 0.0
    assert reduction(candidate, baseline) == pytest.approx(1.0)
