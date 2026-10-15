import numpy as np
import torch

from experiments.run_route_a_vtse_soft_strong_anchor_v3 import (
    _continuous_visibility_loss,
    _continuous_visibility_severity,
)


def test_visibility_severity_is_monotonic_and_excludes_singletons_and_negatives():
    labels = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    power = np.array([
        [-20, 0, np.nan, np.nan, np.nan, np.nan],
        [-5, 0, np.nan, np.nan, np.nan, np.nan],
        [0, np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    severity = _continuous_visibility_severity(labels, power, -10.0, 2.5)
    assert severity[0, 0] > severity[1, 0]
    assert severity[0, 0] > 0.95
    assert severity[1, 0] < 0.15
    assert severity[2].sum() == 0
    assert np.all(severity[labels == 0] == 0)


def test_continuous_visibility_loss_is_finite():
    logits = torch.zeros(4, 3, requires_grad=True)
    targets = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    severity = torch.tensor([[1, 0, .5], [.2, .8, 0], [0, .1, 0], [0, 0, .7]], dtype=torch.float32)
    loss = _continuous_visibility_loss(logits, targets, severity, torch.ones(3), 2.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
