import numpy as np
import torch

from experiments.run_route_a_predicted_visibility_conditioned_fusion_v3 import (
    PredictedVisibilityConditionedFusion,
    _predicted_visibility_severity,
    _shuffle_by_role,
)


def test_predicted_visibility_severity_is_monotonic():
    gap = np.array([[0.0], [-0.25], [-0.50]], dtype=np.float32)
    severity = _predicted_visibility_severity(gap, -10.0, 2.5, 40.0).ravel()
    assert severity[0] < severity[1] < severity[2]
    assert np.isclose(severity[1], 0.5)


def test_conditioned_fusion_is_bounded_and_has_twelve_parameters():
    model = PredictedVisibilityConditionedFusion(0.5)
    base = torch.zeros(4, 6)
    presence = torch.full((4, 3), 100.0)
    severity = torch.full((4, 3), 100.0)
    logits, residual = model(base, presence, severity, return_residual=True)
    assert logits.shape == (4, 6)
    assert int(sum(parameter.numel() for parameter in model.parameters())) == 12
    assert residual[:, [0, 1, 2]].abs().sum() == 0
    assert residual[:, [3, 4, 5]].abs().max() <= 0.5


def test_role_shuffle_is_deterministic_and_preserves_rows():
    values = {"fit": np.arange(24, dtype=np.float32).reshape(8, 3)}
    first, receipt1 = _shuffle_by_role(values, 2039)
    second, receipt2 = _shuffle_by_role(values, 2039)
    assert np.array_equal(first["fit"], second["fit"])
    assert receipt1 == receipt2
    assert np.array_equal(np.sort(first["fit"], axis=0), np.sort(values["fit"], axis=0))

