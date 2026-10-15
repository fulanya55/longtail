import numpy as np
import torch

from experiments.run_route_b_c2_local_parameter_probe import (
    ScalarParameterProbe,
    _metrics,
    _quantile_split,
    temporal_local_distribution,
)


def test_local_distribution_has_expected_shape_and_preserves_gradients():
    temporal = torch.randn(3, 5, 20, requires_grad=True)
    features = temporal_local_distribution(temporal, topk=4, lags=(1, 4, 16))
    assert features.shape == (3, 35)
    features.sum().backward()
    assert temporal.grad is not None
    assert torch.isfinite(temporal.grad).all()


def test_local_distribution_is_time_shift_tolerant_for_summary_components():
    temporal = torch.randn(2, 4, 32)
    shifted = torch.roll(temporal, shifts=7, dims=-1)
    first = temporal_local_distribution(temporal, topk=8, lags=())
    second = temporal_local_distribution(shifted, topk=8, lags=())
    assert torch.allclose(first, second, atol=1.0e-6)


def test_quantile_split_is_disjoint_complete_and_deterministic():
    target = np.linspace(0.0, 1.0, 101)
    fit_a, validation_a = _quantile_split(target, seed=2041)
    fit_b, validation_b = _quantile_split(target, seed=2041)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(validation_a, validation_b)
    assert len(set(fit_a) & set(validation_a)) == 0
    assert sorted(np.concatenate([fit_a, validation_a]).tolist()) == list(range(101))


def test_probe_and_metrics_return_finite_values():
    probe = ScalarParameterProbe(input_dim=7, hidden_dim=8, dropout=0.0)
    output = probe(torch.randn(5, 7))
    assert output.shape == (5,)
    metrics = _metrics(
        prediction=np.asarray([1.0, 2.0, 3.0]),
        target=np.asarray([1.5, 2.5, 2.0]),
        training_median=2.0,
        scale=2.0,
    )
    assert metrics["count"] == 3
    assert np.isfinite(metrics["normalized_mae"])
