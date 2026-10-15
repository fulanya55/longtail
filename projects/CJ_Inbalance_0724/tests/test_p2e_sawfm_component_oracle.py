import numpy as np
import pytest

from cj0724.p2e_sawfm_component_oracle import metrics, reconstruct_sawfm_component, reduction


def test_reconstruct_sawfm_component_contract():
    iq = reconstruct_sawfm_component(
        np.asarray([2.0e6, 5.0e6]),
        np.asarray([20.0e-6, 80.0e-6]),
        np.asarray([0.0, 10.0e-6]),
    )
    assert iq.shape == (2, 2, 3000)
    power = np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1)
    assert np.allclose(power, 1.0, atol=1.0e-5)


def test_metrics_and_reduction():
    target = np.asarray([1.0, 2.0])
    baseline = metrics(np.asarray([0.0, 3.0]), target)
    candidate = metrics(target, target)
    assert reduction(candidate, baseline) == pytest.approx(1.0)
