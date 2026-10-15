import numpy as np

from cj0724.p2e_sawfm_component_oracle import reconstruct_sawfm_component
from cj0724.p2m_analytic_saw_period import analytic_period_estimate


def test_analytic_period_estimator_recovers_exact_components():
    target = np.asarray([8e-6, 15e-6, 50e-6, 90e-6, 130e-6])
    iq = reconstruct_sawfm_component(
        np.full(len(target), 3.0e6), target, np.full(len(target), 20e-6),
    )
    prediction = analytic_period_estimate(iq)["prediction"]
    assert np.max(np.abs(prediction - target)) <= 0.5e-6
