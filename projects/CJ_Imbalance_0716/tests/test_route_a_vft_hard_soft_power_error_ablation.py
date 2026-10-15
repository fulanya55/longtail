import numpy as np

from experiments.run_route_a_vft_hard_soft_power_error_ablation import relative_component_power, soft_visibility_weight


def test_relative_power_and_soft_visibility_follow_10db_boundary():
    labels = np.array([[1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    power = np.array([[10.0, 20.0, np.nan, np.nan, np.nan, np.nan]])
    relative = relative_component_power(labels, power)
    assert np.isclose(relative[0, 0], -10.0)
    assert np.isclose(relative[0, 1], 10.0)
    assert np.isclose(soft_visibility_weight(np.array([-10.0]))[0], 0.5)
    assert soft_visibility_weight(np.array([-20.0]))[0] > 0.9
    assert soft_visibility_weight(np.array([0.0]))[0] < 0.05
