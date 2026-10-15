import numpy as np

from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features, robust_reference


def test_bounded_router_stays_between_e5_and_physics():
    e5 = np.asarray([0.2, 0.8, 0.4])
    physics = np.asarray([0.9, 0.1, 0.6])
    actions = np.asarray([[0.0, 0.9, 0.1], [0.0, 0.1, 0.9], [0.8, 0.1, 0.1]])
    output = bounded_three_state(e5, physics, actions, alpha=1.0, confidence=0.5)
    assert np.all(output >= np.minimum(e5, physics))
    assert np.all(output <= np.maximum(e5, physics))
    assert output[2] == e5[2]


def test_reliability_features_have_no_power_column():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(5, 320)).astype(np.float32)
    median, scale = robust_reference(matrix)
    output = reliability_features(
        np.linspace(0.1, 0.9, 5), np.linspace(0.2, 0.8, 5),
        np.linspace(0.3, 0.7, 5), matrix, median, scale,
    )
    assert output.shape == (5, 13)
    assert np.isfinite(output).all()
