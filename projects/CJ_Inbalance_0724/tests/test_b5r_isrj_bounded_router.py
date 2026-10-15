import numpy as np

from cj0724.b5r_isrj_bounded_router import (
    action_targets,
    isrj_reliability_features,
    route_variant,
    two_state_route,
)


def test_isrj_features_exclude_explicit_power_and_support_ablations():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(6, 310)).astype(np.float32)
    median = np.median(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0), 1e-3)
    score = np.linspace(0.1, 0.9, 6)
    full = isrj_reliability_features(score, score[::-1], score, matrix, median, scale)
    no_physics = isrj_reliability_features(score, score[::-1], score, matrix, median, scale, include_physics=False)
    no_e5 = isrj_reliability_features(score, score[::-1], score, matrix, median, scale, include_e5=False)
    assert full.shape[1] > no_physics.shape[1]
    assert full.shape[1] > no_e5.shape[1]
    assert np.isfinite(full).all()


def test_action_targets_use_positive_rescue_negative_protection_and_unknown():
    labels = np.asarray([1, 0, 1, 0])
    reference = np.asarray([0.2, 0.8, 0.7, 0.2])
    candidate = np.asarray([0.8, 0.1, 0.3, 0.7])
    assert action_targets(labels, reference, candidate).tolist() == [1, 2, 0, 0]


def test_bounded_two_and_three_state_outputs_stay_in_expert_interval():
    reference = np.asarray([0.2, 0.8, 0.4])
    candidate = np.asarray([0.9, 0.1, 0.6])
    actions = np.asarray([[0.0, 0.9, 0.1], [0.0, 0.1, 0.9], [0.8, 0.1, 0.1]])
    two = two_state_route(reference, candidate, actions, 1.0)
    three = route_variant("bounded_three_state", reference, candidate, actions, 1.0, 0.5)
    for output in (two, three):
        assert np.all(output >= np.minimum(reference, candidate))
        assert np.all(output <= np.maximum(reference, candidate))
    assert three[2] == reference[2]
