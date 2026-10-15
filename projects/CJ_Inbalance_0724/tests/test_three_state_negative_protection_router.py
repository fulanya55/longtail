import numpy as np

from cj0724.three_state_negative_protection_router import (
    ABSTAIN,
    ACCEPT_EXPERT,
    KEEP_MAIN,
    build_three_state_targets,
    fit_three_state_routers,
    predict_three_state,
    three_state_blend,
)


def test_three_state_targets_separate_advantage_and_hard_negative():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[0, 3] = 1
    main = np.full((4, 6), 0.1, dtype=np.float32)
    expert = np.full((4, 3), 0.1, dtype=np.float32)
    main[0, 3], expert[0, 0] = 0.2, 0.9
    main[1, 3], expert[1, 0] = 0.1, 0.8
    main[2, 3], expert[2, 0] = 0.8, 0.1
    state, weight, audit = build_three_state_targets(main, expert, labels, 0, 0.5)
    assert state[0] == ACCEPT_EXPERT
    assert state[1] == ABSTAIN
    assert state[2] == ACCEPT_EXPERT
    assert state[3] == KEEP_MAIN
    assert audit["hard_negative"] == 1
    assert weight[1] >= 2.0


def test_three_state_blend_vetoes_abstain_and_preserves_non_targets():
    main = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
    expert = np.asarray([[0.9, 0.1, 0.8]], dtype=np.float32)
    states = np.zeros((1, 3, 3), dtype=np.float32)
    states[0, 0] = [0.1, 0.8, 0.1]
    states[0, 1] = [0.1, 0.6, 0.5]
    states[0, 2] = [0.1, 0.7, 0.2]
    final, accept = three_state_blend(main, expert, states, 0.5, 0.6, 0.2)
    np.testing.assert_array_equal(final[:, :3], main[:, :3])
    np.testing.assert_array_equal(accept, [[True, False, True]])
    np.testing.assert_allclose(final[:, 3:], [[0.65, 0.5, 0.7]])


def test_three_state_router_outputs_normalized_probabilities():
    rng = np.random.default_rng(13)
    labels = rng.integers(0, 2, size=(120, 6), dtype=np.uint8)
    main = rng.uniform(0.05, 0.95, size=(120, 6)).astype(np.float32)
    expert = rng.uniform(0.05, 0.95, size=(120, 3)).astype(np.float32)
    thresholds = np.full(6, 0.5, dtype=np.float32)
    routers, audits = fit_three_state_routers(main, expert, labels, thresholds, 2041)
    probability = predict_three_state(routers, main, expert, thresholds)
    assert probability.shape == (120, 3, 3)
    np.testing.assert_allclose(probability.sum(axis=2), 1.0, atol=1e-6)
    assert np.isfinite(probability).all()
    assert len(audits) == 3
