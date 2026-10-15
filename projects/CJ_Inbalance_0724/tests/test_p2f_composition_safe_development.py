import numpy as np

from cj0724.p2f_composition_safe_development import (
    clopper_pearson_upper,
    gated_prediction,
    select_fpr_buffered_threshold,
    split_identity_roles,
)


def test_identity_roles_are_disjoint_and_complete():
    identities = np.repeat(np.arange(20), 3)
    roles = split_identity_roles(identities, seed=7)
    combined = np.concatenate(list(roles.values()))
    assert len(np.unique(combined)) == 20
    assert sum(len(values) for values in roles.values()) == 20


def test_fpr_buffered_threshold_controls_every_condition():
    labels = np.zeros((8, 6), dtype=bool)
    labels[[0, 1, 4, 5], 1] = True
    probability = np.zeros((8, 6), dtype=np.float32)
    probability[:, 1] = [0.9, 0.8, 0.7, 0.2, 0.85, 0.75, 0.6, 0.1]
    condition = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = select_fpr_buffered_threshold(
        labels, probability, condition, 1, (0, 1), np.ones(8, dtype=bool), 0.8,
    )
    predicted = probability[:, 1] >= result["threshold"]
    assert not predicted[[2, 3, 6, 7]].any()


def test_clopper_pearson_upper_exceeds_empirical_rate():
    assert clopper_pearson_upper(5, 100, 0.95) > 0.05


def test_complexity_gate_falls_back_for_crowded_prediction():
    saved = {
        "base_prediction": np.zeros((2, 6, 2), dtype=np.float32),
        "joint_prediction": np.ones((2, 6, 2), dtype=np.float32),
    }
    probability = np.array([
        [0.9, 0.9, 0.1, 0.1, 0.9, 0.1],
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.1],
    ], dtype=np.float32)
    prediction, _ = gated_prediction(saved, probability, np.full(6, 0.5), 3, 3)
    assert prediction[0, 1, 1] == 1.0
    assert prediction[1, 1, 1] == 0.0


def test_disagreement_gate_rejects_unreliable_specialist():
    saved = {
        "base_prediction": np.zeros((1, 6, 2), dtype=np.float32),
        "joint_prediction": np.ones((1, 6, 2), dtype=np.float32),
    }
    probability = np.zeros((1, 6), dtype=np.float32)
    probability[0, 1] = 0.9
    prediction, _ = gated_prediction(
        saved, probability, np.full(6, 0.5), 3, 0,
        saw_max_disagreement=0.5,
    )
    assert prediction[0, 1, 1] == 0.0


def test_pattern_gate_only_enables_whitelisted_prediction():
    saved = {
        "base_prediction": np.zeros((2, 6, 2), dtype=np.float32),
        "joint_prediction": np.ones((2, 6, 2), dtype=np.float32),
    }
    probability = np.zeros((2, 6), dtype=np.float32)
    probability[:, 1] = 0.9
    probability[1, 4] = 0.9
    prediction, _ = gated_prediction(
        saved, probability, np.full(6, 0.5), 3, 0,
        saw_allowed_patterns=[2],
    )
    assert prediction[0, 1, 1] == 1.0
    assert prediction[1, 1, 1] == 0.0
