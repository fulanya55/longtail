import numpy as np

from cj0724.p1f_isrj_confidence_bound_router import (
    apply_route,
    grouped_bootstrap_rows,
    pooled_isrj_threshold,
    stratified_train_roles,
)


def test_stratified_train_roles_are_balanced_and_disjoint():
    combinations = np.array([[int(bit) for bit in f"{value:06b}"] for value in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(combinations, 40, axis=0)
    roles = stratified_train_roles(labels, 17)
    assert {name: len(values) for name, values in roles.items()} == {"fit": 1260, "selection": 630, "calibration": 630}
    assert not np.intersect1d(roles["fit"], roles["selection"]).size
    assert not np.intersect1d(roles["fit"], roles["calibration"]).size
    for values, count in ((roles["fit"], 20), (roles["selection"], 10), (roles["calibration"], 10)):
        _, counts = np.unique(labels[values], axis=0, return_counts=True)
        assert np.all(counts == count)


def test_grouped_bootstrap_keeps_both_power_rows_together():
    sample_indices = np.array([0, 1, 2, 0, 1, 2])
    rows = grouped_bootstrap_rows(sample_indices, np.random.default_rng(3))
    sampled = sample_indices[rows]
    _, counts = np.unique(sampled, return_counts=True)
    assert np.all(counts % 2 == 0)
    assert len(rows) == len(sample_indices)


def test_apply_route_changes_only_isrj_probability():
    payload = {
        "main": np.full((3, 6), 0.2, dtype=np.float32),
        "expert": np.full((3, 3), 0.8, dtype=np.float32),
    }
    final, accept = apply_route(payload, np.array([2.0, 0.0, -1.0]), {"score_cutoff": 1.0})
    assert accept.tolist() == [True, False, False]
    assert np.allclose(final[:, :5], payload["main"][:, :5])
    assert final[0, 5] > payload["main"][0, 5]


def test_pooled_threshold_caps_only_the_aggregate_negative_rate():
    labels = np.zeros((24, 6), dtype=np.uint8)
    labels[[10, 11, 22, 23], 5] = 1
    probabilities = np.zeros((24, 6), dtype=np.float32)
    probabilities[:, 5] = np.array([
        0.90, 0.80, *([0.10] * 8), 0.95, 0.94,
        *([0.10] * 10), 0.95, 0.94,
    ])
    payload = {"labels": labels, "power": np.repeat([20, 25], 12)}
    result = pooled_isrj_threshold(payload, probabilities)
    predicted = probabilities[:, 5] >= result["threshold"]
    pooled_fpr = predicted[labels[:, 5] == 0].mean()
    assert pooled_fpr <= 0.10 + 1e-12
    assert len(result["by_power"]) == 2
