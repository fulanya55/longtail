import numpy as np

from experiments.run_route_a_stable_expert_visibility_calibration import (
    _add_c2_features,
    _fit_isotonic,
    _fit_uniform_platt,
    _split_three_by_combo,
)


def test_three_way_combo_split_is_disjoint_and_stratified():
    labels = np.asarray(
        [[1, 1, 0, 1, 0, 0]] * 6 + [[0, 1, 1, 0, 1, 0]] * 6,
        dtype=np.uint8,
    )
    selection = {"positive_base_indices": np.arange(12)}
    roles = _split_three_by_combo(selection, labels, 31)
    assert [len(role) for role in roles] == [4, 4, 4]
    assert not set(roles[0]).intersection(roles[1])
    assert not set(roles[0]).intersection(roles[2])
    assert not set(roles[1]).intersection(roles[2])
    for role in roles:
        assert len({tuple(np.delete(labels[index], 1)) for index in role}) == 2


def test_c2_visibility_features_have_expected_shape():
    group = {
        "expert_raw": np.asarray([0.0, 1.0], dtype=np.float32),
        "base_logits": np.asarray(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
            dtype=np.float32,
        ),
        "evidence_topk": np.asarray([1.0, 2.0], dtype=np.float32),
        "evidence_max": np.asarray([2.0, 3.0], dtype=np.float32),
        "stage2_abs_mean": np.asarray([0.4, 0.5], dtype=np.float32),
        "stage2_temporal_std": np.asarray([0.6, 0.7], dtype=np.float32),
    }
    result = _add_c2_features(group, 2.0, -1.0)
    assert result["visibility_features"].shape == (2, 10)
    assert np.allclose(
        result["c2"],
        0.2 * group["base_logits"][:, 1] + 0.8 * (2.0 * group["expert_raw"] - 1.0),
    )


def test_scalar_calibrators_return_valid_probabilities():
    groups = [
        (np.asarray([-2.0, -1.0, 1.0, 2.0]), np.asarray([0, 0, 1, 1])),
        (np.asarray([-3.0, -0.5, 0.5, 3.0]), np.asarray([0, 0, 1, 1])),
    ]
    coefficient, _ = _fit_uniform_platt(groups)
    isotonic = _fit_isotonic(groups)
    assert coefficient > 0.0
    values = isotonic.predict(np.asarray([-4.0, 0.0, 4.0]))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.all(np.diff(values) >= 0.0)
