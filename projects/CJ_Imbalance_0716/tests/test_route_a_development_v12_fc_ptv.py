from pathlib import Path

import numpy as np
import pytest

from experiments.route_a_fc_ptv_core import (
    exact_functional_distances,
    functional_gram,
    linear_distances,
    robust_logit_scales,
    simultaneous_lower_bounds,
    solve_two_block_projection,
)
from experiments.run_route_a_development_v12_fc_ptv import (
    _safe_mean_ci,
    _validate_config,
)


def test_robust_scales_are_fit_stratum_specific():
    logits = np.asarray([
        [0.0, 0.0],
        [2.0, 4.0],
        [10.0, 20.0],
        [14.0, 28.0],
    ])
    payload = robust_logit_scales(logits, [False, False, True, True])
    np.testing.assert_allclose(payload["medians"], [[1.0, 2.0], [12.0, 24.0]])
    np.testing.assert_allclose(payload["scales"], [[1.4826, 2.9652], [2.9652, 5.9304]])
    np.testing.assert_array_equal(payload["counts"], [2, 2])


def test_robust_scales_reject_degenerate_anchor():
    with pytest.raises(ValueError, match="degenerate"):
        robust_logit_scales(np.zeros((4, 2)), [False, False, True, True])


def test_functional_gram_and_exact_distance_match_known_linear_case():
    base = np.zeros((4, 1))
    shared = np.asarray([[1.0], [1.0], [2.0], [2.0]])
    classifier = np.asarray([[0.0], [0.0], [1.0], [1.0]])
    saw = np.asarray([False, False, True, True])
    scales = np.ones((2, 1))
    gram = functional_gram(base, shared, classifier, saw, scales)
    np.testing.assert_allclose(gram[0, 0], [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(gram[1, 0], [[4.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(linear_distances([0.5, 0.5], gram)[:, 0], [0.5, 1.5])
    candidate = 0.5 * shared + 0.5 * classifier
    np.testing.assert_allclose(exact_functional_distances(base, candidate, saw, scales)[:, 0], [0.5, 1.5])


def test_projection_preserves_feasible_q0():
    gram = np.zeros((2, 5, 2, 2), dtype=np.float64)
    q, diagnostics = solve_two_block_projection(gram)
    np.testing.assert_allclose(q, [1.5, 1.5], atol=1e-8)
    assert diagnostics["max_constraint_violation"] <= 1e-7


def test_projection_enforces_joint_functional_constraint_and_order():
    gram = np.repeat(np.eye(2)[None, None, :, :], 10, axis=0).reshape(2, 5, 2, 2)
    q, diagnostics = solve_two_block_projection(gram)
    assert 0.0 <= q[0] <= q[1] <= 1.5
    assert np.max(linear_distances(q, gram)) <= 1.0 + 1e-7
    np.testing.assert_allclose(q, [2.0 ** -0.5, 2.0 ** -0.5], atol=1e-7)
    assert diagnostics["stationarity_residual"] <= 1e-5


def test_simultaneous_bounds_preserve_three_labels():
    deltas = np.asarray([
        [0.03, 0.02, 0.01],
        [0.04, 0.03, 0.02],
        [0.02, 0.01, 0.01],
        [0.05, 0.04, 0.03],
        [0.03, 0.02, 0.02],
    ])
    result = simultaneous_lower_bounds(deltas, bootstrap_samples=1000, seed=7)
    assert len(result["familywise_95_lower"]) == 3
    assert all(np.isfinite(result["familywise_95_lower"]))


def test_empty_metric_summary_is_an_explicit_invalid_value():
    result = _safe_mean_ci([], 7)
    assert result["count"] == 0
    assert np.isnan(result["mean"])


def _valid_config(tmp_path):
    return {
        "allow_locked_test": False,
        "evidence_role": "development_v12",
        "data_root": str(tmp_path / "development_v12"),
        "train_file": "closedset_6class_train.mat",
        "test_v12_path": str(tmp_path / "development_v12" / "closedset_6class_test.mat"),
        "test_v12_generated": False,
        "test_v12_access_authorized": False,
        "test_v12_used": False,
        "sealed_test_v3_path": str(tmp_path / "sealed_v3" / "closedset_6class_test.mat"),
        "test_v3_sealed": True,
        "test_v3_used": False,
        "forbidden_test_v2_path": str(tmp_path / "consumed_v2" / "closedset_6class_test.mat"),
        "test_v2_consumed": True,
        "development_v11_reuse_for_selection": False,
    }


def test_v12_config_accepts_only_fresh_development_role(tmp_path):
    config = _valid_config(tmp_path)
    _validate_config(config)
    config["development_v11_reuse_for_selection"] = True
    with pytest.raises(ValueError, match="reuse"):
        _validate_config(config)


def test_v12_config_refuses_test_named_input(tmp_path):
    config = _valid_config(tmp_path)
    config["train_file"] = "closedset_6class_test.mat"
    with pytest.raises(ValueError, match="test role"):
        _validate_config(config)
