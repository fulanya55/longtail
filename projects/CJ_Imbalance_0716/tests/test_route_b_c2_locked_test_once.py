import numpy as np
import pytest

from experiments.run_route_b_c2_locked_test_once import (
    _predict_serialized_router,
    _stable_sigmoid,
    _validate_freeze,
)


def test_stable_sigmoid_handles_large_values():
    actual = _stable_sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(actual).all()
    assert actual[0] == 0.0
    assert actual[1] == 0.5
    assert actual[2] == 1.0


def test_serialized_router_matches_manual_standardization():
    state = {
        "type": "standardized_logistic_regression",
        "feature_names": ["a", "b"],
        "scaler_mean": [1.0, 2.0],
        "scaler_scale": [2.0, 4.0],
        "coefficient": [0.5, -1.0],
        "intercept": 0.25,
    }
    features = np.asarray([[3.0, 6.0], [1.0, 2.0]])
    expected_logits = np.asarray([-0.25, 0.25])
    np.testing.assert_allclose(
        _predict_serialized_router(state, features),
        _stable_sigmoid(expected_logits),
    )


def test_serialized_constant_router():
    state = {
        "type": "constant",
        "constant_probability": 0.3,
        "feature_names": ["a"],
    }
    np.testing.assert_allclose(
        _predict_serialized_router(state, np.zeros((3, 1))),
        np.full(3, 0.3),
    )


def test_freeze_rejects_scope_or_seed_drift():
    payload = {
        "status": "locked_test_configuration_frozen",
        "locked_test_authorized": True,
        "protocol": {
            "scope": "closedset_6class_test.mat only",
            "fit_or_selection_on_locked_test": False,
            "identity_consensus_beta": 0,
            "predicted_context_safety": False,
        },
        "seeds": [{"seed": seed} for seed in range(2041, 2047)],
        "implementation_sources": [{"path": "runner.py", "sha256": "0" * 64}],
    }
    _validate_freeze(payload)
    payload["protocol"]["scope"] = "power sweep"
    with pytest.raises(ValueError, match="scope"):
        _validate_freeze(payload)
