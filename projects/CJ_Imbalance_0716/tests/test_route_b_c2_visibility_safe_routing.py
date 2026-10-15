import numpy as np

from experiments.run_route_b_c2_visibility_safe_routing import (
    _apply_identity_consensus,
    _gate_features,
    _identity_split,
    _predicted_context_keys,
    _select_threshold,
)


def test_identity_split_is_disjoint_and_complete():
    indices = np.arange(20, dtype=np.int64)
    identities = np.arange(100, 120, dtype=np.int64)
    fit, selection = _identity_split(indices, identities, 7)
    assert set(fit).isdisjoint(set(selection))
    assert sorted(np.concatenate([fit, selection]).tolist()) == indices.tolist()


def test_select_threshold_falls_back_when_local_is_always_worse():
    payload = {
        "base_indices": np.arange(4, dtype=np.int64),
        "identities": np.arange(100, 104, dtype=np.int64),
        "c2_predictions": np.zeros((4, 2), dtype=np.float32),
        "local_predictions": np.ones((4, 2), dtype=np.float32),
    }
    metadata = {
        "class_index": 0,
        "column_index": 0,
        "scale": 1.0,
        "labels": np.ones((4, 6), dtype=np.uint8),
        "values": np.zeros((4, 79), dtype=np.float32),
        "mask": np.ones((4, 79), dtype=bool),
    }
    selected, _ = _select_threshold(
        {10: payload}, {10: np.ones(4, dtype=np.float64)}, 0, metadata
    )
    assert selected["threshold"] > 1.0
    assert selected["route_coverage"] == 0.0
    assert selected["routed_relative_reduction"] == 0.0


def test_multilabel_context_gate_adds_label_context_features():
    payload = {
        "c2_predictions": np.zeros((3, 2), dtype=np.float32),
        "local_predictions": np.ones((3, 2), dtype=np.float32),
        "local_z_rms": np.ones((3, 2), dtype=np.float32),
        "local_z_max": np.ones((3, 2), dtype=np.float32),
        "class_probabilities": np.full((3, 6), 0.25, dtype=np.float32),
        "raw_rms": np.ones(3, dtype=np.float32),
    }
    base = _gate_features(payload, 0, 0, 0.0, 1.0, "G0_ModelEvidence")
    context = _gate_features(payload, 0, 0, 0.0, 1.0, "G2_MultiLabelContext")
    context_rms = _gate_features(
        payload, 0, 0, 0.0, 1.0, "G3_MultiLabelContextRMS"
    )
    assert base.shape == (3, 6)
    assert context.shape == (3, 17)
    assert context_rms.shape == (3, 18)


def test_identity_consensus_shrinks_probabilities_to_identity_median():
    payloads = {
        10: {"identities": np.asarray([1, 2])},
        20: {"identities": np.asarray([1, 2])},
        30: {"identities": np.asarray([1, 2])},
    }
    probabilities = {
        10: np.asarray([0.1, 0.8]),
        20: np.asarray([0.9, 0.4]),
        30: np.asarray([0.5, 0.6]),
    }
    unchanged = _apply_identity_consensus(payloads, probabilities, 0.0)
    consensus = _apply_identity_consensus(payloads, probabilities, 1.0)
    assert np.allclose(unchanged[10], probabilities[10])
    assert np.allclose(consensus[10], [0.5, 0.6])
    assert np.allclose(consensus[20], [0.5, 0.6])


def test_predicted_context_keys_force_target_present_and_exclude_it():
    payload = {
        "class_probabilities": np.asarray(
            [[0.1, 0.9, 0.2, 0.7, 0.1, 0.8]], dtype=np.float32
        )
    }
    assert _predicted_context_keys(payload, 0).tolist() == ["10101"]
