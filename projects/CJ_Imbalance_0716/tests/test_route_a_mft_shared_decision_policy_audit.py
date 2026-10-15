import numpy as np

from experiments.run_route_a_mft_shared_decision_policy_audit import MFT_INDEX, hybrid_policy


def _calibration():
    return {"parameters": [{"coefficient": 1.0, "intercept": 0.0} for _ in range(6)]}


def test_hybrid_policy_uses_q1_mft_and_q4_other_labels():
    logits = np.asarray([[0.0, 1.0, -1.0, -0.5, 0.3, 0.7]], dtype=np.float32)
    q4 = _calibration()
    q1 = _calibration()
    q1["parameters"][MFT_INDEX] = {"coefficient": 2.0, "intercept": -0.2}
    q4_thresholds = np.full(6, 0.5, dtype=np.float32)
    q1_thresholds = np.full(6, 0.4, dtype=np.float32)
    probability, prediction, thresholds = hybrid_policy(logits, q4, q4_thresholds, q1, q1_thresholds)
    expected_q4 = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(probability[:, [0, 1, 2, 4, 5]], expected_q4[:, [0, 1, 2, 4, 5]])
    assert np.isclose(probability[0, MFT_INDEX], 1.0 / (1.0 + np.exp(1.2)))
    assert thresholds[MFT_INDEX] == np.float32(0.4)
    assert prediction.shape == logits.shape


def test_shared_monotone_policy_cannot_add_positive_after_negative_residual():
    base = np.linspace(-2.0, 2.0, 101, dtype=np.float32)
    residual = -np.linspace(0.0, 0.75, 101, dtype=np.float32)
    q1 = _calibration()
    q4 = _calibration()
    thresholds = np.full(6, 0.5, dtype=np.float32)
    base_logits = np.zeros((101, 6), dtype=np.float32)
    guard_logits = base_logits.copy()
    base_logits[:, MFT_INDEX] = base
    guard_logits[:, MFT_INDEX] = base + residual
    _, base_prediction, _ = hybrid_policy(base_logits, q4, thresholds, q1, thresholds)
    _, guard_prediction, _ = hybrid_policy(guard_logits, q4, thresholds, q1, thresholds)
    assert not np.any((~base_prediction[:, MFT_INDEX]) & guard_prediction[:, MFT_INDEX])
