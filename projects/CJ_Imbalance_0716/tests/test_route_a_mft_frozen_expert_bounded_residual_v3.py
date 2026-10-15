import numpy as np

from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import (
    MFT_INDEX,
    _bounded_mft_residual,
)


def test_bounded_residual_changes_only_mft_and_respects_cap():
    logits = np.zeros((4, 6), dtype=np.float32)
    expert = np.asarray([-10.0, -1.0, 1.0, 10.0], dtype=np.float32)
    output, residual, uncertainty = _bounded_mft_residual(logits, expert, residual_cap=0.5)
    other = [index for index in range(6) if index != MFT_INDEX]
    np.testing.assert_array_equal(output[:, other], logits[:, other])
    assert np.max(np.abs(residual)) <= 0.5 + 1e-7
    np.testing.assert_allclose(uncertainty, 1.0)


def test_uncertainty_gate_protects_confident_logits():
    logits = np.zeros((3, 6), dtype=np.float32)
    logits[:, MFT_INDEX] = np.asarray([-20.0, 0.0, 20.0])
    _, residual, _ = _bounded_mft_residual(logits, np.ones(3, dtype=np.float32))
    assert abs(float(residual[0])) < 1e-3
    assert abs(float(residual[2])) < 1e-3
    assert float(residual[1]) > 0.3

