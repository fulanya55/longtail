import numpy as np

from experiments.run_route_a_mft_predicted_context_gate_v3 import _context_gate, _context_residual


def test_context_gate_rewards_sawfm_and_suppresses_vft_isrj():
    probabilities = np.zeros((2, 6), dtype=np.float32)
    probabilities[0, [1,4,5]] = [0.9, 0.1, 0.1]
    probabilities[1, [1,4,5]] = [0.1, 0.9, 0.9]
    gate = _context_gate(probabilities)
    assert gate[0] > gate[1]
    assert np.all((gate >= 0) & (gate <= 1))


def test_context_residual_is_no_larger_than_global_cap():
    logits = np.zeros((3, 6), dtype=np.float32)
    output, residual, _ = _context_residual(
        logits, np.ones(3, dtype=np.float32), np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    )
    assert residual[0] == 0.0
    assert 0.0 < residual[1] < residual[2] <= 0.5
    np.testing.assert_array_equal(output[:, [0,1,2,4,5]], logits[:, [0,1,2,4,5]])

