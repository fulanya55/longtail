import numpy as np

from experiments.run_route_a_h1_b2_mft_peak_guard_v3 import guarded_logits


def test_guard_changes_only_mft_and_is_negative():
    logits = np.zeros((4, 6), dtype=np.float32)
    evidence = np.asarray([-3.0, -0.5, 0.0, 2.0], dtype=np.float32)
    output, residual = guarded_logits(logits, evidence, 0.75)
    assert np.array_equal(output[:, [0, 1, 2, 4, 5]], logits[:, [0, 1, 2, 4, 5]])
    assert np.all(residual <= 0.0)
    assert residual[-1] == 0.0
    assert np.allclose(output[:, 3], residual)


def test_guard_is_bounded_by_cap():
    logits = np.zeros((2, 6), dtype=np.float32)
    output, residual = guarded_logits(logits, np.asarray([-100.0, 100.0]), 0.75)
    assert residual[0] >= -0.750001
    assert residual[1] == 0.0
    assert output[0, 3] < output[1, 3]
