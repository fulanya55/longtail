import numpy as np

from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits


def test_guard_is_one_sided_and_preserves_other_logits():
    logits = np.zeros((3, 6), dtype=np.float32)
    evidence = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
    guarded, residual = guarded_logits(logits, evidence, cap=0.75)
    assert residual[0] < 0.0
    assert residual[1] == 0.0
    assert residual[2] == 0.0
    assert np.array_equal(guarded[:, [0, 1, 2, 4, 5]], logits[:, [0, 1, 2, 4, 5]])


def test_guard_is_bounded_by_cap():
    logits = np.zeros((2, 6), dtype=np.float32)
    guarded, residual = guarded_logits(logits, np.asarray([-100.0, 100.0]), cap=0.75)
    assert residual[0] >= -0.750001
    assert residual[1] == 0.0
    assert np.isfinite(guarded).all()

