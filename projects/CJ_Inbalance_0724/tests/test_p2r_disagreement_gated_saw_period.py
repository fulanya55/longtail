import numpy as np

from cj0724.p2r_disagreement_gated_saw_period import disagreement_gated_period


def test_disagreement_gate_only_replaces_catastrophic_high_period_case() -> None:
    legacy = np.asarray([8e-6, 8e-6, 70e-6, 8e-6])
    strongest = np.asarray([100e-6, 70e-6, 100e-6, 100e-6])
    learned = np.asarray([60e-6, 60e-6, 80e-6, 40e-6])
    prediction, gate = disagreement_gated_period(legacy, strongest, learned)
    np.testing.assert_array_equal(gate, [True, False, False, False])
    np.testing.assert_allclose(prediction, [100e-6, 8e-6, 70e-6, 8e-6])


def test_disagreement_gate_rejects_shape_mismatch() -> None:
    try:
        disagreement_gated_period(np.zeros(2), np.zeros(3), np.zeros(2))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
