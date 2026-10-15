import numpy as np

from cj0724.p2w_isrj_width_oracle_ood import reconstruct_isrj_component
from cj0724.p2x_isrj_phase_reset_ood import estimate_width_phase_resets


def test_phase_reset_estimator_recovers_hand_constructed_widths() -> None:
    widths = np.asarray([1.0, 1.1, 2.0, 5.0, 12.0, 20.0, 30.0, 50.0]) * 1.0e-6
    replicas = np.asarray([2, 4, 3, 2, 4, 3, 2, 4])
    arrivals = np.asarray([0.0, 7.0, 3.0, 13.0, 29.0, 11.0, 17.0, 23.0]) * 1.0e-6
    iq = reconstruct_isrj_component(widths, replicas, arrivals)
    result = estimate_width_phase_resets(iq)
    assert np.allclose(result["prediction"], widths, atol=0.11e-6)
    assert np.all(result["confidence"] > 0.0)
    assert np.all(result["reset_count"] >= 2)


def test_phase_reset_estimator_shape_validation() -> None:
    bad = np.zeros((2, 3000), dtype=np.float32)
    try:
        estimate_width_phase_resets(bad)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected shape validation error")
