import numpy as np
import pytest

from experiments.run_route_a_v17_predicted_width_frozen_q4_alignment import _row


def _q4(probability, prediction, threshold=.5):
    return {
        "receiver_noise_seed": np.array([11, 12, 13]),
        "targets": np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]], dtype=np.uint8),
        "probabilities": np.column_stack((np.zeros((3, 4)), np.asarray(probability), np.zeros(3))),
        "predictions": np.column_stack((np.zeros((3, 4), dtype=np.uint8), np.asarray(prediction, dtype=np.uint8), np.zeros(3, dtype=np.uint8))),
        "thresholds": np.full(6, threshold, dtype=np.float32),
    }


def test_row_uses_vft_absent_isrj_present_pair_only():
    narrow = _q4([.10, .20, .90], [0, 0, 1])
    wide = _q4([.30, .50, .80], [0, 1, 1])
    width = {
        "receiver_noise_seed": np.array([11, 12]),
        "narrow_prediction": np.array([10., 11.]),
        "wide_prediction": np.array([20., 21.]),
        "narrow_true": np.array([10., 11.]),
        "wide_true": np.array([20., 21.]),
    }
    row = _row(2039, narrow, wide, width)
    assert row["hard_negative_count_per_regime"] == 2
    assert row["paired_probability_delta"] == pytest.approx(.25)
    assert row["paired_fpr_delta"] == pytest.approx(.5)
    assert row["paired_receivers_exact"] and row["thresholds_exact"]


def test_row_rejects_width_identity_mismatch():
    narrow = _q4([.10, .20, .90], [0, 0, 1])
    wide = _q4([.30, .50, .80], [0, 1, 1])
    width = {
        "receiver_noise_seed": np.array([12, 11]),
        "narrow_prediction": np.array([10., 11.]),
        "wide_prediction": np.array([20., 21.]),
        "narrow_true": np.array([10., 11.]),
        "wide_true": np.array([20., 21.]),
    }
    with pytest.raises(ValueError, match="identities differ"):
        _row(2039, narrow, wide, width)
