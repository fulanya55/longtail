import numpy as np
import pytest

from experiments.run_route_a_vft_isrj_occupancy_intervention_v1_frozen import _paired_effect


def test_paired_effect_requires_identity_and_label_pairing_and_uses_hard_negatives():
    targets = np.zeros((4, 6), dtype=np.uint8)
    targets[[0, 1], 5] = 1
    reference = {"receiver_noise_seed": np.array([1, 2, 3, 4]), "targets": targets, "probabilities": np.zeros((4, 6)), "predictions": np.zeros((4, 6), dtype=np.uint8), "thresholds": np.full(6, .5)}
    intervention = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in reference.items()}
    intervention["probabilities"][0:2, 4] = [.2, .4]
    intervention["predictions"][0:2, 4] = 1
    row = _paired_effect(reference, intervention, 1)
    assert row["hard_negative_count"] == 2
    assert row["paired_probability_delta"]["mean"] == pytest.approx(.3)
    assert row["paired_fpr_delta"]["mean"] == pytest.approx(1.)
    intervention["receiver_noise_seed"][0] = 9
    try:
        _paired_effect(reference, intervention, 1)
    except ValueError as error:
        assert "receiver" in str(error)
    else:
        raise AssertionError("unpaired receiver IDs must fail")
