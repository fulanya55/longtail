import numpy as np
import torch

from cj0724.b6_cross_domain_sufficiency import (
    cross_domain_feature_batch,
    identity_balanced_class_weights,
    screen_slice,
)


def test_cross_domain_features_are_finite_and_physically_separate():
    output = cross_domain_feature_batch(torch.randn(3, 2, 3000))
    assert output["envelope_periodicity"].shape == (3, 88)
    assert output["comb_spectrum"].shape == (3, 155)
    assert torch.isfinite(output["envelope_periodicity"]).all()
    assert torch.isfinite(output["comb_spectrum"]).all()


def test_identity_balanced_weights_equalize_classes():
    labels = np.asarray([1, 1, 0, 0, 0, 0])
    weights = identity_balanced_class_weights(labels)
    assert np.isclose(weights[labels == 1].sum(), weights[labels == 0].sum())


def test_screen_slice_counts_complementary_rescue_and_safe_joint_negative():
    labels = np.asarray([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ], dtype=np.uint8)
    result = screen_slice(
        labels,
        np.asarray([0.1, 0.1, 0.9, 0.9]),
        np.asarray([0.9, 0.1, 0.1, 0.1]),
        np.asarray([0.1, 0.9, 0.1, 0.1]),
        0.5, 0.5, 0.5,
    )
    assert result["union_rescue"] == 1.0
    assert result["envelope_unique_rescue"] == 0.5
    assert result["comb_unique_rescue"] == 0.5
    assert result["joint_hard_false_positive_correction"] == 1.0
    assert result["joint_negative_positive_harm"] == 0.0
