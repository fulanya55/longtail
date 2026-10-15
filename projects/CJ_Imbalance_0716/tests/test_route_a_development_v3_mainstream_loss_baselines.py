import numpy as np
import torch

from experiments.run_route_a_development_v3_mainstream_loss_baselines import (
    LOSS_MODE,
    _cosine_multiplier,
    _elementwise_loss,
    _fit_statistics,
    _paired_directional_statistics,
    _route_rawsaw,
)


def test_all_loss_modes_return_finite_elementwise_values():
    targets = np.asarray([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
                          [1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1]], dtype=np.float32)
    stats = _fit_statistics(targets)
    logits = torch.zeros_like(torch.from_numpy(targets))
    tensor_targets = torch.from_numpy(targets)
    for mode in sorted(set(LOSS_MODE.values())):
        value = _elementwise_loss(mode, logits, tensor_targets, stats, torch.device("cpu"))
        assert value.shape == logits.shape
        assert torch.isfinite(value).all()


def test_fit_statistics_are_finite_and_fit_only_shapes():
    targets = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    stats = _fit_statistics(targets)
    assert stats["positive_counts"].tolist() == [2, 2]
    assert np.isfinite(stats["pos_weight"]).all()
    assert np.isfinite(stats["logit_prior"]).all()
    assert np.isfinite(stats["cb_weight"]).all()


def test_rawsaw_routing_changes_only_sawfm_column():
    logits = {"fit": np.arange(18, dtype=np.float32).reshape(3, 6)}
    routed = _route_rawsaw(logits, {"fit": np.asarray([0.1, 0.2, 0.3], dtype=np.float32)})
    assert np.array_equal(routed["fit"][:, [0, 2, 3, 4, 5]], logits["fit"][:, [0, 2, 3, 4, 5]])
    assert np.allclose(routed["fit"][:, 1], [0.1, 0.2, 0.3])


def test_paired_directional_statistics_record_consistent_positive_directions():
    stats = _paired_directional_statistics([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["positive_n"] == 5
    assert stats["nonzero_n"] == 5
    assert stats["wilcoxon_greater_p"] == 0.03125
    assert stats["sign_test_greater_p"] == 0.03125


def test_cosine_multiplier_reaches_preregistered_minimum():
    assert _cosine_multiplier(0, 72, 0.01) == 1.0
    assert np.isclose(_cosine_multiplier(72, 72, 0.01), 0.01)
    assert _cosine_multiplier(36, 72, 0.01) < 1.0
