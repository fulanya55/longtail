import torch
import numpy as np

from experiments.audit_route_a_development_v4_dataset import _ranges_disjoint
from experiments.run_route_a_development_v4_sawfm_specificity import (
    _apply_direction,
    _clip_logits,
    _direction_cosine,
    _direction_norm,
    _layer_shuffled,
    _per_tensor_norm_error,
    _random_norm_matched,
    _state_direction,
    _validate_v4_config,
)


def _valid_config():
    return {
        "data_root": r"C:\data\v4",
        "train_file": "train.mat",
        "test_v4_path": r"C:\data\v4\test.mat",
        "sealed_test_v3_path": r"C:\data\v3\test.mat",
        "forbidden_test_v2_path": r"C:\data\v2\test.mat",
        "allow_locked_test": False,
        "evidence_role": "development_v4",
        "test_v4_generated": False,
        "test_v4_access_authorized": False,
        "test_v4_used": False,
        "test_v3_sealed": True,
        "test_v3_used": False,
        "test_v2_consumed": True,
        "development_v2_reuse_for_selection": False,
        "development_v3_reuse_for_selection": False,
    }


def test_v4_config_guard_rejects_every_test_or_reuse_boundary_break():
    _validate_v4_config(_valid_config())
    invalid = (
        {"allow_locked_test": True},
        {"evidence_role": "development_v3"},
        {"test_v4_generated": True},
        {"test_v4_access_authorized": True},
        {"test_v4_used": True},
        {"test_v3_sealed": False},
        {"test_v3_used": True},
        {"test_v2_consumed": False},
        {"development_v2_reuse_for_selection": True},
        {"development_v3_reuse_for_selection": True},
        {"train_file": "test.mat"},
    )
    for update in invalid:
        candidate = _valid_config()
        candidate.update(update)
        try:
            _validate_v4_config(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid development-v4 evidence config was accepted")


def test_direction_application_and_norm_matched_random_control():
    base = {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(2, dtype=torch.long)}
    endpoint = {"weight": torch.tensor([3.0, -2.0]), "count": torch.tensor(9, dtype=torch.long)}
    direction = _state_direction(base, endpoint)
    applied = _apply_direction(base, direction, 1.5)
    torch.testing.assert_close(applied["weight"], torch.tensor([4.0, -4.0]))
    assert int(applied["count"]) == 2
    random = _random_norm_matched(direction, 123)
    assert _per_tensor_norm_error(direction, random) < 1e-6
    assert abs(_direction_norm(direction) - _direction_norm(random)) < 1e-6


def test_layer_shuffle_preserves_global_norm_and_changes_direction():
    direction = {
        "layer_a": torch.tensor([1.0, 2.0, 3.0]),
        "layer_b": torch.tensor([-2.0, 4.0, 1.0]),
        "singleton": torch.tensor([[1.0, -3.0], [2.0, 0.0]]),
    }
    shuffled = _layer_shuffled(direction, 99)
    assert abs(_direction_norm(direction) - _direction_norm(shuffled)) < 1e-6
    assert _direction_cosine(direction, shuffled) < 0.999


def test_declared_v4_identity_range_is_disjoint_from_all_frozen_roles():
    current = (232677743, 232702942)
    references = (
        (218938474, 218963673),
        (221938474, 221963673),
        (222677743, 222702942),
        (225677743, 225702942),
    )
    assert all(_ranges_disjoint(current, reference) for reference in references)


def test_fixed_logit_clip_is_uniform_and_rejects_nonfinite_values():
    values = _clip_logits(np.asarray([[-1e20, 0.0, 1e20]], dtype=np.float64))
    np.testing.assert_allclose(values, np.asarray([[-40.0, 0.0, 40.0]]))
    try:
        _clip_logits(np.asarray([[np.inf]]))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite logits were accepted")
