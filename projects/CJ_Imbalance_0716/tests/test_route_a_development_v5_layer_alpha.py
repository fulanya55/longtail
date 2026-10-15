import torch

from experiments.run_route_a_development_v5_layer_alpha import (
    _choose_adaptive,
    _compose_layer_state,
    _direction_cosine,
    _direction_norm,
    _orthogonal_task_subspace,
    _validate_v5_config,
)


def _valid_config():
    return {
        "data_root": r"C:\data\v5",
        "train_file": "train.mat",
        "test_v5_path": r"C:\data\v5\test.mat",
        "sealed_test_v3_path": r"C:\data\v3\test.mat",
        "forbidden_test_v2_path": r"C:\data\v2\test.mat",
        "allow_locked_test": False,
        "evidence_role": "development_v5",
        "test_v5_generated": False,
        "test_v5_access_authorized": False,
        "test_v5_used": False,
        "test_v3_sealed": True,
        "test_v3_used": False,
        "test_v2_consumed": True,
        "development_v4_reuse_for_selection": False,
    }


def test_v5_config_rejects_test_access_and_prior_development_reuse():
    _validate_v5_config(_valid_config())
    for update in (
        {"allow_locked_test": True},
        {"evidence_role": "development_v4"},
        {"test_v5_generated": True},
        {"test_v5_access_authorized": True},
        {"test_v5_used": True},
        {"test_v3_sealed": False},
        {"test_v3_used": True},
        {"test_v2_consumed": False},
        {"development_v4_reuse_for_selection": True},
        {"train_file": "test.mat"},
    ):
        candidate = _valid_config()
        candidate.update(update)
        try:
            _validate_v5_config(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid development-v5 config was accepted")


def test_layer_composition_changes_only_shared_and_classifier_support():
    base = {
        "temporal_backbone.0.weight": torch.tensor([1.0]),
        "classifier.1.weight": torch.tensor([2.0]),
        "frozen.weight": torch.tensor([3.0]),
        "count": torch.tensor(4, dtype=torch.long),
    }
    direction = {
        "temporal_backbone.0.weight": torch.tensor([2.0]),
        "classifier.1.weight": torch.tensor([3.0]),
        "frozen.weight": torch.tensor([100.0]),
        "count": torch.tensor(0.0),
    }
    state = _compose_layer_state(base, direction, 1.5, 0.5)
    torch.testing.assert_close(state["temporal_backbone.0.weight"], torch.tensor([4.0]))
    torch.testing.assert_close(state["classifier.1.weight"], torch.tensor([3.5]))
    torch.testing.assert_close(state["frozen.weight"], torch.tensor([3.0]))
    assert int(state["count"]) == 4


def test_orthogonal_task_subspace_matches_norm_and_is_orthogonal():
    saw = {"w": torch.tensor([1.0, 0.0, 0.0])}
    others = [
        {"w": torch.tensor([1.0, 1.0, 0.0])},
        {"w": torch.tensor([1.0, 0.0, 1.0])},
    ]
    control = _orthogonal_task_subspace(saw, others)
    assert abs(_direction_norm(control) - _direction_norm(saw)) < 1e-6
    assert abs(_direction_cosine(control, saw)) < 1e-6


def test_adaptive_selection_respects_macro_tolerance_then_weak_rank_and_ties():
    rows = [
        {"variant": "a", "valid": True, "macro_aupr": 0.900, "target_weak_auroc": 0.60, "total_alpha": 2.0, "classifier_alpha": 1.0},
        {"variant": "b", "valid": True, "macro_aupr": 0.896, "target_weak_auroc": 0.70, "total_alpha": 2.0, "classifier_alpha": 1.0},
        {"variant": "c", "valid": True, "macro_aupr": 0.894, "target_weak_auroc": 0.99, "total_alpha": 1.0, "classifier_alpha": 0.0},
    ]
    selected, context = _choose_adaptive(rows, tolerance=0.005)
    assert selected["variant"] == "b"
    assert context["eligible"] == ["b", "a"]
