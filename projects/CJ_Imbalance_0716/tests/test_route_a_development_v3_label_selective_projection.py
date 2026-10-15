import torch

from experiments.run_route_a_development_v3_label_selective_projection import (
    _project_saw_gradient,
    _validate_v3_config,
)


def test_projection_removes_negative_target_component_and_keeps_positive_one():
    saw = [torch.tensor([-1.0, 1.0])]
    targets = {
        "MFT": [torch.tensor([1.0, 0.0])],
        "ISRJ": [torch.tensor([0.0, 1.0])],
        "VFT": [torch.tensor([0.0, 1.0])],
    }
    projected, diagnostics = _project_saw_gradient(saw, targets, passes=2)
    assert torch.dot(projected[0], targets["MFT"][0]) >= -1e-7
    assert torch.dot(projected[0], targets["ISRJ"][0]) > 0
    assert diagnostics["applications"]["MFT"] >= 1
    assert diagnostics["applications"]["ISRJ"] == 0
    assert diagnostics["applications"]["VFT"] == 0


def test_projection_is_noop_when_all_target_gradients_are_compatible():
    saw = [torch.tensor([1.0, 2.0])]
    targets = {
        "MFT": [torch.tensor([1.0, 0.0])],
        "ISRJ": [torch.tensor([0.0, 1.0])],
        "VFT": [torch.tensor([1.0, 1.0])],
    }
    projected, diagnostics = _project_saw_gradient(saw, targets, passes=2)
    torch.testing.assert_close(projected[0], saw[0])
    assert diagnostics["removed_norm_fraction"] <= 1e-10
    assert all(value == 0 for value in diagnostics["applications"].values())


def test_v3_config_guard_rejects_evidence_and_test_leakage():
    valid = {
        "data_root": r"C:\data\v3",
        "train_file": "train.mat",
        "test_v3_path": r"C:\data\v3\test.mat",
        "forbidden_test_v2_path": r"C:\data\v2\test.mat",
        "allow_locked_test": False,
        "evidence_role": "development_v3",
        "test_v3_sealed": True,
        "test_v3_access_authorized": False,
        "test_v3_used": False,
        "test_v2_consumed": True,
        "development_v2_reuse_for_selection": False,
    }
    _validate_v3_config(valid)
    invalid_updates = (
        {"allow_locked_test": True},
        {"evidence_role": "development_v2"},
        {"test_v3_sealed": False},
        {"test_v3_access_authorized": True},
        {"test_v3_used": True},
        {"test_v2_consumed": False},
        {"development_v2_reuse_for_selection": True},
        {"data_root": r"C:\data\v3", "train_file": "test.mat"},
        {"data_root": r"C:\data\v2", "train_file": "test.mat"},
    )
    for update in invalid_updates:
        candidate = dict(valid)
        candidate.update(update)
        try:
            _validate_v3_config(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v3 evidence config was accepted")
