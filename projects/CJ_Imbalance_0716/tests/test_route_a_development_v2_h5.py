import numpy as np
import torch

from experiments.run_route_a_development_v2_h5 import (
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_NAMES,
    _paired_heads,
    _promotion_gate,
    _validate_config,
)
from experiments.run_route_a_development_v2_shared_gradient import (
    _configure_partial_scope,
    _loss_components,
)
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_development_v2_gradient_contrast import _interpolate_state


def test_paired_heads_copy_common_rows_exactly():
    c6, h5 = _paired_heads(8, seed=123)
    np.testing.assert_allclose(
        h5.weight.detach().numpy(),
        c6.weight.detach().numpy()[COMMON_INDICES],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        h5.bias.detach().numpy(),
        c6.bias.detach().numpy()[COMMON_INDICES],
        rtol=0.0,
        atol=0.0,
    )
    assert c6.out_features == 6
    assert h5.out_features == 5
    assert COMMON_NAMES == ("FMB", "SinFM", "MFT", "VFT", "ISRJ")


def test_development_v2_config_guard_rejects_test_access_modes():
    valid = {
        "allow_locked_test": False,
        "evidence_role": "development_v2",
        "test_v2_sealed": True,
    }
    _validate_config(valid)
    for update in (
        {"allow_locked_test": True},
        {"evidence_role": "locked_test"},
        {"test_v2_sealed": False},
    ):
        candidate = dict(valid)
        candidate.update(update)
        try:
            _validate_config(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evidence config was accepted")


def test_promotion_gate_checks_all_registered_requirements():
    def stat(value):
        return {"mean": value, "std": 0.0, "count": 5, "bootstrap_95_ci": [value, value]}

    comparison = {
        "aggregate": {
            "macro_aupr": stat(0.0),
            "macro_f1": stat(0.02),
            "target_mean_weak_recall": stat(0.04),
            "target_mean_negative_fpr": stat(-0.01),
            "sawfm_present_macro_f1": stat(0.02),
            "sawfm_absent_macro_f1": stat(0.0),
            "per_label_aupr": {
                name: stat(0.0) for name in COMMON_NAMES
            },
            "positive_target_weak_recall_seeds": 5,
        }
    }
    aggregate = {
        "H5_CommonCal": {
            "target_mean_negative_fpr": stat(0.15),
        }
    }
    result = _promotion_gate(comparison, aggregate)
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert set(TARGET_NAMES) == {"MFT", "VFT", "ISRJ"}


def test_shared_gradient_loss_masks_only_sawfm_term():
    logits = torch.zeros(3, 6, requires_grad=True)
    targets = torch.tensor([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
    ], dtype=torch.float32)
    common, saw = _loss_components(logits, targets, torch.ones(6))
    common.backward(retain_graph=True)
    assert torch.all(logits.grad[:, 1] == 0)
    assert torch.any(logits.grad[:, COMMON_INDICES] != 0)
    logits.grad.zero_()
    (common + 0.2 * saw).backward()
    assert torch.any(logits.grad[:, 1] != 0)


def test_partial_scope_unfreezes_only_last_temporal_block_and_classifier():
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    scope = _configure_partial_scope(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("temporal_backbone.0") or name.startswith("classifier")
        for name in trainable
    )
    assert sum(parameter.numel() for parameter in scope["shared"]) > 0
    assert sum(parameter.numel() for parameter in scope["classifier"]) == 6 * 128 + 6


def test_gradient_contrast_interpolation_has_exact_endpoints_and_extrapolation():
    c6 = {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(3)}
    m5 = {"weight": torch.tensor([2.0, 0.0]), "count": torch.tensor(4)}
    at_one = _interpolate_state(c6, m5, 1.0)
    at_two = _interpolate_state(c6, m5, 2.0)
    torch.testing.assert_close(at_one["weight"], m5["weight"])
    torch.testing.assert_close(at_two["weight"], torch.tensor([3.0, -2.0]))
    assert at_one["count"].item() == 4
