import torch

from cj0724.multitask import build_multitask_model
from cj0724.p7a_static_equal_mac_control import static_rows


def test_compute_matched_control_preserves_multitask_contract():
    model = build_multitask_model("legacy_mlcnn_global_compute_matched_control", width=24, head_dim=256, parameter_hidden_dim=64)
    logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)


def test_static_control_selects_only_a_simultaneous_parameter_and_mac_match():
    result = static_rows([64, 65, 66, 67, 68, 69, 70])
    assert result["static_passed"]
    assert .98 <= result["selected"]["mac_ratio"] <= 1.02
    assert .98 <= result["selected"]["parameter_ratio"] <= 1.02
