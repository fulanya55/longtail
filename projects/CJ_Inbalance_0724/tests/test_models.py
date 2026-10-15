import torch

from cj0724.models import (
    ComplexConv1d,
    LEGACY_MODEL_NAMES,
    MODEL_NAMES,
    STANDARD_MODEL_NAMES,
    build_benchmark_model,
    build_model,
)


def test_complex_conv_matches_scalar_complex_product():
    layer = ComplexConv1d(1, 1, kernel_size=1)
    with torch.no_grad():
        layer.real_conv.weight.fill_(2.0)
        layer.imag_conv.weight.fill_(3.0)
        layer.real_bias.zero_()
        layer.imag_bias.zero_()
    real, imag = layer(torch.tensor([[[5.0]]]), torch.tensor([[[7.0]]]))
    assert torch.equal(real, torch.tensor([[[-11.0]]]))
    assert torch.equal(imag, torch.tensor([[[29.0]]]))


def test_both_models_accept_iq_and_return_six_logits():
    iq = torch.randn(3, 2, 3000)
    for name in ("complex1d", "iq_grouped"):
        model = build_model(name, width=8)
        assert model(iq).shape == (3, 6)


def test_legacy_models_are_registered_for_the_shared_trainer():
    assert set(LEGACY_MODEL_NAMES).issubset(MODEL_NAMES)


def test_standard_backbones_accept_current_iq_interface():
    iq = torch.randn(1, 2, 3000)
    for name in STANDARD_MODEL_NAMES:
        model = build_model(name).eval()
        with torch.no_grad():
            assert model(iq).shape == (1, 6)


def test_unified_classifier_has_same_fixed_embedding_for_different_backbones():
    iq = torch.randn(2, 2, 3000)
    for name in ("complex1d", "iq_grouped"):
        model = build_benchmark_model(name, width=8, head_dim=64)
        assert model.classifier[-1].in_features == 64
        assert model(iq).shape == (2, 6)


def test_multi_expert_b1_returns_expert_and_ensemble_logits():
    model = build_benchmark_model("efficientnet_b1_moe3_1d", head_dim=64).eval()
    iq = torch.randn(1, 2, 3000)
    with torch.no_grad():
        expert_logits = model.forward_experts(iq)
        ensemble_logits = model(iq)
    assert expert_logits.shape == (1, 3, 6)
    assert ensemble_logits.shape == (1, 6)
    assert torch.allclose(ensemble_logits, expert_logits.mean(dim=1))


def test_vft_residual_moe_only_changes_vft_logit():
    model = build_benchmark_model("efficientnet_b1_vft_residual_moe_1d", head_dim=64).eval()
    iq = torch.randn(1, 2, 3000)
    with torch.no_grad():
        components = model.forward_components(iq)
    assert components["logits"].shape == (1, 6)
    assert torch.allclose(components["logits"][:, :4], components["general_logits"][:, :4])
    assert torch.allclose(components["logits"][:, 5:], components["general_logits"][:, 5:])
    assert components["target_residual"].abs().max() <= 1.0
