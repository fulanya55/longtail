import pytest
import torch

from cjlt.models.evidence_readout import (
    BoundedLocalContrastReadout,
    EarlySawFMEnvelopeIFExpert,
    LabelConditionedEvidenceReadout,
    OrdinalVisibilityAnchorCalibrator,
    SawEvidenceRouter,
    SawStage2LowRankEvidence,
    VisibilityAffineCalibrator,
)


def test_bounded_local_contrast_is_zero_centred_and_hard_bounded():
    model = BoundedLocalContrastReadout(
        channels=8, num_labels=3, hidden_channels=8, topk=4, residual_cap=0.35
    )
    temporal = torch.randn(5, 8, 20)
    base = torch.randn(5, 3)
    logits, diagnostics = model(temporal, base, return_diagnostics=True)
    residual = diagnostics["residual"]
    assert logits.shape == (5, 3)
    assert torch.allclose(diagnostics["heatmap_mean"], torch.zeros_like(diagnostics["heatmap_mean"]), atol=1e-5)
    assert float(residual.detach().abs().max()) <= 0.350001
    assert not hasattr(model, "statistic_bias")


def test_label_conditioned_readout_preserves_base_and_trains_query_and_local_paths():
    torch.manual_seed(41)
    module = LabelConditionedEvidenceReadout(
        channels=16,
        num_labels=6,
        query_dim=8,
        evidence_dim=8,
        topk=4,
        dropout=0.0,
        enable_local=True,
    )
    values = torch.randn(3, 16, 25, requires_grad=True)
    base = torch.randn(3, 6)

    logits, diagnostics = module(values, base, return_diagnostics=True)
    logits.mean().backward()

    assert logits.shape == (3, 6)
    assert diagnostics["attention_entropy"].shape == (3, 6)
    assert diagnostics["topk_indices"].shape == (3, 6, 4)
    assert module.queries.grad is not None
    assert module.query_weight.grad is not None
    assert module.local_weight.grad is not None
    assert values.grad is not None


def test_early_saw_expert_outputs_one_bounded_residual_logit():
    torch.manual_seed(43)
    expert = EarlySawFMEnvelopeIFExpert(
        complex_channels=8,
        hidden_channels=8,
        topk=4,
        dropout=0.0,
    )
    real = torch.randn(2, 8, 128, requires_grad=True)
    imag = torch.randn(2, 8, 128, requires_grad=True)

    residual, diagnostics = expert(real, imag, return_diagnostics=True)
    residual.sum().backward()

    assert residual.shape == (2, 1)
    assert diagnostics["expert_logit"].shape == (2, 1)
    assert abs(float(diagnostics["scale"].detach())) < 0.1
    assert expert.stem[0].weight.grad is not None
    assert real.grad is not None
    assert imag.grad is not None


def test_saw_stage2_low_rank_evidence_is_small_and_temporally_normalized():
    model = SawStage2LowRankEvidence(channels=64, rank=8, topk=16)
    values = torch.randn(4, 64, 75)

    logits, evidence_map, diagnostics = model(values, return_map=True)

    assert logits.shape == (4,)
    assert evidence_map.shape == (4, 75)
    assert torch.allclose(evidence_map.mean(dim=-1), torch.zeros(4), atol=1e-5)
    assert abs(float(diagnostics["adapter_scale"].detach())) <= 0.20
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 5000


def test_saw_evidence_router_is_convex_and_differentiable():
    model = SawEvidenceRouter(feature_dim=8)
    features = torch.randn(4, 8, requires_grad=True)
    h2 = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    expert = torch.tensor([2.0, 1.0, -1.0, -2.0])
    fused, gate = model(features, h2, expert, return_gate=True)
    assert fused.shape == (4,)
    assert gate.shape == (4,)
    assert torch.all(gate > 0.0)
    assert torch.all(gate < 1.0)
    assert torch.all(fused >= torch.minimum(h2, expert))
    assert torch.all(fused <= torch.maximum(h2, expert))
    fused.sum().backward()
    assert features.grad is not None


def test_saw_evidence_router_rejects_bad_feature_shape():
    model = SawEvidenceRouter(feature_dim=8)
    with pytest.raises(ValueError, match="router features"):
        model(torch.randn(2, 7), torch.randn(2), torch.randn(2))


def test_visibility_affine_calibrator_starts_near_identity_and_has_22_parameters():
    model = VisibilityAffineCalibrator(feature_dim=10)
    logits = torch.tensor([-2.0, 0.0, 2.0])
    features = torch.randn(3, 10, requires_grad=True)
    output, scale, bias = model(logits, features, return_parameters=True)
    assert torch.allclose(output, logits, atol=2e-3)
    assert torch.all(scale > 0.0)
    assert torch.allclose(bias, torch.zeros_like(bias))
    assert sum(parameter.numel() for parameter in model.parameters()) == 22
    output.sum().backward()
    assert features.grad is not None


def test_ordinal_visibility_anchor_has_16_parameters_and_ordered_biases():
    model = OrdinalVisibilityAnchorCalibrator(feature_dim=10)
    logits = torch.tensor([-2.0, 0.0, 2.0])
    features = torch.randn(3, 10, requires_grad=True)
    oracle = torch.tensor([1.0, 0.0, -1.0])
    output, diagnostics = model(
        logits, features, oracle_visibility=oracle, return_diagnostics=True
    )
    assert output.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) == 16
    assert diagnostics["anchor_values"][0] == 0.0
    assert diagnostics["anchor_values"][1] > diagnostics["anchor_values"][0]
    assert diagnostics["anchor_values"][2] > diagnostics["anchor_values"][1]
    assert torch.allclose(diagnostics["anchor_weights"].sum(dim=1), torch.ones(3))
    learned_output = model(logits, features)
    learned_output.sum().backward()
    assert features.grad is not None
