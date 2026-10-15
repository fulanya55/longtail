import pytest
import torch

from cjlt.models.backbone_adapter import (
    LightweightComplexIQClassifier,
    build_backbone,
    count_trainable_parameters,
)
from cjlt.models.phase_multiscale import (
    CrossStageComplexFeaturePyramid1d,
    DelayedFusionMultiScaleIQClassifier,
    EvidencePreservingMultiScaleBlock1d,
    FixedBlurPool1d,
    LocalEvidenceStatisticsBypass1d,
    PhaseAwareMultiScaleIQClassifier,
)
from cjlt.models.student_a import (
    StudentAModel,
    configure_last_block_classifier_training,
)


def test_lightweight_complex_model_preserves_default_logits_interface():
    torch.manual_seed(2026)
    model = LightweightComplexIQClassifier(
        num_classes=4,
        width=8,
        dropout=0.0,
    )
    model.eval()
    inputs = torch.randn(2, 2, 3000)

    logits_default = model(inputs)
    logits_feature, features = model(inputs, return_features=True)

    assert logits_default.shape == (2, 4)
    assert features.shape == (2, 32)
    assert model.feature_dim == 32
    assert torch.allclose(logits_default, logits_feature, rtol=0.0, atol=0.0)


def test_model_exposes_temporal_features_for_future_label_conditioning():
    model = LightweightComplexIQClassifier(num_classes=6, width=8, dropout=0.0)
    inputs = torch.randn(2, 2, 3000)

    temporal = model.forward_temporal_features(inputs)
    pooled = model.forward_features(inputs)

    assert temporal.ndim == 3
    assert temporal.shape[0] == 2
    assert temporal.shape[1] == model.feature_dim
    assert pooled.shape == (2, model.feature_dim)


def test_build_backbone_supports_four_and_six_classes():
    four = build_backbone("complex_1d_true", num_classes=4, width=8)
    six = build_backbone("complex_1d_true", num_classes=6, width=8)
    assert four(torch.randn(1, 2, 3000)).shape == (1, 4)
    assert six(torch.randn(1, 2, 3000)).shape == (1, 6)
    assert count_trainable_parameters(four) > 0


def test_model_rejects_non_iq_input_shape():
    model = LightweightComplexIQClassifier(num_classes=4, width=8)
    with pytest.raises(ValueError, match="\\[batch, 2, length\\]"):
        model(torch.randn(2, 1, 3000))


def test_build_backbone_rejects_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("transformer_large", num_classes=4)


def test_phase_multiscale_backbone_preserves_temporal_and_logits_contracts():
    torch.manual_seed(2026)
    model = PhaseAwareMultiScaleIQClassifier(
        num_classes=6, width=8, dropout=0.0
    )
    model.eval()
    inputs = torch.randn(2, 2, 3000)

    temporal = model.forward_temporal_features(inputs)
    logits, features = model(inputs, return_features=True)

    assert temporal.shape == (2, 64, 375)
    assert features.shape == (2, 64)
    assert logits.shape == (2, 6)
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(features).all()


def test_phase_multiscale_backbone_is_phase_aware_and_differentiable():
    torch.manual_seed(2027)
    model = PhaseAwareMultiScaleIQClassifier(
        num_classes=4, width=4, dropout=0.0
    )
    model.eval()
    in_phase = torch.zeros(2, 2, 256)
    in_phase[:, 0, :] = 1.0
    quadrature = torch.zeros(2, 2, 256)
    quadrature[:, 1, :] = 1.0

    in_phase_features = model.forward_features(in_phase)
    quadrature_features = model.forward_features(quadrature)
    assert not torch.allclose(in_phase_features, quadrature_features)

    inputs = torch.randn(2, 2, 256, requires_grad=True)
    model(inputs).square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_build_backbone_supports_phase_multiscale_aliases_and_partial_adaptation():
    backbone = build_backbone("pam_iq", num_classes=6, width=4, dropout=0.0)
    assert isinstance(backbone, PhaseAwareMultiScaleIQClassifier)
    wrapper = StudentAModel(backbone, relation_dim=8)
    trainable = configure_last_block_classifier_training(wrapper)

    assert trainable["total"] > trainable["classifier"]
    assert wrapper(torch.randn(2, 2, 256)).shape == (2, 6)


@pytest.mark.parametrize(
    ("name", "option", "expected"),
    [
        ("pam_iq_no_phase", "include_phase_delta", False),
        ("pam_iq_no_attention", "use_sparse_attention", False),
        ("pam_iq_avg_pool", "attentive_statistics_pooling", False),
    ],
)
def test_phase_multiscale_build_variants_expose_controlled_ablation(
    name, option, expected
):
    model = build_backbone(name, num_classes=6, width=4, dropout=0.0)
    assert model.architecture_options[option] is expected
    assert model(torch.randn(2, 2, 256)).shape == (2, 6)


def test_pam_iq_dual_pool_preserves_feature_contract_and_residual_attention():
    torch.manual_seed(2030)
    model = build_backbone(
        "pam_iq_dual_pool", num_classes=6, width=4, dropout=0.0
    )
    logits, features = model(
        torch.randn(2, 2, 256), return_features=True
    )

    assert logits.shape == (2, 6)
    assert features.shape == (2, 32)
    assert model.architecture_options["pooling_mode"] == "dual_pool"
    assert model.architecture_options["residual_attention"] is True
    logits.square().mean().backward()
    attention_scales = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("attention.scale")
    ]
    assert len(attention_scales) == 3
    assert all(parameter.grad is not None for parameter in attention_scales)


def test_pam_iq_dual_query_uses_label_specific_features_and_classifier():
    torch.manual_seed(2031)
    model = build_backbone(
        "pam_iq_dual_query", num_classes=6, width=4, dropout=0.0
    )
    inputs = torch.randn(2, 2, 256, requires_grad=True)
    logits, features = model(inputs, return_features=True)

    assert logits.shape == (2, 6)
    assert features.shape == (2, 32 + 6 * 32)
    assert model.feature_dim == 224
    assert model.architecture_options["pooling_mode"] == "label_query"
    assert model.architecture_options["query_dim"] == 32
    assert model.architecture_options["attention_floor"] == pytest.approx(0.2)
    logits.mean().backward()
    assert inputs.grad is not None
    assert model.global_pool.query_pool.queries.grad is not None
    assert model.classifier.query_weight.grad is not None


def test_pam_iq_positional_query_preserves_capacity_and_temporal_contract():
    torch.manual_seed(2032)
    content_only = build_backbone(
        "pam_iq_dual_query", num_classes=6, width=4, dropout=0.0
    )
    positional = build_backbone(
        "pam_iq_positional_query", num_classes=6, width=4, dropout=0.0
    )
    inputs = torch.randn(2, 2, 256, requires_grad=True)
    logits, features = positional(inputs, return_features=True)

    assert logits.shape == (2, 6)
    assert features.shape == (2, 32 + 6 * 32)
    assert positional.architecture_options["pooling_mode"] == "positional_label_query"
    assert count_trainable_parameters(positional) == count_trainable_parameters(
        content_only
    )
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_pam_iq_dual_query_supports_student_a_classifier_only_mode():
    backbone = build_backbone(
        "pam_iq_dual_query", num_classes=6, width=4, dropout=0.0
    )
    wrapper = StudentAModel(backbone, relation_dim=8)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.backbone.classifier.parameters():
        parameter.requires_grad_(True)

    logits = wrapper(torch.randn(2, 2, 256))
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in wrapper.backbone.classifier.parameters()
    )


def test_dfms_iq_keeps_complex_streams_until_length_375_and_matches_capacity():
    torch.manual_seed(2032)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.0)
    assert isinstance(model, DelayedFusionMultiScaleIQClassifier)
    inputs = torch.randn(2, 2, 3000)

    real, imag = model.forward_complex_features(inputs)
    temporal = model.forward_temporal_features(inputs)
    logits, features = model(inputs, return_features=True)

    assert real.shape == (2, 64, 375)
    assert imag.shape == (2, 64, 375)
    assert temporal.shape == (2, 128, 375)
    assert features.shape == (2, 128)
    assert logits.shape == (2, 6)
    assert model.architecture_options["delayed_fusion"] is True
    assert model.architecture_options["phase_lags"] == (1, 4, 16)
    assert 230_000 <= count_trainable_parameters(model) <= 300_000


def test_dfms_iq_multiscale_and_phase_residual_are_differentiable():
    torch.manual_seed(2033)
    model = build_backbone("dfms_iq", num_classes=4, width=4, dropout=0.0)
    inputs = torch.randn(2, 2, 256, requires_grad=True)

    logits = model(inputs)
    logits.square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.late_fusion.phase_scale.grad is not None
    first_branch = model.complex_encoder.stage3.branches[0]["conv"]
    assert first_branch.real_conv.weight.grad is not None
    assert first_branch.imag_conv.weight.grad is not None


def test_dfms_iq_no_phase_is_a_controlled_late_fusion_ablation():
    model = build_backbone(
        "dfms_iq_no_phase", num_classes=6, width=4, dropout=0.0
    )
    assert model.architecture_options["include_phase"] is False
    assert model.late_fusion.phase_projection is None
    assert model.late_fusion.phase_scale is None
    assert model(torch.randn(2, 2, 256)).shape == (2, 6)


def test_dfms_iq_exposes_matching_main_and_phase_free_expert_features():
    torch.manual_seed(29)
    model = build_backbone("dfms_iq", num_classes=6, width=4, dropout=0.0)
    model.eval()
    inputs = torch.randn(2, 2, 256)

    with torch.no_grad():
        expected = model.forward_features(inputs)
        main, phase_free = model.forward_feature_branches(inputs)

    assert main.shape == expected.shape == phase_free.shape == (2, 32)
    assert torch.allclose(main, expected, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(main, phase_free)


def test_dfms_iq_exposes_temporal_and_stage2_maps_without_changing_features():
    torch.manual_seed(31)
    model = build_backbone("dfms_iq", num_classes=6, width=4, dropout=0.0)
    model.eval()
    inputs = torch.randn(2, 2, 256)

    with torch.no_grad():
        expected = model.forward_features(inputs)
        maps = model.forward_structure_maps(inputs)

    assert maps["global"].shape == expected.shape == (2, 32)
    assert maps["temporal"].shape == (2, 32, 32)
    assert maps["stage2_real"].shape == maps["stage2_imag"].shape == (2, 8, 128)
    assert maps["stage3_real"].shape == maps["stage3_imag"].shape == (2, 12, 64)
    assert torch.allclose(maps["global"], expected, atol=1e-6, rtol=1e-5)


def test_dfms_iq_rejects_too_short_or_non_iq_inputs():
    model = build_backbone("dfms_iq", num_classes=6, width=4, dropout=0.0)
    with pytest.raises(ValueError, match=r"\[batch, 2, length\]"):
        model(torch.randn(2, 1, 256))
    with pytest.raises(ValueError, match="at least 8"):
        model(torch.randn(2, 2, 7))


def test_fixed_blur_pool_is_parameter_free_shared_channel_lowpass():
    blur = FixedBlurPool1d()
    assert sum(parameter.numel() for parameter in blur.parameters()) == 0
    assert blur.kernel.flatten().tolist() == pytest.approx(
        [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
    )
    constant = torch.ones(2, 3, 32)
    torch.testing.assert_close(blur(constant), constant)
    pattern = torch.tensor([1.0, 0.0, -1.0, 0.0] * 8).view(1, 1, -1)
    assert blur(pattern).abs().mean() < pattern.abs().mean()


def test_dfms_m1_anti_alias_preserves_shapes_parameters_and_gradients():
    torch.manual_seed(20806)
    baseline = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.0)
    candidate = build_backbone(
        "dfms_iq_anti_alias", num_classes=6, width=16, dropout=0.0
    )
    assert count_trainable_parameters(candidate) == count_trainable_parameters(baseline) == 257511
    assert candidate.architecture_options["anti_aliased_downsampling"] is True
    inputs = torch.randn(2, 2, 3000, requires_grad=True)
    maps = candidate.forward_structure_maps(inputs)
    logits = candidate.forward_classifier(maps["global"])
    assert maps["stage2_real"].shape == (2, 32, 1500)
    assert maps["stage3_real"].shape == (2, 48, 750)
    assert maps["temporal"].shape == (2, 128, 375)
    assert logits.shape == (2, 6)
    logits.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_dfms_m1_uses_identical_blur_for_real_and_imaginary_streams():
    model = build_backbone(
        "dfms_iq_anti_alias", num_classes=6, width=4, dropout=0.0
    )
    blur = model.complex_encoder.pre_downsample
    values = torch.randn(2, 4, 128)
    torch.testing.assert_close(blur(values), blur(values.clone()))


def test_csfp_m2_has_preregistered_shapes_bounds_and_parameter_count():
    torch.manual_seed(2080602)
    model = build_backbone("dfms_iq_csfp", num_classes=6, width=16, dropout=0.0)
    assert isinstance(model.cross_stage_pyramid, CrossStageComplexFeaturePyramid1d)
    assert model.architecture_options["cross_stage_pyramid"] is True
    assert count_trainable_parameters(model) == 310376
    inputs = torch.randn(2, 2, 3000, requires_grad=True)
    maps = model.forward_structure_maps(inputs)
    assert maps["stage2_real"].shape == (2, 32, 1500)
    assert maps["stage3_real"].shape == (2, 48, 750)
    assert maps["stage4_real"].shape == (2, 64, 375)
    assert maps["cross_stage_pyramid"].shape == maps["temporal"].shape == (2, 128, 375)
    assert float(model.cross_stage_pyramid.residual_scale.detach()) == 0.0
    logits = model(inputs)
    assert logits.shape == (2, 6)
    logits.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.cross_stage_pyramid.residual_scale.grad is not None


def test_csfp_m2_preserves_all_b0_initial_weights_under_same_seed():
    torch.manual_seed(73039)
    baseline = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    torch.manual_seed(73039)
    candidate = build_backbone("dfms_iq_csfp", num_classes=6, width=16, dropout=0.2)
    candidate_state = candidate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
    baseline.eval()
    candidate.eval()
    inputs = torch.randn(2, 2, 3000)
    with torch.no_grad():
        torch.testing.assert_close(candidate(inputs), baseline(inputs), rtol=0, atol=0)


def test_epb_m3_has_preregistered_scales_shape_and_parameter_count():
    torch.manual_seed(2080603)
    model = build_backbone("dfms_iq_epb", num_classes=6, width=16, dropout=0.0)
    assert isinstance(
        model.evidence_preserving_block, EvidencePreservingMultiScaleBlock1d
    )
    assert model.architecture_options["evidence_preserving_block"] is True
    assert count_trainable_parameters(model) == 313191
    assert [branch.kernel_size[0] for branch in model.evidence_preserving_block.branches] == [3, 15, 31]
    assert [branch.dilation[0] for branch in model.evidence_preserving_block.branches] == [1, 2, 4]
    inputs = torch.randn(2, 2, 3000, requires_grad=True)
    temporal = model.forward_temporal_features(inputs)
    assert temporal.shape == (2, 128, 375)
    logits = model.forward_classifier(
        model.feature_projection(model.global_pool(temporal))
    )
    assert logits.shape == (2, 6)
    logits.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.evidence_preserving_block.fusion[0].weight.grad is not None


def test_epb_m3_preserves_all_b0_initial_weights_under_same_seed():
    torch.manual_seed(73039)
    baseline = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    torch.manual_seed(73039)
    candidate = build_backbone("dfms_iq_epb", num_classes=6, width=16, dropout=0.2)
    candidate_state = candidate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "topk"),
    (("dfms_iq_local_topk8", 8), ("dfms_iq_local_topk16", 16)),
)
def test_local_statistics_m4_preserves_interface_and_parameter_count(name, topk):
    torch.manual_seed(2080604)
    model = build_backbone(name, num_classes=6, width=16, dropout=0.0)
    assert isinstance(model.local_statistics_bypass, LocalEvidenceStatisticsBypass1d)
    assert model.local_statistics_bypass.topk == topk
    assert model.architecture_options["local_statistics_topk"] == topk
    assert count_trainable_parameters(model) == 290664
    inputs = torch.randn(2, 2, 3000, requires_grad=True)
    logits, features = model(inputs, return_features=True)
    assert logits.shape == (2, 6)
    assert features.shape == (2, 128)
    logits.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.local_statistics_bypass.residual_scale.grad is not None


def test_local_statistics_m4_preserves_all_b0_initial_weights_under_same_seed():
    torch.manual_seed(73039)
    baseline = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    torch.manual_seed(73039)
    candidate = build_backbone(
        "dfms_iq_local_topk8", num_classes=6, width=16, dropout=0.2
    )
    candidate_state = candidate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
    baseline.eval()
    candidate.eval()
    inputs = torch.randn(2, 2, 3000)
    with torch.no_grad():
        torch.testing.assert_close(candidate(inputs), baseline(inputs), rtol=0, atol=0)


def test_local_statistics_candidate_checkpoint_reload_is_exact(tmp_path):
    torch.manual_seed(2080605)
    model = build_backbone(
        "dfms_iq_local_topk16", num_classes=6, width=16, dropout=0.0
    )
    model.eval()
    inputs = torch.randn(2, 2, 3000)
    with torch.no_grad():
        expected_logits, expected_features = model(inputs, return_features=True)
    checkpoint = tmp_path / "m4_checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    reloaded = build_backbone(
        "dfms_iq_local_topk16", num_classes=6, width=16, dropout=0.0
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        actual_logits, actual_features = reloaded(inputs, return_features=True)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
