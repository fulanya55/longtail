import numpy as np
import torch

from cj0724.multitask import (
    PARAMETER_COLUMNS,
    ParameterNormalizer,
    build_multitask_model,
    masked_parameter_smooth_l1,
    parameter_metrics,
    reshape_selected_parameters,
    raw_iq_physical_features,
    raw_iq_spectral_features,
)
from cj0724.multitask_benchmark import equal_combo_limit, make_combination_stratified_split
from cj0724.tuned_multitask_sweep import (
    make_five_way_combination_split,
    optimize_class_thresholds,
    parameter_loss_target_weights,
)


def test_multitask_model_has_common_output_shapes():
    iq = torch.randn(2, 2, 3000)
    for name in ("complex1d", "iq_grouped"):
        model = build_multitask_model(name, width=8, head_dim=32, parameter_hidden_dim=16).eval()
        with torch.no_grad():
            logits, parameters = model(iq)
        assert logits.shape == (2, 6)
        assert parameters.shape == (2, 6, 2)


def test_parameter_columns_and_reshape_follow_six_by_two_schema():
    assert PARAMETER_COLUMNS == (0, 2, 11, 14, 16, 18, 27, 28, 48, 49, 52, 53)
    values = torch.arange(24).reshape(2, 12)
    assert reshape_selected_parameters(values).shape == (2, 6, 2)


def test_masked_loss_ignores_negative_classes_and_invalid_targets():
    prediction = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = masked_parameter_smooth_l1(prediction, target, labels, mask)
    expected = torch.nn.functional.smooth_l1_loss(torch.zeros(2), torch.tensor([1.0, 2.0]))
    assert torch.allclose(loss, expected)


def test_weighted_masked_loss_normalizes_by_valid_weight_sum():
    prediction = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    labels = torch.ones(1, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    element = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    expected = (element * weights).sum() / weights.sum()
    actual = masked_parameter_smooth_l1(prediction, target, labels, mask, weights)
    assert torch.allclose(actual, expected)


def test_weighted_masked_loss_never_touches_inactive_nan_targets():
    prediction = torch.zeros(1, 2, 2, requires_grad=True)
    target = torch.tensor([[[1.0, 2.0], [float("nan"), float("nan")]]])
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[[True, True], [False, False]]])
    weights = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    loss = masked_parameter_smooth_l1(prediction, target, labels, mask, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[:, 1], torch.zeros_like(prediction.grad[:, 1]))


def test_p4g_loss_alias_changes_only_sawfm_period_training_weight():
    baseline = parameter_loss_target_weights("mlcnn_iq_grouped_task_routed")
    candidate = parameter_loss_target_weights("mlcnn_iq_grouped_task_routed_saw_loss2")
    assert torch.equal(baseline, torch.ones(6, 2))
    assert candidate[1, 1].item() == 2.0
    candidate = candidate.clone()
    candidate[1, 1] = 1.0
    assert torch.equal(candidate, baseline)


def test_p4g_loss_alias_has_identical_inference_state_schema_and_size():
    baseline = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=8, head_dim=32, parameter_hidden_dim=16
    )
    candidate = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_saw_loss2", width=8, head_dim=32,
        parameter_hidden_dim=16,
    )
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )


def test_normalizer_uses_only_valid_values():
    values = torch.tensor([[[1.0, 10.0]], [[3.0, 999.0]]])
    valid = torch.tensor([[[True, True]], [[True, False]]])
    normalizer = ParameterNormalizer().fit(values, valid)
    assert torch.allclose(normalizer.mean, torch.tensor([[2.0, 10.0]]))
    assert torch.allclose(normalizer.scale, torch.tensor([[1.0, 1.0e-6]]))


def test_combination_split_and_limit_keep_every_combination():
    labels = np.repeat(np.eye(3, dtype=np.uint8), 10, axis=0)
    train, val = make_combination_stratified_split(labels, 0.2, 7)
    limited = equal_combo_limit(train, labels, 2, 8)
    assert len(train) == 24 and len(val) == 6 and len(limited) == 6
    assert len(np.unique(labels[limited], axis=0)) == 3


def test_five_way_split_is_disjoint_and_complete():
    labels = np.repeat(np.eye(3, dtype=np.uint8), 400, axis=0)
    roles = make_five_way_combination_split(labels, 9)
    combined = np.concatenate(list(roles.values()))
    assert len(combined) == len(labels)
    assert len(np.unique(combined)) == len(labels)
    assert {role: len(indices) for role, indices in roles.items()} == {
        "train": 900, "early": 120, "calibration": 60, "tune": 60, "confirm": 60
    }


def test_adaptive_thresholds_are_fitted_per_class():
    probabilities = np.array([[0.4, 0.1], [0.45, 0.8], [0.3, 0.7], [0.2, 0.2]])
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    thresholds = optimize_class_thresholds(probabilities, labels, np.array([0.25, 0.35, 0.5, 0.75]))
    assert np.allclose(thresholds, np.array([0.35, 0.5]))


def test_legacy_backbone_can_use_common_parameter_head():
    model = build_multitask_model("legacy_mlcnn", head_dim=32, parameter_hidden_dim=16).eval()
    with torch.no_grad():
        logits, parameters = model(torch.randn(1, 2, 3000))
    assert logits.shape == (1, 6)
    assert parameters.shape == (1, 6, 2)


def test_mlcnn_parameter_input_variants_keep_common_outputs():
    iq = torch.randn(2, 2, 3000)
    for name in (
        "legacy_mlcnn_global",
        "legacy_mlcnn_shallow_multiscale",
        "legacy_mlcnn_class_conditioned",
        "legacy_mlcnn_shallow_multiscale_detached",
        "legacy_mlcnn_class_conditioned_detached",
    ):
        model = build_multitask_model(name, head_dim=32, parameter_hidden_dim=16).eval()
        with torch.no_grad():
            logits, parameters = model(iq)
        assert logits.shape == (2, 6)
        assert parameters.shape == (2, 6, 2)


def test_parameter_metrics_report_pearson_direction():
    target = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    prediction = target * 2.0 + 1.0
    valid = np.ones_like(target, dtype=bool)
    normalizer = ParameterNormalizer().fit(torch.tensor(target), torch.tensor(valid))
    metrics = parameter_metrics(prediction, target, valid, normalizer)
    assert np.isclose(metrics["oracle_macro_pearson_r"], 1.0)


def test_task_routed_model_combines_mlcnn_classification_and_iq_parameter_paths():
    model = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=8, head_dim=32, parameter_hidden_dim=16
    ).eval()
    with torch.no_grad():
        logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)


def test_physical_feature_bank_is_26d_and_finite_for_current_iq_contract():
    features = raw_iq_physical_features(torch.randn(3, 2, 3000))
    assert features.shape == (3, 26)
    assert torch.isfinite(features).all()


def test_spectral_feature_bank_and_film_route_are_iq_only_and_finite():
    features = raw_iq_spectral_features(torch.randn(3, 2, 3000))
    assert features.shape == (3, 19)
    assert torch.isfinite(features).all()
    model = build_multitask_model(
        "mlcnn_iq_grouped_spectral_film_task_routed", width=8, head_dim=32,
        parameter_hidden_dim=16,
    ).eval()
    with torch.no_grad():
        logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)


def test_physical_film_route_preserves_common_outputs_and_blocks_parameter_gradient_to_classifier():
    model = build_multitask_model(
        "mlcnn_iq_grouped_physical_film_task_routed", width=8, head_dim=32,
        parameter_hidden_dim=16,
    )
    logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)
    parameters.square().mean().backward()
    assert model.classification_backbone.backbone.conv1.weight.grad is None
    assert model.condition_encoder[-1].weight.grad is not None
    assert torch.isfinite(model.condition_encoder[-1].weight.grad).all()


def test_p6b_temporal_query_and_global_control_have_exactly_matched_parameter_count():
    query = build_multitask_model(
        "mlcnn_iq_grouped_temporal_query_task_routed", width=8, head_dim=32,
        parameter_hidden_dim=16,
    )
    control = build_multitask_model(
        "mlcnn_iq_grouped_temporal_global_control_task_routed", width=8, head_dim=32,
        parameter_hidden_dim=16,
    )
    assert sum(parameter.numel() for parameter in query.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    assert query.global_control is False
    assert control.global_control is True


def test_p6b_temporal_query_route_has_common_outputs_and_protects_mlcnn_from_parameter_loss():
    model = build_multitask_model(
        "mlcnn_iq_grouped_temporal_query_task_routed", width=8, head_dim=32,
        parameter_hidden_dim=16,
    )
    logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)
    parameters.square().mean().backward()
    assert model.classification_backbone.backbone.conv1.weight.grad is None
    assert model.parameter_head.queries.grad is not None
    assert torch.isfinite(model.parameter_head.queries.grad).all()


def test_p4h_compact_route_changes_only_iq_grouped_parameter_width():
    compact = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_w4", width=24, head_dim=32, parameter_hidden_dim=16
    ).eval()
    explicit = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=4, head_dim=32, parameter_hidden_dim=16
    ).eval()
    full = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=24, head_dim=32, parameter_hidden_dim=16
    ).eval()
    assert compact.classification_backbone.state_dict().keys() == full.classification_backbone.state_dict().keys()
    assert compact.parameter_backbone.state_dict().keys() == explicit.parameter_backbone.state_dict().keys()
    assert sum(parameter.numel() for parameter in compact.parameter_backbone.parameters()) == sum(
        parameter.numel() for parameter in explicit.parameter_backbone.parameters()
    )
    assert sum(parameter.numel() for parameter in compact.parameter_backbone.parameters()) < sum(
        parameter.numel() for parameter in full.parameter_backbone.parameters()
    )


def test_p4h_compact_route_keeps_common_multitask_outputs():
    model = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_w4", width=24, head_dim=32, parameter_hidden_dim=16
    ).eval()
    with torch.no_grad():
        logits, parameters = model(torch.randn(2, 2, 3000))
    assert logits.shape == (2, 6)
    assert parameters.shape == (2, 6, 2)


def test_p5a_shared_temporal_route_preserves_sequence_and_output_contract():
    model = build_multitask_model(
        "mlcnn_shared_temporal_task_routed", width=24, head_dim=32,
        parameter_hidden_dim=16,
    )
    sample = torch.randn(3, 2, 3000)
    captured = {}

    def remember_shape(_module, inputs, _output):
        captured["shape"] = tuple(inputs[0].shape)

    hook = model.parameter_temporal.register_forward_hook(remember_shape)
    logits, parameters = model(sample)
    hook.remove()
    assert captured["shape"] == (3, 128, 750)
    assert logits.shape == (3, 6)
    assert parameters.shape == (3, 6, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(parameters).all()


def test_p5a_shared_temporal_route_backpropagates_through_shared_stage2():
    model = build_multitask_model(
        "mlcnn_shared_temporal_task_routed", width=24, head_dim=32,
        parameter_hidden_dim=16,
    )
    _logits, parameters = model(torch.randn(2, 2, 3000))
    parameters.square().mean().backward()
    gradient = model.mlcnn.conv2.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_p5a2_detached_shared_temporal_route_protects_shared_stage2():
    model = build_multitask_model(
        "mlcnn_shared_temporal_detached_task_routed", width=24, head_dim=32,
        parameter_hidden_dim=16,
    )
    _logits, parameters = model(torch.randn(2, 2, 3000))
    parameters.square().mean().backward()
    assert model.mlcnn.conv1.weight.grad is None
    assert model.mlcnn.conv2.weight.grad is None
    branch_gradient = model.parameter_temporal[0].weight.grad
    assert branch_gradient is not None
    assert torch.isfinite(branch_gradient).all()
    assert float(branch_gradient.abs().sum()) > 0.0


def test_p4l_budget_route_uses_width_16_parameter_backbone():
    candidate = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_w16", width=24, head_dim=32, parameter_hidden_dim=16
    ).eval()
    explicit = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=16, head_dim=32, parameter_hidden_dim=16
    ).eval()
    full = build_multitask_model(
        "mlcnn_iq_grouped_task_routed", width=24, head_dim=32, parameter_hidden_dim=16
    ).eval()
    assert candidate.parameter_backbone.state_dict().keys() == explicit.parameter_backbone.state_dict().keys()
    assert sum(parameter.numel() for parameter in candidate.parameter_backbone.parameters()) == sum(
        parameter.numel() for parameter in explicit.parameter_backbone.parameters()
    )
    assert sum(parameter.numel() for parameter in candidate.parameter_backbone.parameters()) < sum(
        parameter.numel() for parameter in full.parameter_backbone.parameters()
    )


def test_detached_shallow_parameter_path_does_not_backpropagate_to_mlcnn_convolutions():
    model = build_multitask_model(
        "legacy_mlcnn_shallow_multiscale_detached", head_dim=32, parameter_hidden_dim=16
    )
    _, parameters = model(torch.randn(2, 2, 3000))
    parameters.sum().backward()
    assert model.mlcnn.conv1.weight.grad is None
    assert any(parameter.grad is not None for parameter in model.parameter_head.parameters())


def test_saw_safe_route_detaches_fallback_from_classification_backbone():
    model = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_saw_safe", width=8, head_dim=32, parameter_hidden_dim=16
    )
    _, parameters = model(torch.randn(2, 2, 3000))
    parameters.sum().backward()
    assert parameters.shape == (2, 6, 2)
    assert all(parameter.grad is None for parameter in model.classification_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.parameter_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.saw_period_safe_head.parameters())


def test_shallow_saw_safe_route_detaches_local_fallback_from_mlcnn():
    model = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_saw_shallow_safe",
        width=8,
        head_dim=32,
        parameter_hidden_dim=16,
    )
    _, parameters = model(torch.randn(2, 2, 3000))
    parameters.sum().backward()
    assert parameters.shape == (2, 6, 2)
    assert model.mlcnn.conv1.weight.grad is None
    assert any(parameter.grad is not None for parameter in model.parameter_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.saw_period_safe_head.parameters())
