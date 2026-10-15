import numpy as np
import pytest
import torch

from cjlt.data.parameter_schema import get_parameter_specs
from cjlt.engine.supervised_joint_trainer import (
    JointPredictionOutput,
    compute_parameter_metrics,
    project_conflicting_gradients,
    parameter_weight_at_progress,
)
from cjlt.losses.parameter import (
    ParameterNormalizer,
    masked_parameter_smooth_l1,
    masked_parameter_smooth_l1_by_target,
)
from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.supervised_joint import SupervisedJointParameterModel
from experiments.run_supervised_joint_parameter import (
    _parameter_transfer_configuration,
    _stratified_limit,
)


def test_supervised_joint_model_blocks_only_parameter_gradient_from_backbone():
    torch.manual_seed(2041)
    backbone = LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0)
    model = SupervisedJointParameterModel(
        backbone,
        relation_dim=8,
        parameters_per_label=2,
        enable_parameter_heads=True,
        relation_dropout=0.0,
    )
    output = model(
        torch.randn(3, 2, 64),
        return_aux=True,
        detach_parameter_backbone=True,
    )
    output["parameter_predictions"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameter_heads.parameters())
    assert any(parameter.grad is not None for parameter in model.label_feature_bank.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_typed_and_spectral_parameter_heads_return_finite_outputs_and_gradients():
    torch.manual_seed(2041)
    backbone = LightweightComplexIQClassifier(num_classes=6, width=4, dropout=0.0)
    model = SupervisedJointParameterModel(
        backbone,
        relation_dim=8,
        parameters_per_label=2,
        enable_parameter_heads=True,
        use_mft_count_classifier=True,
        use_vft_spectral_head=True,
    )
    output = model(torch.randn(3, 2, 64), return_aux=True)
    assert output["mft_count_logits"].shape == (3, 6)
    assert output["parameter_predictions"].shape == (3, 6, 2)
    assert torch.isfinite(output["parameter_predictions"]).all()
    loss = output["mft_count_logits"].sum() + output["parameter_predictions"][:, 4, 1].sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.mft_count_head.parameters())
    assert any(parameter.grad is not None for parameter in model.vft_spectral_head.parameters())


def test_local_parameter_head_replaces_only_selected_prediction_and_reaches_backbone():
    torch.manual_seed(2041)
    signal = torch.randn(3, 2, 64)
    baseline = SupervisedJointParameterModel(
        LightweightComplexIQClassifier(num_classes=6, width=4, dropout=0.0),
        relation_dim=8,
        parameters_per_label=2,
        enable_parameter_heads=True,
    )
    selected = SupervisedJointParameterModel(
        LightweightComplexIQClassifier(num_classes=6, width=4, dropout=0.0),
        relation_dim=8,
        parameters_per_label=2,
        enable_parameter_heads=True,
        local_parameter_targets=[(0, 1)],
    )
    compatible = {
        key: value
        for key, value in selected.state_dict().items()
        if key in baseline.state_dict() and baseline.state_dict()[key].shape == value.shape
    }
    baseline.load_state_dict(compatible, strict=False)
    baseline.eval()
    selected.eval()
    original = baseline(signal, return_aux=True)["parameter_predictions"]
    modified = selected(signal, return_aux=True)["parameter_predictions"]
    assert modified.shape == original.shape
    unchanged = torch.ones_like(modified, dtype=torch.bool)
    unchanged[:, 0, 1] = False
    assert torch.allclose(modified[unchanged], original[unchanged])
    modified[:, 0, 1].sum().backward()
    assert any(parameter.grad is not None for parameter in selected.local_parameter_heads.parameters())
    assert any(parameter.grad is not None for parameter in selected.backbone.parameters())


def test_parameter_weight_schedule_has_warmup_and_linear_ramp():
    assert parameter_weight_at_progress(0.1, 0.1, 0.2, 0.4) == 0.0
    assert parameter_weight_at_progress(0.3, 0.1, 0.2, 0.4) == pytest.approx(0.05)
    assert parameter_weight_at_progress(0.6, 0.1, 0.2, 0.4) == 0.1


def test_target_loss_decomposition_preserves_original_global_loss():
    prediction = torch.tensor(
        [[[0.2, -0.1], [0.4, 0.8]], [[0.5, 0.3], [-0.2, 0.1]]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [float("nan"), float("nan")]]]
    )
    labels = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    mask = torch.tensor(
        [[[True, True], [True, True]], [[True, False], [False, False]]]
    )
    global_loss = masked_parameter_smooth_l1(prediction, target, labels, mask)
    decomposed = masked_parameter_smooth_l1_by_target(
        prediction, target, labels, mask
    )
    assert decomposed.shape == (2, 2)
    assert float(decomposed.sum().detach()) == pytest.approx(
        float(global_loss.detach())
    )


def test_conflicting_gradient_projection_removes_negative_component():
    task = [torch.tensor([-2.0, 1.0])]
    reference = [torch.tensor([1.0, 0.0])]
    projected, changed = project_conflicting_gradients(task, reference)
    assert changed is True
    assert torch.dot(projected[0], reference[0]) == pytest.approx(0.0)
    assert projected[0][1] == pytest.approx(1.0)


def test_transfer_configuration_blocks_only_failed_parameter_targets():
    names, mask = _parameter_transfer_configuration(
        ["MFT", "VFT"], [["MFT", "num_targets"], ["VFT", "Doppler"]]
    )
    assert names == ["MFT_JNR", "MFT_num_targets", "VFT_JNR", "VFT_Doppler"]
    assert mask == [[True, False], [True, False]]


def test_parameter_metrics_use_only_positive_valid_targets():
    normalizer = ParameterNormalizer().fit(
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.ones(2, 1, 2, dtype=torch.bool),
    )
    output = JointPredictionOutput(
        logits=np.zeros((2, 1), dtype=np.float32),
        targets=np.asarray([[1], [0]], dtype=np.uint8),
        sample_ids=np.asarray([0, 1], dtype=np.int64),
        parameter_predictions=np.asarray([[[2.0, 4.0]], [[100.0, 100.0]]]),
        parameter_targets=np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]]),
        parameter_masks=np.ones((2, 1, 2), dtype=bool),
    )
    metrics = compute_parameter_metrics(
        output,
        class_names=["FMB"],
        parameter_specs={"FMB": get_parameter_specs("FMB")},
        parameter_normalizer=normalizer,
        class_predictions=np.asarray([[1], [1]], dtype=np.uint8),
    )
    assert metrics["oracle_valid_values"] == 2
    assert metrics["system_parameter_coverage"] == 1.0
    assert metrics["per_target"][0]["mae"] == 1.0
    assert metrics["per_target"][1]["mae"] == 2.0


def test_count_metrics_report_rounded_error_and_accuracy():
    normalizer = ParameterNormalizer().fit(
        torch.tensor([[[1.0, 3.0]], [[2.0, 8.0]]]),
        torch.ones(2, 1, 2, dtype=torch.bool),
    )
    output = JointPredictionOutput(
        logits=np.zeros((2, 1), dtype=np.float32),
        targets=np.ones((2, 1), dtype=np.uint8),
        sample_ids=np.arange(2, dtype=np.int64),
        parameter_predictions=np.asarray([[[1.0, 3.4]], [[2.0, 6.8]]]),
        parameter_targets=np.asarray([[[1.0, 3.0]], [[2.0, 8.0]]]),
        parameter_masks=np.ones((2, 1, 2), dtype=bool),
    )
    specs = get_parameter_specs("MFT")
    metrics = compute_parameter_metrics(
        output,
        class_names=["MFT"],
        parameter_specs={"MFT": specs},
        parameter_normalizer=normalizer,
    )
    count_row = metrics["per_target"][1]
    assert count_row["rounded_mae"] == pytest.approx(0.5)
    assert count_row["rounded_accuracy"] == pytest.approx(0.5)


def test_stratified_limit_keeps_all_available_combinations_before_repeating():
    labels = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.uint8
    )
    selected = _stratified_limit(np.arange(6), labels, maximum=3, seed=2041)
    assert len(selected) == 3
    assert len({tuple(row) for row in labels[selected]}) == 3
