import torch

from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.student_b import StudentBModel, count_student_b_deployment_parameters


def test_student_b_model_exposes_classifier_relations_and_optional_parameter_heads():
    torch.manual_seed(2026)
    backbone = LightweightComplexIQClassifier(num_classes=4, width=4, dropout=0.0)
    model = StudentBModel(
        backbone,
        relation_dim=12,
        parameters_per_label=2,
        enable_parameter_heads=True,
        relation_dropout=0.0,
    )
    signal = torch.randn(3, 2, 64)

    logits = model(signal)
    auxiliary = model(signal, return_aux=True)

    assert logits.shape == (3, 4)
    assert auxiliary["label_embeddings"].shape == (3, 4, 12)
    assert auxiliary["parameter_predictions"].shape == (3, 4, 2)
    assert count_student_b_deployment_parameters(model) == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_student_b_without_parameter_heads_deploys_only_original_classifier_path():
    backbone = LightweightComplexIQClassifier(num_classes=4, width=4, dropout=0.0)
    model = StudentBModel(
        backbone,
        relation_dim=12,
        enable_parameter_heads=False,
    )
    auxiliary = model(torch.randn(2, 2, 64), return_aux=True)

    assert auxiliary["parameter_predictions"] is None
    assert count_student_b_deployment_parameters(model) == sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )

