import torch

from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.student_a import StudentAModel, count_deployment_parameters


def test_student_a_model_preserves_original_logits_and_exposes_label_embeddings():
    torch.manual_seed(2026)
    backbone = LightweightComplexIQClassifier(
        num_classes=4,
        width=4,
        dropout=0.0,
    )
    model = StudentAModel(backbone, relation_dim=12, relation_dropout=0.0)
    signal = torch.randn(3, 2, 64)

    logits = model(signal)
    auxiliary = model(signal, return_aux=True)

    assert logits.shape == (3, 4)
    assert auxiliary["logits"].shape == (3, 4)
    assert auxiliary["features"].shape == (3, backbone.feature_dim)
    assert auxiliary["label_embeddings"].shape == (3, 4, 12)
    assert torch.allclose(logits, auxiliary["logits"], atol=1e-6, rtol=1e-6)
    assert count_deployment_parameters(model) == sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    assert count_deployment_parameters(model) < sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
