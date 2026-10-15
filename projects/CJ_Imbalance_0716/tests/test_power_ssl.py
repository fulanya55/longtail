import torch

from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.power_ssl import (
    PowerContrastiveModel,
    multi_positive_contrastive_loss,
    symmetric_contrastive_loss,
)


def test_power_contrastive_model_and_loss_are_finite():
    torch.manual_seed(2026)
    backbone = LightweightComplexIQClassifier(num_classes=4, width=4, dropout=0.0)
    model = PowerContrastiveModel(backbone, projection_dim=8, hidden_dim=16)
    first = torch.randn(6, 2, 64)
    second = first + 0.01 * torch.randn_like(first)
    features, first_projection = model(first)
    _, second_projection = model(second)
    loss = symmetric_contrastive_loss(first_projection, second_projection)
    loss.backward()
    assert features.shape == (6, backbone.feature_dim)
    assert first_projection.shape == (6, 8)
    assert torch.isfinite(loss)


def test_multi_positive_loss_and_power_gap_head_support_repeated_identities():
    torch.manual_seed(2031)
    backbone = LightweightComplexIQClassifier(num_classes=4, width=4, dropout=0.0)
    model = PowerContrastiveModel(backbone, projection_dim=8, hidden_dim=16)
    first = torch.randn(6, 2, 64)
    second = torch.randn(6, 2, 64)
    first_features, first_projection = model(first)
    second_features, second_projection = model(second)
    identities = torch.tensor([10, 10, 20, 20, 30, 30])
    loss = multi_positive_contrastive_loss(
        first_projection, second_projection, identities, temperature=0.2
    )
    prediction = model.predict_power_gap(first_features, second_features)
    assert torch.isfinite(loss)
    assert prediction.shape == (6,)
