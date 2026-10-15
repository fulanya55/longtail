import numpy as np
import torch

from cjlt.engine.oracle_parameter_trainer import compute_specialist_metrics
from cjlt.models.oracle_parameter_specialists import (
    build_oracle_specialist,
    ordinal_count_loss,
)
from experiments.run_oracle_parameter_specialists import _target_stratified_limit


def test_mft_ordinal_models_are_ordered_finite_and_differentiable():
    torch.manual_seed(2041)
    for variant in ("MFT_RAW_ORD", "MFT_ACF_ORD"):
        model = build_oracle_specialist(variant, width=4)
        output = model(torch.randn(3, 2, 256))
        assert output["ordinal_logits"].shape == (3, 5)
        assert output["prediction"].shape == (3,)
        assert torch.isfinite(output["prediction"]).all()
        thresholds = model.ordinal_head.thresholds()
        assert torch.all(thresholds[1:] > thresholds[:-1])
        loss = ordinal_count_loss(
            output["ordinal_logits"], torch.tensor([3.0, 5.0, 8.0])
        )
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_vft_specialists_return_finite_scalar_predictions():
    torch.manual_seed(2041)
    signal = torch.randn(2, 2, 256)
    for variant in ("VFT_RAW", "VFT_STFT", "VFT_IFREQ"):
        model = build_oracle_specialist(variant, width=4)
        prediction = model(signal)["prediction"]
        assert prediction.shape == (2,)
        assert torch.isfinite(prediction).all()
        prediction.sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_specialist_metrics_compare_to_training_median():
    metrics = compute_specialist_metrics(
        predictions=np.asarray([3.0, 4.0, 7.0, 8.0]),
        targets=np.asarray([3.0, 5.0, 6.0, 8.0]),
        training_median=5.5,
        target_kind="ordinal",
    )
    assert metrics["mae"] == 0.5
    assert metrics["relative_mae_reduction_vs_constant"] > 0.5
    assert metrics["pearson_correlation"] > 0.9
    assert metrics["rounded_accuracy"] == 0.5


def test_target_stratified_limit_covers_all_mft_counts():
    indices = np.arange(60)
    target = np.repeat(np.arange(3, 9), 10).astype(np.float32)
    selected = _target_stratified_limit(
        indices, target, maximum=12, seed=2041, target_kind="ordinal"
    )
    assert len(selected) == 12
    assert set(target[selected].tolist()) == set(range(3, 9))
