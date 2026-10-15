import torch

from cj0724.physics_parameter_specialists import build_physics_specialist, ordinal_count_loss


def test_physics_specialists_preserve_batch_and_expose_sequence():
    iq = torch.randn(2, 2, 3000)
    for name in ("sawfm_period_sequence", "mft_delay_peak_ordinal", "vft_instantaneous_frequency"):
        output = build_physics_specialist(name, width=4)(iq)
        assert output["prediction"].shape == (2,)
        assert output["sequence"].shape[0] == 2
        assert torch.isfinite(output["prediction"]).all()


def test_mft_ordinal_loss_is_finite():
    output = build_physics_specialist("mft_delay_peak_ordinal", width=4)(torch.randn(3, 2, 3000))
    loss = ordinal_count_loss(output["ordinal_logits"], torch.tensor([3.0, 5.0, 8.0]))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
