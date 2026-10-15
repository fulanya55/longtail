import torch

from cj0724.p3ob_dualview_representation import (
    METHODS,
    DualViewMultiTask,
    delay_peak_sequence,
    local_time_frequency,
)


def test_physical_views_have_fixed_finite_shapes() -> None:
    iq = torch.randn(3, 2, 3000)
    delay = delay_peak_sequence(iq)
    time_frequency = local_time_frequency(iq)
    assert delay.shape == (3, 3, 1024)
    assert time_frequency.shape[0:2] == (3, 2)
    assert torch.isfinite(delay).all() and torch.isfinite(time_frequency).all()


def test_all_p3ob_controls_share_multitask_output_contract() -> None:
    iq = torch.randn(2, 2, 3000)
    for method in METHODS:
        logits, parameters, embedding = DualViewMultiTask(method).forward_with_embedding(iq)
        assert logits.shape == (2, 6)
        assert parameters.shape == (2, 6, 2)
        assert embedding.shape == (2, 64)


def test_exclusive_contrast_ignores_mft_isrj_cooccurrence() -> None:
    model = DualViewMultiTask("DF_exclusive_contrast")
    embedding = torch.randn(2, 64)
    labels = torch.zeros(2, 6)
    labels[:, 3] = 1.0
    labels[:, 5] = 1.0
    loss = model.exclusive_contrast_loss(embedding, labels)
    assert float(loss) == 0.0
    labels[0, 5] = 0.0
    labels[1, 3] = 0.0
    assert float(model.exclusive_contrast_loss(embedding, labels).detach()) > 0.0
