import torch

from cj0724.p3ou_local_phase_dynamics import FEATURE_DIM, BoundedISRJResidual, LocalPhaseDynamics, corrected_logits, conditional_residual_loss


def test_local_phase_feature_contract_and_controls() -> None:
    torch.manual_seed(9)
    iq = torch.randn(3, 2, 3000)
    frontend = LocalPhaseDynamics()
    phase = frontend(iq, "phase")
    assert phase.shape == (3, FEATURE_DIM)
    assert torch.isfinite(phase).all()
    assert torch.equal(frontend(iq, "zero"), torch.zeros_like(phase))
    shuffled = frontend(iq, "shuffled")
    assert shuffled.shape == phase.shape
    assert not torch.equal(shuffled, phase)


def test_bounded_residual_only_changes_isrj() -> None:
    torch.manual_seed(10)
    logits = torch.randn(4, 6)
    features = torch.randn(4, FEATURE_DIM)
    residual = BoundedISRJResidual(bound=2.0)
    delta = residual(logits, features)
    corrected = corrected_logits(logits, delta)
    assert torch.all(delta <= 0)
    assert torch.all(delta >= -2.0)
    assert torch.equal(corrected[:, :5], logits[:, :5])
    assert torch.equal(corrected[:, 5], logits[:, 5] + delta)


def test_conditional_loss_preserves_mft_isrj_coexistence_bce() -> None:
    base = torch.zeros(3, 6)
    corrected = base.clone(); corrected[:, 5] = torch.tensor([-1.0, -1.0, 0.0])
    labels = torch.zeros(3, 6); labels[0, 3] = 1; labels[1, 3] = 1; labels[1, 5] = 1
    total, bce, conditional = conditional_residual_loss(corrected, base, labels, 0.25)
    assert torch.isfinite(total) and torch.isfinite(bce) and torch.isfinite(conditional)
    assert conditional.item() > 0
