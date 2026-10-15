import numpy as np
import torch

from cj0724.p3ot_mft_conditional_residual import (
    FixedMFTSupport,
    ISRJConditionalResidual,
    conditional_residual_loss,
    corrected_logits,
)


def test_p3ot_support_frontend_is_observable_finite_and_equal_dimensional() -> None:
    generator = torch.Generator().manual_seed(17)
    iq = torch.randn((3, 2, 3000), generator=generator)
    frontend = FixedMFTSupport()
    support = frontend(iq, "support")
    control = frontend(iq, "control")
    zero = frontend(iq, "zero")
    assert support.shape == control.shape == zero.shape == (3, 32)
    assert torch.isfinite(support).all() and torch.isfinite(control).all()
    assert torch.count_nonzero(zero) == 0
    assert not torch.allclose(support, control)


def test_p3ot_residual_preserves_all_non_isrj_logits_exactly() -> None:
    feature = torch.randn(4, 256)
    base = torch.randn(4, 6)
    support = torch.randn(4, 32)
    branch = ISRJConditionalResidual()
    delta = branch(feature, base, support)
    # The preregistered zero final layer gives an exact T0-equivalent start.
    assert torch.equal(delta, torch.zeros_like(delta))
    corrected = corrected_logits(base, delta)
    assert torch.equal(corrected[:, :5], base[:, :5])
    assert torch.equal(corrected[:, 5], base[:, 5])
    changed = corrected_logits(base, torch.ones(4))
    assert torch.equal(changed[:, :5], base[:, :5])
    assert torch.equal(changed[:, 5], base[:, 5] + 1)


def test_p3ot_conditional_penalty_applies_only_mft_positive_isrj_negative_rows() -> None:
    base = torch.zeros(3, 6)
    corrected = base.clone()
    corrected[:, 5] = torch.tensor([1.0, 1.0, 1.0])
    labels = torch.zeros(3, 6)
    labels[0, 3] = 1.0  # MFT-only: eligible
    labels[1, 3] = 1.0; labels[1, 5] = 1.0  # coexistence: not eligible
    # Row 2 is neither MFT nor ISRJ: not eligible.
    total, bce, conditional = conditional_residual_loss(corrected, base, labels, weight=0.25)
    expected = torch.nn.functional.softplus(torch.tensor(1.0))
    assert torch.allclose(conditional, expected)
    assert torch.allclose(total, bce + 0.25 * conditional)
