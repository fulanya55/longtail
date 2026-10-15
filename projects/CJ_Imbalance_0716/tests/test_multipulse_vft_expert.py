import torch

from cjlt.models.multipulse_vft_expert import MODES, MultiPulseVFTExpertClassifier
from cjlt.models.vft_staggered_pri import record_samples


def test_h2_modes_preserve_six_label_contract() -> None:
    iq = torch.randn(2, 2, record_samples())
    for mode in MODES:
        model = MultiPulseVFTExpertClassifier(mode, width=8, expert_dim=4).eval()
        with torch.inference_mode():
            result = model(iq, return_components=True)
        assert result["logits"].shape == result["global_logits"].shape == (2, 6)
        assert result["sawfm_expert_logit"].shape == result["vft_expert_logit"].shape == (2,)
        assert torch.isfinite(result["logits"]).all()


def test_c3_and_c4_have_equal_trainable_parameter_count() -> None:
    c3 = MultiPulseVFTExpertClassifier("C3-mp", width=8, expert_dim=4)
    c4 = MultiPulseVFTExpertClassifier("C4-mp", width=8, expert_dim=4)
    assert sum(p.numel() for p in c3.parameters()) == sum(p.numel() for p in c4.parameters())


def test_c3_expert_contributions_are_target_only_and_additive() -> None:
    iq = torch.randn(2, 2, record_samples())
    model = MultiPulseVFTExpertClassifier("C3-mp", width=8, expert_dim=4).eval()
    with torch.inference_mode():
        result = model(iq, return_components=True)
    assert torch.allclose(result["logits"][:, 4], result["global_logits"][:, 4] + result["vft_expert_logit"])
    assert torch.allclose(result["logits"][:, 1], result["global_logits"][:, 1] + result["sawfm_expert_logit"])
    for index in (0, 2, 3, 5):
        assert torch.allclose(result["logits"][:, index], result["global_logits"][:, index])
