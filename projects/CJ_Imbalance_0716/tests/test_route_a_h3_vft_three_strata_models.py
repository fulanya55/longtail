import torch

from cjlt.models.multipulse_vft_expert import MultiPulseVFTExpertClassifier
from cjlt.models.vft_staggered_pri import record_samples


def test_h3_true_and_permuted_arms_are_capacity_matched_and_target_only():
    true_model = MultiPulseVFTExpertClassifier("T1-h3", width=8, expert_dim=4).eval()
    permuted_model = MultiPulseVFTExpertClassifier("T2-h3", width=8, expert_dim=4).eval()
    assert sum(parameter.numel() for parameter in true_model.parameters()) == sum(parameter.numel() for parameter in permuted_model.parameters())
    value = torch.randn(2, 2, record_samples())
    with torch.no_grad():
        pieces = true_model(value, return_components=True)
    assert torch.allclose(pieces["logits"][:, [0, 1, 2, 3, 5]], pieces["global_logits"][:, [0, 1, 2, 3, 5]])
    assert torch.allclose(pieces["logits"][:, 4], pieces["global_logits"][:, 4] + pieces["vft_expert_logit"])
