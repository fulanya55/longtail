import torch
from torch import nn

from cjlt.models.dual_axis_sawfm import RoutedSawFMModel


class Core(nn.Module):
    def forward(self, signal):
        base = signal.mean(dim=(1, 2))
        return torch.stack([base + index for index in range(6)], dim=1)


class Expert(nn.Module):
    def forward(self, real, imag):
        return real.mean(dim=(1, 2)) - imag.mean(dim=(1, 2))


def test_routing_replaces_only_sawfm_column():
    signal = torch.randn(4, 2, 32)
    model = RoutedSawFMModel(Core(), Expert())
    routed, parts = model(signal, return_components=True)
    assert routed.shape == (4, 6)
    assert torch.equal(routed[:, 0], parts["core_logits"][:, 0])
    assert torch.equal(routed[:, 2:], parts["core_logits"][:, 2:])
    assert torch.equal(routed[:, 1], parts["sawfm_logit"])


def test_routing_rejects_invalid_signal_shape():
    model = RoutedSawFMModel(Core(), Expert())
    try:
        model(torch.randn(4, 3, 32))
    except ValueError as error:
        assert "[B,2,T]" in str(error)
    else:
        raise AssertionError("invalid input shape was accepted")
