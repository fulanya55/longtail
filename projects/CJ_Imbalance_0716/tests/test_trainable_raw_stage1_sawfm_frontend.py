import torch

from cjlt.models.evidence_readout import RawStage1SawFMFrontend


def test_raw_stage1_frontend_forward_backward_and_parameter_budget():
    model = RawStage1SawFMFrontend(mode="raw_stage1", hidden_channels=16, topk=16)
    raw_real = torch.randn(3, 1, 3000)
    raw_imag = torch.randn(3, 1, 3000)
    stage1_real = torch.randn(3, 16, 750)
    stage1_imag = torch.randn(3, 16, 750)
    logits, features = model(
        raw_real, raw_imag, stage1_real, stage1_imag, return_features=True
    )
    assert logits.shape == (3,)
    assert features.shape == (3, 176)
    logits.mean().backward()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count < 20000
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_frontend_modes_require_only_their_selected_inputs():
    raw_real = torch.randn(2, 1, 300)
    raw_imag = torch.randn(2, 1, 300)
    stage1_real = torch.randn(2, 16, 300)
    stage1_imag = torch.randn(2, 16, 300)
    assert RawStage1SawFMFrontend(mode="raw")(raw_real, raw_imag).shape == (2,)
    assert RawStage1SawFMFrontend(mode="stage1")(
        raw_real, raw_imag, stage1_real, stage1_imag
    ).shape == (2,)


def test_descriptor_ablations_preserve_shape_and_change_features():
    real = torch.randn(2, 1, 300)
    imag = torch.randn(2, 1, 300)
    full = RawStage1SawFMFrontend(mode="raw", descriptor_mode="full")._descriptors(real, imag)
    envelope = RawStage1SawFMFrontend(mode="raw", descriptor_mode="envelope")._descriptors(real, imag)
    phase = RawStage1SawFMFrontend(mode="raw", descriptor_mode="phase")._descriptors(real, imag)
    assert full.shape == envelope.shape == phase.shape == (2, 6, 300)
    assert torch.count_nonzero(envelope[:, 4:]) == 0
    assert torch.count_nonzero(phase[:, :4]) == 0
