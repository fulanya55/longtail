import torch

from cj0724.learned_swept_separator_experiment import ComplexSTFTSeparatorClassifier


def test_separator_forward_shapes_and_single_view_contract():
    model = ComplexSTFTSeparatorClassifier(width=4, classifier_width=4)
    output = model.forward_components(torch.randn(2, 2, 3000))
    assert output["logits"].shape == (2, 3)
    assert output["estimated_iq"].shape == (2, 2, 3000)
    assert output["mask"].ndim == 4
    assert torch.isfinite(output["logits"]).all()


def test_separator_initializes_as_finite_passthrough_and_backpropagates():
    model = ComplexSTFTSeparatorClassifier(width=4, classifier_width=4)
    iq = torch.randn(2, 2, 3000)
    output = model.forward_components(iq)
    cosine = torch.nn.functional.cosine_similarity(
        output["normalized_estimate"].flatten(1), model.normalize_iq(iq).flatten(1), dim=1
    )
    assert float(cosine.detach().min()) > 0.999
    loss = output["logits"].square().mean() + output["normalized_estimate"].square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
