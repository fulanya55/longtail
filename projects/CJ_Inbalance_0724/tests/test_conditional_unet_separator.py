import torch

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator


def test_conditional_unet_initializes_as_passthrough_and_backpropagates():
    model = ConditionalComplexUNetSeparator(width=4, classifier_width=4, condition=True)
    iq = torch.randn(2, 2, 3000)
    output = model.forward_components(iq, torch.randn(2, 6))
    cosine = torch.nn.functional.cosine_similarity(
        output["normalized_estimate"].flatten(1), model.normalize_iq(iq).flatten(1), dim=1
    )
    assert output["logits"].shape == (2, 3)
    assert float(cosine.detach().min()) > 0.999
    output["logits"].square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
