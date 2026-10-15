import torch

from cj0724.p3a_isrj_time_domain_separator import TimeDomainComplexResidualSeparator


def test_time_domain_separator_preserves_shape_and_starts_as_identity():
    model = TimeDomainComplexResidualSeparator()
    values = torch.randn(3, 2, 3000)
    output = model.forward_components(values, None)
    assert output["normalized_estimate"].shape == values.shape
    expected_energy = torch.ones(3)
    actual_energy = output["normalized_estimate"].square().sum(1).mean(-1)
    assert torch.allclose(actual_energy, expected_energy, atol=1.0e-5)


def test_time_domain_separator_is_smaller_than_stft_unet():
    from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator

    candidate = sum(parameter.numel() for parameter in TimeDomainComplexResidualSeparator().parameters())
    baseline = sum(parameter.numel() for parameter in ConditionalComplexUNetSeparator(
        width=12, classifier_width=12, condition=False
    ).parameters())
    assert candidate < 0.25 * baseline
