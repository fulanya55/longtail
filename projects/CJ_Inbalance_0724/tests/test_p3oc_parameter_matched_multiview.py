import torch

from cj0724.p3oc_parameter_matched_multiview import (
    EXPECTED_PARAMETERS,
    METHODS,
    ParameterMatchedMultiView,
    parameter_count,
)


def test_p3oc_models_are_parameter_matched_before_development() -> None:
    counts = {method: parameter_count(ParameterMatchedMultiView(method)) for method in METHODS}
    assert counts == EXPECTED_PARAMETERS
    reference = counts["F_time_frequency_matched"]
    assert all(abs(value / reference - 1.0) <= 0.05 for value in counts.values())


def test_p3oc_models_preserve_multitask_contract() -> None:
    iq = torch.randn(2, 2, 3000)
    for method in METHODS:
        logits, parameters, embedding = ParameterMatchedMultiView(method).forward_with_embedding(iq)
        assert logits.shape == (2, 6)
        assert parameters.shape == (2, 6, 2)
        assert embedding.shape == (2, 64 if method == "DF_dual_matched" else {"T_time_matched": 72, "D_delay_matched": 72, "F_time_frequency_matched": 64}[method])
