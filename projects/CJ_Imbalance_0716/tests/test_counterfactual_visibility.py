from __future__ import annotations

import torch

from cjlt.models.counterfactual_visibility import CounterfactualVisibilityProbe
from cjlt.models.counterfactual_delta_prototype import CounterfactualDeltaPrototypeProbe


def test_counterfactual_visibility_probe_is_single_view_six_logit_model():
    model = CounterfactualVisibilityProbe(width=8)
    value = model(torch.randn(3, 2, 3000))
    assert value.shape == (3, 6)
    assert torch.isfinite(value).all()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_counterfactual_visibility_probe_rejects_pair_metadata_as_input():
    model = CounterfactualVisibilityProbe(width=8)
    try:
        model(torch.randn(2, 3, 3000))
    except ValueError as error:
        assert "[B,2,T]" in str(error)
    else:
        raise AssertionError("invalid channel count must be rejected")


def test_counterfactual_delta_prototype_has_single_view_and_delta_contract():
    model=CounterfactualDeltaPrototypeProbe(width=8,projection_dim=8)
    positive=torch.randn(4,2,3000); negative=torch.randn(4,2,3000)
    output=model(positive,return_features=True)
    delta=torch.nn.functional.normalize(output['embedding']-model(negative,return_features=True)['embedding'],dim=1)
    assert output['logits'].shape==(4,6)
    assert (delta@output['prototypes'].T).shape==(4,3)
    assert torch.isfinite(output['logits']).all() and torch.isfinite(delta).all()
