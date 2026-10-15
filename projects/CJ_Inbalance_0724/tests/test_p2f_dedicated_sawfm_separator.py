import numpy as np
import pytest
import torch

from cj0724.p2f_dedicated_sawfm_separator import metric, separation_objective


def test_metric_reduction_inputs():
    result = metric(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    assert result["mae"] == 0.0
    assert result["pearson"] == pytest.approx(1.0)


def test_separation_objective_period_weight_changes_total():
    class Separator(torch.nn.Module):
        def forward_components(self, iq, _):
            return {"normalized_estimate": iq, "logits": torch.zeros(len(iq), 3)}

    class Expert(torch.nn.Module):
        def forward(self, iq):
            return {"prediction": torch.zeros(len(iq))}

    iq = torch.randn(2, 2, 2, 3000)
    target_iq = iq[:, 0]
    period = torch.ones(2)
    total0, _, _ = separation_objective(Separator(), Expert(), iq, target_iq, period, 0.0, 1.0, 0.0)
    total1, _, _ = separation_objective(Separator(), Expert(), iq, target_iq, period, 0.0, 1.0, 0.25)
    assert total1 > total0
