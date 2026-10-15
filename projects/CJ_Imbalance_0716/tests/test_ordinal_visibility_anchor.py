import numpy as np
import torch

from experiments.run_route_a_ordinal_visibility_anchor import (
    _ordinal_loss,
    _ordinal_metrics,
)


def test_ordinal_loss_rewards_correct_visibility_order():
    ordered = torch.tensor([[1.0, 0.4, -0.3, -1.0]])
    reversed_values = torch.flip(ordered, dims=[1])
    assert _ordinal_loss(ordered) < _ordinal_loss(reversed_values)


def test_ordinal_metrics_report_perfect_and_reversed_rankings():
    ordered = np.asarray([[2.0, 1.0, 0.0, -1.0], [4.0, 3.0, 2.0, 1.0]])
    perfect = _ordinal_metrics(ordered)
    reversed_metrics = _ordinal_metrics(ordered[:, ::-1])
    assert perfect["pairwise_accuracy"] == 1.0
    assert perfect["kendall_tau"] == 1.0
    assert reversed_metrics["pairwise_accuracy"] == 0.0
    assert reversed_metrics["kendall_tau"] == -1.0
