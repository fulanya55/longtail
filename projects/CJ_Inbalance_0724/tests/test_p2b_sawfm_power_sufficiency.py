import numpy as np
import torch

from cj0724.p2b_sawfm_power_sufficiency import (
    paired_saw_loss,
    relative_mae_reduction,
    segment_metrics,
)


def test_paired_loss_zero_for_correct_power_invariant_prediction():
    prediction = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    target = torch.tensor([0.5, -0.5])
    total, supervised, consistency = paired_saw_loss(prediction, target, 0.0, 1.0, 0.05)
    assert torch.isclose(total, torch.tensor(0.0))
    assert torch.isclose(supervised, torch.tensor(0.0))
    assert torch.isclose(consistency, torch.tensor(0.0))


def test_segment_metrics_and_relative_reduction():
    target = np.asarray([1.0, 2.0])
    baseline_prediction = np.tile(np.asarray([[0.0], [3.0]]), (1, 9))
    candidate_prediction = np.tile(target[:, None], (1, 9))
    baseline = segment_metrics(baseline_prediction, target)
    candidate = segment_metrics(candidate_prediction, target)
    assert candidate["all"]["mae"] == 0.0
    assert relative_mae_reduction(candidate, baseline, "high_35_50") == 1.0


def test_segment_metrics_rejects_wrong_power_count():
    target = np.asarray([1.0, 2.0])
    prediction = np.zeros((2, 8))
    try:
        segment_metrics(prediction, target)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("wrong power count was accepted")
