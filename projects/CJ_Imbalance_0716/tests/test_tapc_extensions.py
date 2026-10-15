import numpy as np
import pytest
import torch

from cjlt.analysis.decision_utility import (
    candidate_actions,
    evaluate_decision_utility,
)
from cjlt.data.longtail_manifest import (
    build_mixed_longtail,
    compute_weak_component_mask,
)
from cjlt.data.receiver_perturbation import (
    ReceiverPerturbationConfig,
    apply_receiver_perturbation,
)
from cjlt.engine.combination_metrics import (
    compute_cardinality_metrics,
    compute_combination_metrics,
)
from cjlt.losses.paired_power import (
    directional_power_ordering_loss,
    unchanged_component_stability_loss,
)
from experiments.run_tapc import _target_label_counts, tapc_variant_matrix


def test_directional_ordering_is_zero_only_when_margin_is_satisfied():
    labels = torch.tensor([[1.0, 1.0]])
    stronger = torch.tensor([[30, 10]], dtype=torch.int16)
    good = directional_power_ordering_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        labels,
        stronger,
        reference_level=torch.tensor([30]),
        paired_level=torch.tensor([10]),
        margin=0.2,
    )
    bad = directional_power_ordering_loss(
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        labels,
        stronger,
        reference_level=torch.tensor([30]),
        paired_level=torch.tensor([10]),
        margin=0.2,
    )
    assert good.item() == pytest.approx(0.0)
    assert bad.item() > 0.0


def test_strong_only_ordering_does_not_push_down_weak_logit():
    weak = torch.tensor([[0.0]], requires_grad=True)
    strong = torch.tensor([[0.0]], requires_grad=True)
    loss = directional_power_ordering_loss(
        weak,
        strong,
        torch.ones(1, 1),
        torch.tensor([[30]], dtype=torch.int16),
        reference_level=torch.tensor([10]),
        paired_level=torch.tensor([30]),
        margin=0.2,
        mode="strong_only",
    )
    loss.backward()
    assert weak.grad is None or torch.allclose(weak.grad, torch.zeros_like(weak))
    assert strong.grad is not None
    assert strong.grad.item() < 0.0


def test_unchanged_component_stability_ignores_changed_labels():
    loss = unchanged_component_stability_loss(
        torch.tensor([[4.0, 4.0]]),
        torch.tensor([[-4.0, -4.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[0, 30]], dtype=torch.int16),
        reference_level=torch.tensor([30]),
        paired_level=torch.tensor([10]),
        tolerance=0.1,
    )
    assert loss.item() > 0.0


def test_combination_and_cardinality_metrics_report_strata():
    true = np.asarray([[1, 0], [1, 0], [1, 1]], dtype=np.uint8)
    pred = np.asarray([[1, 0], [0, 0], [1, 0]], dtype=np.uint8)
    combination = compute_combination_metrics(true, pred, ["A", "B"])
    cardinality = compute_cardinality_metrics(true, pred)
    assert combination["summary"]["combination_count"] == 2
    assert len(combination["per_combination"]) == 2
    assert {row["cardinality"] for row in cardinality["per_cardinality"]} == {1, 2}


def test_mixed_manifest_records_weak_counts_and_is_reproducible():
    labels = np.asarray(
        [[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]],
        dtype=np.uint8,
    )
    power = np.asarray(
        [[10, np.nan], [np.nan, 10], [20, 5], [10, np.nan], [np.nan, 10], [5, 20]],
        dtype=np.float32,
    )
    weak = compute_weak_component_mask(labels, power, 10)
    first = build_mixed_longtail(labels, range(6), power, 2, 2, 10, 7)
    repeat = build_mixed_longtail(labels, range(6), power, 2, 2, 10, 7)
    assert weak[2, 1]
    assert weak[5, 0]
    assert first.sha256 == repeat.sha256
    assert first.sample_id_sha256 == repeat.sample_id_sha256
    assert int(first.weak_label_counts.sum()) >= 1


def test_receiver_perturbation_and_decision_interface_are_explicitly_bounded():
    signal = torch.zeros(2, 32)
    signal[0] = 1.0
    shifted = apply_receiver_perturbation(
        signal,
        ReceiverPerturbationConfig(
            iq_gain_imbalance_db=1.0,
            quadrature_error_deg=2.0,
            sample_shift=2,
        ),
    )
    assert shifted.shape == signal.shape
    decision = candidate_actions(["FMB", "ISRJ"], ["ISRJ"], uncertainty=0.6)
    assert decision["risk_level"] == "high"
    assert "not real-radar" in decision["claim_boundary"]
    utility = evaluate_decision_utility(
        [["frequency_domain_filtering"]],
        [["frequency_domain_filtering"]],
    )
    assert utility["action_accuracy"] == pytest.approx(1.0)


def test_tapc_matrix_matches_preregistered_t0_to_t5_stages():
    variants = tapc_variant_matrix()
    assert {"T0", "T0P", "T1", "T2", "T2S", "T2O_V2", "T2D", "T2OD_V2", "T3", "T4", "T5"} == set(variants)
    assert variants["T0P"]["pair_views"] is True
    assert variants["T0P"]["ordering"] is False
    assert variants["T2"]["ordering"] is True
    assert variants["T2S"]["stability"] is True
    assert variants["T4"]["stability"] is True
    assert variants["T5"]["parameter"] is True


def test_controlled_label_targets_use_fixed_tail_anchor_without_clipping():
    labels = np.ones((2000, 4), dtype=np.uint8)
    target = _target_label_counts(
        labels,
        np.arange(len(labels)),
        imbalance_factor=4.0,
        rank_order=[1, 2, 3, 0],
        tail_count_anchor=400,
    )
    assert target.tolist() == [400, 1600, 1008, 635]


def test_controlled_label_targets_report_insufficient_positives():
    labels = np.ones((1000, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds available"):
        _target_label_counts(
            labels,
            np.arange(len(labels)),
            imbalance_factor=4.0,
            rank_order=[0, 1, 2, 3],
            tail_count_anchor=400,
        )
