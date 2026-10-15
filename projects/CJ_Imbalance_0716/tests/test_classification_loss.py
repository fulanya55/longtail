import numpy as np
import pytest
import torch

from cjlt.losses.classification import (
    build_classification_loss,
    effective_number_weights,
)


def test_effective_number_weights_are_mean_normalized_and_capped():
    weights = effective_number_weights(
        positive_counts=[1000, 100, 10, 1],
        beta=0.99,
        minimum=0.5,
        maximum=8.0,
    )
    assert weights.shape == (4,)
    assert weights.mean() == pytest.approx(1.0, rel=1e-6)
    assert np.all(weights >= 0.5)
    assert np.all(weights <= 8.0)
    assert weights[-1] > weights[0]


@pytest.mark.parametrize("name", ["bce", "focal", "asl", "cb_bce", "cb_asl"])
def test_classification_losses_are_finite_and_backpropagate(name):
    logits = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 1.5, -1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    loss_fn = build_classification_loss(
        name,
        positive_counts=[100, 20, 5],
        gamma_pos=0.0,
        gamma_neg=3.0,
        clip=0.05,
        beta=0.99,
    )

    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_class_balanced_loss_requires_positive_counts():
    with pytest.raises(ValueError, match="positive_counts"):
        build_classification_loss("cb_asl")


def test_unknown_loss_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown classification loss"):
        build_classification_loss("dice")


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_asl_db_mix_is_finite_and_backpropagates(weight):
    logits = torch.randn(6, 4, requires_grad=True)
    targets = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    loss_fn = build_classification_loss(
        "asl_db_mix",
        positive_counts=[5, 3, 3, 3],
        sample_count=6,
        db_mix_weight=weight,
    )
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_asl_db_mix_matches_endpoints():
    logits = torch.tensor([[0.2, -0.3], [1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    common = {"positive_counts": [2, 1], "sample_count": 3}
    asl = build_classification_loss("asl", **common)(logits, targets)
    db = build_classification_loss("db", **common)(logits, targets)
    mix_asl = build_classification_loss(
        "asl_db_mix", db_mix_weight=0.0, **common
    )(logits, targets)
    mix_db = build_classification_loss(
        "asl_db_mix", db_mix_weight=1.0, **common
    )(logits, targets)
    assert mix_asl == pytest.approx(asl)
    assert mix_db == pytest.approx(db)


def test_asl_db_mix_rejects_invalid_weight():
    with pytest.raises(ValueError, match="db_weight"):
        build_classification_loss(
            "asl_db_mix",
            positive_counts=[5, 3],
            sample_count=6,
            db_mix_weight=1.1,
        )


def test_tail_asl_db_uses_db_only_for_registered_tail_column():
    logits = torch.tensor(
        [[0.2, -0.3, 0.7], [1.0, -1.0, -0.2]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    loss_fn = build_classification_loss(
        "tail_asl_db",
        positive_counts=[2, 1, 1],
        sample_count=3,
        tail_label_index=1,
        tail_loss_weight=1.5,
    )
    asl = loss_fn.asl.elementwise(logits, targets)
    db = loss_fn.db.elementwise(logits, targets)
    expected = asl.clone()
    expected[:, 1] = 1.5 * db[:, 1]
    loss = loss_fn(logits, targets)
    assert torch.allclose(loss, expected.mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_tail_asl_db_requires_valid_tail_configuration():
    common = {"positive_counts": [5, 3], "sample_count": 6}
    with pytest.raises(ValueError, match="tail_label_index"):
        build_classification_loss("tail_asl_db", **common)
    with pytest.raises(ValueError, match="tail_label_index"):
        build_classification_loss(
            "tail_asl_db", tail_label_index=2, **common
        )
    with pytest.raises(ValueError, match="tail_weight"):
        build_classification_loss(
            "tail_asl_db",
            tail_label_index=1,
            tail_loss_weight=0.0,
            **common,
        )
