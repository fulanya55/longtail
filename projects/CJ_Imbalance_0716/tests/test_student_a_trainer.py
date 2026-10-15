from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cjlt.engine.student_a_trainer import train_student_a
from cjlt.losses.classification import build_classification_loss
from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.student_a import (
    StudentAModel,
    configure_classifier_only_training,
    configure_last_block_classifier_training,
    configure_last_two_blocks_classifier_training,
)


class SyntheticPowerPairDataset(Dataset):
    def __init__(self, samples=16, length=64):
        generator = torch.Generator().manual_seed(2026)
        self.reference = torch.randn(samples, 2, length, generator=generator)
        self.paired = self.reference + 0.05 * torch.randn(
            samples, 2, length, generator=generator
        )
        self.labels = torch.stack(
            [
                (self.reference[:, 0].mean(dim=1) > 0).float(),
                (self.reference[:, 1].mean(dim=1) > 0).float(),
            ],
            dim=1,
        )

    def __len__(self):
        return len(self.reference)

    def __getitem__(self, index):
        stronger = torch.where(
            self.labels[index] > 0.5,
            torch.tensor([30, 10], dtype=torch.int16),
            torch.zeros(2, dtype=torch.int16),
        )
        return {
            "reference_signal": self.reference[index],
            "paired_signal": self.paired[index],
            "label": self.labels[index],
            "stronger_level_by_label": stronger,
            "reference_level": 30,
            "paired_level": 10,
            "relative_power_gap": 20.0,
            "identity": int(index),
        }


def test_student_a_trainer_runs_full_auxiliary_path_and_saves_memory(tmp_path):
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    backbone = LightweightComplexIQClassifier(
        num_classes=2,
        width=4,
        dropout=0.0,
    )
    model = StudentAModel(backbone, relation_dim=8, relation_dropout=0.0)
    checkpoint = tmp_path / "student_a.pth"

    result = train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        checkpoint_path=checkpoint,
        use_prototype=True,
        use_power_consistency=True,
        prototype_weight=0.2,
        power_weight=0.2,
        warmup_fraction=0.0,
        ramp_end_fraction=0.0,
        use_amp=False,
    )

    assert checkpoint.is_file()
    assert result.epochs_ran == 2
    assert len(result.history) == 2
    assert result.prototype_diagnostics["positive_initialized"] == 2
    assert result.prototype_diagnostics["negative_initialized"] == 2
    assert all("classification_loss" in row for row in result.history)
    assert all("prototype_loss" in row for row in result.history)
    assert all("power_loss" in row for row in result.history)
    assert all("directional_margin_mean" in row for row in result.history)
    assert all("unchanged_drift_p90" in row for row in result.history)
    assert all("prototype_positive_cosine_max_offdiag" in row for row in result.history)


def test_pair_view_baseline_uses_both_views_without_power_auxiliaries():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    result = train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=1,
        use_prototype=False,
        use_power_consistency=False,
        use_pair_views=True,
        use_power_distillation=False,
        use_directional_ordering=False,
        use_stability=False,
        use_amp=False,
    )
    row = result.history[0]
    assert row["paired_view_samples"] == 8
    assert row["power_distillation_loss"] == pytest.approx(0.0)
    assert row["power_ordering_loss"] == pytest.approx(0.0)


def test_ordering_can_stop_gradient_at_backbone_features():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    result = train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=1,
        use_prototype=False,
        use_pair_views=True,
        use_power_distillation=False,
        use_directional_ordering=True,
        ordering_gradient_mode="backbone_stop_gradient",
        ordering_mode="strong_only",
        warmup_fraction=0.0,
        ramp_end_fraction=0.0,
        use_amp=False,
    )
    row = result.history[0]
    assert torch.isfinite(torch.tensor(row["ordering_gradient_norm"]))
    assert row["ordering_gradient_norm"] > 0.0


def test_classifier_only_training_preserves_representation_state():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    trainable = configure_classifier_only_training(model)
    assert trainable == sum(
        parameter.numel() for parameter in model.backbone.classifier.parameters()
    )
    train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=1,
        use_prototype=False,
        use_pair_views=False,
        use_power_distillation=False,
        use_directional_ordering=False,
        use_stability=False,
        use_amp=False,
        freeze_representation=True,
    )
    after = model.state_dict()
    for key, value in before.items():
        if not key.startswith("backbone.classifier."):
            assert torch.equal(after[key], value), key
    assert any(
        not torch.equal(after[key], value)
        for key, value in before.items()
        if key.startswith("backbone.classifier.")
    )


def test_last_block_adaptation_preserves_early_representation_and_logs_l2sp():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    groups = configure_last_block_classifier_training(model)
    assert groups["total"] == groups["final_temporal_block"] + groups["classifier"]
    result = train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=1,
        use_prototype=False,
        use_pair_views=False,
        use_power_distillation=False,
        use_directional_ordering=False,
        use_stability=False,
        use_amp=False,
        trainable_scope="last_block_classifier",
        representation_learning_rate_scale=0.1,
        starting_point_weight=1e-3,
    )
    after = model.state_dict()
    for key, value in before.items():
        if not (
            key.startswith("backbone.temporal_backbone.5.")
            or key.startswith("backbone.classifier.")
        ):
            assert torch.equal(after[key], value), key
    assert any(
        not torch.equal(after[key], value)
        for key, value in before.items()
        if key.startswith("backbone.temporal_backbone.5.")
    )
    assert any(
        not torch.equal(after[key], value)
        for key, value in before.items()
        if key.startswith("backbone.classifier.")
    )
    assert result.history[0]["starting_point_loss"] >= 0.0
    assert "weighted_starting_point_loss" in result.history[0]


def test_tail_aupr_can_drive_checkpoint_selection():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    result = train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        use_prototype=False,
        use_pair_views=False,
        use_power_distillation=False,
        use_directional_ordering=False,
        use_stability=False,
        use_amp=False,
        selection_metric="tail_aupr",
        selection_tail_index=0,
    )
    assert result.selection_metric == "tail_aupr"
    assert all(torch.isfinite(torch.tensor(row["tail_aupr"])) for row in result.history)
    assert result.best_selection_value == pytest.approx(
        max(row["tail_aupr"] for row in result.history)
    )


def test_last_two_block_adaptation_preserves_early_temporal_state():
    torch.manual_seed(2026)
    dataset = SyntheticPowerPairDataset(samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = StudentAModel(
        LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0),
        relation_dim=8,
        relation_dropout=0.0,
    )
    before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    groups = configure_last_two_blocks_classifier_training(model)
    assert groups["total"] == (
        groups["penultimate_temporal_block"]
        + groups["final_temporal_block"]
        + groups["classifier"]
    )
    train_student_a(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        classification_loss=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=1,
        use_prototype=False,
        use_pair_views=False,
        use_power_distillation=False,
        use_directional_ordering=False,
        use_stability=False,
        use_amp=False,
        trainable_scope="last_two_blocks_classifier",
        representation_learning_rate_scale=0.1,
        penultimate_representation_learning_rate_scale=0.03,
    )
    after = model.state_dict()
    for key, value in before.items():
        if not (
            key.startswith("backbone.temporal_backbone.3.")
            or key.startswith("backbone.temporal_backbone.5.")
            or key.startswith("backbone.classifier.")
        ):
            assert torch.equal(after[key], value), key
    for prefix in (
        "backbone.temporal_backbone.3.",
        "backbone.temporal_backbone.5.",
        "backbone.classifier.",
    ):
        assert any(
            not torch.equal(after[key], value)
            for key, value in before.items()
            if key.startswith(prefix)
        )
