import torch
from torch.utils.data import DataLoader, Dataset

from cjlt.engine.student_b_trainer import train_student_b
from cjlt.losses.classification import build_classification_loss
from cjlt.losses.parameter import ParameterNormalizer
from cjlt.models.backbone_adapter import LightweightComplexIQClassifier
from cjlt.models.student_b import StudentBModel


class SyntheticLabeledParameterDataset(Dataset):
    def __init__(self, samples=12, length=64):
        generator = torch.Generator().manual_seed(2026)
        self.signal = torch.randn(samples, 2, length, generator=generator)
        self.label = torch.stack(
            [
                (self.signal[:, 0].mean(dim=1) > 0).float(),
                (self.signal[:, 1].mean(dim=1) > 0).float(),
            ],
            dim=1,
        )
        self.parameter = torch.stack(
            [
                torch.stack([self.signal[:, 0].std(dim=1), self.signal[:, 0].abs().mean(dim=1)], dim=1),
                torch.stack([self.signal[:, 1].std(dim=1), self.signal[:, 1].abs().mean(dim=1)], dim=1),
            ],
            dim=1,
        )

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        return {
            "signal": self.signal[index],
            "label": self.label[index],
            "parameter_target": self.parameter[index],
            "parameter_target_mask": torch.ones(2, 2, dtype=torch.bool),
            "sample_index": int(index),
        }


class SyntheticUnlabeledDataset(Dataset):
    def __init__(self, samples=16, length=64):
        generator = torch.Generator().manual_seed(2027)
        self.signal = torch.randn(samples, 2, length, generator=generator)

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        return {"signal": self.signal[index], "sample_index": int(index)}


def test_student_b_trainer_runs_daps_parameter_path(tmp_path):
    torch.manual_seed(2026)
    labeled = SyntheticLabeledParameterDataset()
    unlabeled = SyntheticUnlabeledDataset()
    labeled_loader = DataLoader(labeled, batch_size=4, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled, batch_size=4, shuffle=False)
    backbone = LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0)
    model = StudentBModel(
        backbone,
        relation_dim=8,
        parameters_per_label=2,
        enable_parameter_heads=True,
        relation_dropout=0.0,
    )
    normalizer = ParameterNormalizer().fit(
        labeled.parameter,
        torch.ones_like(labeled.parameter, dtype=torch.bool),
    )
    checkpoint = tmp_path / "student_b.pth"

    result = train_student_b(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        validation_loader=labeled_loader,
        calibration_loader=labeled_loader,
        classification_loss=build_classification_loss("bce"),
        parameter_normalizer=normalizer,
        parameter_names=["JNR", "shape", "JNR", "shape"],
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        checkpoint_path=checkpoint,
        use_ema_teacher=True,
        use_dynamic_threshold=True,
        use_prototype_gate=True,
        use_feature_consistency=True,
        use_parameter_heads=True,
        tau_pos=0.5,
        tau_neg=0.5,
        reliability_threshold=0.0,
        stability_ceiling=1.0,
        unsupervised_warmup_fraction=0.0,
        threshold_update_interval=1,
        selection_metric="macro_aupr",
        use_amp=False,
    )

    assert checkpoint.is_file()
    assert result.epochs_ran == 2
    assert len(result.history) == 2
    assert any(row["pseudo_coverage"] > 0.0 for row in result.history)
    assert all(torch.isfinite(torch.tensor(row["supervised_parameter_loss"])) for row in result.history)
    assert result.prototype_diagnostics["positive_initialized"] == 2
    assert result.final_thresholds["tau_pos"] != [0.5, 0.5]
    assert result.selection_metric == "macro_aupr"
    assert all("negative_count" in row for row in result.pseudo_label_statistics)
    assert all("probability_quantiles" in row for row in result.pseudo_label_statistics)
    assert all("confidence_candidate_count" in row for row in result.pseudo_label_statistics)
    assert all(row["labeled_examples"] == len(labeled) for row in result.history)
