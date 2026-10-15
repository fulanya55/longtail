from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cjlt.data.augmentations import strong_complex_augmentation, weak_complex_augmentation
from cjlt.data.semi_supervised_split import (
    HiddenLabelDataset,
    build_nested_labeled_splits,
)
from cjlt.data.registry import select_labels, valid_sample_mask
from cjlt.data.split_builder import build_stratified_roles
from cjlt.data.v73_dataset import read_labels
from experiments.run_shared import load_config
from experiments.run_student_b import build_student_b_plan


TRAIN_PATH = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"
)


class TinyLabeledDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "signal": torch.ones(2, 32) * float(index + 1),
            "label": torch.tensor([1.0, 0.0]),
            "parameter_values": torch.ones(79),
            "parameter_mask": torch.ones(79, dtype=torch.bool),
            "sample_index": int(index),
        }


def test_nested_labeled_splits_are_stratified_and_monotonic():
    labels6 = read_labels(TRAIN_PATH)
    labels = select_labels(labels6, "four")
    eligible = np.flatnonzero(valid_sample_mask(labels6, "four"))
    roles = build_stratified_roles(labels, eligible, seed=2026)

    split = build_nested_labeled_splits(
        labels,
        roles.optimization,
        ratios=[0.05, 0.10, 0.20, 0.50, 1.00],
        seed=2026,
    )

    labeled_5 = set(split.labeled[0.05].tolist())
    labeled_10 = set(split.labeled[0.10].tolist())
    labeled_20 = set(split.labeled[0.20].tolist())
    labeled_50 = set(split.labeled[0.50].tolist())
    assert labeled_5 < labeled_10 < labeled_20 < labeled_50
    assert len(labeled_5) == 210
    assert len(labeled_20) == 840
    assert len(split.labeled[1.00]) == 4200
    assert set(split.labeled[0.20]).isdisjoint(set(split.unlabeled[0.20]))
    assert set(split.labeled[0.20]) | set(split.unlabeled[0.20]) == set(
        roles.optimization.tolist()
    )
    assert len(split.sha256) == 64


def test_hidden_label_dataset_never_returns_training_truth():
    hidden = HiddenLabelDataset(TinyLabeledDataset())

    item = hidden[0]

    assert "signal" in item
    assert "sample_index" in item
    assert "label" not in item
    assert "parameter_values" not in item
    assert "parameter_mask" not in item


def test_complex_augmentations_return_parameter_safety_metadata():
    signal = torch.randn(4, 2, 128)
    parameter_names = ["JNR", "bandwidth"]
    generator = torch.Generator().manual_seed(2026)

    weak = weak_complex_augmentation(signal, parameter_names, generator=generator)
    strong = strong_complex_augmentation(signal, parameter_names, generator=generator)

    assert weak.signal.shape == signal.shape
    assert strong.signal.shape == signal.shape
    assert weak.preserves_parameter_mask.shape == (4, 2)
    assert strong.preserves_parameter_mask.shape == (4, 2)
    assert weak.parameter_transform == "identity_except_jnr"
    assert strong.parameter_transform == "identity_except_jnr"
    assert not bool(weak.preserves_parameter_mask[:, 0].any())
    assert not bool(strong.preserves_parameter_mask[:, 0].any())
    assert bool(strong.preserves_parameter_mask[:, 1].all())


def test_capped_unlabeled_subset_preserves_all_combinations_and_label_priors():
    config = load_config(Path("configs/student_b/daps_no_param.yaml"))
    plan = build_student_b_plan(
        config,
        labeled_ratio=0.20,
        seed=2026,
        max_unlabeled_samples=1024,
        max_eval_samples=300,
    )
    labels = select_labels(read_labels(TRAIN_PATH), "four")
    selected = labels[plan["unlabeled_indices"]]
    code = (selected.astype(np.int64) * (1 << np.arange(4))[None, :]).sum(axis=1)
    assert len(np.unique(code)) == 15
    assert np.allclose(selected.mean(axis=0), 0.533, atol=0.02)


def test_weak_augmentation_is_a_small_perturbation():
    timeline = torch.linspace(0.0, 8.0 * torch.pi, 512)
    signal = torch.stack([torch.cos(timeline), torch.sin(timeline)], dim=0)
    signal = signal.unsqueeze(0).repeat(32, 1, 1)
    weak = weak_complex_augmentation(
        signal, ["bandwidth"], generator=torch.Generator().manual_seed(2026)
    ).signal
    relative_error = torch.sqrt((weak - signal).pow(2).mean()) / torch.sqrt(
        signal.pow(2).mean()
    )
    assert relative_error < 0.40
    energy_ratio = torch.sqrt(weak.pow(2).mean()) / torch.sqrt(signal.pow(2).mean())
    assert 0.85 < float(energy_ratio) < 1.15
