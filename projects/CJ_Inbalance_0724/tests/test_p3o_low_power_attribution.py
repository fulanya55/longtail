import numpy as np
import torch

from cj0724.p3o_low_power_attribution import (
    ISRJ_INDEX,
    ParameterNormalizer,
    evaluate_outputs,
    make_identity_roles,
    select_checkpoint_epochs,
    select_cp_safe_isrj_threshold,
)


def test_p3o_roles_are_nested_and_identity_safe():
    labels = np.repeat(
        np.asarray([[(value >> bit) & 1 for bit in range(6)] for value in range(1, 64)], dtype=np.uint8),
        33, axis=0,
    )
    roles = make_identity_roles(labels, 7)
    assert {name: len(values) for name, values in roles.items()} == {
        "small_fit": 441, "full_fit": 1701, "early": 126, "calibration": 126, "selection": 126,
    }
    assert set(roles["small_fit"]).issubset(set(roles["full_fit"]))
    disjoint = np.concatenate((roles["full_fit"], roles["early"], roles["calibration"], roles["selection"]))
    assert len(np.unique(disjoint)) == len(disjoint) == len(labels)


def test_joint_checkpoint_obeys_parameter_tolerance_and_maximizes_aupr():
    history = [
        {"epoch": 1, "early_parameter_macro_nmae": 0.40, "early_macro_aupr": 0.80},
        {"epoch": 2, "early_parameter_macro_nmae": 0.404, "early_macro_aupr": 0.90},
        {"epoch": 3, "early_parameter_macro_nmae": 0.41, "early_macro_aupr": 0.99},
    ]
    selected = select_checkpoint_epochs(history, tolerance=1.01)
    assert selected["C_param_epoch"] == 1
    assert selected["C_joint_epoch"] == 2
    assert selected["eligible_epochs"] == [1, 2]


def test_safe_threshold_controls_every_regime_power_cp_cell():
    groups, labels, probabilities = [], [], []
    for regime in range(2):
        for power in (25, 35):
            # With 40 negatives, zero false positives has CP95 upper < 0.10.
            groups.extend([(regime, power)] * 50)
            labels.extend([0] * 40 + [1] * 10)
            probabilities.extend([0.8] * 2 + [0.2] * 38 + [0.9] * 10)
    label_matrix = np.zeros((len(labels), 6), dtype=np.uint8)
    label_matrix[:, ISRJ_INDEX] = labels
    score_matrix = np.zeros((len(labels), 6), dtype=np.float64)
    score_matrix[:, ISRJ_INDEX] = probabilities
    regime = np.asarray([row[0] for row in groups], dtype=np.int8)
    power = np.asarray([row[1] for row in groups], dtype=np.int16)
    selected = select_cp_safe_isrj_threshold(score_matrix, label_matrix, regime, power, cap=0.10)
    assert selected["feasible"]
    assert selected["maximum_cp95_upper"] <= 0.10
    assert selected["threshold"] > 0.8


def test_predicted_gate_uses_each_class_threshold_not_isrj_for_all_parameters():
    # classification_metrics expects every regime-power cell to be populated.
    regime = np.repeat(np.arange(5, dtype=np.int8), 6)
    power = np.tile(np.repeat(np.asarray([25, 35, 50], dtype=np.int16), 2), 5)
    labels = np.zeros((len(regime), 6), dtype=np.float32)
    labels[:, 0] = 1.0
    labels[:, ISRJ_INDEX] = np.tile([0.0, 1.0], len(regime) // 2)
    logits = np.full((len(regime), 6), -10.0, dtype=np.float32)
    logits[:, 0] = 10.0
    logits[:, ISRJ_INDEX] = np.where(labels[:, ISRJ_INDEX] > 0, 10.0, -10.0)
    normalizer = ParameterNormalizer()
    normalizer.mean = torch.zeros((6, 2), dtype=torch.float32)
    normalizer.scale = torch.ones((6, 2), dtype=torch.float32)
    outputs = {
        "logits": torch.from_numpy(logits), "labels": torch.from_numpy(labels),
        "prediction": np.zeros((len(regime), 6, 2), dtype=np.float32), "target": np.zeros((len(regime), 6, 2), dtype=np.float32),
        "mask": np.ones((len(regime), 6, 2), dtype=bool), "total_loss": 0.0, "classification_loss": 0.0, "parameter_loss": 0.0,
    }
    metadata = {"regime": regime, "power": power}
    metrics = evaluate_outputs(outputs, np.full(6, 0.5, dtype=np.float64), metadata, normalizer)
    # Class 0 parameters are gated for both samples although ISRJ is positive once.
    assert metrics["predicted_gate"]["predicted_gate_parameter_coverage"] == 1.0
