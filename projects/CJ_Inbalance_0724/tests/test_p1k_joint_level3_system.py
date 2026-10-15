import numpy as np

from cj0724.p1k_joint_level3_system import (
    SAW_PARAMETER_INDEX,
    VFT_PARAMETER_INDEX,
    fused_prediction,
    role_payload,
    worst_combo_predicted_gate_nmae,
)


def test_fused_prediction_replaces_only_promoted_parameters():
    base = np.zeros((4, 6, 2), dtype=np.float32)
    payload = {
        "prediction": base,
        "saw_prediction": np.arange(4, dtype=np.float32) + 10,
        "vft_prediction": np.arange(4, dtype=np.float32) + 20,
    }
    fused = fused_prediction(payload)
    assert np.array_equal(fused[:, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]], payload["saw_prediction"])
    assert np.array_equal(fused[:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]], payload["vft_prediction"])
    untouched = np.ones((6, 2), dtype=bool)
    untouched[SAW_PARAMETER_INDEX] = False
    untouched[VFT_PARAMETER_INDEX] = False
    assert np.all(fused[:, untouched] == 0)


def test_role_payload_preserves_paired_power_views():
    records = []
    for power in (20, 25):
        records.append({
            "sample_indices": np.arange(4),
            "power": np.full(4, power),
            "labels": np.zeros((4, 6), dtype=np.uint8),
        })
    selected = role_payload(records, np.array([1, 3]))
    assert selected["sample_indices"].tolist() == [1, 3, 1, 3]
    assert selected["power"].tolist() == [20, 20, 25, 25]


def test_worst_combo_metric_uses_predicted_valid_parameters():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[:2, 0] = 1
    labels[2:, 1] = 1
    prediction = np.zeros((4, 6, 2), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[2:, 1, 0] = 2.0
    mask = np.zeros_like(prediction, dtype=np.uint8)
    mask[:2, 0, 0] = 1
    mask[2:, 1, 0] = 1
    record = {
        "labels": labels, "probabilities": labels.astype(np.float32),
        "thresholds": np.full(6, 0.5), "prediction": prediction,
        "target": target, "mask": mask, "scale": np.ones((6, 2)),
    }
    assert worst_combo_predicted_gate_nmae(record) == 2.0
