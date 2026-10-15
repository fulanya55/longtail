import numpy as np

from experiments.run_route_b_c2_frozen_local_replacement import _joint_metrics


def test_joint_metrics_uses_positive_valid_targets_and_detected_subset():
    prediction = np.zeros((6, 6, 2), dtype=np.float32)
    target = np.zeros((6, 6, 2), dtype=np.float32)
    prediction[0, 0] = [2.0, 4.0]
    target[0, 0] = [1.0, 2.0]
    prediction[1, 1] = [4.0, 8.0]
    target[1, 1] = [4.0, 6.0]
    parameter_mask = np.ones_like(target, dtype=bool)
    labels = np.eye(6, dtype=np.uint8)
    class_prediction = labels.copy()
    scale = np.ones((6, 2), dtype=np.float32) * 2.0
    metrics = _joint_metrics(
        prediction,
        target,
        parameter_mask,
        labels,
        class_prediction,
        scale,
    )
    assert metrics["system_parameter_coverage"] == 1.0
    assert metrics["oracle_micro_nmae"] == (0.5 + 1.0 + 0.0 + 1.0) / 12.0
    assert metrics["oracle_macro_target_nmae"] == (0.5 + 1.0 + 0.0 + 1.0) / 12.0
