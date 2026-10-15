from __future__ import annotations

import numpy as np

from experiments.analyze_route_a_h5o_frozen_ood_failure import _label_metrics, _stratum_fpr


def _item() -> dict[str, np.ndarray]:
    return {
        "targets": np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8),
        "weak_mask": np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=bool),
        "probabilities": np.array([[.9, .1], [.8, .2], [.1, .8], [.2, .7]], dtype=np.float32),
        "predictions": np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8),
        "thresholds": np.array([.5, .5], dtype=np.float32),
    }


def test_label_metrics_separates_weak_strong_and_negative_roles() -> None:
    metrics = _label_metrics(_item(), 0)
    assert metrics["weak_positive_count"] == 1
    assert metrics["strong_positive_count"] == 1
    assert metrics["negative_count"] == 2
    assert metrics["weak_recall"] == 1.0


def test_stratum_fpr_only_uses_target_negatives() -> None:
    item = _item()
    result = _stratum_fpr(item, item, 0, np.array([True, True, True, False]))
    assert result["count"] == 1
    assert result["q0_fpr"] == 0.0 and result["q6_fpr"] == 0.0
