import numpy as np

from experiments.run_route_a_gradient_contrast_test_v2_once import (
    _evaluate_frozen,
    _gate,
)


def test_frozen_evaluation_uses_supplied_temperature_and_thresholds():
    logits6 = np.zeros((8, 6), dtype=np.float32)
    targets6 = np.asarray([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ], dtype=np.uint8)
    weak6 = targets6.astype(bool)
    result = _evaluate_frozen(
        "candidate",
        logits6,
        targets6,
        np.arange(8),
        weak6,
        temperature=2.0,
        thresholds=[0.5] * 5,
    )
    assert result["temperature"] == 2.0
    assert result["thresholds"] == [0.5] * 5
    assert result["predictions"]["predictions"].shape == (8, 5)


def test_independent_gate_requires_every_registered_check():
    def stat(value):
        return {"mean": value, "std": 0.0, "count": 5, "bootstrap_95_ci": [value, value]}

    comparison = {"aggregate": {
        "macro_aupr": stat(0.0),
        "macro_f1": stat(0.02),
        "target_mean_weak_recall": stat(0.04),
        "target_mean_negative_fpr": stat(0.0),
        "sawfm_present_macro_f1": stat(0.02),
        "sawfm_absent_macro_f1": stat(0.0),
        "per_label_aupr": {name: stat(0.0) for name in ("FMB", "SinFM", "MFT", "VFT", "ISRJ")},
        "positive_target_weak_recall_seeds": 5,
    }}
    aggregate = {"GradientContrastFixed1p5": {"target_mean_negative_fpr": stat(0.15)}}
    assert _gate(comparison, aggregate, hashes_match=True, finite=True)["passed"]
    assert not _gate(comparison, aggregate, hashes_match=False, finite=True)["passed"]
