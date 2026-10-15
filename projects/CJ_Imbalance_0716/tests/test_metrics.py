import numpy as np
import pytest

from cjlt.engine.calibration import apply_temperature, fit_temperature
from cjlt.engine.evaluator import compute_multilabel_metrics
from cjlt.engine.thresholding import (
    apply_thresholds,
    select_global_threshold,
    select_per_label_fbeta_thresholds,
    select_per_label_f1_thresholds,
    select_per_label_weak_recall_fpr_thresholds,
    select_per_label_youden_thresholds,
    select_shrinkage_per_label_thresholds,
)


def test_compute_multilabel_metrics_reports_core_tail_and_weak_metrics():
    y_true = np.asarray(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    y_pred = np.asarray(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    probability = np.asarray(
        [[0.9, 0.1, 0.4], [0.1, 0.8, 0.2], [0.8, 0.4, 0.1], [0.1, 0.2, 0.9]],
        dtype=np.float32,
    )
    weak_mask = np.asarray(
        [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype=bool,
    )

    result = compute_multilabel_metrics(
        y_true,
        y_pred,
        probabilities=probability,
        class_names=["head", "tail", "other"],
        tail_indices=[1],
        weak_component_mask=weak_mask,
    )

    assert result["subset_acc"] == pytest.approx(0.5)
    assert result["hamming_loss"] == pytest.approx(2.0 / 12.0)
    assert result["label_acc"] == pytest.approx(10.0 / 12.0)
    assert result["tail_label_recall"] == pytest.approx(0.5)
    assert result["weak_component_recall"] == pytest.approx(0.0)
    assert result["cardinality_mae"] == pytest.approx(0.5)
    assert len(result["per_label"]) == 3
    assert result["per_label"][1]["label"] == "tail"
    assert 0.0 <= result["macro_aupr"] <= 1.0


def test_metric_validation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_multilabel_metrics(
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        )


def test_cardinality_mae_handles_overprediction_without_unsigned_underflow():
    y_true = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    y_pred = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)

    result = compute_multilabel_metrics(y_true, y_pred)

    assert result["cardinality_mae"] == pytest.approx(1.5)


def test_temperature_fit_does_not_increase_calibration_bce():
    logits = np.asarray(
        [[5.0, -5.0], [-4.0, 4.0], [3.0, -3.0], [-2.0, 2.0]],
        dtype=np.float32,
    )
    labels = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)

    calibration = fit_temperature(
        logits,
        labels,
        source_role="calibration",
        max_iter=20,
    )

    assert calibration.temperature > 0.0
    assert calibration.loss_after <= calibration.loss_before + 1e-8
    scaled = apply_temperature(logits, calibration.temperature)
    assert scaled.shape == logits.shape


def test_temperature_fit_rejects_non_calibration_role():
    with pytest.raises(ValueError, match="calibration"):
        fit_temperature(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            source_role="locked_test",
        )


def test_global_and_shrinkage_threshold_selection_use_threshold_role_only():
    labels = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]],
        dtype=np.uint8,
    )
    probabilities = np.asarray(
        [[0.80, 0.20], [0.70, 0.30], [0.40, 0.65], [0.35, 0.75], [0.65, 0.70], [0.30, 0.20]],
        dtype=np.float32,
    )
    grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    global_result = select_global_threshold(
        labels,
        probabilities,
        grid=grid,
        source_role="threshold_selection",
    )
    shrinkage = select_shrinkage_per_label_thresholds(
        labels,
        probabilities,
        positive_counts=[1000, 100],
        beta=100,
        grid=grid,
        source_role="threshold_selection",
    )
    per_label_f1 = select_per_label_f1_thresholds(
        labels,
        probabilities,
        grid=grid,
        source_role="threshold_selection",
    )
    per_label_f2 = select_per_label_fbeta_thresholds(
        labels,
        probabilities,
        beta=2.0,
        grid=grid,
        source_role="threshold_selection",
    )
    per_label_youden = select_per_label_youden_thresholds(
        labels,
        probabilities,
        grid=grid,
        source_role="threshold_selection",
    )

    assert len(global_result.thresholds) == 2
    assert len(shrinkage.thresholds) == 2
    assert len(per_label_f1.thresholds) == 2
    assert len(per_label_f2.thresholds) == 2
    assert len(per_label_youden.thresholds) == 2
    assert all(0.0 < value < 1.0 for value in shrinkage.thresholds)
    assert apply_thresholds(probabilities, shrinkage.thresholds).shape == labels.shape
    with pytest.raises(ValueError, match="threshold_selection"):
        select_global_threshold(
            labels,
            probabilities,
            grid=grid,
            source_role="locked_test",
        )


def test_per_label_fbeta_thresholds_reject_invalid_beta():
    labels = np.asarray([[1], [0]], dtype=np.uint8)
    probabilities = np.asarray([[0.8], [0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="beta"):
        select_per_label_fbeta_thresholds(
            labels,
            probabilities,
            beta=0.0,
            source_role="threshold_selection",
        )


def test_weak_recall_threshold_respects_negative_fpr_and_uses_weak_positives():
    labels = np.asarray(
        [[1], [1], [1], [0], [0], [0], [0], [0]], dtype=np.uint8
    )
    probabilities = np.asarray(
        [[0.82], [0.35], [0.25], [0.60], [0.20], [0.10], [0.05], [0.01]],
        dtype=np.float32,
    )
    weak = np.asarray(
        [[0], [1], [1], [0], [0], [0], [0], [0]], dtype=bool
    )

    result = select_per_label_weak_recall_fpr_thresholds(
        labels,
        probabilities,
        weak_component_mask=weak,
        max_negative_fpr=0.20,
        source_role="threshold_selection",
        grid=[0.20, 0.30, 0.40, 0.50, 0.60],
    )

    assert result.mode == "per_label_weak_recall_fpr"
    assert result.thresholds == [0.30]
    prediction = probabilities[:, 0] >= result.thresholds[0]
    assert prediction[labels[:, 0] == 0].mean() <= 0.20
    assert prediction[weak[:, 0]].mean() == 0.5
