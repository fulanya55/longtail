import numpy as np

from experiments.run_route_a_raw_platt_five_seed_label_routing import (
    _multilabel_metrics,
)


def test_multilabel_metrics_uses_per_label_thresholds():
    targets = np.asarray([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
    ])
    probability = np.where(targets, 0.8, 0.2).astype(np.float64)
    metrics = _multilabel_metrics(probability, targets, np.full(6, 0.5))
    assert metrics["macro_f1"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["hamming_loss"] == 0.0


def test_replacing_one_label_does_not_change_other_label_fpr():
    targets = np.asarray([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
    ])
    baseline = np.where(targets, 0.8, 0.2).astype(np.float64)
    routed = baseline.copy()
    routed[:, 1] = np.asarray([0.1, 0.9, 0.2, 0.95])
    thresholds = np.full(6, 0.5)
    base_metrics = _multilabel_metrics(baseline, targets, thresholds)
    route_metrics = _multilabel_metrics(routed, targets, thresholds)
    assert route_metrics["per_label"]["FMB"]["fpr"] == base_metrics["per_label"]["FMB"]["fpr"]
