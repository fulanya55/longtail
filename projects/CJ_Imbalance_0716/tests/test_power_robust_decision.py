import numpy as np

from experiments.run_route_a_power_robust_decision import (
    _select_robust_threshold,
    _split_matched_positions_by_combo,
)


def test_combo_split_keeps_every_combo_in_both_roles():
    labels = np.asarray(
        [
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    selection = {"positive_base_indices": np.arange(8)}
    calibration, threshold = _split_matched_positions_by_combo(selection, labels, 17)
    assert set(calibration).isdisjoint(set(threshold))
    assert len(calibration) == len(threshold) == 4
    for positions in (calibration, threshold):
        keys = {tuple(np.delete(labels[position], 1)) for position in positions}
        assert len(keys) == 2


def test_robust_threshold_respects_all_fpr_caps():
    ordinary_probability = np.asarray([0.8, 0.7, 0.2, 0.1])
    ordinary_targets = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    power = {
        "40": {"positive": np.asarray([0.75, 0.65]), "negative": np.asarray([0.30, 0.20])},
        "45": {"positive": np.asarray([0.60, 0.55]), "negative": np.asarray([0.35, 0.25])},
        "50": {"positive": np.asarray([0.52, 0.48]), "negative": np.asarray([0.38, 0.28])},
    }
    threshold, rows = _select_robust_threshold(
        ordinary_probability, ordinary_targets, power, max_fpr=0.20
    )
    selected = next(row for row in rows if np.isclose(row["threshold"], threshold))
    assert selected["feasible"]
    assert selected["ordinary_fpr"] <= 0.20
    assert selected["max_power_fpr"] <= 0.20
