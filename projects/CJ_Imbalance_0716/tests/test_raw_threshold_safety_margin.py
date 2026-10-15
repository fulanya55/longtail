from experiments.run_route_a_raw_threshold_safety_margin import (
    _select_margin,
    _select_shared,
)


def _row(threshold, ordinary_fpr, power_fpr, recall):
    return {
        "threshold": threshold,
        "ordinary_fpr": ordinary_fpr,
        "max_power_fpr": power_fpr,
        "ordinary_recall": recall,
        "mean_power_recall": recall,
        "power": {"50": {"recall": recall}},
    }


def test_margin_selection_rejects_boundary_violations():
    rows = [
        _row(0.3, 0.19, 0.05, 0.99),
        _row(0.4, 0.14, 0.05, 0.95),
        _row(0.5, 0.08, 0.04, 0.80),
    ]
    assert _select_margin(rows, 0.15)["threshold"] == 0.4
    assert _select_margin(rows, 0.10)["threshold"] == 0.5


def test_margin_selection_prioritizes_recall_then_higher_threshold():
    rows = [
        _row(0.4, 0.14, 0.05, 0.90),
        _row(0.5, 0.10, 0.05, 0.90),
    ]
    assert _select_margin(rows, 0.15)["threshold"] == 0.5


def test_shared_selection_requires_every_seed_to_be_feasible():
    searches = {
        1: [_row(0.4, 0.14, 0.04, 0.95), _row(0.5, 0.08, 0.02, 0.85)],
        2: [_row(0.4, 0.17, 0.03, 0.96), _row(0.5, 0.09, 0.02, 0.86)],
    }
    selected = _select_shared(searches, 0.15)
    assert selected["threshold"] == 0.5
