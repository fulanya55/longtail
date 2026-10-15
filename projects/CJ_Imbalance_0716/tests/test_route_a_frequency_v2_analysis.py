import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/analyze_route_a_frequency_reconstruction_v2.py"
SPEC = importlib.util.spec_from_file_location("frequency_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_weighted_average_precision_matches_unweighted_reference():
    truth = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    scores = np.asarray([0.1, 0.9, 0.2, 0.7, 0.6, 0.3], dtype=np.float64)
    counts = np.ones((1, len(truth)), dtype=np.uint16)
    actual = MODULE.weighted_average_precision(truth, scores, counts)[0]
    expected = average_precision_score(truth, scores)
    assert abs(actual - expected) < 1e-12


def test_mean_summary_counts_direction():
    summary = MODULE.mean_summary([-1.0, 0.0, 1.0, 2.0])
    assert summary["positive_count"] == 2
    assert summary["nonnegative_count"] == 3
