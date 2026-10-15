import numpy as np

from experiments.run_route_a_h5_background_ablation_v2 import _metrics


def test_metrics_reports_saw_subsets_and_target_rates():
    targets = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=np.uint8)
    probabilities = np.asarray([[0.9, 0.1, 0.8, 0.2, 0.7],
                                [0.1, 0.9, 0.2, 0.8, 0.3]], dtype=np.float32)
    weak = targets.astype(bool)
    result = _metrics(targets, probabilities, weak, [0.5] * 5,
                      np.asarray([True, False]))
    assert result["macro_f1"] == 1.0
    assert result["target_weak_recall"] == 1.0
    assert result["target_negative_fpr"] == 0.0
    assert "saw_present_macro_f1" in result
    assert "saw_absent_macro_f1" in result

