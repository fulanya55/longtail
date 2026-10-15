import numpy as np

from cj0724.evaluation import detailed_metrics


def test_detailed_metrics_report_labels_and_combinations():
    labels = np.asarray([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    logits = np.where(labels == 1, 10.0, -10.0).astype(np.float32)
    label_rows, combination_rows, summary = detailed_metrics("perfect", logits, labels, 0.5)
    assert len(label_rows) == 6
    assert len(combination_rows) == 2
    assert summary["overall_subset_accuracy"] == 1.0
    assert all(row["exact_accuracy"] == 1.0 for row in combination_rows)
