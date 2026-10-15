import numpy as np

from cj0724.independent_worst_power_calibration import (
    buffered_threshold,
    clopper_pearson_upper,
)


def test_clopper_pearson_upper_is_conservative_and_monotone():
    assert clopper_pearson_upper(0, 620) > 0.0
    assert clopper_pearson_upper(20, 620) < clopper_pearson_upper(40, 620)
    assert clopper_pearson_upper(620, 620) == 1.0


def test_buffered_threshold_controls_every_power():
    records = []
    for level in (10, 50):
        labels = np.zeros((200, 6), dtype=np.uint8)
        labels[:100, 3] = 1
        probabilities = np.zeros((200, 6), dtype=np.float32)
        probabilities[:100, 3] = 0.8
        probabilities[100:105, 3] = 0.7
        probabilities[105:, 3] = 0.1
        records.append({
            "labels": labels,
            "probabilities": probabilities,
            "power": np.full(200, level, dtype=np.int16),
        })
    result = buffered_threshold(records, 3, "probabilities")
    assert result["feasible"]
    assert result["threshold"] <= 0.8
    assert all(row["observed_fpr"] <= 0.08 for row in result["by_power"])
    assert all(row["cp95_upper"] <= 0.10 for row in result["by_power"])
