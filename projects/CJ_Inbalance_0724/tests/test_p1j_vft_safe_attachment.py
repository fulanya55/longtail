import numpy as np

from cj0724.p1j_vft_safe_attachment import buffered_vft_threshold


def test_buffered_vft_threshold_controls_each_power():
    labels = np.zeros((200, 6), dtype=np.uint8)
    labels[:50, 4] = 1
    labels[100:150, 4] = 1
    probability = np.zeros((200, 6), dtype=np.float32)
    probability[:, 4] = 0.1
    probability[labels[:, 4] == 1, 4] = 0.9
    payload = {"labels": labels, "probability": probability,
               "power": np.repeat([20, 25], 100)}
    result = buffered_vft_threshold(payload)
    assert result["feasible"]
    assert all(row["observed_fpr"] <= 0.06 for row in result["by_power"])


def test_buffered_vft_threshold_accepts_explicit_power_grid():
    powers = (10, 20, 30)
    labels = np.zeros((300, 6), dtype=np.uint8)
    probability = np.zeros((300, 6), dtype=np.float32)
    power = np.repeat(powers, 100)
    for offset in range(0, 300, 100):
        labels[offset:offset + 50, 4] = 1
        probability[offset:offset + 50, 4] = 0.9
        probability[offset + 50:offset + 100, 4] = 0.1
    result = buffered_vft_threshold({"labels": labels, "probability": probability, "power": power}, powers)
    assert result["feasible"]
    assert [row["power_db"] for row in result["by_power"]] == list(powers)
