import numpy as np

from cj0724.p2w_vft_component_isolation_oracle import _to_iq, gate


def _row(power, oracle=20.0, raw=40.0, pearson=0.8, wrong=30.0, no_delay=30.0):
    return {"power_db": power, "C0_raw": {"mae_hz": raw}, "C1_oracle": {"mae_hz": oracle, "finite_rate": 1.0, "pearson": pearson}, "C2_wrong_reference": {"mae_hz": wrong}, "C3_no_delay": {"mae_hz": no_delay}}


def test_p2w_gates_require_weak_power_and_mechanism_controls():
    rows = [_row(power) for _ in range(3) for power in (10, 30, 50)]
    result = gate(rows)
    assert all(value for key, value in result.items() if isinstance(value, bool))
    rows[0] = _row(10, oracle=25.0)
    assert not gate(rows)["C1_10dB_mae_each_seed_vs_C0_ratio_max"]
    rows = [_row(10, wrong=22.0) if row["power_db"] == 10 else row for row in rows]
    assert not gate(rows)["C1_mean_mae_each_power_vs_C2_ratio_max"]


def test_p2w_sample_first_legacy_iq_keeps_i_and_q_channel_order():
    legacy = np.zeros((1, 6000), dtype=np.float32); legacy[0, :3000] = 1.0; legacy[0, 3000:] = 2.0
    iq = _to_iq(legacy).numpy()
    assert iq.shape == (1, 2, 3000)
    assert np.all(iq[0, 0] == 1.0) and np.all(iq[0, 1] == 2.0)
