from cj0724.p5e_five_model_seed_confirmation import data_gates


def test_p5e_confirmation_requires_every_model_safety_gate():
    rows = [{"invariant": True} for _ in range(5)]
    summary = {
        "MFT.JNR": {"oracle_gain_by_model": [0.1] * 5, "predicted_gated_delta_by_model": [0.0] * 5},
        "ISRJ.JNR": {"oracle_gain_by_model": [0.1] * 5, "predicted_gated_delta_by_model": [-0.1] * 5},
    }
    bootstrap = {"MFT.JNR": [{"ci95_low": 0.01}] * 5, "ISRJ.JNR": [{"ci95_low": 0.01}] * 5}

    assert all(data_gates(rows, summary, bootstrap, []).values())

    summary["ISRJ.JNR"]["predicted_gated_delta_by_model"][3] = 0.0001
    assert not data_gates(rows, summary, bootstrap, [])["isrj_predicted_not_worse"]
