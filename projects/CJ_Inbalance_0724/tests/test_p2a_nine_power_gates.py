from cj0724.p1k_finalize_saved_predictions import nine_power_recall_gates


def test_nine_power_recall_gates_accept_eight_nonnegative_and_small_decline():
    values = {str(power): 0.01 for power in range(10, 51, 5)}
    values["10"] = -0.01
    assert nine_power_recall_gates(values) == {
        "isrj_at_least_eight_of_nine_powers_nonnegative": True,
        "no_power_recall_decline_below_minus_0p02": True,
    }


def test_nine_power_recall_gates_reject_two_negative_or_large_decline():
    values = {str(power): 0.01 for power in range(10, 51, 5)}
    values["10"], values["15"] = -0.03, -0.01
    gates = nine_power_recall_gates(values)
    assert not gates["isrj_at_least_eight_of_nine_powers_nonnegative"]
    assert not gates["no_power_recall_decline_below_minus_0p02"]
