from cj0724.p2o_confirmation_aggregate import DATA_SEEDS, REGIMES, build_gates


def test_confirmation_gates_require_all_rows():
    physical = [{"promotion_passed": True}, {"promotion_passed": True}]
    systems = []
    rows = []
    for data_seed in DATA_SEEDS:
        per_regime = {}
        for regime in REGIMES:
            per_regime[regime] = {
                "mean_full_improvement_vs_P2k": 0.01,
                "mean_miss_delta_vs_P2k": -0.01,
                "mean_worst_combo_delta_vs_P2k": -0.01,
                "bootstrap": {"ci95_high": -0.001},
            }
            for seed in (2041, 2042, 2043):
                rows.append({
                    "data_seed": data_seed, "regime": regime, "seed": seed,
                    "full_improvement_vs_safe": 0.06, "full_improvement_vs_P2k": 0.01,
                })
        systems.append({
            "promotion_passed": True, "per_regime": per_regime,
            "promotion_gates": {
                "classification_arrays_unchanged": True,
                "other_11_parameter_outputs_unchanged": True,
            },
        })
    gates = build_gates(physical, systems, rows, 0, True)
    assert all(gates.values())
    rows[0]["full_improvement_vs_safe"] = 0.049
    gates = build_gates(physical, systems, rows, 0, True)
    assert not gates["all_30_full_improvement_vs_safe_at_least_0p05"]
