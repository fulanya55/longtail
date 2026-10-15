from cj0724.b2_power_boundary import summarize_gates


def test_b2_gate_summary_uses_all_nine_levels():
    rows = []
    for power in (10, 15, 20, 25, 30, 35, 40, 45, 50):
        rows.append({"power_dB": power, "variant": "E5", "macro_aupr": 0.8,
                     "hamming": 0.2, "weak_recall": 0.8, "MFT_fpr": 0.5, "ISRJ_fpr": 0.6})
        rows.append({"power_dB": power, "variant": "B2-Hard", "macro_aupr": 0.82,
                     "hamming": 0.18, "weak_recall": 0.8, "MFT_fpr": 0.4, "ISRJ_fpr": 0.5})
    result = summarize_gates(rows)
    assert result["B2_passed"]
    assert len(result["power_aggregates"]) == 9
