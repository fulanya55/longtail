from experiments.run_route_a_fullalpha_periodicity_development_v7_frozen import _promotion_checks


def _ci(value):
    return {"mean": value, "ci_low": value, "ci_high": value}


def test_v7_gate_requires_all_frozen_confirmation_conditions():
    labels = ("MFT", "VFT", "ISRJ", "SawFM")
    base_per = {
        name: {
            "aupr": 0.80,
            "weak_recall": 0.30,
            "strong_recall": 0.80,
            "negative_fpr": 0.10,
            "weak_vs_negative_auroc": 0.60,
        }
        for name in labels
    }
    candidate_per = {name: dict(values) for name, values in base_per.items()}
    for name in ("MFT", "VFT", "ISRJ"):
        candidate_per[name]["aupr"] += 0.01
        candidate_per[name]["weak_recall"] += 0.05
        candidate_per[name]["weak_vs_negative_auroc"] += 0.02
    u0 = {"variant": "U0_C6RawSaw", "metrics": {"target_mean_negative_fpr": 0.10, "per_label": base_per}}
    u1 = {"variant": "U1_FullAlphaRawSaw", "metrics": {"target_mean_negative_fpr": 0.11, "per_label": base_per}}
    u2 = {"variant": "U2_FullAlphaAlignedPeriodicity", "metrics": {"target_mean_negative_fpr": 0.12, "per_label": candidate_per}}
    u3 = {"variant": "U3_FullAlphaBlockShuffled", "metrics": {"target_mean_negative_fpr": 0.12, "per_label": base_per}}
    rows = [{"seed": seed, "variants": [u0, u1, u2, u3]} for seed in range(5)]
    comparisons = {
        "U2_vs_U0": {"aggregate": {
            "macro_aupr": _ci(0.01), "macro_f1": _ci(0.02),
            "target_mean_weak_recall": _ci(0.05), "target_weak_vs_negative_auroc": _ci(0.02),
            "target_mean_strong_recall": _ci(0.0), "positive_target_weak_seeds": 5,
            "SawFM__aupr": _ci(0.0), "SawFM__weak_recall": _ci(0.0),
        }},
        "U2_vs_U1": {"aggregate": {
            "ISRJ__weak_vs_negative_auroc": _ci(0.02), "ISRJ__weak_recall": _ci(0.05),
        }},
        "U2_vs_U3": {"aggregate": {"ISRJ__weak_vs_negative_auroc": _ci(0.02)}},
    }
    _, checks = _promotion_checks(rows, comparisons, external_refit=False, test_v3_used=False)
    assert all(checks.values())
    _, failed = _promotion_checks(rows, comparisons, external_refit=True, test_v3_used=False)
    assert not failed["external_refit_false"]
