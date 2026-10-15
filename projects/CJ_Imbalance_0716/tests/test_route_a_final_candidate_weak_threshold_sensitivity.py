import numpy as np

from experiments.analyze_route_a_final_candidate_weak_threshold_sensitivity import _delta


def test_threshold_delta_uses_candidate_minus_baseline():
    per_label = {name: {"weak_recall": .2, "weak_vs_negative_auroc": .6} for name in ("MFT", "VFT", "ISRJ")}
    candidate = {"macro_aupr": .8, "macro_f1": .7, "target_mean_weak_recall": .4, "target_weak_vs_negative_auroc": .7, "target_mean_negative_fpr": .2, "per_label": per_label}
    baseline = {**candidate, "macro_aupr": .7, "target_mean_weak_recall": .3}
    values = _delta(candidate, baseline)
    assert np.isclose(values["macro_aupr"], .1)
    assert np.isclose(values["target_weak_recall"], .1)
