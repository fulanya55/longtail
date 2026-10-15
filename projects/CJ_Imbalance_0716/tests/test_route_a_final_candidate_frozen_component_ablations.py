import numpy as np

from experiments.audit_route_a_final_candidate_frozen_component_ablations import _delta


def test_component_delta_tracks_candidate_minus_lesion():
    base = {"metrics": {"macro_aupr": .8, "macro_f1": .7, "target_mean_weak_recall": .4, "target_weak_vs_negative_auroc": .6, "target_mean_negative_fpr": .2, "per_label": {name: {"aupr": .5, "weak_recall": .4, "weak_vs_negative_auroc": .6, "negative_fpr": .2} for name in ("MFT", "VFT", "ISRJ")}}, "mft_hard": {"hard_fpr": .1, "order5_fpr": .1}}
    lesion = {"metrics": {**base["metrics"], "macro_aupr": .7}, "mft_hard": {"hard_fpr": .2, "order5_fpr": .3}}
    values = _delta(base, lesion)
    assert np.isclose(values["macro_aupr"], .1)
    assert np.isclose(values["mft_hard_fpr"], -.1)
