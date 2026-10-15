import json
from argparse import Namespace

import numpy as np

from experiments.summarize_route_a_h3_vft_three_strata_models import run


def _receipt(mode, seed, parameter_count, weak_auc, strong_auc):
    return {
        "status": "completed_development_unaggregated", "smoke": False, "modes": [mode], "seeds": [seed],
        "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        "config": {"calibration": "none", "threshold_selection": "none"},
        "results": [{"mode": mode, "seed": seed, "parameter_count": parameter_count, "metrics": {
            "macro_aupr": .8, "per_label": [{"aupr": .8}] * 6,
            "weak_vft_vs_absent": {"auroc": weak_auc}, "strong_vft_vs_absent": {"auroc": strong_auc},
        }}],
    }


def test_h3_summary_recomputes_immutable_predictions_and_uses_t1_t2_capacity_control(tmp_path):
    seed = 7
    # arms: absent negative, weak positive, strong positive. T1 ranks weak and
    # strong correctly; T2 reverses weak only, which makes every fixed gate pass.
    labels = np.zeros((6, 6), dtype=bool); labels[2:, 4] = True
    arm = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint8)
    for mode, score, count in (("T0-h3", [.2, .3, .7, .8, .8, .9], 10), ("T1-h3", [.1, .2, .8, .9, .8, .9], 11), ("T2-h3", [.8, .9, .1, .2, .8, .9], 11)):
        root = tmp_path / f"route_{mode.replace('-', '_')}_seed{seed}"; artifact = root / f"{mode}_seed{seed}"; artifact.mkdir(parents=True)
        probability = np.tile(np.asarray(score, dtype=float)[:, None], (1, 6))
        weak_auc = 1.0 if mode != "T2-h3" else 0.0
        strong_auc = .5 if mode == "T2-h3" else 1.0
        (root / "run_receipt.json").write_text(json.dumps(_receipt(mode, seed, count, weak_auc, strong_auc)), encoding="utf-8")
        np.savez_compressed(artifact / "development_predictions.npz", probabilities=probability, labels=labels, vft_visibility_arm=arm)
    output = tmp_path / "summary.json"
    value = run(Namespace(root_template=str(tmp_path / "route_{mode}_seed{seed}"), seeds=[seed], bootstrap_draws=100, bootstrap_seed=1, output=output))
    assert value["gate"]["passed"]
    assert value["comparison"]["T1_minus_T2_weak_vft_auroc"]["mean"] == 1.0
    assert json.loads(output.read_text(encoding="utf-8"))["test_v3_used"] is False
