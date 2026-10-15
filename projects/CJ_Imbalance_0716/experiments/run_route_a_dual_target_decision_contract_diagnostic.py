"""Replay a fixed MFT/VFT parent-decision contract on three diagnostic sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch  # Load Windows DLLs before sklearn.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _write_json  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _hard_metrics  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402


MFT_INDEX = 3
VFT_INDEX = 4
SEEDS = (2039, 2040, 2041, 2042, 2043)
VARIANTS = ("D0_H1B2", "D1_AlignedContract", "D2_CapacityContract", "D3_ShuffledContract")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(root, seed, name):
    return {key: np.asarray(value) for key, value in np.load(root / f"seed_{seed}" / name / "predictions.npz").items()}


def contract_logits(q1, q2, expert, mft_cap=0.50, source_mft_cap=0.75, vft_cap=0.50):
    output = np.asarray(q1, dtype=np.float32).copy()
    output[:, MFT_INDEX] = q1[:, MFT_INDEX] + (float(mft_cap) / float(source_mft_cap)) * (q2[:, MFT_INDEX] - q1[:, MFT_INDEX])
    base = q1[:, VFT_INDEX]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    delta = np.asarray(expert[:, VFT_INDEX] - q2[:, VFT_INDEX], dtype=np.float32)
    output[:, VFT_INDEX] = base + float(vft_cap) * uncertainty * np.tanh(delta)
    return output


def _evaluate(logits, calibration, thresholds, labels, weak):
    probability = _apply_frozen_calibration(logits, calibration)
    prediction = probability >= np.asarray(thresholds, dtype=np.float32)[None, :]
    return {"metrics": _metrics(labels, weak, probability, prediction), "mft_hard": _hard_metrics(labels, weak, probability, prediction), "probabilities": probability, "predictions": prediction}


def _dataset(args, name, root, macro_floor):
    rows = []
    for seed in SEEDS:
        q0 = _load(root, seed, "Q0_H1")
        q1 = _load(root, seed, "Q1_H1B2")
        q2 = _load(root, seed, "Q2_H1B2_MFTGuard")
        controls = {
            VARIANTS[1]: _load(root, seed, "Q4_VFTAlignedRidgeFusion"),
            VARIANTS[2]: _load(root, seed, "Q3_VFTBaseOnlyCapacity"),
            VARIANTS[3]: _load(root, seed, "Q5_VFTShuffledRidgeFusion"),
        }
        labels = np.asarray(q0["targets"], dtype=np.uint8)
        weak = np.asarray(q0["weak_mask"], dtype=bool)
        source = args.construction_root / f"seed_{seed}" / "Q1_H1B2"
        calibration = _json(source / "calibration.json")
        thresholds = _json(source / "thresholds.json")["thresholds"]
        evaluated = {VARIANTS[0]: _evaluate(np.asarray(q1["logits"], dtype=np.float32), calibration, thresholds, labels, weak)}
        logits = {}
        for variant, expert in controls.items():
            logits[variant] = contract_logits(np.asarray(q1["logits"], dtype=np.float32), np.asarray(q2["logits"], dtype=np.float32), np.asarray(expert["logits"], dtype=np.float32))
            evaluated[variant] = _evaluate(logits[variant], calibration, thresholds, labels, weak)
        base = _evaluate(np.asarray(q0["logits"], dtype=np.float32), _json(args.construction_root / f"seed_{seed}" / "Q0_H1/calibration.json"), _json(args.construction_root / f"seed_{seed}" / "Q0_H1/thresholds.json")["thresholds"], labels, weak)
        aligned = evaluated[VARIANTS[1]]
        row = {
            "seed": seed, "variants": {key: {"metrics": value["metrics"], "mft_hard": value["mft_hard"]} for key, value in evaluated.items()},
            "macro_aupr_vs_h1": aligned["metrics"]["macro_aupr"] - base["metrics"]["macro_aupr"],
            "target_weak_recall_vs_h1": aligned["metrics"]["target_mean_weak_recall"] - base["metrics"]["target_mean_weak_recall"],
            "target_fpr": aligned["metrics"]["target_mean_negative_fpr"],
            "mft_hard_fpr_vs_q1": aligned["mft_hard"]["hard_fpr"] - evaluated[VARIANTS[0]]["mft_hard"]["hard_fpr"],
            "mft_weak_recall_vs_q1": aligned["metrics"]["per_label"]["MFT"]["weak_recall"] - evaluated[VARIANTS[0]]["metrics"]["per_label"]["MFT"]["weak_recall"],
            "vft_fpr": aligned["metrics"]["per_label"]["VFT"]["negative_fpr"],
            "vft_weak_auroc_vs_q1": aligned["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"] - evaluated[VARIANTS[0]]["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"],
            "vft_weak_auroc_vs_capacity": aligned["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"] - evaluated[VARIANTS[2]]["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"],
            "vft_weak_auroc_vs_shuffled": aligned["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"] - evaluated[VARIANTS[3]]["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"],
            "sawfm_probability_max_abs": float(np.max(np.abs(aligned["probabilities"][:, 1] - base["probabilities"][:, 1]))),
            "non_target_prediction_mismatch": int((aligned["predictions"][:, [0, 1, 2, 5]] != evaluated[VARIANTS[0]]["predictions"][:, [0, 1, 2, 5]]).sum()),
        }
        rows.append(row)
    keys = ("macro_aupr_vs_h1", "target_weak_recall_vs_h1", "target_fpr", "mft_hard_fpr_vs_q1", "mft_weak_recall_vs_q1", "vft_fpr", "vft_weak_auroc_vs_q1", "vft_weak_auroc_vs_capacity", "vft_weak_auroc_vs_shuffled")
    aggregate = {key: _mean_ci([row[key] for row in rows], 20431000 + index) for index, key in enumerate(keys)}
    checks = {
        "target_fpr_le_0p20": aggregate["target_fpr"]["mean"] <= 0.20,
        "target_weak_recall_gain_ge_0p030": aggregate["target_weak_recall_vs_h1"]["mean"] >= 0.030,
        "macro_aupr_gain_meets_floor": aggregate["macro_aupr_vs_h1"]["mean"] >= macro_floor,
        "mft_hard_fpr_delta_le_minus0p010": aggregate["mft_hard_fpr_vs_q1"]["mean"] <= -0.010,
        "mft_weak_recall_loss_le_0p020": aggregate["mft_weak_recall_vs_q1"]["mean"] >= -0.020,
        "vft_fpr_le_0p20": aggregate["vft_fpr"]["mean"] <= 0.20,
        "vft_weak_auroc_gain_ge_0p010": aggregate["vft_weak_auroc_vs_q1"]["mean"] >= 0.010,
        "vft_beats_capacity_ge_0p005": aggregate["vft_weak_auroc_vs_capacity"]["mean"] >= 0.005,
        "vft_beats_shuffled_ge_0p005": aggregate["vft_weak_auroc_vs_shuffled"]["mean"] >= 0.005,
        "sawfm_probability_exact": max(row["sawfm_probability_max_abs"] for row in rows) == 0.0,
        "non_target_predictions_exact": sum(row["non_target_prediction_mismatch"] for row in rows) == 0,
    }
    return {"dataset": name, "rows": rows, "aggregate": aggregate, "checks": checks, "passed": all(checks.values())}


def run(args):
    seal = _json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    datasets = {
        "construction_v3": _dataset(args, "construction_v3", args.v3_root, 0.005),
        "development_v12_diagnostic": _dataset(args, "development_v12_diagnostic", args.v12_root, 0.005),
        "isrj_parameter_ood_diagnostic": _dataset(args, "isrj_parameter_ood_diagnostic", args.ood_root, 0.0),
    }
    result = {
        "status": "completed", "experiment": "RouteA_DualTargetDecisionContract_Diagnostic",
        "fixed_caps": {"mft": 0.50, "vft": 0.50}, "calibration_and_threshold_source": "Q1_H1B2",
        "datasets": datasets, "passed": all(item["passed"] for item in datasets.values()),
        "external_refit": False, "hyperparameter_search": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_dual_target_decision_contract_diagnostic_summary.json"
    _write_json(output, result)
    print(json.dumps({"passed": result["passed"], "datasets": {name: {"passed": item["passed"], "checks": item["checks"], "aggregate": item["aggregate"]} for name, item in datasets.items()}, "output": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--v12-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
    parser.add_argument("--ood-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dual_target_decision_contract_diagnostic")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

