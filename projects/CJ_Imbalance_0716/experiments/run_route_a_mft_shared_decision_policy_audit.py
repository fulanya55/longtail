"""Audit a monotone MFT decision guard by sharing Q1 calibration/threshold."""

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
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _hard_metrics  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402


MFT_INDEX = 3
SAW_INDEX = 1
SEEDS = (2039, 2040, 2041, 2042, 2043)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hybrid_policy(
    logits, q4_calibration, q4_thresholds, q1_calibration, q1_thresholds,
    frozen_q4_probabilities=None, frozen_q4_predictions=None,
):
    """Use Q4 policy except for MFT, which reuses the monotone Q1 policy."""
    probabilities = _apply_frozen_calibration(logits, q4_calibration)
    q1_probabilities = _apply_frozen_calibration(logits, q1_calibration)
    thresholds = np.asarray(q4_thresholds, dtype=np.float32).copy()
    probabilities[:, MFT_INDEX] = q1_probabilities[:, MFT_INDEX]
    thresholds[MFT_INDEX] = np.asarray(q1_thresholds, dtype=np.float32)[MFT_INDEX]
    predictions = probabilities >= thresholds[None, :]
    non_mft = [0, 1, 2, 4, 5]
    if frozen_q4_probabilities is not None:
        probabilities[:, non_mft] = np.asarray(frozen_q4_probabilities, dtype=np.float32)[:, non_mft]
    if frozen_q4_predictions is not None:
        predictions[:, non_mft] = np.asarray(frozen_q4_predictions, dtype=bool)[:, non_mft]
    return probabilities, predictions, thresholds


def _dataset_rows(name, prediction_root, calibration_root, output_root):
    rows = []
    for seed in SEEDS:
        pred_seed = prediction_root / f"seed_{seed}"
        cal_seed = calibration_root / f"seed_{seed}"
        q0 = np.load(pred_seed / "Q0_H1" / "predictions.npz")
        q1 = np.load(pred_seed / "Q1_H1B2" / "predictions.npz")
        q4 = np.load(pred_seed / "Q4_VFTAlignedRidgeFusion" / "predictions.npz")
        labels = np.asarray(q4["targets"], dtype=np.uint8)
        weak = np.asarray(q4["weak_mask"], dtype=bool)
        logits = np.asarray(q4["logits"], dtype=np.float32)
        q4_cal = _load_json(cal_seed / "Q4_VFTAlignedRidgeFusion" / "calibration.json")
        q4_thr = _load_json(cal_seed / "Q4_VFTAlignedRidgeFusion" / "thresholds.json")["thresholds"]
        q1_cal = _load_json(cal_seed / "Q1_H1B2" / "calibration.json")
        q1_thr = _load_json(cal_seed / "Q1_H1B2" / "thresholds.json")["thresholds"]
        base_probability = np.asarray(q1["probabilities"], dtype=np.float32)
        base_prediction = np.asarray(q1["predictions"], dtype=bool)
        q4_probability = np.asarray(q4["probabilities"], dtype=np.float32)
        q4_prediction = np.asarray(q4["predictions"], dtype=bool)
        probability, prediction, thresholds = hybrid_policy(
            logits, q4_cal, q4_thr, q1_cal, q1_thr, q4_probability, q4_prediction
        )
        negative_to_positive = int(((~base_prediction[:, MFT_INDEX]) & prediction[:, MFT_INDEX]).sum())
        non_mft = [0, 1, 2, 4, 5]
        non_mft_probability_max_abs = float(np.max(np.abs(probability[:, non_mft] - q4_probability[:, non_mft])))
        non_mft_prediction_mismatch = int((prediction[:, non_mft] != q4_prediction[:, non_mft]).sum())
        metrics = _metrics(labels, weak, probability, prediction)
        base_metrics = _metrics(labels, weak, base_probability, base_prediction)
        h1_metrics = _metrics(
            labels, weak, np.asarray(q0["probabilities"], dtype=np.float32), np.asarray(q0["predictions"], dtype=bool)
        )
        hard = _hard_metrics(labels, weak, probability, prediction)
        base_hard = _hard_metrics(labels, weak, base_probability, base_prediction)
        row = {
            "seed": seed,
            "negative_to_positive": negative_to_positive,
            "non_mft_probability_max_abs": non_mft_probability_max_abs,
            "non_mft_prediction_mismatch": non_mft_prediction_mismatch,
            "mft_hard_fpr_delta_vs_q1": hard["hard_fpr"] - base_hard["hard_fpr"],
            "mft_order5_fpr_delta_vs_q1": hard["order5_fpr"] - base_hard["order5_fpr"],
            "mft_weak_auroc_delta_vs_q1": metrics["per_label"]["MFT"]["weak_vs_negative_auroc"] - base_metrics["per_label"]["MFT"]["weak_vs_negative_auroc"],
            "mft_weak_recall_delta_vs_q1": metrics["per_label"]["MFT"]["weak_recall"] - base_metrics["per_label"]["MFT"]["weak_recall"],
            "macro_aupr_delta_vs_h1": metrics["macro_aupr"] - h1_metrics["macro_aupr"],
            "macro_f1_delta_vs_h1": metrics["macro_f1"] - h1_metrics["macro_f1"],
            "target_weak_recall_delta_vs_h1": metrics["target_mean_weak_recall"] - h1_metrics["target_mean_weak_recall"],
            "sawfm_aupr_delta_vs_h1": metrics["per_label"]["SawFM"]["aupr"] - h1_metrics["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall_delta_vs_h1": metrics["per_label"]["SawFM"]["weak_recall"] - h1_metrics["per_label"]["SawFM"]["weak_recall"],
            "metrics": metrics,
            "mft_hard": hard,
        }
        seed_root = output_root / name / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        _write_json(seed_root / "metrics.json", row)
        np.savez_compressed(
            seed_root / "predictions.npz", targets=labels, weak_mask=weak.astype(np.uint8), logits=logits,
            probabilities=probability, predictions=prediction.astype(np.uint8), thresholds=thresholds,
        )
        rows.append(row)
    return rows


def _audit(name, rows):
    scalar = (
        "mft_hard_fpr_delta_vs_q1", "mft_order5_fpr_delta_vs_q1", "mft_weak_auroc_delta_vs_q1",
        "mft_weak_recall_delta_vs_q1", "macro_aupr_delta_vs_h1", "macro_f1_delta_vs_h1",
        "target_weak_recall_delta_vs_h1", "sawfm_aupr_delta_vs_h1", "sawfm_weak_recall_delta_vs_h1",
    )
    aggregate = {key: _mean_ci([row[key] for row in rows], 20426000 + i) for i, key in enumerate(scalar)}
    checks = {
        "zero_negative_to_positive": sum(row["negative_to_positive"] for row in rows) == 0,
        "non_mft_probability_exact": max(row["non_mft_probability_max_abs"] for row in rows) == 0.0,
        "non_mft_prediction_exact": sum(row["non_mft_prediction_mismatch"] for row in rows) == 0,
        "hard_fpr_delta_le_minus0p010": aggregate["mft_hard_fpr_delta_vs_q1"]["mean"] <= -0.010,
        "order5_fpr_delta_le_minus0p010": aggregate["mft_order5_fpr_delta_vs_q1"]["mean"] <= -0.010,
        "mft_weak_auroc_gain_ge_0p010": aggregate["mft_weak_auroc_delta_vs_q1"]["mean"] >= 0.010,
        "mft_weak_recall_loss_le_0p020": aggregate["mft_weak_recall_delta_vs_q1"]["mean"] >= -0.020,
        "macro_aupr_gain_ge_0p005": aggregate["macro_aupr_delta_vs_h1"]["mean"] >= 0.005,
        "target_weak_recall_gain_ge_0p030": aggregate["target_weak_recall_delta_vs_h1"]["mean"] >= 0.030,
        "sawfm_aupr_exact": max(abs(row["sawfm_aupr_delta_vs_h1"]) for row in rows) <= 1e-12,
        "sawfm_weak_recall_exact": max(abs(row["sawfm_weak_recall_delta_vs_h1"]) for row in rows) <= 1e-12,
    }
    return {"dataset": name, "passed": all(checks.values()), "checks": checks, "aggregate": aggregate}


def run(args):
    seal = _load_json(args.test_v3_manifest)
    if seal.get("access_authorized") or seal.get("analysis_allowed") or seal.get("used"):
        raise ValueError("test-v3 seal is not closed")
    frozen = _load_json(args.frozen_manifest)
    if frozen["test_v3"] != {"access_authorized": False, "analysis_allowed": False, "used": False}:
        raise ValueError("frozen candidate test-v3 state is not sealed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = {
        "construction_v3": (args.construction_root, args.construction_root),
        "diagnostic_v11": (args.v11_root, args.construction_root),
    }
    audits = {}
    for name, (prediction_root, calibration_root) in datasets.items():
        rows = _dataset_rows(name, prediction_root, calibration_root, args.output_root)
        audits[name] = _audit(name, rows)
    summary = {
        "status": "completed", "experiment": "RouteA_MFT_SharedDecisionPolicy_Audit",
        "evidence_role": {"construction_v3": "construction", "development_v11": "diagnostic_reuse"},
        "external_refit": False, "hyperparameter_search": False, "locked_test_used": False,
        "test_v3_used": False, "frozen_manifest_sha256": _sha256(args.frozen_manifest),
        "test_v3_manifest_sha256": _sha256(args.test_v3_manifest), "audits": audits,
        "passed": all(item["passed"] for item in audits.values()),
    }
    _write_json(args.output_root / "route_a_mft_shared_decision_policy_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--v11-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v11_frozen")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_unified_v1/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_shared_decision_policy_audit")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
