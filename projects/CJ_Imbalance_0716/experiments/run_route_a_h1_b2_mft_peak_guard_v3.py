"""Attach the frozen MFT peak-count negative guard to the strong H1+B2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch  # Load Windows DLLs before sklearn/scipy imports in dependencies.
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits  # noqa: E402


MFT_INDEX = 3
TARGET_INDICES = (2, 3, 5)
NON_MFT = (0, 1, 2, 4, 5)
VARIANTS = (
    "P0_H1",
    "P1_H1B2",
    "P2_H1B2_MFTPeakGuard",
    "P3_H1B2_WrongChirpGuard",
    "P4_H1B2_ShuffledGuard",
)
EVIDENCE_KEYS = {
    VARIANTS[2]: "K1_AlignedPeakNegativeGuard__early",
    VARIANTS[3]: "K2_WrongChirpNegativeGuard__early",
    VARIANTS[4]: "K3_ShuffledAlignedGuard__early",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(logits, targets, weak, calibration, thresholds):
    probabilities = _apply_frozen_calibration(logits, calibration)
    predictions = probabilities >= thresholds[None, :]
    metrics = _metrics(targets, weak, probabilities, predictions)
    order = targets.sum(axis=1)
    positive = targets[:, MFT_INDEX] == 1
    weak_positive = positive & weak[:, MFT_INDEX]
    hard_negative = (~positive) & (order >= 4)
    ranked = weak_positive | hard_negative
    metrics["mft_order_ge4_fpr"] = float(predictions[hard_negative, MFT_INDEX].mean())
    metrics["mft_weak_vs_order_ge4_negative_auroc"] = float(
        roc_auc_score(weak_positive[ranked].astype(np.uint8), probabilities[ranked, MFT_INDEX])
    )
    saw_present = targets[:, 1] == 1
    metrics["subgroups"] = {}
    for name, mask in (("sawfm_present", saw_present), ("sawfm_absent", ~saw_present)):
        metrics["subgroups"][name] = {
            "macro_f1": float(
                f1_score(targets[mask], predictions[mask], average="macro", zero_division=0)
            )
        }
    return metrics, probabilities, predictions


def per_label_aupr_delta(candidate, baseline):
    return {
        label: float(candidate["per_label"][label]["aupr"] - baseline["per_label"][label]["aupr"])
        for label in ("MFT", "VFT", "ISRJ")
    }


def run_seed(args, seed):
    source_seed = args.h1_b2_root / f"seed_{seed}"
    evidence_path = args.evidence_root / f"seed_{seed}" / "experts" / "all_role_expert_evidence.npz"
    base_npz = np.load(source_seed / "B0_H1" / "predictions.npz")
    b2_npz = np.load(source_seed / "B2_H1AlignedPeriodicity" / "predictions.npz")
    evidence_npz = np.load(evidence_path)
    if not np.array_equal(base_npz["sample_ids"], b2_npz["sample_ids"]):
        raise ValueError(f"sample identity mismatch for seed {seed}")
    targets = np.asarray(b2_npz["targets"], dtype=np.uint8)
    weak = np.asarray(b2_npz["weak_mask"], dtype=bool)
    p0_logits = np.asarray(base_npz["logits"], dtype=np.float32)
    p1_logits = np.asarray(b2_npz["logits"], dtype=np.float32)
    logits = {VARIANTS[0]: p0_logits, VARIANTS[1]: p1_logits}
    residuals = {
        VARIANTS[0]: np.zeros(len(targets), dtype=np.float32),
        VARIANTS[1]: np.zeros(len(targets), dtype=np.float32),
    }
    for variant in VARIANTS[2:]:
        evidence = np.asarray(evidence_npz[EVIDENCE_KEYS[variant]], dtype=np.float32)
        logits[variant], residuals[variant] = guarded_logits(p1_logits, evidence, args.residual_cap)
    p0_calibration = load_json(source_seed / "B0_H1" / "calibration.json")
    p0_thresholds = np.asarray(
        load_json(source_seed / "B0_H1" / "thresholds.json")["thresholds"],
        dtype=np.float32,
    )
    b2_calibration = load_json(source_seed / "B2_H1AlignedPeriodicity" / "calibration.json")
    b2_thresholds = np.asarray(
        load_json(source_seed / "B2_H1AlignedPeriodicity" / "thresholds.json")["thresholds"],
        dtype=np.float32,
    )
    output_seed = args.output_root / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant in VARIANTS:
        calibration = p0_calibration if variant == VARIANTS[0] else b2_calibration
        thresholds = p0_thresholds if variant == VARIANTS[0] else b2_thresholds
        metrics, probabilities, predictions = evaluate(
            logits[variant], targets, weak, calibration, thresholds
        )
        out = output_seed / variant
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "metrics.json", metrics)
        np.savez_compressed(
            out / "predictions.npz",
            sample_ids=b2_npz["sample_ids"],
            targets=targets,
            weak_mask=weak.astype(np.uint8),
            logits=logits[variant],
            probabilities=probabilities,
            predictions=predictions.astype(np.uint8),
            thresholds=thresholds,
            residual=residuals[variant],
        )
        variants.append({"variant": variant, "metrics": metrics})
    row = {
        "seed": int(seed),
        "status": "completed",
        "source_b0_sha256": _sha256(source_seed / "B0_H1" / "predictions.npz"),
        "source_b2_sha256": _sha256(source_seed / "B2_H1AlignedPeriodicity" / "predictions.npz"),
        "evidence_sha256": _sha256(evidence_path),
        "variants": variants,
        "non_mft_logit_max_abs": {
            variant: float(np.max(np.abs(logits[variant][:, NON_MFT] - p1_logits[:, NON_MFT])))
            for variant in VARIANTS[2:]
        },
        "calibration_reused": True,
        "thresholds_reused": True,
        "residual_cap": float(args.residual_cap),
        "locked_test_used": False,
        "test_v3_used": False,
    }
    _write_json(output_seed / "seed_summary.json", row)
    return row


def aggregate(rows):
    by_seed = []
    promotion = []
    for row in rows:
        variants = {item["variant"]: item["metrics"] for item in row["variants"]}
        p0, p1, p2, p3, p4 = (variants[name] for name in VARIANTS)
        by_seed.append(
            {
                "seed": row["seed"],
                "macro_aupr": p2["macro_aupr"] - p1["macro_aupr"],
                "macro_f1": p2["macro_f1"] - p1["macro_f1"],
                "target_weak_recall": p2["target_mean_weak_recall"] - p1["target_mean_weak_recall"],
                "target_weak_auroc": p2["target_weak_vs_negative_auroc"] - p1["target_weak_vs_negative_auroc"],
                "hard_fpr": p2["mft_order_ge4_fpr"] - p1["mft_order_ge4_fpr"],
                "hard_auroc": p2["mft_weak_vs_order_ge4_negative_auroc"] - p1["mft_weak_vs_order_ge4_negative_auroc"],
                "vs_wrong_hard_auroc": p2["mft_weak_vs_order_ge4_negative_auroc"] - p3["mft_weak_vs_order_ge4_negative_auroc"],
                "vs_shuffled_hard_auroc": p2["mft_weak_vs_order_ge4_negative_auroc"] - p4["mft_weak_vs_order_ge4_negative_auroc"],
            }
        )
        promotion.append(
            {
                "seed": row["seed"],
                "macro_aupr": p2["macro_aupr"] - p0["macro_aupr"],
                "macro_f1": p2["macro_f1"] - p0["macro_f1"],
                "target_weak_recall": p2["target_mean_weak_recall"] - p0["target_mean_weak_recall"],
                "target_weak_auroc": p2["target_weak_vs_negative_auroc"] - p0["target_weak_vs_negative_auroc"],
                "target_fpr": p2["target_mean_negative_fpr"],
                "target_strong_recall": p2["target_mean_strong_recall"] - p0["target_mean_strong_recall"],
                "sawfm_aupr": p2["per_label"]["SawFM"]["aupr"] - p0["per_label"]["SawFM"]["aupr"],
                "sawfm_weak_recall": p2["per_label"]["SawFM"]["weak_recall"] - p0["per_label"]["SawFM"]["weak_recall"],
                "sawfm_present_macro_f1": p2["subgroups"]["sawfm_present"]["macro_f1"] - p0["subgroups"]["sawfm_present"]["macro_f1"],
                "sawfm_absent_macro_f1": p2["subgroups"]["sawfm_absent"]["macro_f1"] - p0["subgroups"]["sawfm_absent"]["macro_f1"],
                "target_aupr_delta": per_label_aupr_delta(p2, p0),
            }
        )
    mean = {key: float(np.mean([item[key] for item in by_seed])) for key in by_seed[0] if key != "seed"}
    promotion_mean = {
        key: float(np.mean([item[key] for item in promotion]))
        for key in promotion[0]
        if key not in ("seed", "target_aupr_delta")
    }
    target_aupr = {
        label: float(np.mean([item["target_aupr_delta"][label] for item in promotion]))
        for label in ("MFT", "VFT", "ISRJ")
    }
    construction_checks = {
        "macro_aupr_decline_le_0p002": mean["macro_aupr"] >= -0.002,
        "target_weak_recall_decline_le_0p010": mean["target_weak_recall"] >= -0.010,
        "target_weak_auroc_decline_le_0p002": mean["target_weak_auroc"] >= -0.002,
        "hard_fpr_decrease_ge_0p020": mean["hard_fpr"] <= -0.020,
        "hard_auroc_gain_ge_0p010": mean["hard_auroc"] >= 0.010,
        "aligned_vs_wrong_hard_auroc_ge_0p005": mean["vs_wrong_hard_auroc"] >= 0.005,
        "aligned_vs_shuffled_hard_auroc_ge_0p005": mean["vs_shuffled_hard_auroc"] >= 0.005,
        "non_mft_logits_exact": all(row["non_mft_logit_max_abs"][VARIANTS[2]] == 0.0 for row in rows),
        "all_seeds_reduce_hard_fpr": all(item["hard_fpr"] < 0.0 for item in by_seed),
    }
    promotion_checks = {
        "macro_aupr_gain_ge_0p005": promotion_mean["macro_aupr"] >= 0.005,
        "macro_f1_gain_ge_0p010": promotion_mean["macro_f1"] >= 0.010,
        "target_weak_recall_gain_ge_0p030": promotion_mean["target_weak_recall"] >= 0.030,
        "target_weak_auroc_gain_ge_0p015": promotion_mean["target_weak_auroc"] >= 0.015,
        "sawfm_aupr_preserved": promotion_mean["sawfm_aupr"] >= -1e-12,
        "sawfm_weak_recall_preserved": promotion_mean["sawfm_weak_recall"] >= -1e-12,
        "target_aupr_decline_le_0p010": min(target_aupr.values()) >= -0.010,
        "target_mean_fpr_le_0p20": promotion_mean["target_fpr"] <= 0.20,
        "strong_recall_decline_le_0p010": promotion_mean["target_strong_recall"] >= -0.010,
        "positive_weak_recall_seeds_ge_4": sum(item["target_weak_recall"] > 0 for item in promotion) >= 4,
        "sawfm_present_macro_f1_not_regressed": promotion_mean["sawfm_present_macro_f1"] >= -0.010,
        "sawfm_absent_macro_f1_not_regressed": promotion_mean["sawfm_absent_macro_f1"] >= -0.010,
    }
    return {
        "construction_gate": {
            "passed": bool(all(construction_checks.values())),
            "checks": {key: bool(value) for key, value in construction_checks.items()},
            "mean_delta_p2_vs_p1": mean,
            "per_seed": by_seed,
        },
        "unified_promotion_gate": {
            "passed": bool(all(promotion_checks.values())),
            "checks": {key: bool(value) for key, value in promotion_checks.items()},
            "mean_p2_vs_p0": promotion_mean,
            "target_aupr_delta": target_aupr,
            "per_seed": promotion,
        },
    }


def run(args):
    seal = load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    rows = [run_seed(args, seed) for seed in args.seeds]
    gates = aggregate(rows)
    result = {
        "status": "completed",
        "experiment": "RouteA_H1B2_MFTPeakGuard_v3",
        "candidate": VARIANTS[2],
        "seeds": args.seeds,
        "rows": rows,
        **gates,
        "next_action": "fresh_confirmation" if gates["construction_gate"]["passed"] else "stop_integration",
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_h1_b2_mft_peak_guard_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"summary": str(output), **gates}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-b2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_h1_periodicity_increment")
    parser.add_argument("--evidence-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_peak_guard_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--residual-cap", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
