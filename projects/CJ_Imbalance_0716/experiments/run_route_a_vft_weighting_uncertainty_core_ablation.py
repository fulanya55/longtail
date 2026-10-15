"""Direct VFT weighting and uncertainty-gate ablation on frozen v3 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _fit_head, _head_scores, _replace_vft, _ridge_scores  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _fit_score_stats, _standardize_scores  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT_INDEX = 4
NON_VFT = (0, 1, 2, 3, 5)
ARMS = ("A0_UniformBCE_NoGate", "A1_WeakWeight_NoGate", "A2_ThreeStratum_NoGate", "A3_WeakWeight_SqrtGate")


def _features(logits: np.ndarray, z: np.ndarray, gate: str) -> np.ndarray:
    base = np.asarray(logits, dtype=np.float32)[:, VFT_INDEX]
    if gate == "constant":
        value = np.ones_like(base)
    elif gate == "sqrt":
        probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
        value = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    else:
        raise ValueError(f"unknown gate {gate}")
    return np.stack((base, z, value * z, np.tanh(base) * z), axis=1).astype(np.float32)


def _weights(labels: np.ndarray, weak: np.ndarray, arm: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    weak = np.asarray(weak, dtype=bool)
    if arm == "A0_UniformBCE_NoGate":
        return np.ones(len(labels), dtype=np.float64)
    if arm in ("A1_WeakWeight_NoGate", "A3_WeakWeight_SqrtGate"):
        return np.where((labels == 1) & weak, 4.0, 1.0).astype(np.float64)
    if arm == "A2_ThreeStratum_NoGate":
        masks = ((labels == 1) & weak, (labels == 1) & ~weak, labels == 0)
        if any(not mask.any() for mask in masks):
            raise ValueError("three-stratum arm requires weak, strong, and negative fit samples")
        value = np.zeros(len(labels), dtype=np.float64)
        for mask in masks:
            value[mask] = len(labels) / (3.0 * float(mask.sum()))
        return value
    raise ValueError(f"unknown arm {arm}")


def _metric(item: dict, name: str) -> float:
    return float(item["metrics"]["per_label"]["VFT"][name])


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary = root / "seed_summary.json"
    if args.resume and summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    previous = np.load(args.construction_root / f"seed_{seed}" / "all_role_logits.npz")
    q2 = {role: np.asarray(previous[f"Q2_H1B2_MFTGuard__{role}"], dtype=np.float32) for role in ALL_ROLES}
    reference = np.asarray(previous["Q4_VFTAlignedRidgeFusion__early"], dtype=np.float32)[:, VFT_INDEX]
    ridge = np.load(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz")
    raw_score = {role: _ridge_scores(_extract_vft(roles[role]["signals"], device, args.feature_batch_size), ridge) for role in ALL_ROLES}
    fit_labels = roles["fit"]["targets6"][:, VFT_INDEX].astype(np.uint8)
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, VFT_INDEX]
    strict = ((fit_labels == 1) & fit_weak) | (fit_labels == 0)
    stats = _fit_score_stats(raw_score["fit"][strict])
    z = {role: _standardize_scores(raw_score[role], stats) for role in ALL_ROLES}
    rows, arrays = [], {}
    for index, arm in enumerate(ARMS):
        gate = "sqrt" if arm == "A3_WeakWeight_SqrtGate" else "constant"
        feature_sets = {role: _features(q2[role], z[role], gate) for role in ALL_ROLES}
        model, mean, std = _fit_head(feature_sets["fit"], fit_labels, _weights(fit_labels, fit_weak, arm), 20422000 + int(seed) + (1 if arm == "A3_WeakWeight_SqrtGate" else 100 + index), args.c_value)
        logits = {role: _replace_vft(q2[role], _head_scores(model, feature_sets[role], mean, std)) for role in ALL_ROLES}
        if any(np.max(np.abs(logits[role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0.0 for role in ALL_ROLES):
            raise AssertionError("an ablation changed a non-VFT logit")
        output = root / arm; output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output / "model.joblib")
        np.savez_compressed(output / "normalization.npz", mean=mean, std=std, score_mean=stats["mean"], score_std=stats["std"])
        result = _evaluate(arm, logits, roles, weak6, output, args)
        result.update({"feature_dim": int(feature_sets["fit"].shape[1]), "fit_weight_sum": float(_weights(fit_labels, fit_weak, arm).sum())})
        rows.append(result)
        arrays[f"{arm}_early_vft_logits"] = logits["early"][:, VFT_INDEX]
    reproduced = arrays["A3_WeakWeight_SqrtGate_early_vft_logits"]
    row = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
           "ridge_sha256": _sha256(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz"), "q2_q4_source_sha256": _sha256(args.construction_root / f"seed_{seed}" / "all_role_logits.npz"),
           "a3_q4_reproduction_max_abs": float(np.max(np.abs(reproduced - reference))), "non_vft_logits_exact": True, "variants": rows,
           "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    np.savez_compressed(root / "early_vft_logits.npz", **arrays)
    _write_json(summary, row); return row


def _variant_map(row: dict) -> dict:
    return {item["variant"]: item for item in row["variants"]}


def _delta(rows: list[dict], left: str, right: str, metric: str, seed_base: int):
    values = [_metric(_variant_map(row)[left], metric) - _metric(_variant_map(row)[right], metric) for row in rows]
    return {"summary": _mean_ci(values, seed_base), "positive_seeds": int(sum(value > 0 for value in values)), "values": [float(value) for value in values]}


def _mechanism(rows: list[dict], left: str, right: str, kind: str, seed_base: int):
    weak_auroc = _delta(rows, left, right, "weak_vs_negative_auroc", seed_base)
    weak_recall = _delta(rows, left, right, "weak_recall", seed_base + 1)
    strong_recall = _delta(rows, left, right, "strong_recall", seed_base + 2)
    fpr = _delta(rows, left, right, "negative_fpr", seed_base + 3)
    left_fpr = float(np.mean([_metric(_variant_map(row)[left], "negative_fpr") for row in rows]))
    if kind == "gain":
        checks = {"weak_auroc_gain_ge_0p015": weak_auroc["summary"]["mean"] >= .015,
                  "weak_recall_gain_ge_0p030": weak_recall["summary"]["mean"] >= .030,
                  "weak_auroc_positive_four_of_five": weak_auroc["positive_seeds"] >= 4,
                  "weak_recall_positive_four_of_five": weak_recall["positive_seeds"] >= 4,
                  "vft_fpr_le_0p20": left_fpr <= .20,
                  "strong_recall_loss_le_0p010": strong_recall["summary"]["mean"] >= -.010}
    else:
        checks = {"weak_auroc_loss_le_0p010": weak_auroc["summary"]["mean"] >= -.010,
                  "fpr_increase_le_0p010": fpr["summary"]["mean"] <= .010,
                  "strong_recall_loss_le_0p010": strong_recall["summary"]["mean"] >= -.010,
                  "vft_fpr_le_0p20": left_fpr <= .20}
    return {"comparison": f"{left}_minus_{right}", "weak_auroc": weak_auroc, "weak_recall": weak_recall, "strong_recall": strong_recall, "negative_fpr": fpr, "left_vft_fpr": left_fpr, "checks": checks, "passed": bool(all(checks.values()))}


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(data_path); values, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in seeds]
    checks = {} if args.smoke else {"a3_reproduces_q4": max(row["a3_q4_reproduction_max_abs"] for row in rows) <= 1e-5,
                                    "non_vft_logits_exact": all(row["non_vft_logits_exact"] for row in rows)}
    comparisons = {} if args.smoke else {"weak_weighting": _mechanism(rows, ARMS[1], ARMS[0], "gain", 2026082100),
                                         "three_stratum": _mechanism(rows, ARMS[2], ARMS[0], "gain", 2026082200),
                                         "three_stratum_vs_weak_only": _mechanism(rows, ARMS[2], ARMS[1], "gain", 2026082300),
                                         "sqrt_gate_safety": _mechanism(rows, ARMS[3], ARMS[1], "safety", 2026082400)}
    payload = {"status": "completed", "experiment": "RouteA_VFTWeightingUncertaintyCoreAblation", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "arms": list(ARMS), "implementation_checks": checks, "mechanism_comparisons": comparisons,
               "scope": "current_vft_head_core_ablation_only_no_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_weighting_uncertainty_core_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--vft-ridge-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_ridge_stft_separability_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_weighting_uncertainty_core_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--c-value", type=float, default=.1); parser.add_argument("--max-negative-fpr", type=float, default=.2); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
