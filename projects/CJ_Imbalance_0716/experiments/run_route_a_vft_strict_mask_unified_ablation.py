"""Direct unified-policy strict weak-mask versus all-positive VFT Ridge study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _fit_head, _fusion_features, _head_scores, _replace_vft, _ridge_scores  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _fit_score_stats, _standardize_scores  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_route_a_vft_weak_mixed_positive_ablation import TARGET_INDEX, _all_positive_matched_view  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT = int(TARGET_INDEX)
NON_VFT = (0, 1, 2, 3, 5)
ARMS = ("StrictWeak_Q4", "AllPositive_NoWeakMask")


def _fit_ridge(features: np.ndarray, labels: np.ndarray, alpha: float) -> RidgeClassifier:
    positives = int(labels.sum()); negatives = int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError("all-positive ridge view requires both classes")
    weight = np.where(labels == 1, .5 / positives, .5 / negatives).astype(np.float64)
    model = RidgeClassifier(alpha=float(alpha), fit_intercept=True)
    model.fit(features, labels, sample_weight=weight)
    return model


def _metric(variant: dict, name: str) -> float:
    return float(variant["metrics"]["per_label"]["VFT"][name])


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary = root / "seed_summary.json"
    if args.resume and summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    stored = np.load(args.construction_root / f"seed_{seed}" / "all_role_logits.npz")
    q2 = {role: np.asarray(stored[f"Q2_H1B2_MFTGuard__{role}"], dtype=np.float32) for role in ALL_ROLES}
    q4 = {role: np.asarray(stored[f"Q4_VFTAlignedRidgeFusion__{role}"], dtype=np.float32) for role in ALL_ROLES}
    features = {role: _extract_vft(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    mean = features["fit"].mean(axis=0).astype(np.float32); std = np.maximum(features["fit"].std(axis=0), 1e-5).astype(np.float32)
    normalised = {role: ((features[role] - mean[None, :]) / std[None, :]).astype(np.float32) for role in ALL_ROLES}
    selected, labels = _all_positive_matched_view(roles["fit"]["targets6"], VFT)[:2]
    ridge = _fit_ridge(normalised["fit"][selected], labels.astype(np.uint8), args.ridge_alpha)
    raw = {role: ridge.decision_function(normalised[role]).astype(np.float32) for role in ALL_ROLES}
    fit_label = roles["fit"]["targets6"][:, VFT].astype(np.uint8)
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, VFT]
    strict = ((fit_label == 1) & fit_weak) | (fit_label == 0)
    stats = _fit_score_stats(raw["fit"][strict])
    z = {role: _standardize_scores(raw[role], stats) for role in ALL_ROLES}
    sets = {role: _fusion_features(q2[role], z[role]) for role in ALL_ROLES}
    weight = np.where((fit_label == 1) & fit_weak, float(args.weak_positive_weight), 1.0).astype(np.float64)
    head, head_mean, head_std = _fit_head(sets["fit"], fit_label, weight, 20422000 + int(seed) + 1, args.c_value)
    all_positive = {role: _replace_vft(q2[role], _head_scores(head, sets[role], head_mean, head_std)) for role in ALL_ROLES}
    if any(np.max(np.abs(all_positive[role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0.0 for role in ALL_ROLES):
        raise AssertionError("all-positive VFT arm changed non-VFT logits")
    variants = []
    for name, logits in ((ARMS[0], q4), (ARMS[1], all_positive)):
        variants.append(_evaluate(name, logits, roles, weak6, root / name, args))
    joblib.dump(ridge, root / "all_positive_ridge.joblib")
    np.savez_compressed(root / "all_positive_ridge_normalization.npz", mean=mean, std=std, score_mean=stats["mean"], score_std=stats["std"], head_mean=head_mean, head_std=head_std)
    joblib.dump(head, root / "all_positive_fusion_head.joblib")
    reference = q4["early"][:, VFT]
    result = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
              "q4_source_sha256": _sha256(args.construction_root / f"seed_{seed}" / "all_role_logits.npz"), "all_positive_view": {"fit_selected": int(selected.sum()), "fit_positive": int(labels.sum())},
              "strict_q4_source_exact": True, "strict_q4_early_logit_sha256": _sha256(root / ARMS[0] / "predictions.npz"), "q4_early_logit_reference_count": int(len(reference)),
              "non_vft_logits_exact": True, "variants": variants, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary, result); return result


def _variant_map(row: dict) -> dict:
    return {item["variant"]: item for item in row["variants"]}


def run(args):
    config = load_config(args.config); _validate_config(config)
    path = _data_path(config); data_sha = _sha256(path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(path); values, _ = read_parameters(path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in seeds]
    comparison = {}
    if not args.smoke:
        for offset, metric in enumerate(("weak_vs_negative_auroc", "weak_recall", "strong_recall", "negative_fpr", "aupr")):
            values = [_metric(_variant_map(row)[ARMS[1]], metric) - _metric(_variant_map(row)[ARMS[0]], metric) for row in rows]
            comparison[metric] = {"summary": _mean_ci(values, 2026083000 + offset), "positive_seeds": int(sum(value > 0 for value in values)), "values": [float(value) for value in values]}
        left_fpr = float(np.mean([_metric(_variant_map(row)[ARMS[1]], "negative_fpr") for row in rows]))
        checks = {"weak_auroc_gain_ge_0p015": comparison["weak_vs_negative_auroc"]["summary"]["mean"] >= .015,
                  "weak_recall_gain_ge_0p030": comparison["weak_recall"]["summary"]["mean"] >= .030,
                  "weak_auroc_positive_four_of_five": comparison["weak_vs_negative_auroc"]["positive_seeds"] >= 4,
                  "weak_recall_positive_four_of_five": comparison["weak_recall"]["positive_seeds"] >= 4,
                  "vft_fpr_le_0p20": left_fpr <= .20,
                  "strong_recall_loss_le_0p010": comparison["strong_recall"]["summary"]["mean"] >= -.010,
                  "non_vft_logits_exact": all(row["non_vft_logits_exact"] for row in rows)}
    else:
        checks = {}
    payload = {"status": "completed", "experiment": "RouteA_VFTStrictMaskUnifiedAblation", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "comparison_all_positive_minus_strict_q4": comparison, "promotion_gate": {"checks": checks, "passed": bool(checks and all(checks.values()))},
               "scope": "current_vft_strict_mask_core_ablation_only_no_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_strict_mask_unified_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_strict_mask_unified_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--c-value", type=float, default=.1); parser.add_argument("--weak-positive-weight", type=float, default=4.0); parser.add_argument("--max-negative-fpr", type=float, default=.2); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
