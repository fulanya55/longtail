"""Bounded integration of the frozen ISRJ joint-periodicity HGB into C4."""

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
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import (  # noqa: E402
    _block_permutation, _block_shuffle, _extract,
)
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import (  # noqa: E402
    ALL_ROLES, _load_c4_components, _reconstruct_c4,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256, _evaluate, _extract_backbone, _load_roles, _validate_config,
)
from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import (  # noqa: E402
    _fit_score_stats, _standardize_scores,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


ISRJ_INDEX = 5
VARIANTS = ("I0_C4_Reconstructed", "I1_ISRJ_AlignedPeriodicityResidual",
            "I2_ISRJ_BlockShuffledResidual")


METRIC_PATHS = {
    "macro_aupr": ("metrics", "macro_aupr"),
    "macro_f1": ("metrics", "macro_f1"),
    "isrj_aupr": ("metrics", "per_label", "ISRJ", "aupr"),
    "isrj_weak_recall": ("metrics", "per_label", "ISRJ", "weak_recall"),
    "isrj_strong_recall": ("metrics", "per_label", "ISRJ", "strong_recall"),
    "isrj_negative_fpr": ("metrics", "per_label", "ISRJ", "negative_fpr"),
    "isrj_weak_auroc": ("metrics", "per_label", "ISRJ", "weak_vs_negative_auroc"),
    "sawfm_aupr": ("metrics", "per_label", "SawFM", "aupr"),
    "sawfm_weak_recall": ("metrics", "per_label", "SawFM", "weak_recall"),
    "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
    "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
}


def _clipped_logit(probability):
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def _bounded_isrj_residual(c4_logits, expert_z, residual_cap=0.50, uncertainty_power=0.50):
    logits = np.asarray(c4_logits, dtype=np.float32)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[:, ISRJ_INDEX], -30.0, 30.0)))
    uncertainty = np.maximum(4.0 * probability * (1.0 - probability), 0.0) ** float(uncertainty_power)
    residual = float(residual_cap) * uncertainty * np.tanh(np.asarray(expert_z, dtype=np.float32))
    output = logits.copy(); output[:, ISRJ_INDEX] += residual.astype(np.float32)
    return output, residual.astype(np.float32), uncertainty.astype(np.float32)


def _metric(result, path):
    value = result
    for key in path: value = value[key]
    return float(value)


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _compare(rows, candidate, baseline, seed_offset):
    per_seed = []
    for row in rows:
        variants = _variant_map(row)
        deltas = {name: _metric(variants[candidate], path) - _metric(variants[baseline], path)
                  for name, path in METRIC_PATHS.items()}
        per_seed.append({"seed": row["seed"], **deltas})
    aggregate = {name: _mean_ci([row[name] for row in per_seed], seed_offset + offset)
                 for offset, name in enumerate(METRIC_PATHS)}
    aggregate["positive_isrj_weak_recall_seeds"] = int(sum(row["isrj_weak_recall"] > 0 for row in per_seed))
    aggregate["positive_isrj_weak_auroc_seeds"] = int(sum(row["isrj_weak_auroc"] > 0 for row in per_seed))
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _load_expert(source_root, seed, name):
    root = source_root / f"seed_{seed}" / name
    model = joblib.load(root / "model.joblib")
    normalization = np.load(root / "normalization.npz")
    return model, np.asarray(normalization["mean"], dtype=np.float32), np.asarray(normalization["std"], dtype=np.float32), {
        "model_sha256": _sha256(root / "model.joblib"),
        "normalization_sha256": _sha256(root / "normalization.npz"),
    }


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_path, map_location="cpu", weights_only=False)["model_state_dict"]
    backbone = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state)
    extracted = _extract_backbone(backbone, roles, device, args.backbone_batch_size, args.topk)
    del backbone
    if device.type == "cuda": torch.cuda.empty_cache()
    raw, _, probe, probe_payload, fusion, fusion_payload, c4_hashes = _load_c4_components(args, seed, device)
    c4 = _reconstruct_c4(args, roles, extracted, raw, probe, probe_payload, fusion, fusion_payload, device)
    source_prediction = np.load(args.unified_root / f"seed_{seed}" / "C4_SameCapacity_NoVisibility" / "predictions.npz")
    reconstruction_error = float(np.max(np.abs(c4["early"] - source_prediction["logits"])))
    if reconstruction_error > args.maximum_reconstruction_error:
        raise AssertionError(f"C4 reconstruction mismatch: {reconstruction_error}")

    aligned_model, aligned_mean, aligned_std, aligned_hashes = _load_expert(
        args.periodicity_root, seed, "H_AlignedHGB")
    shuffled_model, shuffled_mean, shuffled_std, shuffled_hashes = _load_expert(
        args.periodicity_root, seed, "S_BlockShuffledHGB")
    permutation = _block_permutation(120, 20301000 + int(seed))[0]
    aligned_features = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    shuffled_features = {role: _extract(_block_shuffle(roles[role]["signals"], permutation, args.block_size),
                                        device, args.feature_batch_size) for role in ALL_ROLES}
    aligned_raw, shuffled_raw = {}, {}
    for role in ALL_ROLES:
        aligned_x = ((aligned_features[role] - aligned_mean[None, :]) / np.maximum(aligned_std[None, :], 1e-5)).astype(np.float32)
        shuffled_x = ((shuffled_features[role] - shuffled_mean[None, :]) / np.maximum(shuffled_std[None, :], 1e-5)).astype(np.float32)
        aligned_raw[role] = _clipped_logit(aligned_model.predict_proba(aligned_x)[:, 1])
        shuffled_raw[role] = _clipped_logit(shuffled_model.predict_proba(shuffled_x)[:, 1])
    fit_selected, _, fit_receipt = _matched_view(
        roles["fit"]["targets6"], weak6[roles["fit"]["sample_ids"]], ISRJ_INDEX)
    aligned_stats = _fit_score_stats(aligned_raw["fit"][fit_selected])
    shuffled_stats = _fit_score_stats(shuffled_raw["fit"][fit_selected])
    aligned_z = {role: _standardize_scores(aligned_raw[role], aligned_stats) for role in ALL_ROLES}
    shuffled_z = {role: _standardize_scores(shuffled_raw[role], shuffled_stats) for role in ALL_ROLES}
    aligned_logits, shuffled_logits, arrays = {}, {}, {}
    for role in ALL_ROLES:
        aligned_logits[role], aligned_residual, uncertainty = _bounded_isrj_residual(
            c4[role], aligned_z[role], args.residual_cap, args.uncertainty_power)
        shuffled_logits[role], shuffled_residual, _ = _bounded_isrj_residual(
            c4[role], shuffled_z[role], args.residual_cap, args.uncertainty_power)
        arrays.update({f"{role}_sample_ids": roles[role]["sample_ids"],
                       f"{role}_aligned_z": aligned_z[role], f"{role}_shuffled_z": shuffled_z[role],
                       f"{role}_aligned_residual": aligned_residual,
                       f"{role}_shuffled_residual": shuffled_residual,
                       f"{role}_uncertainty": uncertainty})
    np.savez_compressed(root / "isrj_expert_residuals.npz", **arrays)
    logits = {VARIANTS[0]: c4, VARIANTS[1]: aligned_logits, VARIANTS[2]: shuffled_logits}
    results = [_evaluate(name, logits[name], roles, weak6, root / name, args) for name in VARIANTS]
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "c6_checkpoint_sha256": _sha256(c6_path), "c4_component_sha256": c4_hashes,
               "c4_reconstruction_max_abs_error": reconstruction_error,
               "periodicity_expert_sha256": {"aligned": aligned_hashes, "block_shuffled": shuffled_hashes},
               "fit_matched_receipt": fit_receipt,
               "score_stats": {"aligned": aligned_stats, "block_shuffled": shuffled_stats},
               "residual": {"cap": args.residual_cap, "uncertainty_power": args.uncertainty_power,
                            "learned_integration_parameters": 0},
               "residual_arrays_sha256": _sha256(root / "isrj_expert_residuals.npz"),
               "variants": results, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def run(args):
    config = load_config(args.config); _validate_config(config); data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_ISRJ_PeriodicityBoundedResidual_v3", "seeds": list(seeds),
        "residual_cap": args.residual_cap, "uncertainty_power": args.uncertainty_power,
        "learned_integration_parameters": 0,
        "gates": {"weak_recall_gain": .030, "weak_auroc_gain": .015,
                  "vs_control_weak_recall": .015, "vs_control_weak_auroc": .010,
                  "fpr_cap": .20, "positive_seeds": 4},
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [_variant_map(row)[variant] for row in rows]
        aggregate[variant] = {name: _mean_ci([_metric(item, path) for item in selected], 20305000 + vi * 100 + i)
                              for i, (name, path) in enumerate(METRIC_PATHS.items())}
    comparisons = {"I1_vs_I0": _compare(rows, VARIANTS[1], VARIANTS[0], 20306000),
                   "I1_vs_I2": _compare(rows, VARIANTS[1], VARIANTS[2], 20307000)}
    if args.smoke:
        gate = {"checks": {}, "passed": False, "evidence_level": "smoke_only"}
    else:
        c4 = comparisons["I1_vs_I0"]["aggregate"]; control = comparisons["I1_vs_I2"]["aggregate"]
        candidate = aggregate[VARIANTS[1]]
        checks = {"isrj_weak_recall_gain_ge_0p030": c4["isrj_weak_recall"]["mean"] >= .030,
                  "isrj_weak_auroc_gain_ge_0p015": c4["isrj_weak_auroc"]["mean"] >= .015,
                  "isrj_aupr_nonnegative": c4["isrj_aupr"]["mean"] >= 0.0,
                  "isrj_strong_recall_noninferior": c4["isrj_strong_recall"]["mean"] >= -.010,
                  "isrj_fpr_le_0p20": candidate["isrj_negative_fpr"]["mean"] <= .20,
                  "four_of_five_positive_weak_recall": c4["positive_isrj_weak_recall_seeds"] >= 4,
                  "four_of_five_positive_weak_auroc": c4["positive_isrj_weak_auroc_seeds"] >= 4,
                  "aligned_vs_control_weak_recall_ge_0p015": control["isrj_weak_recall"]["mean"] >= .015,
                  "aligned_vs_control_weak_auroc_ge_0p010": control["isrj_weak_auroc"]["mean"] >= .010,
                  "macro_aupr_positive": c4["macro_aupr"]["mean"] > 0.0,
                  "macro_f1_positive": c4["macro_f1"]["mean"] > 0.0,
                  "sawfm_preserved": c4["sawfm_aupr"]["mean"] >= -1e-8 and c4["sawfm_weak_recall"]["mean"] >= -1e-8,
                  "sawfm_subgroups_noninferior": c4["sawfm_present_macro_f1"]["mean"] >= -.005 and c4["sawfm_absent_macro_f1"]["mean"] >= -.005,
                  "reconstruction_and_test_boundary_safe": all(row["c4_reconstruction_max_abs_error"] <= args.maximum_reconstruction_error and not row["test_v3_used"] for row in rows)}
        gate = {"checks": checks, "passed": bool(all(checks.values())),
                "evidence_level": "five_seed_development_v3_component_integration"}
    payload = {"status": "completed", "experiment": "RouteA_ISRJ_PeriodicityBoundedResidual_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "comparisons": comparisons, "component_promotion_gate": gate,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(args.output_root / "route_a_isrj_periodicity_bounded_residual_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_periodicity_bounded_residual_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043]); parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--backbone-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=64); parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--c4-residual-cap", type=float, default=.50); parser.add_argument("--residual-cap", type=float, default=.50)
    parser.add_argument("--uncertainty-power", type=float, default=.50); parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--max-negative-fpr", type=float, default=.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1e-5); return parser.parse_args()


if __name__ == "__main__": run(parse_args())

