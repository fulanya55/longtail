"""Inference-available predicted co-label context gate for frozen MFT residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import (  # noqa: E402
    ALL_ROLES,
    METRIC_PATHS,
    MFT_INDEX,
    _bounded_mft_residual,
    _load_c4_components,
    _metric,
    _reconstruct_c4,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    SAW_INDEX,
    _apply_platt,
    _evaluate,
    _extract_backbone,
    _fit_platt,
    _load_roles,
    _validate_config,
)
from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import ROLE_OFFSETS, _fixed_permutation  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT_INDEX = 4
ISRJ_INDEX = 5
VARIANTS = ("K0_C4_Reconstructed", "K1_MFT_GlobalResidual",
            "K2_MFT_PredictedContextResidual", "K3_MFT_ContextShuffledResidual")


def _context_gate(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    product = (values[:, SAW_INDEX] * (1.0 - values[:, VFT_INDEX]) *
               (1.0 - values[:, ISRJ_INDEX]))
    return np.cbrt(np.clip(product, 0.0, 1.0)).astype(np.float32)


def _context_residual(c4_logits, expert_z, context_gate, residual_cap=0.50,
                      uncertainty_power=0.50):
    global_logits, global_residual, uncertainty = _bounded_mft_residual(
        c4_logits, expert_z, residual_cap, uncertainty_power
    )
    del global_logits
    residual = global_residual * np.asarray(context_gate, dtype=np.float32)
    output = np.asarray(c4_logits, dtype=np.float32).copy()
    output[:, MFT_INDEX] += residual
    return output, residual.astype(np.float32), uncertainty


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
    aggregate["positive_mft_weak_recall_seeds"] = int(sum(row["mft_weak_recall"] > 0 for row in per_seed))
    aggregate["positive_mft_weak_auroc_seeds"] = int(sum(row["mft_weak_auroc"] > 0 for row in per_seed))
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _run_seed(args, config, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state)
    extracted = _extract_backbone(model, roles, device, args.feature_batch_size, args.topk)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw, _, probe, probe_payload, fusion, fusion_payload, hashes = _load_c4_components(args, seed, device)
    c4 = _reconstruct_c4(args, roles, extracted, raw, probe, probe_payload, fusion, fusion_payload, device)
    source = np.load(args.unified_root / f"seed_{seed}" / "C4_SameCapacity_NoVisibility" / "predictions.npz")
    reconstruction_error = float(np.max(np.abs(c4["early"] - source["logits"])))
    if reconstruction_error > args.maximum_reconstruction_error:
        raise AssertionError(f"C4 reconstruction mismatch: {reconstruction_error}")

    baseline_calibrator = _fit_platt(c4["calibration"], roles["calibration"]["targets6"])
    context = {role: _context_gate(_apply_platt(baseline_calibrator, c4[role])) for role in ALL_ROLES}
    prior = np.load(args.mft_source_root / f"seed_{seed}" / "expert_residuals.npz")
    expert_z = {}
    for role in ALL_ROLES:
        np.testing.assert_array_equal(prior[f"{role}_sample_ids"], roles[role]["sample_ids"])
        expert_z[role] = prior[f"{role}_expert_z"].astype(np.float32)

    global_logits, context_logits, shuffled_logits = {}, {}, {}
    arrays, permutations = {}, {}
    for role in ALL_ROLES:
        global_logits[role] = _bounded_mft_residual(
            c4[role], expert_z[role], args.residual_cap, args.uncertainty_power
        )[0]
        aligned, residual, uncertainty = _context_residual(
            c4[role], expert_z[role], context[role], args.residual_cap, args.uncertainty_power
        )
        permutation_seed = 20292000 + int(seed) * 100 + ROLE_OFFSETS.get(role, 41 if role == "calibration" else 53)
        permutation, permutation_sha = _fixed_permutation(len(expert_z[role]), permutation_seed)
        shuffled, shuffled_residual, _ = _context_residual(
            c4[role], expert_z[role][permutation], context[role],
            args.residual_cap, args.uncertainty_power,
        )
        context_logits[role], shuffled_logits[role] = aligned, shuffled
        arrays.update({f"{role}_sample_ids": roles[role]["sample_ids"],
                       f"{role}_context_gate": context[role],
                       f"{role}_aligned_residual": residual,
                       f"{role}_shuffled_residual": shuffled_residual,
                       f"{role}_uncertainty": uncertainty,
                       f"{role}_permutation": permutation})
        permutations[role] = {"seed": int(permutation_seed), "sha256": permutation_sha}
    np.savez_compressed(seed_root / "context_residuals.npz", **arrays)
    logits = {VARIANTS[0]: c4, VARIANTS[1]: global_logits,
              VARIANTS[2]: context_logits, VARIANTS[3]: shuffled_logits}
    results = [_evaluate(name, logits[name], roles, weak6, seed_root / name, args) for name in VARIANTS]
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "c6_checkpoint_sha256": _sha256(c6_path), "c4_component_sha256": hashes,
               "mft_source_summary_sha256": _sha256(args.mft_source_root / f"seed_{seed}" / "seed_summary.json"),
               "c4_reconstruction_max_abs_error": reconstruction_error,
               "context_gate": {"formula": "cuberoot(p_sawfm*(1-p_vft)*(1-p_isrj))",
                                "learned_parameters": 0,
                                "fit_mean": float(context["fit"].mean()),
                                "early_mean": float(context["early"].mean())},
               "permutations": permutations, "variants": results,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload)
    return payload


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_MFT_PredictedContextGate_v3", "seeds": list(seeds),
        "formula": "cuberoot(p_sawfm*(1-p_vft)*(1-p_isrj))", "residual_cap": args.residual_cap,
        "uncertainty_power": args.uncertainty_power, "learned_parameters": 0,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [_variant_map(row)[variant] for row in rows]
        aggregate[variant] = {name: _mean_ci([_metric(item, path) for item in selected], 20293000 + vi * 100 + i)
                              for i, (name, path) in enumerate(METRIC_PATHS.items())}
    comparisons = {"K2_vs_K0": _compare(rows, VARIANTS[2], VARIANTS[0], 20294000),
                   "K2_vs_K1": _compare(rows, VARIANTS[2], VARIANTS[1], 20295000),
                   "K2_vs_K3": _compare(rows, VARIANTS[2], VARIANTS[3], 20296000)}
    if args.smoke:
        gate = {"checks": {}, "passed": False, "evidence_level": "smoke_only"}
    else:
        c4 = comparisons["K2_vs_K0"]["aggregate"]
        global_delta = comparisons["K2_vs_K1"]["aggregate"]
        shuffled = comparisons["K2_vs_K3"]["aggregate"]
        candidate = aggregate[VARIANTS[2]]
        checks = {
            "mft_weak_recall_gain_ge_0p030": c4["mft_weak_recall"]["mean"] >= 0.030,
            "mft_weak_auroc_gain_ge_0p015": c4["mft_weak_auroc"]["mean"] >= 0.015,
            "mft_aupr_nonnegative": c4["mft_aupr"]["mean"] >= 0.0,
            "mft_strong_recall_noninferior": c4["mft_strong_recall"]["mean"] >= -0.010,
            "mft_fpr_le_0p20": candidate["mft_negative_fpr"]["mean"] <= 0.20,
            "four_of_five_positive_weak_recall": c4["positive_mft_weak_recall_seeds"] >= 4,
            "four_of_five_positive_weak_auroc": c4["positive_mft_weak_auroc_seeds"] >= 4,
            "weak_auroc_gain_vs_global_ge_0p005": global_delta["mft_weak_auroc"]["mean"] >= 0.005,
            "fpr_not_higher_than_global": global_delta["mft_negative_fpr"]["mean"] <= 0.0,
            "weak_recall_vs_shuffled_ge_0p015": shuffled["mft_weak_recall"]["mean"] >= 0.015,
            "weak_auroc_vs_shuffled_ge_0p010": shuffled["mft_weak_auroc"]["mean"] >= 0.010,
            "macro_aupr_positive": c4["macro_aupr"]["mean"] > 0.0,
            "macro_f1_positive": c4["macro_f1"]["mean"] > 0.0,
            "sawfm_preserved": c4["sawfm_aupr"]["mean"] >= -1e-8 and c4["sawfm_weak_recall"]["mean"] >= -1e-8,
            "sawfm_subgroups_noninferior": c4["sawfm_present_macro_f1"]["mean"] >= -0.005 and c4["sawfm_absent_macro_f1"]["mean"] >= -0.005,
            "reconstruction_and_test_boundary_safe": all(row["c4_reconstruction_max_abs_error"] <= args.maximum_reconstruction_error and not row["test_v3_used"] for row in rows),
        }
        gate = {"checks": checks, "passed": bool(all(checks.values())),
                "evidence_level": "five_seed_development_v3_component_gate",
                "final_unified_gate_not_evaluated": True}
    payload = {"status": "completed", "experiment": "RouteA_MFT_PredictedContextGate_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "comparisons": comparisons, "component_promotion_gate": gate,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(args.output_root / "route_a_mft_predicted_context_gate_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--mft-source-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_mft_frozen_expert_bounded_residual_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_mft_predicted_context_gate_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--c4-residual-cap", type=float, default=0.50)
    parser.add_argument("--residual-cap", type=float, default=0.50); parser.add_argument("--uncertainty-power", type=float, default=0.50)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1e-5); return parser.parse_args()


if __name__ == "__main__": run(parse_args())

