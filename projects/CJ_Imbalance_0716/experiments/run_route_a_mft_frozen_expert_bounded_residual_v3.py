"""Frozen MFT expert bounded residual integration into reconstructed C4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view, _scores  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    GenericBoundedFusion,
    SAW_INDEX,
    TARGET_INDICES,
    _apply_fusion,
    _evaluate,
    _extract_backbone,
    _load_roles,
    _probe_scores,
    _raw_scores,
    _validate_config,
)
from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import (  # noqa: E402
    ROLES,
    ROLE_OFFSETS,
    _extract_final_temporal,
    _fit_score_stats,
    _fixed_permutation,
    _load_probe,
    _normalize_features,
    _standardize_scores,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
MFT_NAME = "MFT"
VARIANTS = ("M0_C4_Reconstructed", "M1_MFT_AlignedBoundedResidual", "M2_MFT_ShuffledBoundedResidual")
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")


def _bounded_mft_residual(c4_logits, expert_z, residual_cap=0.50, uncertainty_power=0.50):
    logits = np.asarray(c4_logits, dtype=np.float32)
    expert = np.asarray(expert_z, dtype=np.float32)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[:, MFT_INDEX], -30.0, 30.0)))
    uncertainty = np.maximum(4.0 * probability * (1.0 - probability), 0.0) ** float(uncertainty_power)
    residual = float(residual_cap) * uncertainty * np.tanh(expert)
    output = logits.copy()
    output[:, MFT_INDEX] += residual.astype(np.float32)
    return output, residual.astype(np.float32), uncertainty.astype(np.float32)


def _load_c4_components(args, seed, device):
    source = args.unified_root / f"seed_{seed}" / "checkpoints"
    raw_payload = torch.load(source / "rawsaw.pth", map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.raw_hidden_channels,
        topk=args.raw_topk, dropout=args.raw_dropout,
    ).to(device)
    raw.load_state_dict(raw_payload["model_state_dict"])
    raw.eval()

    probe_payload = torch.load(source / "capacity_probe.pth", map_location="cpu", weights_only=False)
    weight = probe_payload["model_state_dict"]["weight"]
    probe = nn.Linear(int(weight.shape[1]), int(weight.shape[0])).to(device)
    probe.load_state_dict(probe_payload["model_state_dict"])
    probe.eval()

    fusion_payload = torch.load(source / "capacity_fusion.pth", map_location="cpu", weights_only=False)
    fusion = GenericBoundedFusion(args.c4_residual_cap).to(device)
    fusion.load_state_dict(fusion_payload["model_state_dict"])
    fusion.eval()
    hashes = {name: _sha256(source / filename) for name, filename in {
        "rawsaw": "rawsaw.pth", "capacity_probe": "capacity_probe.pth",
        "capacity_fusion": "capacity_fusion.pth",
    }.items()}
    return raw, raw_payload, probe, probe_payload, fusion, fusion_payload, hashes


def _reconstruct_c4(args, roles, extracted, raw, probe, probe_payload, fusion,
                    fusion_payload, device):
    raw_by_role = {role: _raw_scores(raw, roles[role]["signals"], device, args.raw_batch_size)
                   for role in ALL_ROLES}
    probe_mean = np.asarray(probe_payload["mean"], dtype=np.float32)
    probe_std = np.maximum(np.asarray(probe_payload["std"], dtype=np.float32), 1e-6)
    generic_scores = {
        role: _probe_scores(
            probe, ((extracted[role]["temporal_stats"] - probe_mean[None, :]) /
                    probe_std[None, :]).astype(np.float32), device
        ) for role in ALL_ROLES
    }
    center = np.asarray(fusion_payload["center"], dtype=np.float32)
    scale = np.maximum(np.asarray(fusion_payload["scale"], dtype=np.float32), 1e-6)
    generic_logits = {
        role: _apply_fusion(fusion, extracted[role]["base_logits"],
                            ((generic_scores[role] - center[None, :]) / scale[None, :]).astype(np.float32),
                            device)[0]
        for role in ALL_ROLES
    }
    c4 = {}
    for role in ALL_ROLES:
        logits = extracted[role]["base_logits"].copy()
        logits[:, SAW_INDEX] = raw_by_role[role]
        logits[:, TARGET_INDICES] = generic_logits[role][:, TARGET_INDICES]
        c4[role] = logits.astype(np.float32)
    return c4


def _metric(result, path):
    value = result
    for key in path:
        value = value[key]
    return float(value)


METRIC_PATHS = {
    "macro_aupr": ("metrics", "macro_aupr"),
    "macro_f1": ("metrics", "macro_f1"),
    "mft_aupr": ("metrics", "per_label", "MFT", "aupr"),
    "mft_weak_recall": ("metrics", "per_label", "MFT", "weak_recall"),
    "mft_strong_recall": ("metrics", "per_label", "MFT", "strong_recall"),
    "mft_negative_fpr": ("metrics", "per_label", "MFT", "negative_fpr"),
    "mft_weak_auroc": ("metrics", "per_label", "MFT", "weak_vs_negative_auroc"),
    "sawfm_aupr": ("metrics", "per_label", "SawFM", "aupr"),
    "sawfm_weak_recall": ("metrics", "per_label", "SawFM", "weak_recall"),
    "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
    "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
}


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _compare(rows, candidate, baseline, seed_offset):
    per_seed = []
    for row in rows:
        variants = _variant_map(row)
        values = {name: _metric(variants[candidate], path) - _metric(variants[baseline], path)
                  for name, path in METRIC_PATHS.items()}
        per_seed.append({"seed": row["seed"], **values})
    aggregate = {name: _mean_ci([row[name] for row in per_seed], seed_offset + offset)
                 for offset, name in enumerate(METRIC_PATHS)}
    aggregate["positive_mft_weak_recall_seeds"] = int(sum(row["mft_weak_recall"] > 0 for row in per_seed))
    aggregate["positive_mft_weak_auroc_seeds"] = int(sum(row["mft_weak_auroc"] > 0 for row in per_seed))
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
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
    strict_features = {
        role: _extract_final_temporal(model, roles[role]["signals"], device, args.feature_batch_size)
        for role in ALL_ROLES
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw, _, capacity_probe, capacity_probe_payload, capacity_fusion, capacity_fusion_payload, c4_hashes = (
        _load_c4_components(args, seed, device)
    )
    c4 = _reconstruct_c4(
        args, roles, extracted, raw, capacity_probe, capacity_probe_payload,
        capacity_fusion, capacity_fusion_payload, device,
    )
    source_predictions = np.load(
        args.unified_root / f"seed_{seed}" / "C4_SameCapacity_NoVisibility" / "predictions.npz"
    )
    reconstruction_error = float(np.max(np.abs(c4["early"] - source_predictions["logits"])))
    if reconstruction_error > args.maximum_reconstruction_error:
        raise AssertionError(f"C4 reconstruction mismatch: {reconstruction_error}")

    expert_path = args.layerwise_root / f"seed_{seed}" / "T_FinalTemporal" / MFT_NAME / "best_probe.pth"
    expert_probe, expert_payload = _load_probe(expert_path, device)
    expert_raw = {
        role: _scores(expert_probe, _normalize_features(strict_features[role], expert_payload), device)
        for role in ALL_ROLES
    }
    fit_selected, _, fit_receipt = _matched_view(
        roles["fit"]["targets6"], weak6[roles["fit"]["sample_ids"]], MFT_INDEX
    )
    score_stats = _fit_score_stats(expert_raw["fit"][fit_selected])
    expert_z = {role: _standardize_scores(expert_raw[role], score_stats) for role in ALL_ROLES}

    aligned_logits, shuffled_logits = {}, {}
    residual_payload, permutations = {}, {}
    for role in ALL_ROLES:
        permutation_seed = 20286000 + int(seed) * 100 + ROLE_OFFSETS.get(role, 41 if role == "calibration" else 53)
        permutation, permutation_sha = _fixed_permutation(len(expert_z[role]), permutation_seed)
        aligned, aligned_residual, uncertainty = _bounded_mft_residual(
            c4[role], expert_z[role], args.residual_cap, args.uncertainty_power
        )
        shuffled, shuffled_residual, _ = _bounded_mft_residual(
            c4[role], expert_z[role][permutation], args.residual_cap, args.uncertainty_power
        )
        aligned_logits[role], shuffled_logits[role] = aligned, shuffled
        residual_payload.update({
            f"{role}_sample_ids": np.asarray(roles[role]["sample_ids"], dtype=np.int64),
            f"{role}_expert_z": expert_z[role], f"{role}_aligned_residual": aligned_residual,
            f"{role}_shuffled_residual": shuffled_residual, f"{role}_uncertainty": uncertainty,
            f"{role}_permutation": permutation,
        })
        permutations[role] = {"seed": int(permutation_seed), "sha256": permutation_sha,
                              "count": int(len(permutation))}
    np.savez_compressed(seed_root / "expert_residuals.npz", **residual_payload)

    logits = {
        VARIANTS[0]: c4,
        VARIANTS[1]: aligned_logits,
        VARIANTS[2]: shuffled_logits,
    }
    results = [_evaluate(variant, logits[variant], roles, weak6, seed_root / variant, args)
               for variant in VARIANTS]
    summary = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
        "c6_checkpoint_sha256": _sha256(c6_path), "c4_component_sha256": c4_hashes,
        "mft_expert_sha256": _sha256(expert_path),
        "mft_fit_matched_receipt": fit_receipt, "mft_expert_score_stats": score_stats,
        "c4_reconstruction_max_abs_error": reconstruction_error,
        "residual": {"cap": args.residual_cap, "uncertainty_power": args.uncertainty_power,
                     "learned_parameters": 0, "target_index": MFT_INDEX},
        "permutations": permutations, "expert_residuals_sha256": _sha256(seed_root / "expert_residuals.npz"),
        "variants": results, "identity_audit_passed": True,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, summary)
    return summary


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path)
    values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_MFT_FrozenExpert_BoundedResidual_v3", "seeds": list(seeds),
        "smoke": bool(args.smoke), "residual_cap": args.residual_cap,
        "uncertainty_power": args.uncertainty_power, "learned_parameters": 0,
        "gates": {"macro_aupr_gain": 0.005, "macro_f1_gain": 0.010,
                  "mft_weak_recall_gain": 0.030, "mft_weak_auroc_gain": 0.015,
                  "aligned_vs_shuffled_weak_recall": 0.015,
                  "aligned_vs_shuffled_weak_auroc": 0.010,
                  "mft_fpr_cap": 0.20, "positive_seeds": 4},
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]

    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        variants = [_variant_map(row)[variant] for row in rows]
        aggregate[variant] = {name: _mean_ci([_metric(item, path) for item in variants], 20287000 + vi * 100 + offset)
                              for offset, (name, path) in enumerate(METRIC_PATHS.items())}
    comparisons = {
        "M1_vs_M0": _compare(rows, VARIANTS[1], VARIANTS[0], 20288000),
        "M1_vs_M2": _compare(rows, VARIANTS[1], VARIANTS[2], 20289000),
    }
    if args.smoke:
        gate = {"checks": {}, "passed": False, "evidence_level": "smoke_only"}
    else:
        vs_c4 = comparisons["M1_vs_M0"]["aggregate"]
        vs_shuffle = comparisons["M1_vs_M2"]["aggregate"]
        candidate = aggregate[VARIANTS[1]]
        checks = {
            "macro_aupr_gain_ge_0p005": vs_c4["macro_aupr"]["mean"] >= 0.005,
            "macro_f1_gain_ge_0p010": vs_c4["macro_f1"]["mean"] >= 0.010,
            "mft_weak_recall_gain_ge_0p030": vs_c4["mft_weak_recall"]["mean"] >= 0.030,
            "mft_weak_auroc_gain_ge_0p015": vs_c4["mft_weak_auroc"]["mean"] >= 0.015,
            "mft_strong_recall_noninferior": vs_c4["mft_strong_recall"]["mean"] >= -0.010,
            "mft_aupr_noninferior": vs_c4["mft_aupr"]["mean"] >= -0.010,
            "mft_negative_fpr_le_0p20": candidate["mft_negative_fpr"]["mean"] <= 0.20,
            "four_of_five_positive_weak_recall": vs_c4["positive_mft_weak_recall_seeds"] >= 4,
            "four_of_five_positive_weak_auroc": vs_c4["positive_mft_weak_auroc_seeds"] >= 4,
            "aligned_vs_shuffled_weak_recall_ge_0p015": vs_shuffle["mft_weak_recall"]["mean"] >= 0.015,
            "aligned_vs_shuffled_weak_auroc_ge_0p010": vs_shuffle["mft_weak_auroc"]["mean"] >= 0.010,
            "sawfm_aupr_preserved": vs_c4["sawfm_aupr"]["mean"] >= -1e-8,
            "sawfm_weak_recall_preserved": vs_c4["sawfm_weak_recall"]["mean"] >= -1e-8,
            "sawfm_present_noninferior": vs_c4["sawfm_present_macro_f1"]["mean"] >= -0.005,
            "sawfm_absent_noninferior": vs_c4["sawfm_absent_macro_f1"]["mean"] >= -0.005,
            "c4_reconstruction_exact": all(row["c4_reconstruction_max_abs_error"] <= args.maximum_reconstruction_error for row in rows),
            "identity_and_test_boundary_safe": all(row["identity_audit_passed"] and not row["test_v3_used"] for row in rows),
        }
        gate = {"checks": checks, "passed": bool(all(checks.values())),
                "evidence_level": "five_seed_development_v3"}
    payload = {
        "status": "completed", "experiment": "RouteA_MFT_FrozenExpert_BoundedResidual_v3",
        "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
        "aggregate": aggregate, "comparisons": comparisons, "promotion_gate": gate,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "route_a_mft_frozen_expert_bounded_residual_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--layerwise-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_layerwise_matched_negative_separability_v3_strict")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_mft_frozen_expert_bounded_residual_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--c4-residual-cap", type=float, default=0.50)
    parser.add_argument("--residual-cap", type=float, default=0.50)
    parser.add_argument("--uncertainty-power", type=float, default=0.50)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
