"""Low-capacity frozen-score fusion head for the ISRJ periodicity expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _block_permutation, _block_shuffle, _extract  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import (  # noqa: E402
    ISRJ_INDEX, METRIC_PATHS, _bounded_isrj_residual, _clipped_logit,
    _compare, _fit_score_stats, _load_expert, _metric, _standardize_scores,
    _variant_map,
)
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES, _load_c4_components, _reconstruct_c4  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256, _evaluate, _extract_backbone, _load_roles, _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "F0_C4_Reconstructed",
    "F1_FixedAlignedResidual",
    "F2_C4OnlyWeakWeightedHead",
    "F3_AlignedPeriodicityFusionHead",
    "F4_BlockShuffledFusionHead",
)


def _fusion_features(c4_logits, expert_z=None):
    base = np.asarray(c4_logits, dtype=np.float32)[:, ISRJ_INDEX]
    if expert_z is None:
        return base[:, None]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    z = np.asarray(expert_z, dtype=np.float32)
    return np.stack([base, z, uncertainty * z, np.tanh(base) * z], axis=1).astype(np.float32)


def _sample_weights(targets6, weak6, weak_positive_weight=4.0):
    target = np.asarray(targets6)[:, ISRJ_INDEX] == 1
    weak = np.asarray(weak6, dtype=bool)[:, ISRJ_INDEX]
    return np.where(target & weak, float(weak_positive_weight), 1.0).astype(np.float64)


def _fit_head(features, labels, weights, seed, c_value):
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
    normalized = ((features - mean[None, :]) / std[None, :]).astype(np.float32)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=1000,
                               random_state=int(seed))
    model.fit(normalized, labels, sample_weight=weights)
    return model, mean, std


def _head_scores(model, features, mean, std):
    normalized = ((features - mean[None, :]) / np.maximum(std[None, :], 1e-5)).astype(np.float32)
    return model.decision_function(normalized).astype(np.float32)


def _replace_isrj(c4_logits, scores):
    output = np.asarray(c4_logits, dtype=np.float32).copy()
    output[:, ISRJ_INDEX] = np.asarray(scores, dtype=np.float32)
    return output


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_path, map_location="cpu", weights_only=False)["model_state_dict"]
    backbone = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state)
    extracted = _extract_backbone(backbone, roles, device, args.backbone_batch_size, args.topk)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw, _, probe, probe_payload, fusion, fusion_payload, c4_hashes = _load_c4_components(args, seed, device)
    c4 = _reconstruct_c4(args, roles, extracted, raw, probe, probe_payload, fusion, fusion_payload, device)
    source_path = args.unified_root / f"seed_{seed}" / "C4_SameCapacity_NoVisibility" / "predictions.npz"
    source = np.load(source_path)
    reconstruction_error = float(np.max(np.abs(c4["early"] - source["logits"])))
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
        ax = ((aligned_features[role] - aligned_mean[None, :]) / np.maximum(aligned_std[None, :], 1e-5)).astype(np.float32)
        sx = ((shuffled_features[role] - shuffled_mean[None, :]) / np.maximum(shuffled_std[None, :], 1e-5)).astype(np.float32)
        aligned_raw[role] = _clipped_logit(aligned_model.predict_proba(ax)[:, 1])
        shuffled_raw[role] = _clipped_logit(shuffled_model.predict_proba(sx)[:, 1])
    fit_selected, _, fit_receipt = _matched_view(
        roles["fit"]["targets6"], weak6[roles["fit"]["sample_ids"]], ISRJ_INDEX)
    aligned_stats = _fit_score_stats(aligned_raw["fit"][fit_selected])
    shuffled_stats = _fit_score_stats(shuffled_raw["fit"][fit_selected])
    aligned_z = {role: _standardize_scores(aligned_raw[role], aligned_stats) for role in ALL_ROLES}
    shuffled_z = {role: _standardize_scores(shuffled_raw[role], shuffled_stats) for role in ALL_ROLES}

    fit_labels = roles["fit"]["targets6"][:, ISRJ_INDEX].astype(np.uint8)
    weights = _sample_weights(roles["fit"]["targets6"], weak6[roles["fit"]["sample_ids"]],
                              args.weak_positive_weight)
    feature_sets = {
        "c4_only": {role: _fusion_features(c4[role]) for role in ALL_ROLES},
        "aligned": {role: _fusion_features(c4[role], aligned_z[role]) for role in ALL_ROLES},
        "shuffled": {role: _fusion_features(c4[role], shuffled_z[role]) for role in ALL_ROLES},
    }
    heads = {}
    for offset, name in enumerate(("c4_only", "aligned", "shuffled")):
        model, mean, std = _fit_head(feature_sets[name]["fit"], fit_labels, weights,
                                     20310000 + int(seed) + offset, args.c_value)
        out = root / f"head_{name}"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
        heads[name] = {"model": model, "mean": mean, "std": std,
                       "model_sha256": _sha256(out / "model.joblib"),
                       "normalization_sha256": _sha256(out / "normalization.npz"),
                       "coefficient": model.coef_.reshape(-1).tolist(),
                       "intercept": model.intercept_.reshape(-1).tolist()}

    fixed = {}
    for role in ALL_ROLES:
        fixed[role] = _bounded_isrj_residual(c4[role], aligned_z[role],
                                             args.residual_cap, args.uncertainty_power)[0]
    logits = {
        VARIANTS[0]: c4,
        VARIANTS[1]: fixed,
        VARIANTS[2]: {role: _replace_isrj(c4[role], _head_scores(heads["c4_only"]["model"], feature_sets["c4_only"][role], heads["c4_only"]["mean"], heads["c4_only"]["std"])) for role in ALL_ROLES},
        VARIANTS[3]: {role: _replace_isrj(c4[role], _head_scores(heads["aligned"]["model"], feature_sets["aligned"][role], heads["aligned"]["mean"], heads["aligned"]["std"])) for role in ALL_ROLES},
        VARIANTS[4]: {role: _replace_isrj(c4[role], _head_scores(heads["shuffled"]["model"], feature_sets["shuffled"][role], heads["shuffled"]["mean"], heads["shuffled"]["std"])) for role in ALL_ROLES},
    }
    for variant in VARIANTS[1:]:
        for role in ALL_ROLES:
            if np.max(np.abs(logits[variant][role][:, :ISRJ_INDEX] - c4[role][:, :ISRJ_INDEX])) > 0:
                raise AssertionError("non-ISRJ logits changed")
    results = [_evaluate(name, logits[name], roles, weak6, root / name, args) for name in VARIANTS]
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
        "c6_checkpoint_sha256": _sha256(c6_path), "c4_component_sha256": c4_hashes,
        "c4_reconstruction_max_abs_error": reconstruction_error,
        "source_prediction_sha256": _sha256(source_path),
        "periodicity_expert_sha256": {"aligned": aligned_hashes, "block_shuffled": shuffled_hashes},
        "fit_matched_receipt": fit_receipt,
        "score_stats": {"aligned": aligned_stats, "block_shuffled": shuffled_stats},
        "head": {name: {key: value for key, value in row.items() if key != "model"}
                 for name, row in heads.items()},
        "training": {"c_value": args.c_value, "weak_positive_weight": args.weak_positive_weight,
                     "fit_weak_positive_count": int(np.sum(weights > 1.0))},
        "variants": results,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    labels6 = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_ISRJ_FrozenScoreFusionHead_v3", "seeds": seeds,
        "features": ["c4_logit", "periodicity_z", "uncertainty_times_z", "tanh_c4_times_z"],
        "logistic_regression": {"C": args.c_value, "solver": "lbfgs", "max_iter": 1000},
        "weak_positive_weight": args.weak_positive_weight,
        "gates": {"vs_c4_weak_recall": .030, "vs_c4_weak_auroc": .015,
                  "vs_c4only_weak_recall": .015, "vs_c4only_weak_auroc": .010,
                  "vs_block_weak_recall": .010, "vs_block_weak_auroc": .005,
                  "vs_fixed_weak_auroc": .005, "fpr_cap": .20, "positive_seeds": 4},
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [_variant_map(row)[variant] for row in rows]
        aggregate[variant] = {
            name: _mean_ci([_metric(item, path) for item in selected], 20311000 + vi * 100 + i)
            for i, (name, path) in enumerate(METRIC_PATHS.items())
        }
    comparisons = {
        "F3_vs_F0": _compare(rows, VARIANTS[3], VARIANTS[0], 20312000),
        "F3_vs_F1": _compare(rows, VARIANTS[3], VARIANTS[1], 20313000),
        "F3_vs_F2": _compare(rows, VARIANTS[3], VARIANTS[2], 20314000),
        "F3_vs_F4": _compare(rows, VARIANTS[3], VARIANTS[4], 20315000),
    }
    if args.smoke:
        gate = {"passed": False, "checks": {}, "evidence_level": "smoke_only"}
    else:
        c4 = comparisons["F3_vs_F0"]["aggregate"]
        fixed = comparisons["F3_vs_F1"]["aggregate"]
        c4only = comparisons["F3_vs_F2"]["aggregate"]
        block = comparisons["F3_vs_F4"]["aggregate"]
        candidate = aggregate[VARIANTS[3]]
        checks = {
            "weak_recall_vs_c4_ge_0p030": c4["isrj_weak_recall"]["mean"] >= .030,
            "weak_auroc_vs_c4_ge_0p015": c4["isrj_weak_auroc"]["mean"] >= .015,
            "isrj_aupr_nonnegative": c4["isrj_aupr"]["mean"] >= 0,
            "strong_recall_noninferior": c4["isrj_strong_recall"]["mean"] >= -.010,
            "fpr_le_0p20": candidate["isrj_negative_fpr"]["mean"] <= .20,
            "weak_recall_vs_c4only_ge_0p015": c4only["isrj_weak_recall"]["mean"] >= .015,
            "weak_auroc_vs_c4only_ge_0p010": c4only["isrj_weak_auroc"]["mean"] >= .010,
            "weak_recall_vs_block_ge_0p010": block["isrj_weak_recall"]["mean"] >= .010,
            "weak_auroc_vs_block_ge_0p005": block["isrj_weak_auroc"]["mean"] >= .005,
            "weak_auroc_vs_fixed_ge_0p005": fixed["isrj_weak_auroc"]["mean"] >= .005,
            "four_of_five_positive_weak_recall": c4["positive_isrj_weak_recall_seeds"] >= 4,
            "four_of_five_positive_weak_auroc": c4["positive_isrj_weak_auroc_seeds"] >= 4,
            "macro_aupr_positive": c4["macro_aupr"]["mean"] > 0,
            "macro_f1_positive": c4["macro_f1"]["mean"] > 0,
            "sawfm_preserved": c4["sawfm_aupr"]["mean"] >= -1e-8 and c4["sawfm_weak_recall"]["mean"] >= -1e-8,
            "sawfm_subgroups_noninferior": c4["sawfm_present_macro_f1"]["mean"] >= -.005 and c4["sawfm_absent_macro_f1"]["mean"] >= -.005,
            "reconstruction_and_test_boundary_safe": all(row["c4_reconstruction_max_abs_error"] <= args.maximum_reconstruction_error and not row["test_v3_used"] for row in rows),
        }
        gate = {"passed": bool(all(checks.values())), "checks": checks,
                "evidence_level": "five_seed_development_v3_frozen_fusion_head"}
    payload = {
        "status": "completed", "experiment": "RouteA_ISRJ_FrozenScoreFusionHead_v3",
        "smoke": bool(args.smoke), "seeds": seeds, "data_sha256": data_sha,
        "aggregate": aggregate, "comparisons": comparisons, "promotion_gate": gate,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "route_a_isrj_frozen_score_fusion_head_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--backbone-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--c4-residual-cap", type=float, default=0.50)
    parser.add_argument("--residual-cap", type=float, default=0.50)
    parser.add_argument("--uncertainty-power", type=float, default=0.50)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--c-value", type=float, default=0.10)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
