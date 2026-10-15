"""Frozen full-candidate confirmation on identity-isolated development-v11."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import SCORE_NAMES, extract_scores  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import FrozenFeatureHead, _extract as _extract_backbone_features, _predict_head  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_model_from_state, _load_rawsaw, _route_rawsaw  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import (  # noqa: E402
    VARIANTS, _extract_vft, _fusion_features as _vft_fusion_features,
    _goal_rows, _head_scores, _replace_vft, _ridge_scores, _vft_gate,
)
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features as _isrj_fusion_features, _replace_isrj  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _fit_score_stats, _load_expert, _standardize_scores  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration, _load_signals  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import expert_logit, guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _load_roles, _raw_scores, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
VFT_INDEX = 4
PEAK_INDEX = SCORE_NAMES.index("nms_count_gt_bg_q99")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_record(record):
    path = Path(record["path"])
    if not path.is_file() or _sha256(path).lower() != record["sha256"].lower():
        raise ValueError(f"frozen artifact mismatch: {path}")


def _subgroups(metrics, labels, predictions):
    saw = labels[:, 1] == 1
    metrics["subgroups"] = {}
    for name, mask in (("sawfm_present", saw), ("sawfm_absent", ~saw)):
        metrics["subgroups"][name] = {
            "macro_f1": float(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0))
        }


def _hard_metrics(labels, weak, probabilities, predictions):
    order = labels.sum(axis=1)
    positive = labels[:, MFT_INDEX] == 1
    weak_positive = positive & weak[:, MFT_INDEX]
    hard_negative = (~positive) & (order >= 4)
    order5_negative = (~positive) & (order == 5)
    selected = weak_positive | hard_negative
    return {
        "hard_auroc": float(roc_auc_score(weak_positive[selected].astype(np.uint8), probabilities[selected, MFT_INDEX])),
        "hard_fpr": float(predictions[hard_negative, MFT_INDEX].mean()),
        "order5_fpr": float(predictions[order5_negative, MFT_INDEX].mean()),
    }


def _evaluate_frozen(variant, logits, labels, weak, receiver, source, output):
    calibration = _load_json(source / "calibration.json")
    thresholds = np.asarray(_load_json(source / "thresholds.json")["thresholds"], dtype=np.float32)
    probabilities = _apply_frozen_calibration(logits, calibration)
    predictions = probabilities >= thresholds[None, :]
    metrics = _metrics(labels, weak, probabilities, predictions)
    _subgroups(metrics, labels, predictions)
    hard = _hard_metrics(labels, weak, probabilities, predictions)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", {"metrics": metrics, "mft_hard": hard})
    np.savez_compressed(
        output / "predictions.npz", receiver_noise_seed=receiver, targets=labels,
        weak_mask=weak.astype(np.uint8), logits=logits, probabilities=probabilities,
        predictions=predictions.astype(np.uint8), thresholds=thresholds,
    )
    return {"variant": variant, "metrics": metrics, "mft_hard": hard}


def _load_vft_head(root, name):
    model = joblib.load(root / f"head_{name}" / "model.joblib")
    normalization = np.load(root / f"head_{name}" / "normalization.npz")
    return model, np.asarray(normalization["mean"], dtype=np.float32), np.asarray(normalization["std"], dtype=np.float32)


def _run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic_features, vft_features, mft_count, seed, device, manifest):
    artifacts = manifest["seed_artifacts"][str(seed)]
    for record in artifacts.values():
        if isinstance(record, dict) and "sha256" in record:
            _verify_record(record)
    roles3, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)

    focal_path = Path(artifacts["focal_checkpoint"]["path"])
    state = torch.load(focal_path, map_location="cpu", weights_only=False)["model_state_dict"]
    backbone = _load_model_from_state(args.f2_root / "runs/F2_DFMS_Full" / f"seed_{seed}", device, state)
    fit_features = _extract_backbone_features(backbone, roles3["fit"]["signals"], device, args.backbone_batch_size)
    mean = fit_features.mean(0, keepdims=True).astype(np.float32)
    std = np.maximum(fit_features.std(0, keepdims=True), 1e-5).astype(np.float32)
    head_receipt = _load_json(args.head_root / f"seed_{seed}" / "seed_summary.json")
    if hashlib.sha256(mean.tobytes()).hexdigest() != head_receipt["feature_mean_sha256"]:
        raise ValueError(f"seed {seed} H1 mean reconstruction mismatch")
    if hashlib.sha256(std.tobytes()).hexdigest() != head_receipt["feature_std_sha256"]:
        raise ValueError(f"seed {seed} H1 std reconstruction mismatch")
    fresh_features = _extract_backbone_features(backbone, signals, device, args.backbone_batch_size)
    del backbone, fit_features
    fresh_features = ((fresh_features - mean) / std).astype(np.float32)
    h1_payload = torch.load(Path(artifacts["h1_head"]["path"]), map_location="cpu", weights_only=False)
    h1 = FrozenFeatureHead(128, "linear").to(device)
    h1.load_state_dict(h1_payload["model_state_dict"])
    base_logits = _predict_head(h1, fresh_features, device, args.eval_batch_size)
    del h1, fresh_features
    raw, _ = _load_rawsaw(args, seed, device)
    raw_score = _raw_scores(raw, signals, device, args.raw_batch_size)
    del raw
    q0 = _route_rawsaw({"fresh": base_logits}, {"fresh": raw_score})["fresh"]

    periodic_model, periodic_mean, periodic_std, _ = _load_expert(args.periodicity_root, seed, "H_AlignedHGB")
    fit_periodic = _extract_periodicity(roles3["fit"]["signals"], device, args.periodicity_batch_size)
    fit_raw = _clipped_logit(periodic_model.predict_proba(((fit_periodic - periodic_mean) / np.maximum(periodic_std, 1e-5)).astype(np.float32))[:, 1])
    selected, _, _ = _matched_view(roles3["fit"]["targets6"], weak3[roles3["fit"]["sample_ids"]], ISRJ_INDEX)
    periodic_stats = _fit_score_stats(fit_raw[selected])
    fresh_raw = _clipped_logit(periodic_model.predict_proba(((periodic_features - periodic_mean) / np.maximum(periodic_std, 1e-5)).astype(np.float32))[:, 1])
    periodic_z = _standardize_scores(fresh_raw, periodic_stats)
    isrj_model = joblib.load(Path(artifacts["isrj_fusion_model"]["path"]))
    isrj_norm = np.load(Path(artifacts["isrj_fusion_normalization"]["path"]))
    isrj_score = _head_scores(isrj_model, _isrj_fusion_features(q0, periodic_z), isrj_norm["mean"], isrj_norm["std"])
    q1 = _replace_isrj(q0, isrj_score)

    guard_model = joblib.load(Path(artifacts["mft_guard_model"]["path"]))
    guard_norm = np.load(Path(artifacts["mft_guard_normalization"]["path"]))
    mft_evidence = expert_logit(guard_model, mft_count, guard_norm["aligned_mean"], guard_norm["aligned_std"])
    q2, _ = guarded_logits(q1, mft_evidence, manifest["fixed_hyperparameters"]["mft_residual_cap"])

    ridge = np.load(Path(artifacts["vft_ridge"]["path"]))
    vft_raw = _ridge_scores(vft_features, ridge)
    construction_receipt = _load_json(args.construction_root / f"seed_{seed}" / "seed_summary.json")
    vft_z = _standardize_scores(vft_raw, construction_receipt["vft_score_stats"])
    permutation = np.random.default_rng(20424000 + seed).permutation(len(vft_z))
    shuffled_z = vft_z[permutation]
    vft_heads = {
        "base": _load_vft_head(args.construction_root / f"seed_{seed}", "base"),
        "aligned": _load_vft_head(args.construction_root / f"seed_{seed}", "aligned"),
        "shuffled": _load_vft_head(args.construction_root / f"seed_{seed}", "shuffled"),
    }
    q3 = _replace_vft(q2, _head_scores(vft_heads["base"][0], _vft_fusion_features(q2), vft_heads["base"][1], vft_heads["base"][2]))
    q4 = _replace_vft(q2, _head_scores(vft_heads["aligned"][0], _vft_fusion_features(q2, vft_z), vft_heads["aligned"][1], vft_heads["aligned"][2]))
    q5 = _replace_vft(q2, _head_scores(vft_heads["shuffled"][0], _vft_fusion_features(q2, shuffled_z), vft_heads["shuffled"][1], vft_heads["shuffled"][2]))
    logits = {VARIANTS[0]: q0, VARIANTS[1]: q1, VARIANTS[2]: q2, VARIANTS[3]: q3, VARIANTS[4]: q4, VARIANTS[5]: q5}
    source_seed = args.construction_root / f"seed_{seed}"
    root = args.output_root / f"seed_{seed}"
    results = [_evaluate_frozen(name, logits[name], labels, weak, receiver, source_seed / name, root / name) for name in VARIANTS]
    row = {
        "seed": int(seed), "status": "completed", "role_indices_sha256": _sha256(role_path),
        "variants": results, "external_refit": False, "calibration_reused": True,
        "thresholds_reused": True, "non_vft_logit_max_abs": float(np.max(np.abs(q4[:, [0, 1, 2, 3, 5]] - q2[:, [0, 1, 2, 3, 5]]))),
        "test_v3_used": False, "test_v11_used": False,
    }
    _write_json(root / "seed_summary.json", row)
    return row


def _mft_gate(rows):
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        base, guard = variants[VARIANTS[1]], variants[VARIANTS[2]]
        per_seed.append({
            "seed": row["seed"],
            "hard_fpr": guard["mft_hard"]["hard_fpr"] - base["mft_hard"]["hard_fpr"],
            "order5_fpr": guard["mft_hard"]["order5_fpr"] - base["mft_hard"]["order5_fpr"],
            "hard_auroc": guard["mft_hard"]["hard_auroc"] - base["mft_hard"]["hard_auroc"],
        })
    aggregate = {key: _mean_ci([item[key] for item in per_seed], 20425000 + i) for i, key in enumerate(("hard_fpr", "order5_fpr", "hard_auroc"))}
    checks = {
        "hard_fpr_delta_le_minus0p020": aggregate["hard_fpr"]["mean"] <= -0.020,
        "order5_fpr_delta_le_minus0p020": aggregate["order5_fpr"]["mean"] <= -0.020,
        "hard_auroc_delta_ge_0p010": aggregate["hard_auroc"]["mean"] >= 0.010,
        "all_seeds_reduce_hard_fpr": all(item["hard_fpr"] < 0 for item in per_seed),
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "aggregate": aggregate, "per_seed": per_seed}


def run(args):
    audit = _load_json(args.audit)
    if not audit["identity_audit_passed"] or audit["evidence_role"] != "development_v11" or audit["test_files_opened"]:
        raise ValueError("development-v11 audit failed")
    seal = _load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = _load_json(args.frozen_manifest)
    if manifest["candidate_name"] != "H1_B2_MFTPeakGuard_VFTRidgeFusion_v1":
        raise ValueError("wrong frozen candidate")
    data_path = Path(audit["development_v11_path"])
    if _sha256(data_path).lower() != audit["sha256"].lower():
        raise ValueError("development-v11 hash changed")
    config = load_config(args.config)
    _validate_config(config)
    labels3 = read_labels(_data_path(config))
    params3, _ = read_parameters(_data_path(config))
    weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    labels = read_labels(data_path)
    params, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(params), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    periodic_features = _extract_periodicity(signals, device, args.periodicity_batch_size)
    vft_features = _extract_vft(signals, device, args.vft_batch_size)
    physical = extract_scores(signals, device, args.mft_batch_size)
    mft_count = physical["MF_AlignedLFM"][:, PEAK_INDEX]
    rows = [_run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic_features, vft_features, mft_count, seed, device, manifest) for seed in args.seeds]
    vft_gate = _vft_gate(rows)
    unified_gate = _goal_rows(rows)
    mft_gate = _mft_gate(rows)
    result = {
        "status": "completed", "experiment": "RouteA_H1B2_MFT_VFT_DevelopmentV11_Frozen",
        "evidence_role": "fresh_identity_frozen_confirmation", "development_v11_sha256": audit["sha256"],
        "frozen_manifest_sha256": _sha256(args.frozen_manifest), "rows": rows,
        "vft_mechanism_gate": vft_gate, "mft_mechanism_gate": mft_gate,
        "unified_promotion_gate": unified_gate,
        "confirmation_passed": bool(vft_gate["passed"] and mft_gate["passed"] and unified_gate["passed"]),
        "external_refit": False, "locked_test_used": False, "test_v3_used": False, "test_v11_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_h1_b2_mft_vft_development_v11_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"summary": str(output), "vft_gate": vft_gate, "mft_gate": mft_gate, "unified_gate": unified_gate, "passed": result["confirmation_passed"]}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v11_generation/identity_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_unified_v1/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v11_frozen")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--backbone-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--periodicity-batch-size", type=int, default=64)
    parser.add_argument("--vft-batch-size", type=int, default=32)
    parser.add_argument("--mft-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
