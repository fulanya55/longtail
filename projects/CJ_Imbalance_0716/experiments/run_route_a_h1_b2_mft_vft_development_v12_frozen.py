"""Fresh frozen confirmation of the shared-policy unified candidate on v12."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import extract_scores  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import (  # noqa: E402
    PEAK_INDEX, _hard_metrics, _load_json, _run_seed, _subgroups,
)
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import (  # noqa: E402
    VARIANTS, _extract_vft, _goal_rows, _vft_gate,
)
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_mft_shared_decision_policy_audit import MFT_INDEX, hybrid_policy  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


SHARED_VARIANT = "Q6_SharedMFTDecision_VFTAligned"


def _shared_variant(args, seed, row):
    root = args.output_root / f"seed_{seed}"
    q0 = np.load(root / "Q0_H1" / "predictions.npz")
    q1 = np.load(root / "Q1_H1B2" / "predictions.npz")
    q4 = np.load(root / "Q4_VFTAlignedRidgeFusion" / "predictions.npz")
    source = args.construction_root / f"seed_{seed}"
    probability, prediction, thresholds = hybrid_policy(
        np.asarray(q4["logits"], dtype=np.float32),
        _load_json(source / "Q4_VFTAlignedRidgeFusion" / "calibration.json"),
        _load_json(source / "Q4_VFTAlignedRidgeFusion" / "thresholds.json")["thresholds"],
        _load_json(source / "Q1_H1B2" / "calibration.json"),
        _load_json(source / "Q1_H1B2" / "thresholds.json")["thresholds"],
        np.asarray(q4["probabilities"], dtype=np.float32),
        np.asarray(q4["predictions"], dtype=bool),
    )
    labels = np.asarray(q4["targets"], dtype=np.uint8)
    weak = np.asarray(q4["weak_mask"], dtype=bool)
    base_prediction = np.asarray(q1["predictions"], dtype=bool)
    metrics = _metrics(labels, weak, probability, prediction)
    _subgroups(metrics, labels, prediction)
    hard = _hard_metrics(labels, weak, probability, prediction)
    output = root / SHARED_VARIANT
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", receiver_noise_seed=np.asarray(q4["receiver_noise_seed"]),
        targets=labels, weak_mask=weak.astype(np.uint8), logits=np.asarray(q4["logits"], dtype=np.float32),
        probabilities=probability, predictions=prediction.astype(np.uint8), thresholds=thresholds,
    )
    variant = {
        "variant": SHARED_VARIANT, "metrics": metrics, "mft_hard": hard,
        "negative_to_positive": int(((~base_prediction[:, MFT_INDEX]) & prediction[:, MFT_INDEX]).sum()),
        "non_mft_probability_max_abs": float(np.max(np.abs(probability[:, [0, 1, 2, 4, 5]] - q4["probabilities"][:, [0, 1, 2, 4, 5]]))),
        "non_mft_prediction_mismatch": int((prediction[:, [0, 1, 2, 4, 5]] != q4["predictions"][:, [0, 1, 2, 4, 5]]).sum()),
    }
    _write_json(output / "metrics.json", variant)
    row["variants"].append(variant)
    return row


def _variant(row, name):
    return next(item for item in row["variants"] if item["variant"] == name)


def _mft_gate(rows):
    per_seed = []
    for row in rows:
        base = _variant(row, VARIANTS[1])
        candidate = _variant(row, SHARED_VARIANT)
        per_seed.append({
            "seed": row["seed"],
            "hard_fpr": candidate["mft_hard"]["hard_fpr"] - base["mft_hard"]["hard_fpr"],
            "order5_fpr": candidate["mft_hard"]["order5_fpr"] - base["mft_hard"]["order5_fpr"],
            "hard_auroc": candidate["mft_hard"]["hard_auroc"] - base["mft_hard"]["hard_auroc"],
            "weak_recall": candidate["metrics"]["per_label"]["MFT"]["weak_recall"] - base["metrics"]["per_label"]["MFT"]["weak_recall"],
            "negative_to_positive": candidate["negative_to_positive"],
            "non_mft_probability_max_abs": candidate["non_mft_probability_max_abs"],
            "non_mft_prediction_mismatch": candidate["non_mft_prediction_mismatch"],
        })
    aggregate = {
        key: _mean_ci([item[key] for item in per_seed], 20427000 + index)
        for index, key in enumerate(("hard_fpr", "order5_fpr", "hard_auroc", "weak_recall"))
    }
    checks = {
        "hard_fpr_delta_le_minus0p010": aggregate["hard_fpr"]["mean"] <= -0.010,
        "order5_fpr_delta_le_minus0p010": aggregate["order5_fpr"]["mean"] <= -0.010,
        "hard_auroc_delta_ge_0p010": aggregate["hard_auroc"]["mean"] >= 0.010,
        "weak_recall_loss_le_0p020": aggregate["weak_recall"]["mean"] >= -0.020,
        "zero_negative_to_positive": sum(item["negative_to_positive"] for item in per_seed) == 0,
        "non_mft_probability_exact": max(item["non_mft_probability_max_abs"] for item in per_seed) == 0.0,
        "non_mft_prediction_exact": sum(item["non_mft_prediction_mismatch"] for item in per_seed) == 0,
        "all_seeds_reduce_hard_fpr": all(item["hard_fpr"] < 0.0 for item in per_seed),
    }
    return {"passed": all(checks.values()), "checks": checks, "aggregate": aggregate, "per_seed": per_seed}


def _promote_shared_rows(rows):
    promoted = copy.deepcopy(rows)
    for row in promoted:
        shared = _variant(row, SHARED_VARIANT)
        shared["variant"] = VARIANTS[4]
        row["variants"] = [item for item in row["variants"] if item["variant"] != VARIANTS[4] and item["variant"] != SHARED_VARIANT]
        row["variants"].append(shared)
    return promoted


def run(args):
    audit = _load_json(args.audit)
    if not audit["identity_audit_passed"] or audit["evidence_role"] != "development_v12" or audit["test_files_opened"]:
        raise ValueError("development-v12 audit failed")
    seal = _load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = _load_json(args.frozen_manifest)
    if manifest["candidate_name"] != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen candidate")
    data_path = Path(audit["development_v12_path"])
    if _sha256(data_path).lower() != audit["sha256"].lower():
        raise ValueError("development-v12 hash changed")
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
    rows = []
    for seed in args.seeds:
        row = _run_seed(
            args, config, labels3, weak3, signals, labels, weak, receiver,
            periodic_features, vft_features, mft_count, seed, device, manifest,
        )
        rows.append(_shared_variant(args, seed, row))
    vft_gate = _vft_gate(rows)
    mft_gate = _mft_gate(rows)
    unified_gate = _goal_rows(_promote_shared_rows(rows))
    result = {
        "status": "completed", "experiment": "RouteA_H1B2_MFT_VFT_DevelopmentV12_Frozen",
        "candidate_name": manifest["candidate_name"], "evidence_role": "fresh_identity_frozen_confirmation",
        "development_v12_sha256": audit["sha256"], "frozen_manifest_sha256": _sha256(args.frozen_manifest),
        "rows": rows, "vft_mechanism_gate": vft_gate, "mft_shared_policy_gate": mft_gate,
        "unified_promotion_gate": unified_gate,
        "confirmation_passed": bool(vft_gate["passed"] and mft_gate["passed"] and unified_gate["passed"]),
        "external_refit": False, "hyperparameter_search": False, "locked_test_used": False,
        "test_v3_used": False, "test_v12_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_h1_b2_mft_vft_development_v12_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({
        "summary": str(output), "vft_gate": vft_gate, "mft_gate": mft_gate,
        "unified_gate": unified_gate, "passed": result["confirmation_passed"],
    }, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
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

