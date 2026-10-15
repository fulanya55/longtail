"""Evaluate the complete frozen candidate on external ISRJ parameter OOD."""

from __future__ import annotations

import argparse
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
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import PEAK_INDEX, _load_json, _run_seed  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v12_frozen import (  # noqa: E402
    SHARED_VARIANT, _mft_gate, _promote_shared_rows, _shared_variant,
)
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _goal_rows, _vft_gate  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


def _variant(row, name):
    return next(item for item in row["variants"] if item["variant"] == name)


def _external_gate(rows):
    per_seed = []
    for row in rows:
        base = _variant(row, "Q0_H1")["metrics"]
        candidate = _variant(row, SHARED_VARIANT)["metrics"]
        per_seed.append({
            "seed": row["seed"], "macro_aupr": candidate["macro_aupr"] - base["macro_aupr"],
            "macro_f1": candidate["macro_f1"] - base["macro_f1"],
            "isrj_aupr": candidate["per_label"]["ISRJ"]["aupr"] - base["per_label"]["ISRJ"]["aupr"],
            "isrj_weak_recall": candidate["per_label"]["ISRJ"]["weak_recall"] - base["per_label"]["ISRJ"]["weak_recall"],
            "isrj_weak_auroc": candidate["per_label"]["ISRJ"]["weak_vs_negative_auroc"] - base["per_label"]["ISRJ"]["weak_vs_negative_auroc"],
            "isrj_fpr": candidate["per_label"]["ISRJ"]["negative_fpr"],
            "sawfm_aupr": candidate["per_label"]["SawFM"]["aupr"] - base["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall": candidate["per_label"]["SawFM"]["weak_recall"] - base["per_label"]["SawFM"]["weak_recall"],
        })
    aggregate = {
        key: _mean_ci([item[key] for item in per_seed], 20430000 + index)
        for index, key in enumerate(("macro_aupr", "macro_f1", "isrj_aupr", "isrj_weak_recall", "isrj_weak_auroc", "isrj_fpr"))
    }
    checks = {
        "macro_aupr_nonnegative": aggregate["macro_aupr"]["mean"] >= 0.0,
        "macro_f1_nonnegative": aggregate["macro_f1"]["mean"] >= 0.0,
        "isrj_aupr_positive": aggregate["isrj_aupr"]["mean"] > 0.0,
        "isrj_weak_recall_gain_ge_0p030": aggregate["isrj_weak_recall"]["mean"] >= 0.030,
        "isrj_weak_auroc_positive": aggregate["isrj_weak_auroc"]["mean"] > 0.0,
        "isrj_fpr_le_0p25": aggregate["isrj_fpr"]["mean"] <= 0.25,
        "sawfm_aupr_exact": max(abs(item["sawfm_aupr"]) for item in per_seed) <= 1e-12,
        "sawfm_weak_recall_exact": max(abs(item["sawfm_weak_recall"]) for item in per_seed) <= 1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks, "aggregate": aggregate, "per_seed": per_seed}


def run(args):
    audit = _load_json(args.identity_audit)
    if not audit["identity_audit_passed"] or not audit["parameter_ood_passed"] or audit["test_files_opened"]:
        raise ValueError("ISRJ OOD identity/parameter audit failed")
    seal = _load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = _load_json(args.frozen_manifest)
    if manifest["candidate_name"] != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen candidate")
    data_path = Path(audit["path"])
    if _sha256(data_path).lower() != audit["sha256"].lower():
        raise ValueError("external data hash changed")
    config = load_config(args.config); _validate_config(config)
    labels3 = read_labels(_data_path(config)); params3, _ = read_parameters(_data_path(config))
    weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    labels = read_labels(data_path); params, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(params), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    periodic = _extract_periodicity(signals, device, args.periodicity_batch_size)
    vft = _extract_vft(signals, device, args.vft_batch_size)
    mft = extract_scores(signals, device, args.mft_batch_size)["MF_AlignedLFM"][:, PEAK_INDEX]
    rows = []
    for seed in args.seeds:
        row = _run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic, vft, mft, seed, device, manifest)
        rows.append(_shared_variant(args, seed, row))
    external_gate = _external_gate(rows)
    vft_gate = _vft_gate(rows)
    mft_gate = _mft_gate(rows)
    unified_gate = _goal_rows(_promote_shared_rows(rows))
    result = {
        "status": "completed", "experiment": "RouteA_FinalSharedPolicy_ISRJParameterOOD_Frozen",
        "evidence_role": "external_controlled_generator_parameter_ood", "data_sha256": audit["sha256"],
        "rows": rows, "external_gate": external_gate, "vft_gate": vft_gate, "mft_gate": mft_gate,
        "unified_gate": unified_gate,
        "passed": bool(external_gate["passed"] and vft_gate["passed"] and mft_gate["passed"] and unified_gate["passed"]),
        "scope_limit": "same controlled generator; ISRJ parameter range OOD only, not new noise or real data",
        "external_refit": False, "hyperparameter_search": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_final_shared_policy_isrj_ood_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"passed": result["passed"], "external_gate": external_gate, "vft_gate": vft_gate, "mft_gate": mft_gate, "unified_gate": unified_gate, "output": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_parameter_ood_v1_generation/identity_parameter_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen")
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

