"""Audit whether frozen v16 predicted ISRJ width aligns with fixed Q4 VFT risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _mean_ci, _read as _load_json, _write  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _run_seed  # noqa: E402
from experiments.run_route_a_v17_isrj_predicted_width_transport import ISRJ, VFT  # noqa: E402
from experiments.run_route_a_vft_isrj_occupancy_intervention_v1_frozen import Q4, _load_regime, _read_q4  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402


NARROW = "W10_15_R3_4"
WIDE = "W22_30_R3_4"


def _sealed(seal: dict) -> bool:
    return not any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used"))


def _load_width_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as item:
        return {key: np.asarray(item[key]) for key in ("receiver_noise_seed", "narrow_prediction", "wide_prediction", "narrow_true", "wide_true")}


def _hard_q4(record: dict) -> dict:
    hard = (record["targets"][:, VFT] == 0) & (record["targets"][:, ISRJ] == 1)
    if not hard.any():
        raise ValueError("no VFT-absent / ISRJ-present hard negatives")
    return {
        "receiver": record["receiver_noise_seed"][hard],
        "targets": record["targets"][hard],
        "probability": record["probabilities"][hard, VFT],
        "prediction": record["predictions"][hard, VFT].astype(np.float64),
        "threshold": record["thresholds"],
    }


def _row(seed: int, narrow: dict, wide: dict, width: dict) -> dict:
    n = _hard_q4(narrow)
    w = _hard_q4(wide)
    if not np.array_equal(n["receiver"], w["receiver"]):
        raise ValueError("Q4 hard-negative receiver pairing changed")
    if not np.array_equal(n["targets"], w["targets"]):
        raise ValueError("Q4 hard-negative labels changed")
    if not np.array_equal(n["threshold"], w["threshold"]):
        raise ValueError("frozen Q4 thresholds changed across paired regimes")
    if not np.array_equal(n["receiver"], width["receiver_noise_seed"]):
        raise ValueError("width predictor and Q4 narrow hard-negative identities differ")
    if len(n["receiver"]) != len(width["narrow_prediction"]):
        raise ValueError("width/Q4 hard-negative cardinality differs")
    probability_delta = w["probability"] - n["probability"]
    fpr_delta = w["prediction"] - n["prediction"]
    widths = np.concatenate((width["narrow_prediction"], width["wide_prediction"]))
    probabilities = np.concatenate((n["probability"], w["probability"]))
    association = float(spearmanr(widths, probabilities).statistic)
    return {
        "seed": int(seed),
        "hard_negative_count_per_regime": int(len(n["receiver"])),
        "paired_probability_delta": float(probability_delta.mean()),
        "paired_fpr_delta": float(fpr_delta.mean()),
        "width_q4_probability_spearman": association,
        "narrow_fpr": float(n["prediction"].mean()),
        "wide_fpr": float(w["prediction"].mean()),
        "thresholds_exact": True,
        "paired_receivers_exact": True,
    }


def _run_q4(args, seeds: list[int], output_root: Path) -> None:
    audit = _load_json(args.audit)
    config = load_config(args.config)
    _validate_config(config)
    labels3 = read_labels(_data_path(config))
    params3, _ = read_parameters(_data_path(config))
    weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    regime_inputs = {}
    for name in (NARROW, WIDE):
        path = args.data_root / name / "closedset_6class_train.mat"
        if _sha256(path).lower() != audit["regimes"][name]["sha256"].lower():
            raise ValueError(f"data hash changed for {name}")
        regime_inputs[name] = _load_regime(path, args.io_batch_size, device, args.periodicity_batch_size, args.vft_batch_size, args.mft_batch_size, args.weak_threshold_db)
    manifest = _load_json(args.frozen_manifest)
    for name in (NARROW, WIDE):
        labels, weak, receiver, signals, periodic, vft, mft = regime_inputs[name]
        args.output_root = output_root / name
        for seed in seeds:
            _run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic, vft, mft, seed, device, manifest)


def run(args):
    audit = _load_json(args.audit)
    transport = _load_json(args.transport_summary)
    seal = _load_json(args.test_v3_manifest)
    manifest = _load_json(args.frozen_manifest)
    if not audit.get("identity_audit_passed") or audit.get("test_files_opened"):
        raise ValueError("v17 identity audit failed")
    if not transport.get("gate", {}).get("passed"):
        raise ValueError("v17 predicted-width transport gate failed")
    if manifest.get("candidate_name") != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen Q4 candidate")
    if not _sealed(seal):
        raise ValueError("test-v3 seal is not intact")
    output_root = args.output_root
    if not args.collate_only:
        _run_q4(args, list(args.seeds), output_root)
    rows = []
    for seed in args.seeds:
        narrow = _read_q4(output_root / NARROW / f"seed_{seed}" / Q4 / "predictions.npz")
        wide = _read_q4(output_root / WIDE / f"seed_{seed}" / Q4 / "predictions.npz")
        width = _load_width_prediction(args.width_root / f"seed_{seed}" / "predictions.npz")
        rows.append(_row(seed, narrow, wide, width))
    aggregate = {key: _mean_ci([row[key] for row in rows], 2026080500 + index) for index, key in enumerate(("paired_probability_delta", "paired_fpr_delta", "width_q4_probability_spearman"))}
    aggregate["positive_probability_delta_seeds"] = int(sum(row["paired_probability_delta"] > 0 for row in rows))
    aggregate["positive_fpr_delta_seeds"] = int(sum(row["paired_fpr_delta"] > 0 for row in rows))
    aggregate["positive_spearman_seeds"] = int(sum(row["width_q4_probability_spearman"] > 0 for row in rows))
    checks = {
        "q4_probability_delta_ge_0p020_and_4_of_5_positive": aggregate["paired_probability_delta"]["mean"] >= .020 and aggregate["positive_probability_delta_seeds"] >= 4,
        "q4_fpr_delta_ge_0p050_and_4_of_5_positive": aggregate["paired_fpr_delta"]["mean"] >= .050 and aggregate["positive_fpr_delta_seeds"] >= 4,
        "width_q4_spearman_ge_0p20_and_4_of_5_positive": aggregate["width_q4_probability_spearman"]["mean"] >= .20 and aggregate["positive_spearman_seeds"] >= 4,
        "all_paired_receivers_exact": all(row["paired_receivers_exact"] for row in rows),
        "all_thresholds_exact": all(row["thresholds_exact"] for row in rows),
        "test_v3_sealed": True,
    }
    passed = bool(all(checks.values()))
    result = {
        "status": "completed",
        "experiment": "route_a_v17_predicted_width_frozen_q4_alignment",
        "evidence_role": "fresh_external_physics_frozen_risk_alignment_diagnostic",
        "candidate": manifest["candidate_name"],
        "q4_variant": Q4,
        "data_audit_sha256": _sha256(args.audit),
        "frozen_manifest_sha256": _sha256(args.frozen_manifest),
        "transport_summary_sha256": _sha256(args.transport_summary),
        "per_seed": rows,
        "aggregate": aggregate,
        "gate": {"checks": checks, "passed": passed, "decision": "permit_fresh_data_bounded_width_residual_design" if passed else "stop_width_conditioned_correction"},
        "external_refit": False,
        "calibration_reused": True,
        "thresholds_reused": True,
        "hyperparameter_search": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v17_used": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "route_a_v17_predicted_width_frozen_q4_alignment_summary.json", result)
    _write(output_root / "run_receipt.json", {"status": "completed", "gate_passed": passed, "decision": result["gate"]["decision"], "external_refit": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v17_used": False})
    print(json.dumps({"passed": passed, "checks": checks, "output": str(output_root)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v17_isrj_width_transport"))
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v17_isrj_width_transport_generation/identity_parameter_pair_audit.json")
    parser.add_argument("--transport-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v17_isrj_predicted_width_transport/route_a_v17_isrj_predicted_width_transport_summary.json")
    parser.add_argument("--width-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v17_isrj_predicted_width_transport")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v17_predicted_width_frozen_q4_alignment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--collate-only", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--backbone-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--periodicity-batch-size", type=int, default=64)
    parser.add_argument("--vft-batch-size", type=int, default=32)
    parser.add_argument("--mft-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--weak-threshold-db", type=float, default=10.)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
