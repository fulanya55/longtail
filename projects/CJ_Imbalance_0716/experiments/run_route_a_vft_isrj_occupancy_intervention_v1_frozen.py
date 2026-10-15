"""Frozen Q4 evaluation on fresh paired ISRJ width-by-replica interventions."""

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
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import PEAK_INDEX, _load_json, _run_seed  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


REGIMES = ("W10_15_R3_4", "W22_30_R3_4", "W10_15_R6_8", "W22_30_R6_8")
REFERENCE = REGIMES[0]
Q4 = "Q4_VFTAlignedRidgeFusion"
VFT, ISRJ = 4, 5


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int = 2000) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("empty paired hard-negative view")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[sample].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "n": int(len(values))}


def _read_q4(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as item:
        return {name: np.asarray(item[name]) for name in ("receiver_noise_seed", "targets", "probabilities", "predictions", "thresholds")}


def _paired_effect(reference: dict, intervention: dict, seed: int) -> dict:
    if not np.array_equal(reference["receiver_noise_seed"], intervention["receiver_noise_seed"]):
        raise ValueError("receiver pairing/order changed")
    if not np.array_equal(reference["targets"], intervention["targets"]):
        raise ValueError("labels changed across intervention")
    targets = reference["targets"]
    hard = (targets[:, VFT] == 0) & (targets[:, ISRJ] == 1)
    if hard.sum() == 0:
        raise ValueError("no VFT-absent/ISRJ-present hard negatives")
    score_delta = intervention["probabilities"][hard, VFT] - reference["probabilities"][hard, VFT]
    prediction_delta = intervention["predictions"][hard, VFT].astype(float) - reference["predictions"][hard, VFT].astype(float)
    return {
        "hard_negative_count": int(hard.sum()),
        "paired_probability_delta": _bootstrap_mean(score_delta, 20801000 + int(seed)),
        "paired_fpr_delta": _bootstrap_mean(prediction_delta, 20802000 + int(seed)),
        "reference_fpr": float(reference["predictions"][hard, VFT].mean()),
        "intervention_fpr": float(intervention["predictions"][hard, VFT].mean()),
        "threshold_exact": bool(np.array_equal(reference["thresholds"], intervention["thresholds"])),
    }


def _aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key]["mean"] for row in rows], dtype=float)
    return _bootstrap_mean(values, 20803000 + sum(map(ord, key))) | {"positive_seed_count": int((values > 0).sum())}


def _load_regime(path: Path, batch_size: int, device: torch.device, periodic_batch: int, vft_batch: int, mft_batch: int, weak_threshold: float):
    labels = read_labels(path)
    params, _ = read_parameters(path)
    weak = compute_weak_component_mask(labels, _component_power(params), weak_threshold)
    signals = _load_signals(path, np.arange(len(labels)), batch_size)
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    return labels, weak, receiver, signals, _extract_periodicity(signals, device, periodic_batch), _extract_vft(signals, device, vft_batch), extract_scores(signals, device, mft_batch)["MF_AlignedLFM"][:, PEAK_INDEX]


def run(args):
    audit = _load_json(args.audit)
    if not audit.get("passed") or audit.get("test_files_opened"):
        raise ValueError("paired intervention data audit failed")
    seal = _load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = _load_json(args.frozen_manifest)
    if manifest.get("candidate_name") != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen shared-policy candidate")
    records = {name: {} for name in REGIMES}
    original_output = args.output_root
    if args.collate_only:
        for name in REGIMES:
            for model_seed in args.seeds:
                prediction_path = original_output / name / f"seed_{model_seed}" / Q4 / "predictions.npz"
                if not prediction_path.is_file():
                    raise FileNotFoundError(f"cannot collate missing frozen prediction: {prediction_path}")
                records[name][model_seed] = _read_q4(prediction_path)
    else:
        config = load_config(args.config)
        _validate_config(config)
        labels3 = read_labels(_data_path(config)); params3, _ = read_parameters(_data_path(config))
        weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        regime_inputs = {}
        for name in REGIMES:
            path = args.data_root / name / "closedset_6class_train.mat"
            if _sha256(path).lower() != audit["regimes"][name]["sha256"].lower():
                raise ValueError(f"data hash changed for {name}")
            regime_inputs[name] = _load_regime(path, args.io_batch_size, device, args.periodicity_batch_size, args.vft_batch_size, args.mft_batch_size, args.weak_threshold_db)
        for name in REGIMES:
            labels, weak, receiver, signals, periodic, vft, mft = regime_inputs[name]
            args.output_root = original_output / name
            for model_seed in args.seeds:
                _run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic, vft, mft, model_seed, device, manifest)
                prediction_path = args.output_root / f"seed_{model_seed}" / Q4 / "predictions.npz"
                records[name][model_seed] = _read_q4(prediction_path)
        args.output_root = original_output
    comparisons = {
        "width": "W22_30_R3_4",
        "replicas": "W10_15_R6_8",
        "joint": "W22_30_R6_8",
    }
    effects = {}
    for comparison, regime in comparisons.items():
        rows = []
        for model_seed in args.seeds:
            rows.append({"seed": int(model_seed), **_paired_effect(records[REFERENCE][model_seed], records[regime][model_seed], model_seed)})
        effects[comparison] = {"intervention": regime, "per_seed": rows, "probability": _aggregate(rows, "paired_probability_delta"), "fpr": _aggregate(rows, "paired_fpr_delta")}
    gates = {
        "width_probability_delta_ge_0p020": effects["width"]["probability"]["mean"] >= .020 and effects["width"]["probability"]["positive_seed_count"] >= 4,
        "replica_probability_delta_ge_0p020": effects["replicas"]["probability"]["mean"] >= .020 and effects["replicas"]["probability"]["positive_seed_count"] >= 4,
        "joint_fpr_delta_ge_0p050": effects["joint"]["fpr"]["mean"] >= .050 and effects["joint"]["fpr"]["positive_seed_count"] >= 4,
        "all_thresholds_exact": all(row["threshold_exact"] for item in effects.values() for row in item["per_seed"]),
        "test_v3_sealed": True,
    }
    result = {
        "status": "completed", "experiment": "RouteA_VFTISRJ_OccupancyInterventionV1_Frozen",
        "evidence_role": "fresh_paired_external_physics_diagnostic", "candidate": manifest["candidate_name"],
        "q4_variant": Q4, "reference_regime": REFERENCE, "effects": effects, "gates": gates,
        "occupancy_causal_screen_passed": bool(all(gates.values())), "external_refit": False, "hyperparameter_search": False,
        "collated_existing_frozen_predictions": bool(args.collate_only),
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    original_output.mkdir(parents=True, exist_ok=True)
    output = original_output / "route_a_vft_isrj_occupancy_intervention_v1_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"passed": result["occupancy_causal_screen_passed"], "gates": gates, "output": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_generation/identity_parameter_pair_audit.json")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_occupancy_v1"))
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_frozen")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--collate-only", action="store_true", help="Write the receipt from already complete frozen Q4 predictions; never performs inference.")
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--backbone-batch-size", type=int, default=128); parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--periodicity-batch-size", type=int, default=64)
    parser.add_argument("--vft-batch-size", type=int, default=32); parser.add_argument("--mft-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1); parser.add_argument("--weak-threshold-db", type=float, default=10.)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
