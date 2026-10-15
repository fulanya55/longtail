"""Capacity-controlled STFT plus raw-source complementarity diagnostic."""

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
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _rank_metrics  # noqa: E402
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _extract_all, _fit_and_score, _hard_source_view  # noqa: E402


ARMS = ("RidgeSTFT", "RidgeSTFTPlusRawSource", "RidgeSTFTPlusPhaseScrambled")
ROLES = ("fit", "validation", "early")


def _standardize(fit: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = fit.mean(axis=0).astype(np.float32)
    std = np.maximum(fit.std(axis=0), 1e-5).astype(np.float32)
    return ((values - mean[None, :]) / std[None, :]).astype(np.float32), mean, std


def _compose(features: dict[str, dict[str, np.ndarray]], arm: str):
    ridge = {}
    raw = {}
    control = {}
    ridge_stats = raw_stats = control_stats = None
    for role in ROLES:
        ridge[role], ridge_mean, ridge_std = _standardize(features["fit"]["RidgeSTFT"], features[role]["RidgeSTFT"])
        raw[role], raw_mean, raw_std = _standardize(features["fit"]["RawSource"], features[role]["RawSource"])
        control[role], control_mean, control_std = _standardize(features["fit"]["PhaseScrambledControl"], features[role]["PhaseScrambledControl"])
        ridge_stats = (ridge_mean, ridge_std); raw_stats = (raw_mean, raw_std); control_stats = (control_mean, control_std)
    if arm == "RidgeSTFT":
        return ridge, {"ridge_mean": ridge_stats[0], "ridge_std": ridge_stats[1]}
    if arm == "RidgeSTFTPlusRawSource":
        return {role: np.concatenate((ridge[role], raw[role]), axis=1).astype(np.float32) for role in ROLES}, {
            "ridge_mean": ridge_stats[0], "ridge_std": ridge_stats[1], "added_mean": raw_stats[0], "added_std": raw_stats[1],
        }
    if arm == "RidgeSTFTPlusPhaseScrambled":
        return {role: np.concatenate((ridge[role], control[role]), axis=1).astype(np.float32) for role in ROLES}, {
            "ridge_mean": ridge_stats[0], "ridge_std": ridge_stats[1], "added_mean": control_stats[0], "added_std": control_stats[1],
        }
    raise ValueError(f"unknown arm {arm}")


def _run_seed(args, labels, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary = root / "seed_summary.json"
    if args.resume and summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    indices_path = args.role_root / f"seed_{seed}" / "role_indices.npz"
    indices = np.load(indices_path)
    role_ids = {role: np.asarray(indices[role], dtype=np.int64) for role in ROLES}
    signals = {role: _load_signals(args.data_path, role_ids[role], args.io_batch_size) for role in ROLES}
    features = {role: _extract_all(signals[role], device, args.feature_batch_size) for role in ROLES}
    views, counts = {}, {}
    for role in ROLES:
        selected, target, receipt = _hard_source_view(labels[role_ids[role]], weak[role_ids[role]])
        if len(np.unique(target)) != 2:
            raise ValueError(f"source view is not estimable: seed={seed}, role={role}")
        views[role] = (selected, target); counts[role] = receipt
    results, arrays = {}, {}
    for arm in ARMS:
        composed, statistics = _compose(features, arm)
        selected, target = views["fit"]
        model, _ = _fit_and_score(composed["fit"][selected], target, composed["fit"][selected], args.ridge_alpha)
        results[arm] = {"feature_dim": int(composed["fit"].shape[1])}
        for role in ("validation", "early"):
            selected, target = views[role]
            results[arm][role] = _rank_metrics(target, model.decision_function(composed[role][selected]))
        arrays[f"{arm}_coef"] = model.coef_.astype(np.float32); arrays[f"{arm}_intercept"] = model.intercept_.astype(np.float32)
        arrays.update({f"{arm}_{name}": value for name, value in statistics.items()})
    deltas = {
        "raw_augmented_minus_ridge": {role: {metric: float(results["RidgeSTFTPlusRawSource"][role][metric] - results["RidgeSTFT"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "raw_augmented_minus_control": {role: {metric: float(results["RidgeSTFTPlusRawSource"][role][metric] - results["RidgeSTFTPlusPhaseScrambled"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "raw_augmented_validation_early_auroc_gap": float(results["RidgeSTFTPlusRawSource"]["validation"]["auroc"] - results["RidgeSTFTPlusRawSource"]["early"]["auroc"]),
    }
    np.savez_compressed(root / "ridge_models.npz", **arrays)
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(indices_path),
               "ridge_alpha": float(args.ridge_alpha), "weak_threshold_db": float(args.weak_threshold_db), "results": results, "deltas": deltas, "matched_counts": counts,
               "validation_used_for_selection": False, "threshold_selected": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary, payload); return payload


def run(args):
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("test_v3_used") or audit.get("test_files_opened"):
        raise ValueError("source-v1 identity/role audit is not valid")
    data_sha = _sha256(args.data_path)
    if data_sha.lower() != audit["data_sha256"].lower():
        raise ValueError("source-v1 data hash changed")
    labels = read_labels(args.data_path); values, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, labels, weak, seed, device, data_sha) for seed in seeds]
    aggregate = {arm: {role: {metric: _mean_ci([row["results"][arm][role][metric] for row in rows], 2026081400 + 100 * ARMS.index(arm) + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for arm in ARMS}
    deltas = {name: {role: {metric: _mean_ci([row["deltas"][name][role][metric] for row in rows], 2026081800 + 100 * index + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for index, name in enumerate(("raw_augmented_minus_ridge", "raw_augmented_minus_control"))}
    raw_positive = int(sum(row["deltas"]["raw_augmented_minus_ridge"]["early"]["auroc"] >= .010 for row in rows))
    gap = _mean_ci([row["deltas"]["raw_augmented_validation_early_auroc_gap"] for row in rows], 2026081999)
    checks = {} if args.smoke else {
        "raw_vs_control_early_auroc_ge_0p010": deltas["raw_augmented_minus_control"]["early"]["auroc"]["mean"] >= .010,
        "raw_vs_control_early_aupr_ge_0p010": deltas["raw_augmented_minus_control"]["early"]["aupr"]["mean"] >= .010,
        "raw_vs_ridge_early_auroc_ge_0p010": deltas["raw_augmented_minus_ridge"]["early"]["auroc"]["mean"] >= .010,
        "raw_vs_ridge_four_of_five_ge_0p010": raw_positive >= 4,
        "raw_augmented_validation_early_gap_lt_0p150": gap["mean"] < .150,
    }
    payload = {"status": "completed", "experiment": "RouteA_VFTISRJ_SourceV1_STFTRawComplementarity", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "deltas": deltas, "validation_early_raw_augmented_auroc_gap": gap,
               "promotion_gate": {"checks": checks, "passed": bool(checks and all(checks.values())), "raw_vs_ridge_early_auroc_ge_0p010_seeds": raw_positive},
               "scope": "fresh_source_fixed_feature_complementarity_only_no_unified_candidate_calibration_or_threshold", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_isrj_source_v1_stft_raw_complementarity_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_source_v1\closedset_6class_train.mat"))
    parser.add_argument("--role-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation")
    parser.add_argument("--source-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation/source_v1_identity_role_audit.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_stft_raw_complementarity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
