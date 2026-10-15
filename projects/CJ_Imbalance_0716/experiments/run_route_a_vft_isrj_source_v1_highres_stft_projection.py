"""Equal-classifier-capacity high-resolution VFT--ISRJ STFT diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa: E402
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _fit_and_score, _hard_source_view, _phase_scramble  # noqa: E402


ARMS = ("Native16", "High32Projected", "High32PhaseScrambledProjected")
ROLES = ("fit", "validation", "early")
PROJECTOR_SEED = 2026082501
NATIVE_DIM = 16 * 16 + 12


def _projector(input_dim: int, output_dim: int = NATIVE_DIM, seed: int = PROJECTOR_SEED) -> np.ndarray:
    values = np.random.default_rng(int(seed)).normal(0.0, 1.0 / np.sqrt(float(input_dim)), size=(int(input_dim), int(output_dim)))
    return values.astype(np.float32)


def _extract(signals: np.ndarray, device: torch.device, batch_size: int, projector: np.ndarray) -> dict[str, np.ndarray]:
    rows = {name: [] for name in ARMS}
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            native = _vft_features(batch, grid_size=16).cpu().numpy().astype(np.float32)
            high = _vft_features(batch, grid_size=32).cpu().numpy().astype(np.float32)
            scrambled = _vft_features(_phase_scramble(batch), grid_size=32).cpu().numpy().astype(np.float32)
            rows["Native16"].append(native)
            rows["High32Projected"].append((high @ projector).astype(np.float32))
            rows["High32PhaseScrambledProjected"].append((scrambled @ projector).astype(np.float32))
    return {name: np.concatenate(value, axis=0) for name, value in rows.items()}


def _normalise(features: dict[str, dict[str, np.ndarray]], arm: str):
    mean = features["fit"][arm].mean(axis=0).astype(np.float32)
    std = np.maximum(features["fit"][arm].std(axis=0), 1e-5).astype(np.float32)
    return {role: ((features[role][arm] - mean[None, :]) / std[None, :]).astype(np.float32) for role in ROLES}, mean, std


def _run_seed(args, labels, weak, seed, device, data_sha, projector, projector_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    indices_path = args.role_root / f"seed_{seed}" / "role_indices.npz"
    loaded = np.load(indices_path)
    ids = {role: np.asarray(loaded[role], dtype=np.int64) for role in ROLES}
    features = {role: _extract(_load_signals(args.data_path, ids[role], args.io_batch_size), device, args.feature_batch_size, projector) for role in ROLES}
    views, counts = {}, {}
    for role in ROLES:
        selected, target, detail = _hard_source_view(labels[ids[role]], weak[ids[role]])
        if len(np.unique(target)) != 2:
            raise ValueError(f"source view is not estimable: seed={seed}, role={role}")
        views[role] = (selected, target); counts[role] = detail
    results, arrays = {}, {}
    for arm in ARMS:
        normalised, mean, std = _normalise(features, arm)
        selected, target = views["fit"]
        model, _ = _fit_and_score(normalised["fit"][selected], target, normalised["fit"][selected], args.ridge_alpha)
        results[arm] = {"feature_dim": int(normalised["fit"].shape[1])}
        for role in ("validation", "early"):
            selected, target = views[role]
            results[arm][role] = _rank_metrics(target, model.decision_function(normalised[role][selected]))
        arrays[f"{arm}_coef"] = model.coef_.astype(np.float32)
        arrays[f"{arm}_intercept"] = model.intercept_.astype(np.float32)
        arrays[f"{arm}_mean"] = mean; arrays[f"{arm}_std"] = std
    deltas = {
        "high_minus_native": {role: {metric: float(results["High32Projected"][role][metric] - results["Native16"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "high_minus_scrambled": {role: {metric: float(results["High32Projected"][role][metric] - results["High32PhaseScrambledProjected"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "high_validation_early_auroc_gap": float(results["High32Projected"]["validation"]["auroc"] - results["High32Projected"]["early"]["auroc"]),
    }
    np.savez_compressed(root / "ridge_models.npz", projector=projector, **arrays)
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(indices_path), "projector_sha256": projector_sha,
               "ridge_alpha": float(args.ridge_alpha), "results": results, "deltas": deltas, "matched_counts": counts,
               "validation_used_for_selection": False, "threshold_selected": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(receipt, payload); return payload


def run(args):
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("test_v3_used") or audit.get("test_files_opened"):
        raise ValueError("source-v1 identity audit is not valid")
    data_sha = _sha256(args.data_path)
    if data_sha.lower() != audit["data_sha256"].lower():
        raise ValueError("source-v1 hash changed")
    labels = read_labels(args.data_path); values, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    high_dim = 32 * 32 + 12; projector = _projector(high_dim)
    projector_sha = hashlib.sha256(projector.tobytes(order="C")).hexdigest()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, labels, weak, seed, device, data_sha, projector, projector_sha) for seed in seeds]
    aggregate = {arm: {role: {metric: _mean_ci([row["results"][arm][role][metric] for row in rows], 2026082600 + 100 * ARMS.index(arm) + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for arm in ARMS}
    deltas = {name: {role: {metric: _mean_ci([row["deltas"][name][role][metric] for row in rows], 2026082700 + 100 * index + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for index, name in enumerate(("high_minus_native", "high_minus_scrambled"))}
    positive = int(sum(row["deltas"]["high_minus_native"]["early"]["auroc"] >= .020 for row in rows))
    gap = _mean_ci([row["deltas"]["high_validation_early_auroc_gap"] for row in rows], 2026082899)
    checks = {} if args.smoke else {
        "high_vs_native_early_auroc_ge_0p020": deltas["high_minus_native"]["early"]["auroc"]["mean"] >= .020,
        "high_vs_native_four_of_five_ge_0p020": positive >= 4,
        "high_vs_scrambled_early_auroc_ge_0p020": deltas["high_minus_scrambled"]["early"]["auroc"]["mean"] >= .020,
        "high_validation_early_gap_lt_0p150": gap["mean"] < .150,
    }
    payload = {"status": "completed", "experiment": "RouteA_VFTISRJ_SourceV1_HighResSTFTProjection", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "native_feature_dim": NATIVE_DIM, "high_feature_dim_before_projection": high_dim, "projector_sha256": projector_sha, "aggregate": aggregate, "deltas": deltas, "high_validation_early_auroc_gap": gap,
               "promotion_gate": {"checks": checks, "passed": bool(checks and all(checks.values())), "high_vs_native_early_auroc_ge_0p020_seeds": positive},
               "scope": "fresh_source_equal_classifier_capacity_resolution_diagnostic_only", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_vft_isrj_source_v1_highres_stft_projection_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_source_v1\closedset_6class_train.mat"))
    parser.add_argument("--role-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation")
    parser.add_argument("--source-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation/source_v1_identity_role_audit.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_highres_stft_projection")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
