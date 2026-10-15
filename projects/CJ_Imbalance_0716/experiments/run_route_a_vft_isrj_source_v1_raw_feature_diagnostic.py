"""Run the fixed VFT-ISRJ source diagnostic on fresh balanced source-v1 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

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
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import ARMS, _extract_all, _fit_and_score, _hard_source_view, _normalise  # noqa: E402


def _run_seed(args, labels, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    indices = np.load(args.role_root / f"seed_{seed}" / "role_indices.npz")
    roles = {role: np.asarray(indices[role], dtype=np.int64) for role in ("fit", "validation", "early")}
    signals = {role: _load_signals(args.data_path, role_ids, args.io_batch_size) for role, role_ids in roles.items()}
    features = {role: _extract_all(value, device, args.feature_batch_size) for role, value in signals.items()}
    views, counts = {}, {}
    for role, role_ids in roles.items():
        selected, target, count = _hard_source_view(labels[role_ids], weak[role_ids])
        if not target.any() or np.all(target == 1):
            raise ValueError(f"seed {seed} role {role} source view is not estimable")
        views[role] = (selected, target); counts[role] = count
    results, model_arrays = {}, {}
    for arm in ARMS:
        normalized, mean, std = _normalise(features, arm)
        train_selected, train_y = views["fit"]
        model, _ = _fit_and_score(normalized["fit"][train_selected], train_y, normalized["fit"][train_selected], args.ridge_alpha)
        results[arm] = {"feature_dim": int(mean.size)}
        for role in ("validation", "early"):
            selected, target = views[role]
            results[arm][role] = _rank_metrics(target, model.decision_function(normalized[role][selected]))
        model_arrays[f"{arm}_coef"] = model.coef_.astype(np.float32); model_arrays[f"{arm}_intercept"] = model.intercept_.astype(np.float32)
        model_arrays[f"{arm}_mean"] = mean; model_arrays[f"{arm}_std"] = std
    deltas = {
        "raw_minus_ridge": {role: {metric: float(results["RawSource"][role][metric] - results["RidgeSTFT"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "raw_minus_scrambled": {role: {metric: float(results["RawSource"][role][metric] - results["PhaseScrambledControl"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")},
        "validation_early_raw_auroc_gap": float(results["RawSource"]["validation"]["auroc"] - results["RawSource"]["early"]["auroc"]),
    }
    np.savez_compressed(root / "ridge_models.npz", **model_arrays)
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(args.role_root / f"seed_{seed}" / "role_indices.npz"),
               "ridge_alpha": float(args.ridge_alpha), "weak_threshold_db": float(args.weak_threshold_db), "results": results, "deltas": deltas,
               "matched_counts": counts, "validation_used_for_selection": False, "threshold_selected": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(receipt, payload); return payload


def run(args):
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("test_v3_used") or audit.get("test_files_opened"):
        raise ValueError("source-v1 identity/estimability audit failed")
    data_sha = _sha256(args.data_path)
    if data_sha.lower() != audit["data_sha256"].lower():
        raise ValueError("source-v1 data hash changed")
    labels = read_labels(args.data_path); parameters, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, labels, weak, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for name in ("raw_minus_ridge", "raw_minus_scrambled"):
        aggregate[name] = {role: {metric: _mean_ci([row["deltas"][name][role][metric] for row in rows], 2026081100 + 100 * (name == "raw_minus_scrambled") + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")}
    aggregate["validation_early_raw_auroc_gap"] = _mean_ci([row["deltas"]["validation_early_raw_auroc_gap"] for row in rows], 2026081199)
    positive = int(sum(row["deltas"]["raw_minus_ridge"]["early"]["auroc"] > 0 for row in rows))
    checks = {} if args.smoke else {"early_raw_beats_ridge_ge_0p010": aggregate["raw_minus_ridge"]["early"]["auroc"]["mean"] >= .010,
                                    "early_raw_beats_ridge_four_of_five": positive >= 4,
                                    "early_raw_beats_phase_scrambled_ge_0p020": aggregate["raw_minus_scrambled"]["early"]["auroc"]["mean"] >= .020,
                                    "raw_validation_early_gap_lt_0p150": aggregate["validation_early_raw_auroc_gap"]["mean"] < .150}
    payload = {"status": "completed", "experiment": "RouteA_VFTISRJ_SourceV1_RawFeatureDiagnostic", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "promotion_gate": {"checks": checks, "passed": bool(checks and all(checks.values())), "positive_early_vs_ridge_seeds": positive},
               "scope": "fresh_source_identifiability_only_not_cross_domain_or_unified_candidate", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_isrj_source_v1_raw_feature_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_source_v1\closedset_6class_train.mat"))
    parser.add_argument("--role-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation")
    parser.add_argument("--source-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation/source_v1_identity_role_audit.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_raw_feature_diagnostic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
