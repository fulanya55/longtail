"""Construction ablation: weak-only versus mixed-positive VFT Ridge training."""

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
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles, _matched_view, _rank_metrics  # noqa: E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_route_a_vft_ridge_stft_separability_v3 import _extract  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


TARGET_INDEX = int(TARGET_INDICES[1])
ROLES = ("fit", "validation", "early")


def _all_positive_matched_view(targets6: np.ndarray, target_index: int):
    """Return all target positives and negatives with mutually available co-labels."""
    targets = np.asarray(targets6, dtype=np.uint8)
    other = [value for value in range(targets.shape[1]) if value != int(target_index)]
    positive = targets[:, int(target_index)] == 1
    negative = ~positive
    positive_compositions = {tuple(row) for row in targets[positive][:, other].tolist()}
    negative_compositions = {tuple(row) for row in targets[negative][:, other].tolist()}
    matched = positive_compositions & negative_compositions
    positive_keep = np.asarray([bool(positive[row]) and tuple(targets[row, other].tolist()) in matched for row in range(len(targets))])
    negative_keep = np.asarray([bool(negative[row]) and tuple(targets[row, other].tolist()) in matched for row in range(len(targets))])
    selected = positive_keep | negative_keep
    return selected, positive_keep[selected].astype(np.uint8), {
        "positive_count": int(positive_keep.sum()),
        "matched_negative_count": int(negative_keep.sum()),
        "matched_composition_count": int(len(matched)),
    }


def _fit_score(train_x, train_y, eval_x, alpha: float):
    positives = max(int(train_y.sum()), 1)
    negatives = max(int((train_y == 0).sum()), 1)
    weights = np.where(train_y == 1, 0.5 / positives, 0.5 / negatives).astype(np.float64)
    model = RidgeClassifier(alpha=float(alpha), fit_intercept=True)
    model.fit(train_x, train_y, sample_weight=weights)
    return model.decision_function(eval_x).astype(np.float32)


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    summary = root / "seed_summary.json"
    if args.resume and summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, indices_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    features = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ROLES}
    mean = features["fit"].mean(axis=0).astype(np.float32)
    std = np.maximum(features["fit"].std(axis=0), 1e-5).astype(np.float32)
    normalized = {role: ((value - mean[None, :]) / std[None, :]).astype(np.float32) for role, value in features.items()}
    weak_views, mixed_views, receipts = {}, {}, {}
    for role in ROLES:
        selected, labels, weak_receipt = _matched_view(roles[role]["targets6"], weak6[roles[role]["sample_ids"]], TARGET_INDEX)
        weak_views[role] = (normalized[role][selected], labels.astype(np.uint8))
        selected, labels, mixed_receipt = _all_positive_matched_view(roles[role]["targets6"], TARGET_INDEX)
        mixed_views[role] = (normalized[role][selected], labels)
        receipts[role] = {"weak_evaluation_view": weak_receipt, "mixed_training_view": mixed_receipt}
    metrics = {}
    for name, train in (("weak_only", weak_views["fit"]), ("mixed_positive", mixed_views["fit"])):
        metrics[name] = {}
        for role in ("validation", "early"):
            score = _fit_score(train[0], train[1], weak_views[role][0], args.ridge_alpha)
            metrics[name][role] = _rank_metrics(weak_views[role][1], score)
    result = {
        "weak_only": metrics["weak_only"], "mixed_positive": metrics["mixed_positive"],
        "mixed_minus_weak": {
            role: {key: float(metrics["mixed_positive"][role][key] - metrics["weak_only"][role][key]) for key in ("auroc", "aupr")}
            for role in ("validation", "early")
        },
        "feature_dim": int(normalized["fit"].shape[1]),
    }
    payload = {
        "status": "completed", "seed": int(seed), "data_sha256": data_sha,
        "role_indices_sha256": _sha256(indices_path), "ridge_alpha": float(args.ridge_alpha),
        "weak_threshold_db": float(args.weak_threshold_db), "receipts": receipts, "result": result,
        "validation_used_for_selection": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    path = _data_path(config)
    data_sha = _sha256(path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(path)
    values, _ = read_parameters(path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for role in ("validation", "early"):
        aggregate[role] = {}
        for metric in ("auroc", "aupr"):
            values = [row["result"]["mixed_minus_weak"][role][metric] for row in rows]
            aggregate[role][f"mixed_minus_weak_{metric}"] = _mean_ci(values, 2026080100 + 10 * (role == "early") + (metric == "aupr"))
    payload = {
        "status": "completed", "experiment": "RouteA_VFTWeakMixedPositiveAblation_v1", "smoke": bool(args.smoke),
        "seeds": list(seeds), "data_sha256": data_sha, "ridge_alpha": float(args.ridge_alpha),
        "weak_threshold_db": float(args.weak_threshold_db), "aggregate": aggregate,
        "interpretation": "construction_ablation_only_no_promotion_or_threshold_selection",
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_vft_weak_mixed_positive_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_weak_mixed_positive_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
