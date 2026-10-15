"""No-parameter VFT final-temporal plus STFT score complementarity test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_dfms_iq_feature_sufficiency import _temporal_summary  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (  # noqa: E402
    _load_roles,
    _matched_view,
    _rank_metrics,
    _scores,
)
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


TARGET_INDEX = 4
TARGET_NAME = "VFT"
ROLES = ("fit", "validation", "early")
ROLE_OFFSETS = {"fit": 11, "validation": 23, "early": 37}


def _extract_final_temporal(model, signals, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            temporal = model.forward_structure_maps(batch)["temporal"].float().cpu().numpy()
            rows.append(_temporal_summary(temporal, chunk_size=int(batch_size)))
    return np.concatenate(rows).astype(np.float32)


def _extract_vft_stft(signals, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(_vft_features(batch, grid_size=16).cpu().numpy().astype(np.float32))
    return np.concatenate(rows).astype(np.float32)


def _load_probe(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weight = payload["model_state_dict"]["weight"]
    probe = nn.Linear(int(weight.shape[1]), 1).to(device)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    return probe, payload


def _normalize_features(values, payload):
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(payload["std"], dtype=np.float32), 1e-5)
    if values.shape[1] != len(mean):
        raise ValueError(f"feature dimension mismatch: {values.shape[1]} != {len(mean)}")
    return ((values - mean[None, :]) / std[None, :]).astype(np.float32)


def _fit_score_stats(scores):
    values = np.asarray(scores, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(max(values.std(), 1e-8))}


def _standardize_scores(scores, stats):
    return ((np.asarray(scores, dtype=np.float64) - float(stats["mean"])) /
            float(stats["std"])).astype(np.float32)


def _fixed_permutation(length, seed):
    permutation = np.random.default_rng(int(seed)).permutation(int(length)).astype(np.int64)
    digest = hashlib.sha256(permutation.tobytes(order="C")).hexdigest()
    return permutation, digest


def _fused_scores(final_scores, stft_scores):
    return (0.5 * np.asarray(final_scores) + 0.5 * np.asarray(stft_scores)).astype(np.float32)


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)

    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    c6_checkpoint = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    model = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state)

    final_features = {
        role: _extract_final_temporal(model, roles[role]["signals"], device, args.feature_batch_size)
        for role in ROLES
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stft_features = {
        role: _extract_vft_stft(roles[role]["signals"], device, args.stft_batch_size)
        for role in ROLES
    }

    final_checkpoint = args.layerwise_root / f"seed_{seed}" / "T_FinalTemporal" / TARGET_NAME / "best_probe.pth"
    stft_checkpoint = args.physics_root / f"seed_{seed}" / TARGET_NAME / "best_probe.pth"
    final_probe, final_payload = _load_probe(final_checkpoint, device)
    stft_probe, stft_payload = _load_probe(stft_checkpoint, device)

    selections, labels, receipts = {}, {}, {}
    final_raw_scores, stft_raw_scores = {}, {}
    sample_ids = {}
    for role in ROLES:
        selected, target, receipt = _matched_view(
            roles[role]["targets6"], weak6[roles[role]["sample_ids"]], TARGET_INDEX
        )
        if len(np.unique(target)) != 2:
            raise ValueError(f"strict matched role lacks two classes: {role}")
        selections[role], labels[role], receipts[role] = selected, target, receipt
        sample_ids[role] = np.asarray(roles[role]["sample_ids"])[selected].astype(np.int64)
        final_normalized = _normalize_features(final_features[role][selected], final_payload)
        stft_normalized = _normalize_features(stft_features[role][selected], stft_payload)
        final_raw_scores[role] = _scores(final_probe, final_normalized, device)
        stft_raw_scores[role] = _scores(stft_probe, stft_normalized, device)

    score_stats = {
        "final_temporal": _fit_score_stats(final_raw_scores["fit"]),
        "vft_stft": _fit_score_stats(stft_raw_scores["fit"]),
    }
    final_scores = {role: _standardize_scores(final_raw_scores[role], score_stats["final_temporal"])
                    for role in ROLES}
    stft_scores = {role: _standardize_scores(stft_raw_scores[role], score_stats["vft_stft"])
                   for role in ROLES}

    metrics, permutation_receipts = {}, {}
    prediction_payload = {}
    for role in ROLES:
        permutation_seed = 20284000 + int(seed) * 100 + ROLE_OFFSETS[role]
        permutation, permutation_sha = _fixed_permutation(len(labels[role]), permutation_seed)
        aligned = _fused_scores(final_scores[role], stft_scores[role])
        shuffled = _fused_scores(final_scores[role], stft_scores[role][permutation])
        metrics[role] = {
            "final_temporal": _rank_metrics(labels[role], final_scores[role]),
            "vft_stft": _rank_metrics(labels[role], stft_scores[role]),
            "aligned_fusion": _rank_metrics(labels[role], aligned),
            "shuffled_fusion": _rank_metrics(labels[role], shuffled),
        }
        permutation_receipts[role] = {
            "seed": int(permutation_seed), "sha256": permutation_sha, "count": int(len(permutation))
        }
        prediction_payload.update({
            f"{role}_sample_ids": sample_ids[role],
            f"{role}_labels": labels[role].astype(np.uint8),
            f"{role}_final_temporal": final_scores[role],
            f"{role}_vft_stft": stft_scores[role],
            f"{role}_aligned_fusion": aligned,
            f"{role}_shuffled_fusion": shuffled,
            f"{role}_stft_permutation": permutation,
        })
    np.savez_compressed(seed_root / "predictions_scores.npz", **prediction_payload)

    early = metrics["early"]
    deltas = {
        "aligned_minus_final_temporal_auroc": early["aligned_fusion"]["auroc"] - early["final_temporal"]["auroc"],
        "aligned_minus_vft_stft_auroc": early["aligned_fusion"]["auroc"] - early["vft_stft"]["auroc"],
        "aligned_minus_best_single_auroc": early["aligned_fusion"]["auroc"] - max(
            early["final_temporal"]["auroc"], early["vft_stft"]["auroc"]
        ),
        "aligned_minus_shuffled_auroc": early["aligned_fusion"]["auroc"] - early["shuffled_fusion"]["auroc"],
        "aligned_validation_early_auroc_gap": (
            metrics["validation"]["aligned_fusion"]["auroc"] - early["aligned_fusion"]["auroc"]
        ),
    }
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
        "c6_checkpoint_sha256": _sha256(c6_checkpoint),
        "final_temporal_probe_sha256": _sha256(final_checkpoint),
        "vft_stft_probe_sha256": _sha256(stft_checkpoint),
        "feature_dimensions": {"final_temporal": int(final_features["fit"].shape[1]),
                               "vft_stft": int(stft_features["fit"].shape[1])},
        "fusion": {"learned_parameters": 0, "weights": [0.5, 0.5],
                   "score_standardization_role": "strict_matched_fit_only"},
        "score_stats": score_stats, "matched_counts": receipts,
        "permutations": permutation_receipts, "metrics": metrics, "deltas": deltas,
        "predictions_sha256": _sha256(seed_root / "predictions_scores.npz"),
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
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")

    labels6 = read_labels(data_path)
    values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "experiment": "RouteA_VFT_StrictComplementaryScoreFusion_v3",
        "seeds": list(seeds), "smoke": bool(args.smoke), "weak_threshold_db": args.weak_threshold_db,
        "target": TARGET_NAME, "fusion_weights": [0.5, 0.5], "learned_fusion_parameters": 0,
        "gates": {
            "absolute_early_auroc": 0.620, "gain_vs_final_temporal": 0.030,
            "gain_vs_vft_stft": 0.010, "gain_vs_shuffled": 0.015,
            "positive_seeds": 4, "maximum_validation_early_gap": 0.150,
        },
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "experiment_config.json", config_payload)
    rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]

    def values(path):
        first, second = path
        return [row[first][second] for row in rows]

    aggregate = {
        "early_auroc": {
            name: _mean_ci([row["metrics"]["early"][name]["auroc"] for row in rows], 20285000 + offset)
            for offset, name in enumerate(("final_temporal", "vft_stft", "aligned_fusion", "shuffled_fusion"))
        },
        "early_aupr": {
            name: _mean_ci([row["metrics"]["early"][name]["aupr"] for row in rows], 20285100 + offset)
            for offset, name in enumerate(("final_temporal", "vft_stft", "aligned_fusion", "shuffled_fusion"))
        },
        "deltas": {
            name: _mean_ci(values(("deltas", name)), 20285200 + offset)
            for offset, name in enumerate((
                "aligned_minus_final_temporal_auroc", "aligned_minus_vft_stft_auroc",
                "aligned_minus_best_single_auroc", "aligned_minus_shuffled_auroc",
                "aligned_validation_early_auroc_gap",
            ))
        },
    }
    positive_best = int(sum(row["deltas"]["aligned_minus_best_single_auroc"] > 0 for row in rows))
    positive_shuffled = int(sum(row["deltas"]["aligned_minus_shuffled_auroc"] > 0 for row in rows))
    if args.smoke:
        gate = {"checks": {}, "passed": False, "evidence_level": "smoke_only"}
    else:
        checks = {
            "aligned_early_auroc_ge_0p620": aggregate["early_auroc"]["aligned_fusion"]["mean"] >= 0.620,
            "gain_vs_final_ge_0p030": aggregate["deltas"]["aligned_minus_final_temporal_auroc"]["mean"] >= 0.030,
            "gain_vs_stft_ge_0p010": aggregate["deltas"]["aligned_minus_vft_stft_auroc"]["mean"] >= 0.010,
            "gain_vs_shuffled_ge_0p015": aggregate["deltas"]["aligned_minus_shuffled_auroc"]["mean"] >= 0.015,
            "four_of_five_better_than_best_single": positive_best >= 4,
            "four_of_five_better_than_shuffled": positive_shuffled >= 4,
            "validation_early_gap_lt_0p150": aggregate["deltas"]["aligned_validation_early_auroc_gap"]["mean"] < 0.150,
        }
        gate = {"checks": checks, "passed": bool(all(checks.values())),
                "positive_seeds_vs_best_single": positive_best,
                "positive_seeds_vs_shuffled": positive_shuffled,
                "evidence_level": "five_seed_development_v3"}
    payload = {
        "status": "completed", "experiment": "RouteA_VFT_StrictComplementaryScoreFusion_v3",
        "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
        "aggregate": aggregate, "promotion_gate": gate,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "route_a_vft_strict_complementary_score_fusion_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--layerwise-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_layerwise_matched_negative_separability_v3_strict")
    parser.add_argument("--physics-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_iq_physics_separability_v3_strict")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_vft_strict_complementary_score_fusion_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--stft-batch-size", type=int, default=32)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

