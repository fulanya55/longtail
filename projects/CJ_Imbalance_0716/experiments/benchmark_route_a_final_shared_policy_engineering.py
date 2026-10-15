"""CPU/GPU benchmark for the complete frozen shared-policy inference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import SCORE_NAMES, extract_scores  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import FrozenFeatureHead, _extract as _extract_backbone, _predict_head  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_model_from_state, _load_rawsaw, _route_rawsaw  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _load_json, _load_vft_head  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _fusion_features as _vft_features, _head_scores, _replace_vft, _ridge_scores  # noqa: E402
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features as _isrj_features, _replace_isrj  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _fit_score_stats, _load_expert, _standardize_scores  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view  # noqa: E402
from experiments.run_route_a_mft_shared_decision_policy_audit import hybrid_policy  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import expert_logit, guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _load_roles, _raw_scores, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


PEAK_INDEX = SCORE_NAMES.index("nms_count_gt_bg_q99")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency(values, count):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return {
        "raw_ms": values.tolist(), "median_ms": median, "p95_ms": float(np.quantile(values, 0.95)),
        "median_ms_per_sample": median / count, "median_samples_per_second": count * 1000.0 / median,
    }


def _tree_nodes(model):
    predictors = getattr(model, "_predictors", [])
    return int(sum(tree.nodes.shape[0] for stage in predictors for tree in stage))


def _linear_size(model):
    return int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size)


def _profile_flops(backbone, h1, raw, signal, mean, std, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    x = torch.from_numpy(np.asarray(signal[:1], dtype=np.float32)).to(device)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
        feature = backbone.forward_features(x).float()
        h1((feature - mean_t) / std_t)
        raw(x[:, 0:1], x[:, 1:2])
    return int(sum(int(getattr(item, "flops", 0) or 0) for item in profile.key_averages()))


def run(args):
    seal = _load_json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = _load_json(args.frozen_manifest)
    if manifest["candidate_name"] != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen candidate")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(args.config)
    _validate_config(config)
    roles, role_path = _load_roles(args.c6_root, args.seed, _data_path(config), config["normalization"], args.io_batch_size)
    labels = read_labels(_data_path(config))
    parameters, _ = read_parameters(_data_path(config))
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = np.asarray(roles["early"]["signals"][: max(args.batch_sizes)], dtype=np.float32)
    artifacts = manifest["seed_artifacts"][str(args.seed)]
    state = torch.load(Path(artifacts["focal_checkpoint"]["path"]), map_location="cpu", weights_only=False)["model_state_dict"]
    backbone = _load_model_from_state(args.f2_root / "runs/F2_DFMS_Full" / f"seed_{args.seed}", device, state)
    fit_feature = _extract_backbone(backbone, roles["fit"]["signals"], device, args.io_batch_size)
    mean = fit_feature.mean(0, keepdims=True).astype(np.float32)
    std = np.maximum(fit_feature.std(0, keepdims=True), 1e-5).astype(np.float32)
    head_payload = torch.load(Path(artifacts["h1_head"]["path"]), map_location="cpu", weights_only=False)
    h1 = FrozenFeatureHead(128, "linear").to(device)
    h1.load_state_dict(head_payload["model_state_dict"]); h1.eval()
    raw, _ = _load_rawsaw(args, args.seed, device); raw.eval()
    periodic_model, periodic_mean, periodic_std, _ = _load_expert(args.periodicity_root, args.seed, "H_AlignedHGB")
    fit_periodic = _extract_periodicity(roles["fit"]["signals"], device, args.io_batch_size)
    fit_periodic_score = _clipped_logit(periodic_model.predict_proba(((fit_periodic - periodic_mean) / np.maximum(periodic_std, 1e-5)).astype(np.float32))[:, 1])
    selected, _, _ = _matched_view(roles["fit"]["targets6"], weak[roles["fit"]["sample_ids"]], ISRJ_INDEX)
    periodic_stats = _fit_score_stats(fit_periodic_score[selected])
    isrj_model = joblib.load(Path(artifacts["isrj_fusion_model"]["path"]))
    isrj_norm = np.load(Path(artifacts["isrj_fusion_normalization"]["path"]))
    mft_model = joblib.load(Path(artifacts["mft_guard_model"]["path"]))
    mft_norm = np.load(Path(artifacts["mft_guard_normalization"]["path"]))
    ridge = np.load(Path(artifacts["vft_ridge"]["path"]))
    vft_model, vft_mean, vft_std = _load_vft_head(args.construction_root / f"seed_{args.seed}", "aligned")
    vft_stats = _load_json(args.construction_root / f"seed_{args.seed}" / "seed_summary.json")["vft_score_stats"]
    q0_cal = _load_json(args.construction_root / f"seed_{args.seed}" / "Q0_H1/calibration.json")
    q0_thr = _load_json(args.construction_root / f"seed_{args.seed}" / "Q0_H1/thresholds.json")["thresholds"]
    q1_cal = _load_json(args.construction_root / f"seed_{args.seed}" / "Q1_H1B2/calibration.json")
    q1_thr = _load_json(args.construction_root / f"seed_{args.seed}" / "Q1_H1B2/thresholds.json")["thresholds"]
    q4_cal = _load_json(args.construction_root / f"seed_{args.seed}" / "Q4_VFTAlignedRidgeFusion/calibration.json")
    q4_thr = _load_json(args.construction_root / f"seed_{args.seed}" / "Q4_VFTAlignedRidgeFusion/thresholds.json")["thresholds"]

    def base_pipeline(batch):
        feature = _extract_backbone(backbone, batch, device, len(batch))
        logits = _predict_head(h1, ((feature - mean) / std).astype(np.float32), device, len(batch))
        q0 = _route_rawsaw({"x": logits}, {"x": _raw_scores(raw, batch, device, len(batch))})["x"]
        from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
        probability = _apply_frozen_calibration(q0, q0_cal)
        return probability >= np.asarray(q0_thr, dtype=np.float32)[None, :]

    def full_pipeline(batch):
        feature = _extract_backbone(backbone, batch, device, len(batch))
        logits = _predict_head(h1, ((feature - mean) / std).astype(np.float32), device, len(batch))
        q0 = _route_rawsaw({"x": logits}, {"x": _raw_scores(raw, batch, device, len(batch))})["x"]
        periodic = _extract_periodicity(batch, device, len(batch))
        periodic_raw = _clipped_logit(periodic_model.predict_proba(((periodic - periodic_mean) / np.maximum(periodic_std, 1e-5)).astype(np.float32))[:, 1])
        periodic_z = _standardize_scores(periodic_raw, periodic_stats)
        q1 = _replace_isrj(q0, _head_scores(isrj_model, _isrj_features(q0, periodic_z), isrj_norm["mean"], isrj_norm["std"]))
        physical = extract_scores(batch, device, len(batch))
        evidence = expert_logit(mft_model, physical["MF_AlignedLFM"][:, PEAK_INDEX], mft_norm["aligned_mean"], mft_norm["aligned_std"])
        q2, _ = guarded_logits(q1, evidence, manifest["fixed_hyperparameters"]["mft_residual_cap"])
        vft_raw = _ridge_scores(_extract_vft(batch, device, len(batch)), ridge)
        vft_z = _standardize_scores(vft_raw, vft_stats)
        q4 = _replace_vft(q2, _head_scores(vft_model, _vft_features(q2, vft_z), vft_mean, vft_std))
        _, prediction, _ = hybrid_policy(q4, q4_cal, q4_thr, q1_cal, q1_thr)
        return prediction

    latency = {"baseline": {}, "complete": {}}
    for count in args.batch_sizes:
        batch = signals[:count]
        for _ in range(args.warmup):
            base_pipeline(batch); full_pipeline(batch); _sync(device)
        for name, function in (("baseline", base_pipeline), ("complete", full_pipeline)):
            times = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(args.repeats):
                _sync(device); started = time.perf_counter(); function(batch); _sync(device)
                times.append((time.perf_counter() - started) * 1000.0)
            latency[name][f"batch_{count}"] = _latency(times, count)
    neural_flops = _profile_flops(backbone, h1, raw, signals, mean, std, device)
    torch_parameters = {
        "backbone": int(sum(p.numel() for p in backbone.parameters())),
        "h1_head": int(sum(p.numel() for p in h1.parameters())),
        "rawsaw": int(sum(p.numel() for p in raw.parameters())),
    }
    ridge_scalars = int(sum(np.asarray(ridge[key]).size for key in ridge.files))
    increments = {
        "isrj_fusion": _linear_size(isrj_model), "mft_guard": _linear_size(mft_model),
        "vft_ridge_arrays": ridge_scalars, "vft_fusion": _linear_size(vft_model),
    }
    unique_artifacts = {record["path"]: record for record in artifacts.values() if isinstance(record, dict) and "path" in record}
    result = {
        "status": "completed", "experiment": "RouteA_FinalSharedPolicy_EngineeringBenchmark",
        "device": str(device), "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "seed": args.seed, "batch_sizes": args.batch_sizes, "warmup": args.warmup, "repeats": args.repeats,
        "latency": latency,
        "memory": {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
        "parameters": {"torch": torch_parameters, "neural_baseline_total": sum(torch_parameters.values()), "non_tree_increment": increments, "non_tree_increment_total": sum(increments.values()), "periodicity_hgb_tree_nodes": _tree_nodes(periodic_model)},
        "flops": {"profiler_supported_backbone_h1_rawsaw_per_sample": neural_flops, "scope_note": "Profiler-supported neural operators only; FFT, matched filtering, tree traversal, calibration, and threshold comparisons are timed but do not have a portable FLOP count."},
        "artifact_bytes": {"unique_total": int(sum(item["size_bytes"] for item in unique_artifacts.values())), "unique_file_count": len(unique_artifacts)},
        "role_indices_sha256": _sha256(role_path), "frozen_manifest_sha256": _sha256(args.frozen_manifest),
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"benchmark_{device.type}.json"
    _write_json(output, result)
    print(json.dumps({"device": str(device), "latency": latency, "memory": result["memory"], "parameters": result["parameters"], "flops": result["flops"], "artifact_bytes": result["artifact_bytes"], "output": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_engineering")
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

