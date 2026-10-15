"""Engineering benchmark for frozen U0 and U2 unified candidates."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _predict
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_rawsaw
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _head_scores, _replace_isrj
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _clipped_logit, _load_expert, _standardize_scores
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import SAW_INDEX, _raw_scores, _validate_config
from experiments.run_shared import _data_path, load_config


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values, sample_count):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return {
        "repeats": int(len(values)), "raw_ms": values.tolist(), "median_ms": median,
        "p95_ms": float(np.quantile(values, 0.95)),
        "median_ms_per_sample": median / sample_count,
        "median_samples_per_second": sample_count * 1000.0 / median,
    }


def _state_count(path):
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    return int(sum(value.numel() for value in state.values()))


def _profile_flops(call, device):
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            call()
            _sync(device)
        return int(sum(event.flops for event in profiler.key_averages() if event.flops is not None))
    except Exception as error:
        return {"value": None, "error": repr(error)}


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    config = load_config(args.config)
    _validate_config(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    roles, role_path = _load_roles(args.c6_root, args.seed, _data_path(config), config["normalization"], args.io_batch_size)
    signals = roles["early"]["signals"][:args.sample_count]
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{args.seed}"
    checkpoint_paths = {
        "c6": args.c6_root / f"seed_{args.seed}" / "C6_Joint" / "best_model.pth",
        "fullalpha": args.c6_root / f"seed_{args.seed}" / "GradientContrastFixed1p5" / "best_model.pth",
    }
    models = {}
    for name, path in checkpoint_paths.items():
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        models[name] = _load_model_from_state(f2_dir, device, state).eval()
    raw, raw_path = _load_rawsaw(args, args.seed, device)
    expert, expert_mean, expert_std, expert_hashes = _load_expert(args.periodicity_root, args.seed, "H_AlignedHGB")
    fusion_root = args.source_root / f"seed_{args.seed}" / "head_aligned"
    fusion = joblib.load(fusion_root / "model.joblib")
    fusion_norm = np.load(fusion_root / "normalization.npz")
    old = json.loads((args.score_stats_root / f"seed_{args.seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    score_stats = old["score_stats"]["aligned"]
    calibration = {
        variant: json.loads((args.source_root / f"seed_{args.seed}" / variant / "calibration.json").read_text(encoding="utf-8"))
        for variant in (VARIANTS[0], VARIANTS[2])
    }
    thresholds = {
        variant: np.asarray(json.loads((args.source_root / f"seed_{args.seed}" / variant / "thresholds.json").read_text(encoding="utf-8"))["thresholds"], dtype=np.float32)
        for variant in (VARIANTS[0], VARIANTS[2])
    }

    def base(name):
        logits = _predict(models[name], signals, device, args.batch_size)
        saw = _raw_scores(raw, signals, device, args.batch_size)
        logits[:, SAW_INDEX] = saw
        return logits

    def periodic(base_logits):
        feature = _extract(signals, device, args.batch_size)
        normalized = ((feature - expert_mean) / np.maximum(expert_std, 1e-5)).astype(np.float32)
        raw_score = _clipped_logit(expert.predict_proba(normalized)[:, 1])
        z_score = _standardize_scores(raw_score, score_stats)
        score = _head_scores(fusion, _fusion_features(base_logits, z_score), fusion_norm["mean"], fusion_norm["std"])
        return _replace_isrj(base_logits, score)

    def postprocess(logits, variant):
        probability = _apply_frozen_calibration(logits, calibration[variant])
        return probability >= thresholds[variant][None, :]

    def u0():
        logits = base("c6")
        postprocess(logits, VARIANTS[0])
        return logits

    def u2():
        logits = periodic(base("fullalpha"))
        postprocess(logits, VARIANTS[2])
        return logits

    for _ in range(args.warmup):
        u0(); u2(); _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = {name: [] for name in ("c6_backbone", "fullalpha_backbone", "rawsaw", "periodicity_feature", "hgb", "fusion", "u0_end_to_end", "u2_end_to_end")}
    for _ in range(args.repeats):
        _sync(device); start = time.perf_counter(); _predict(models["c6"], signals, device, args.batch_size); _sync(device); timings["c6_backbone"].append((time.perf_counter()-start)*1000)
        _sync(device); start = time.perf_counter(); full_logits = _predict(models["fullalpha"], signals, device, args.batch_size); _sync(device); timings["fullalpha_backbone"].append((time.perf_counter()-start)*1000)
        _sync(device); start = time.perf_counter(); _raw_scores(raw, signals, device, args.batch_size); _sync(device); timings["rawsaw"].append((time.perf_counter()-start)*1000)
        _sync(device); start = time.perf_counter(); feature = _extract(signals, device, args.batch_size); _sync(device); after_feature = time.perf_counter(); timings["periodicity_feature"].append((after_feature-start)*1000)
        normalized = ((feature - expert_mean) / np.maximum(expert_std, 1e-5)).astype(np.float32)
        start = time.perf_counter(); raw_score = _clipped_logit(expert.predict_proba(normalized)[:, 1]); after_hgb = time.perf_counter(); timings["hgb"].append((after_hgb-start)*1000)
        z_score = _standardize_scores(raw_score, score_stats)
        start = time.perf_counter(); _head_scores(fusion, _fusion_features(full_logits, z_score), fusion_norm["mean"], fusion_norm["std"]); timings["fusion"].append((time.perf_counter()-start)*1000)
        _sync(device); start = time.perf_counter(); u0(); _sync(device); timings["u0_end_to_end"].append((time.perf_counter()-start)*1000)
        _sync(device); start = time.perf_counter(); u2(); _sync(device); timings["u2_end_to_end"].append((time.perf_counter()-start)*1000)

    sample_tensor = torch.from_numpy(signals[:1]).to(device)
    neural_flops = {
        "c6_backbone_profiler_supported": _profile_flops(lambda: models["c6"](sample_tensor), device),
        "fullalpha_backbone_profiler_supported": _profile_flops(lambda: models["fullalpha"](sample_tensor), device),
        "rawsaw_profiler_supported": _profile_flops(lambda: raw(sample_tensor[:, 0:1], sample_tensor[:, 1:2]), device),
    }
    files = {
        "c6_checkpoint": checkpoint_paths["c6"], "fullalpha_checkpoint": checkpoint_paths["fullalpha"],
        "rawsaw": raw_path,
        "periodicity_hgb": args.periodicity_root / f"seed_{args.seed}" / "H_AlignedHGB" / "model.joblib",
        "periodicity_normalization": args.periodicity_root / f"seed_{args.seed}" / "H_AlignedHGB" / "normalization.npz",
        "fusion_head": fusion_root / "model.joblib", "fusion_normalization": fusion_root / "normalization.npz",
    }
    artifacts = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for name, path in files.items()}
    result = {
        "status": "completed", "experiment": "RouteA_U2_EngineeringBenchmark",
        "seed": args.seed, "sample_count": len(signals), "batch_size": args.batch_size,
        "warmup": args.warmup, "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "latency": {name: _summary(values, len(signals)) for name, values in timings.items()},
        "u2_increment_over_u0_median_ms": float(np.median(timings["u2_end_to_end"]) - np.median(timings["u0_end_to_end"])),
        "memory": {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        },
        "parameter_counts": {
            "c6_backbone_state": _state_count(checkpoint_paths["c6"]),
            "fullalpha_backbone_state": _state_count(checkpoint_paths["fullalpha"]),
            "rawsaw_state": _state_count(raw_path),
            "fusion_coefficients_plus_intercept": int(fusion.coef_.size + fusion.intercept_.size),
        },
        "profiler_supported_flops_per_sample": neural_flops,
        "flops_scope_note": "PyTorch profiler counts supported neural operators only; HGB and NumPy FFT/ACF operations are timed but not represented as FLOPs.",
        "inference_structure_note": "FullAlpha is one merged checkpoint at inference; C6 is not required by deployed U2.",
        "artifacts": artifacts, "expert_hashes": expert_hashes,
        "role_indices_sha256": _sha256(role_path),
        "scope_note": "preloaded signals; includes frozen calibration and threshold postprocessing; excludes disk I/O and model loading",
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_u2_engineering_benchmark.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "latency": result["latency"], "parameters": result["parameter_counts"], "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--score-stats-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_engineering_benchmark")
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
