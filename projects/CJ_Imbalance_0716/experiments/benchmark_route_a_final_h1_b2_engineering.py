"""Engineering benchmark for the frozen Route-A H1+B2 final candidate."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
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
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _head_scores
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _clipped_logit, _load_expert
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config
from experiments.run_shared import _data_path, load_config


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values, sample_count):
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "repeats": int(len(array)),
        "raw_ms": array.tolist(),
        "median_ms": median,
        "p95_ms": float(np.quantile(array, 0.95)),
        "median_ms_per_sample": median / float(sample_count),
        "median_samples_per_second": float(sample_count) * 1000.0 / median,
    }


def _state_count(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    return int(sum(value.numel() for value in state.values()))


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    config = load_config(args.config)
    _validate_config(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    roles, role_path = _load_roles(args.c6_root, args.seed, _data_path(config), config["normalization"], args.io_batch_size)
    signals = roles["early"]["signals"][: args.sample_count]
    expert, expert_mean, expert_std, expert_hashes = _load_expert(args.periodicity_root, args.seed, "H_AlignedHGB")
    fusion_root = args.increment_root / f"seed_{args.seed}" / "head_aligned"
    fusion = joblib.load(fusion_root / "model.joblib")
    fusion_norm = np.load(fusion_root / "normalization.npz")
    source_logits = np.load(args.increment_root / f"seed_{args.seed}" / "all_role_logits.npz")["B0_H1__early"][: args.sample_count]

    for _ in range(args.warmup):
        feature = _extract(signals, device, args.batch_size)
        normalized = ((feature - expert_mean) / np.maximum(expert_std, 1e-5)).astype(np.float32)
        expert_score = _clipped_logit(expert.predict_proba(normalized)[:, 1])
        fusion_feature = _fusion_features(source_logits, expert_score)
        _head_scores(fusion, fusion_feature, fusion_norm["mean"], fusion_norm["std"])
        _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    feature_times, hgb_times, fusion_times, total_times = [], [], [], []
    for _ in range(args.repeats):
        _sync(device); start = time.perf_counter()
        feature = _extract(signals, device, args.batch_size)
        _sync(device); after_feature = time.perf_counter()
        normalized = ((feature - expert_mean) / np.maximum(expert_std, 1e-5)).astype(np.float32)
        expert_score = _clipped_logit(expert.predict_proba(normalized)[:, 1])
        after_hgb = time.perf_counter()
        fusion_feature = _fusion_features(source_logits, expert_score)
        _head_scores(fusion, fusion_feature, fusion_norm["mean"], fusion_norm["std"])
        after_fusion = time.perf_counter()
        feature_times.append((after_feature - start) * 1000.0)
        hgb_times.append((after_hgb - after_feature) * 1000.0)
        fusion_times.append((after_fusion - after_hgb) * 1000.0)
        total_times.append((after_fusion - start) * 1000.0)

    focal = args.convergence_root / f"seed_{args.seed}" / "E2_Focal_36" / "best_model.pth"
    h1 = args.head_root / f"seed_{args.seed}" / "H1_Linear_BCE" / "best_model.pth"
    raw = args.unified_root / f"seed_{args.seed}" / "checkpoints" / "rawsaw.pth"
    files = {
        "focal36_backbone": focal,
        "h1_linear_head": h1,
        "rawsaw": raw,
        "periodicity_hgb": args.periodicity_root / f"seed_{args.seed}" / "H_AlignedHGB" / "model.joblib",
        "periodicity_normalization": args.periodicity_root / f"seed_{args.seed}" / "H_AlignedHGB" / "normalization.npz",
        "b2_fusion_head": fusion_root / "model.joblib",
        "b2_fusion_normalization": fusion_root / "normalization.npz",
    }
    artifacts = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for name, path in files.items()}
    result = {
        "status": "completed",
        "experiment": "RouteA_Final_H1B2_EngineeringBenchmark",
        "seed": args.seed,
        "sample_count": len(signals),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "latency": {
            "periodicity_feature": _summary(feature_times, len(signals)),
            "hgb_predict": _summary(hgb_times, len(signals)),
            "fusion_head": _summary(fusion_times, len(signals)),
            "b2_increment_total": _summary(total_times, len(signals)),
        },
        "memory": {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        },
        "parameter_counts": {
            "focal36_backbone_state": _state_count(focal),
            "h1_linear_head_state": _state_count(h1),
            "rawsaw_state": _state_count(raw),
            "b2_fusion_coefficients_plus_intercept": int(fusion.coef_.size + fusion.intercept_.size),
        },
        "artifacts": artifacts,
        "expert_hashes": expert_hashes,
        "role_indices_sha256": _sha256(role_path),
        "locked_test_used": False,
        "test_v3_used": False,
        "scope_note": "single-GPU incremental B2 latency; excludes H1 base latency and data loading",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "route_a_final_h1_b2_engineering_benchmark.json"
    _write_json(out, result)
    print(json.dumps({"status": "completed", "latency": result["latency"], "parameters": result["parameter_counts"], "summary": str(out)}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--convergence-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_loss_convergence_audit"); p.add_argument("--head-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines"); p.add_argument("--unified-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--periodicity-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); p.add_argument("--increment-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_h1_periodicity_increment"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_final_h1_b2_engineering_benchmark"); p.add_argument("--seed",type=int,default=2039); p.add_argument("--sample-count",type=int,default=512); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--io-batch-size",type=int,default=256); p.add_argument("--warmup",type=int,default=2); p.add_argument("--repeats",type=int,default=10); p.add_argument("--device",default=None); run(p.parse_args())
