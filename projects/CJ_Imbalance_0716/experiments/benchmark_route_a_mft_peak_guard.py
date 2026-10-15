"""Benchmark deployable MFT aligned peak-count guard incremental cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import morphology_scores  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_mft_matched_filter_delay_profile_v3 import build_lfm_reference, circular_correlation_profile  # noqa: E402


PEAK_INDEX = 5


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    data_path = Path(audit["development_v10_path"])
    signals = _load_signals(data_path, np.arange(args.sample_count), args.io_batch_size)
    base_prediction = np.load(args.prediction_root / f"seed_{args.seed}/K0_U2Frozen/predictions.npz")
    base_logits = torch.from_numpy(np.asarray(base_prediction["logits"][: args.sample_count], dtype=np.float32))
    expert_root = args.guard_root / f"seed_{args.seed}/experts"
    model = joblib.load(expert_root / "aligned_model.joblib")
    normalization = np.load(expert_root / "normalization.npz")
    coefficient = float(model.coef_.reshape(-1)[0])
    intercept = float(model.intercept_.reshape(-1)[0])
    mean = float(normalization["aligned_mean"].reshape(-1)[0])
    std = float(normalization["aligned_std"].reshape(-1)[0])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = build_lfm_reference(device)
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, args.sample_count, args.batch_size):
        signal = torch.from_numpy(signals[start : start + args.batch_size]).to(device)
        logits = base_logits[start : start + len(signal)].to(device)
        for repeat in range(args.warmup + args.repeats):
            _synchronize(device)
            began = time.perf_counter()
            profile = circular_correlation_profile(signal, reference)
            count = morphology_scores(profile)[:, PEAK_INDEX]
            evidence = coefficient * ((count - mean) / max(std, 1e-5)) + intercept
            probability = torch.sigmoid(logits[:, 3])
            uncertainty = torch.sqrt(torch.clamp(4.0 * probability * (1.0 - probability), min=0.0))
            guarded_mft = logits[:, 3] + 0.75 * uncertainty * torch.minimum(torch.tanh(evidence), torch.zeros_like(evidence))
            _ = guarded_mft.sum()
            _synchronize(device)
            elapsed = (time.perf_counter() - began) * 1000.0
            if repeat >= args.warmup:
                timings.append(elapsed)
    median_ms = float(statistics.median(timings))
    batch = min(args.batch_size, args.sample_count)
    result = {
        "status": "completed", "experiment": "RouteA_MFT_PeakGuard_EngineeringBenchmark",
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": batch, "sample_count": int(args.sample_count), "warmup": int(args.warmup), "repeats_per_batch": int(args.repeats),
        "incremental_median_ms_per_batch": median_ms,
        "incremental_samples_per_second": float(batch / (median_ms / 1000.0)),
        "prior_u2_gpu_median_ms_batch64": 104.968 if device.type == "cuda" and batch == 64 else None,
        "estimated_u2_plus_guard_ms_batch64": 104.968 + median_ms if device.type == "cuda" and batch == 64 else None,
        "estimated_increment_percent_vs_prior_u2": median_ms / 104.968 * 100.0 if device.type == "cuda" and batch == 64 else None,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "deployable_learned_parameters": 2,
        "feature_learned_parameters": 0,
        "artifacts": {
            "aligned_model_bytes": (expert_root / "aligned_model.joblib").stat().st_size,
            "normalization_bytes": (expert_root / "normalization.npz").stat().st_size,
            "aligned_model_sha256": _sha256(expert_root / "aligned_model.joblib"),
            "normalization_sha256": _sha256(expert_root / "normalization.npz"),
        },
        "fft_flops_note": "Torch profiler does not provide complete complex FFT FLOPs; latency and memory are the primary engineering measures.",
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v10_generation/identity_audit.json")
    parser.add_argument("--prediction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_development_v10_frozen")
    parser.add_argument("--guard-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_evidence_bundle/engineering_gpu.json")
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

