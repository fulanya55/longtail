"""Engineering benchmark for the frozen H2 multi-pulse C0--C4 checkpoints.

The benchmark deliberately uses a deterministic synthetic I/Q tensor.  It
therefore neither reads any development/test sample nor changes a checkpoint,
calibrator, or threshold.  Run it only after the formal H2 queue has finished
and the artifact audit has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.multipulse_vft_expert import MODES, MultiPulseVFTExpertClassifier
from cjlt.models.vft_staggered_pri import record_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_summary(milliseconds: list[float], batch_size: int) -> dict[str, float | int | list[float]]:
    values = np.asarray(milliseconds, dtype=np.float64)
    median = float(np.median(values))
    return {
        "repeats": int(values.size),
        "raw_ms": values.tolist(),
        "median_ms": median,
        "p95_ms": float(np.quantile(values, 0.95)),
        "median_ms_per_sample": median / float(batch_size),
        "median_samples_per_second": float(batch_size) * 1000.0 / median,
    }


def _checkpoint(root: Path, mode: str, seed: int) -> Path:
    arm = mode.lower().replace("-mp", "")
    return root / f"route_a_h2_multipulse_{arm}_seed{seed}" / f"{mode}_seed{seed}" / "checkpoint.pt"


def _profile_supported_flops(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> int | None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            model(sample[:1])
            _sync(device)
        return int(sum(int(event.flops or 0) for event in profiler.key_averages()))
    except (RuntimeError, NotImplementedError):
        return None


def _load(mode: str, checkpoint: Path, width: int, expert_dim: int, device: torch.device) -> MultiPulseVFTExpertClassifier:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("mode") != mode:
        raise ValueError(f"checkpoint mode mismatch: {checkpoint}")
    model = MultiPulseVFTExpertClassifier(mode, width=width, expert_dim=expert_dim)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rng = torch.Generator(device="cpu").manual_seed(args.input_seed)
    sample_cpu = torch.randn(args.batch_size, 2, record_samples(), generator=rng, dtype=torch.float32)
    sample = sample_cpu.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    arms: dict[str, object] = {}
    for mode in args.modes:
        checkpoint = _checkpoint(args.root, mode, args.seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = _load(mode, checkpoint, args.width, args.expert_dim, device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(sample)
                _sync(device)
            values: list[float] = []
            for _ in range(args.repeats):
                _sync(device)
                start = time.perf_counter()
                model(sample)
                _sync(device)
                values.append((time.perf_counter() - start) * 1000.0)
        arms[mode] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_bytes": int(checkpoint.stat().st_size),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "profiler_supported_flops_per_sample": _profile_supported_flops(model, sample, device),
            "latency": _latency_summary(values, args.batch_size),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result: dict[str, object] = {
        "status": "completed_engineering_benchmark",
        "experiment": "RouteA_H2_MultiPulseSixLabel",
        "seed": args.seed,
        "modes": list(args.modes),
        "input": {"kind": "deterministic_synthetic_iq", "seed": args.input_seed, "shape": list(sample_cpu.shape)},
        "protocol": {"warmup": args.warmup, "repeats": args.repeats, "batch_size": args.batch_size},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "arms": arms,
        "memory": {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
        "flops_scope_note": "PyTorch profiler reports only supported neural operators; preprocessing, data loading, calibration and threshold comparison are excluded.",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "tapc")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expert-dim", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--input-seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tapc" / "route_a_h2_multipulse_engineering_benchmark.json")
    value = run(parser.parse_args())
    print(json.dumps({"status": value["status"], "output": str(value["arms"])}, ensure_ascii=False))
