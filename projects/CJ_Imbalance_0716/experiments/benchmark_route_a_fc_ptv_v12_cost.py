"""Post-decision engineering cost benchmark for the frozen v12 FC-PTV artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    floating_parameters = sum(value.numel() for value in state.values() if torch.is_floating_point(value))
    state_bytes = sum(value.numel() * value.element_size() for value in state.values())

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state

    f2_dir = Path(args.f2_root) / "runs/F2_DFMS_Full/seed_2039"
    model = _load_model_from_state(f2_dir, device, state).eval()
    sample = torch.randn(int(args.batch_size), 2, 3000, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(int(args.warmup)):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = []
        for _ in range(int(args.repeats)):
            started = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    result = {
        "status": "completed",
        "evidence_role": "post_decision_engineering_cost_only",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "floating_parameter_count": int(floating_parameters),
        "state_tensor_bytes": int(state_bytes),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "latency_ms_per_batch_mean": 1000.0 * statistics.mean(elapsed),
        "latency_ms_per_batch_median": 1000.0 * statistics.median(elapsed),
        "latency_ms_per_sample_mean": 1000.0 * statistics.mean(elapsed) / int(args.batch_size),
        "throughput_samples_per_second": int(args.batch_size) / statistics.mean(elapsed),
        "inference_peak_allocated_bytes": peak,
        "training_peak_gpu_memory_recorded": False,
        "method_selection_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v12_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv/seed_2039/FC_PTV/best_model.pth")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv/cost_benchmark.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
