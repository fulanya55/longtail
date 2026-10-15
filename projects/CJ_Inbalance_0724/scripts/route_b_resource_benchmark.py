"""Measure Route-B model resources without data or test access."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from cj0724.multitask import build_multitask_model
from cj0724.models import build_benchmark_model


def flops_hook(counter):
    def hook(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            x = inputs[0]
            out = output
            counter[0] += int(2 * x.shape[0] * out.shape[1] * out.shape[2] * module.in_channels * module.kernel_size[0] / module.groups)
        elif isinstance(module, nn.Linear):
            x = inputs[0]
            counter[0] += int(2 * x.shape[0] * module.in_features * module.out_features)
    return hook


def benchmark(name: str, device: torch.device, repeats: int) -> dict[str, object]:
    if name == "B0":
        model = build_benchmark_model("legacy_mlcnn", num_classes=6, width=16, head_dim=256)
    else:
        model_name = {"B1": "legacy_mlcnn_global", "B2": "legacy_mlcnn_class_conditioned", "B3": "legacy_mlcnn_class_conditioned_detached", "B4": "legacy_mlcnn_class_conditioned"}[name]
        model = build_multitask_model(model_name, width=16, head_dim=256, parameter_hidden_dim=64)
    model = model.to(device).eval()
    x = torch.randn(1, 2, 3000, device=device)
    counter = [0]
    handles = [module.register_forward_hook(flops_hook(counter)) for module in model.modules() if isinstance(module, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        for _ in range(3):
            model(x)
    counter[0] = 0
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            started = time.perf_counter(); model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000.0)
    for handle in handles:
        handle.remove()
    memory = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    result = {
        "variant": name, "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()),
        "approx_flops_batch1": int(counter[0] / max(repeats, 1)), "latency_ms_median": statistics.median(times),
        "latency_ms_mean": statistics.mean(times), "throughput_samples_per_s": 1000.0 / statistics.median(times),
        "peak_gpu_memory_mib": memory, "repeats": repeats, "test_mat_opened": False,
        "evidence_status": "verified_resource_benchmark",
    }
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["B0", "B1", "B2", "B3", "B4"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("docs"))
    args = parser.parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    rows = [benchmark(variant, device, args.repeats) for device in devices for variant in args.variants]
    output_root = args.output_root.resolve()
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "receipts").mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["variant"]
    with (output_root / "results" / "route_b_resource_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    receipt = {"stage": "R3_resource", "status": "verified_resource_benchmark", "rows": rows, "test_mat_opened": False, "locked_test_used": False}
    (output_root / "receipts" / "route_b_resource_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
