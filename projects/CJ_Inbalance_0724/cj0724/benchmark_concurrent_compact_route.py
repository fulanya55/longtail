"""P4i benchmark for serial versus two-stream compact task routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.benchmark_multitask_efficiency import count_macs
from cj0724.concurrent_task_route import ConcurrentTaskRouteRunner
from cj0724.multitask import build_multitask_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed_calls(function, sample: torch.Tensor, warmup: int, repeats: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function(sample)
        torch.cuda.synchronize(sample.device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function(sample)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    return values


def peak_memory(function, sample: torch.Tensor) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(sample.device)
    with torch.inference_mode():
        function(sample)
    torch.cuda.synchronize(sample.device)
    return int(torch.cuda.max_memory_allocated(sample.device))


def summarize(values: list[float], block_medians: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    blocks = np.asarray(block_medians, dtype=np.float64)
    return {
        "pooled_median_ms": float(np.median(array)),
        "pooled_p95_ms": float(np.quantile(array, 0.95)),
        "block_median_mean_ms": float(blocks.mean()),
        "block_median_std_ms": float(blocks.std(ddof=1)) if len(blocks) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P4i requires CUDA")

    preregister = Path(args.preregister)
    protocol = json.loads(preregister.read_text(encoding="utf-8"))
    benchmark = protocol["benchmark_protocol"]
    device = torch.device("cuda")
    torch.manual_seed(int(benchmark["model_initialization_seed"]))
    global_model = build_multitask_model("legacy_mlcnn_global").to(device).eval()
    torch.manual_seed(int(benchmark["model_initialization_seed"]))
    compact_model = build_multitask_model("mlcnn_iq_grouped_task_routed_w4").to(device).eval()
    concurrent = ConcurrentTaskRouteRunner(compact_model)
    functions = {
        "legacy_mlcnn_global_serial": global_model,
        "compact_route_serial": compact_model,
        "compact_route_two_stream": concurrent,
    }

    rng = np.random.default_rng(int(benchmark["block_order_seed"]))
    rows = []
    equivalence = []
    for batch_size in benchmark["batch_sizes"]:
        generator = torch.Generator(device=device).manual_seed(8100 + int(batch_size))
        sample = torch.randn(int(batch_size), 2, int(benchmark["input_length"]), generator=generator, device=device)
        with torch.inference_mode():
            serial_logits, serial_parameters = compact_model(sample)
            concurrent_logits, concurrent_parameters = concurrent(sample)
        torch.cuda.synchronize(device)
        equivalence.append({
            "batch_size": int(batch_size),
            "logits_max_abs_difference": float((serial_logits - concurrent_logits).abs().max().item()),
            "parameters_max_abs_difference": float(
                (serial_parameters - concurrent_parameters).abs().max().item()
            ),
        })

        pooled = {name: [] for name in functions}
        block_medians = {name: [] for name in functions}
        for block in range(int(benchmark["measurement_blocks"])):
            order = list(rng.permutation(list(functions)))
            for name in order:
                values = timed_calls(
                    functions[name], sample, int(benchmark["warmup_per_block"]),
                    int(benchmark["repeats_per_block"]),
                )
                pooled[name].extend(values)
                block_medians[name].append(float(np.median(values)))

        for name, function in functions.items():
            row = {
                "mode": name,
                "batch_size": int(batch_size),
                **summarize(pooled[name], block_medians[name]),
                "peak_cuda_memory_bytes": peak_memory(function, sample),
            }
            row["throughput_samples_per_second"] = (
                1000.0 * int(batch_size) / row["pooled_median_ms"]
            )
            rows.append(row)

    by_key = {(row["mode"], row["batch_size"]): row for row in rows}
    global_b1 = by_key[("legacy_mlcnn_global_serial", 1)]
    serial_b1 = by_key[("compact_route_serial", 1)]
    concurrent_b1 = by_key[("compact_route_two_stream", 1)]
    global_b64 = by_key[("legacy_mlcnn_global_serial", 64)]
    serial_b64 = by_key[("compact_route_serial", 64)]
    concurrent_b64 = by_key[("compact_route_two_stream", 64)]
    comparisons = {
        "batch1_concurrent_median_increase_vs_global": (
            concurrent_b1["pooled_median_ms"] / global_b1["pooled_median_ms"] - 1.0
        ),
        "batch1_concurrent_p95_increase_vs_global": (
            concurrent_b1["pooled_p95_ms"] / global_b1["pooled_p95_ms"] - 1.0
        ),
        "batch1_concurrent_latency_reduction_vs_compact_serial": (
            1.0 - concurrent_b1["pooled_median_ms"] / serial_b1["pooled_median_ms"]
        ),
        "batch64_concurrent_throughput_retention_vs_global": (
            concurrent_b64["throughput_samples_per_second"]
            / global_b64["throughput_samples_per_second"]
        ),
        "batch64_concurrent_peak_memory_increase_vs_compact_serial": (
            concurrent_b64["peak_cuda_memory_bytes"] / serial_b64["peak_cuda_memory_bytes"] - 1.0
        ),
    }
    validity = protocol["validity_gates"]
    promotion = protocol["promotion_gates"]
    gate_results = {
        "logits_equivalent": all(
            row["logits_max_abs_difference"]
            <= validity["sequential_and_concurrent_logits_max_abs_difference_at_most"]
            for row in equivalence
        ),
        "parameters_equivalent": all(
            row["parameters_max_abs_difference"]
            <= validity["sequential_and_concurrent_parameters_max_abs_difference_at_most"]
            for row in equivalence
        ),
        "batch1_median_latency": comparisons["batch1_concurrent_median_increase_vs_global"]
        <= promotion["batch1_concurrent_median_latency_increase_vs_global_at_most"],
        "batch1_p95_latency": comparisons["batch1_concurrent_p95_increase_vs_global"]
        <= promotion["batch1_concurrent_p95_latency_increase_vs_global_at_most"],
        "batch1_reduction_vs_serial": comparisons[
            "batch1_concurrent_latency_reduction_vs_compact_serial"
        ] >= promotion["batch1_concurrent_latency_reduction_vs_compact_serial_at_least"],
        "batch64_throughput": comparisons["batch64_concurrent_throughput_retention_vs_global"]
        >= promotion["batch64_concurrent_throughput_retention_vs_global_at_least"],
        "batch64_peak_memory": comparisons[
            "batch64_concurrent_peak_memory_increase_vs_compact_serial"
        ] <= promotion["batch64_concurrent_peak_memory_increase_vs_compact_serial_at_most"],
    }
    result = {
        "experiment": protocol["experiment"],
        "preregister_sha256": sha256_file(preregister),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "float32",
            "amp": False,
        },
        "architecture": {
            "global_parameters": sum(parameter.numel() for parameter in global_model.parameters()),
            "compact_parameters": sum(parameter.numel() for parameter in compact_model.parameters()),
            "global_macs": count_macs(global_model, torch.randn(1, 2, 3000, device=device)),
            "compact_macs": count_macs(compact_model, torch.randn(1, 2, 3000, device=device)),
        },
        "equivalence": equivalence,
        "rows": rows,
        "comparisons": comparisons,
        "gate_results": gate_results,
        "promotion_passed": bool(all(gate_results.values())),
        "data_generated": False,
        "training_started": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
