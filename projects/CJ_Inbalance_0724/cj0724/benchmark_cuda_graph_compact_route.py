"""P4j matched CUDA Graph benchmark for global and compact routed models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.benchmark_multitask_efficiency import count_macs
from cj0724.cuda_graph_runner import FixedShapeCUDAGraphRunner
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


def summarize(values: list[float], block_medians: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    blocks = np.asarray(block_medians, dtype=np.float64)
    return {
        "pooled_median_ms": float(np.median(array)),
        "pooled_p95_ms": float(np.quantile(array, 0.95)),
        "block_median_mean_ms": float(blocks.mean()),
        "block_median_std_ms": float(blocks.std(ddof=1)) if len(blocks) > 1 else 0.0,
    }


def memory_footprint(
    model_name: str, batch_size: int, seed: int, input_length: int, capture_warmup: int
) -> dict[str, int]:
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    before = int(torch.cuda.memory_allocated(device))
    torch.manual_seed(seed)
    model = build_multitask_model(model_name).to(device).eval()
    sample = torch.randn(batch_size, 2, input_length, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(sample)
    torch.cuda.synchronize(device)
    eager_peak = int(torch.cuda.max_memory_allocated(device) - before)
    runner = FixedShapeCUDAGraphRunner(model, sample, capture_warmup)
    torch.cuda.synchronize(device)
    graph_resident = int(torch.cuda.memory_allocated(device) - before)
    del runner, model, sample
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "eager_peak_bytes": eager_peak,
        "graph_resident_bytes": graph_resident,
        "graph_resident_increase_fraction": float(graph_resident / eager_peak - 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compact-model", default="mlcnn_iq_grouped_task_routed_w4")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P4j requires CUDA")

    preregister = Path(args.preregister)
    protocol = json.loads(preregister.read_text(encoding="utf-8"))
    benchmark = protocol["benchmark_protocol"]
    seed = int(benchmark["model_initialization_seed"])
    input_length = int(benchmark["input_length"])
    capture_warmup = int(benchmark["capture_warmup"])
    compact_model_name = str(args.compact_model)
    device = torch.device("cuda")

    memory = {}
    for batch_size in benchmark["batch_sizes"]:
        for model_name in ("legacy_mlcnn_global", compact_model_name):
            memory[f"{model_name}_batch{batch_size}"] = memory_footprint(
                model_name, int(batch_size), seed, input_length, capture_warmup
            )

    torch.manual_seed(seed)
    global_model = build_multitask_model("legacy_mlcnn_global").to(device).eval()
    torch.manual_seed(seed)
    compact_model = build_multitask_model(compact_model_name).to(device).eval()
    architecture = {
        "global_parameters": sum(parameter.numel() for parameter in global_model.parameters()),
        "compact_parameters": sum(parameter.numel() for parameter in compact_model.parameters()),
        "global_macs": count_macs(global_model, torch.randn(1, 2, input_length, device=device)),
        "compact_macs": count_macs(compact_model, torch.randn(1, 2, input_length, device=device)),
    }

    rng = np.random.default_rng(int(benchmark["block_order_seed"]))
    rows = []
    equivalence = []
    for batch_size in benchmark["batch_sizes"]:
        batch_size = int(batch_size)
        generator = torch.Generator(device=device).manual_seed(9100 + batch_size)
        sample = torch.randn(batch_size, 2, input_length, generator=generator, device=device)
        global_graph = FixedShapeCUDAGraphRunner(global_model, sample, capture_warmup)
        compact_graph = FixedShapeCUDAGraphRunner(compact_model, sample, capture_warmup)
        functions = {
            "legacy_mlcnn_global_eager": global_model,
            "legacy_mlcnn_global_graph": global_graph,
            "compact_route_eager": compact_model,
            "compact_route_graph": compact_graph,
        }

        with torch.inference_mode():
            global_eager = global_model(sample)
            global_captured = global_graph(sample)
            compact_eager = compact_model(sample)
            compact_captured = compact_graph(sample)
        torch.cuda.synchronize(device)
        equivalence.append({
            "batch_size": batch_size,
            "global_logits_max_abs_difference": float(
                (global_eager[0] - global_captured[0]).abs().max().item()
            ),
            "global_parameters_max_abs_difference": float(
                (global_eager[1] - global_captured[1]).abs().max().item()
            ),
            "compact_logits_max_abs_difference": float(
                (compact_eager[0] - compact_captured[0]).abs().max().item()
            ),
            "compact_parameters_max_abs_difference": float(
                (compact_eager[1] - compact_captured[1]).abs().max().item()
            ),
        })

        pooled = {name: [] for name in functions}
        block_medians = {name: [] for name in functions}
        for _ in range(int(benchmark["measurement_blocks"])):
            order = list(rng.permutation(list(functions)))
            for name in order:
                values = timed_calls(
                    functions[name], sample,
                    int(benchmark["measurement_warmup_per_block"]),
                    int(benchmark["repeats_per_block"]),
                )
                pooled[name].extend(values)
                block_medians[name].append(float(np.median(values)))
        for name in functions:
            row = {
                "mode": name,
                "batch_size": batch_size,
                **summarize(pooled[name], block_medians[name]),
            }
            row["throughput_samples_per_second"] = 1000.0 * batch_size / row["pooled_median_ms"]
            rows.append(row)
        del global_graph, compact_graph

    by_key = {(row["mode"], row["batch_size"]): row for row in rows}
    global_graph_b1 = by_key[("legacy_mlcnn_global_graph", 1)]
    compact_eager_b1 = by_key[("compact_route_eager", 1)]
    compact_graph_b1 = by_key[("compact_route_graph", 1)]
    global_graph_b64 = by_key[("legacy_mlcnn_global_graph", 64)]
    compact_graph_b64 = by_key[("compact_route_graph", 64)]
    compact_memory_b1 = memory[f"{compact_model_name}_batch1"]
    compact_memory_b64 = memory[f"{compact_model_name}_batch64"]
    comparisons = {
        "batch1_compact_graph_median_increase_vs_global_graph": (
            compact_graph_b1["pooled_median_ms"] / global_graph_b1["pooled_median_ms"] - 1.0
        ),
        "batch1_compact_graph_p95_increase_vs_global_graph": (
            compact_graph_b1["pooled_p95_ms"] / global_graph_b1["pooled_p95_ms"] - 1.0
        ),
        "batch1_compact_graph_latency_reduction_vs_compact_eager": (
            1.0 - compact_graph_b1["pooled_median_ms"] / compact_eager_b1["pooled_median_ms"]
        ),
        "batch64_compact_graph_throughput_retention_vs_global_graph": (
            compact_graph_b64["throughput_samples_per_second"]
            / global_graph_b64["throughput_samples_per_second"]
        ),
        "maximum_compact_graph_memory_increase_vs_eager": max(
            compact_memory_b1["graph_resident_increase_fraction"],
            compact_memory_b64["graph_resident_increase_fraction"],
        ),
    }
    validity = protocol["validity_gates"]
    promotion = protocol["promotion_gates"]
    cost_gates = protocol.get("cost_gates_before_training", {})

    def resolve_gate(primary_key: str, cost_key: str) -> float:
        if primary_key in promotion:
            return float(promotion[primary_key])
        return float(cost_gates[cost_key])

    logits_tolerance = validity.get(
        "eager_and_graph_logits_max_abs_difference_at_most",
        cost_gates.get("eager_and_graph_outputs_equivalent_at_tolerance", 1.0e-6),
    )
    parameters_tolerance = validity.get(
        "eager_and_graph_parameters_max_abs_difference_at_most",
        cost_gates.get("eager_and_graph_outputs_equivalent_at_tolerance", 1.0e-6),
    )
    median_limit = resolve_gate(
        "batch1_compact_graph_median_latency_increase_vs_global_graph_at_most",
        "cuda_graph_batch1_median_latency_increase_vs_global_graph_at_most",
    )
    p95_limit = resolve_gate(
        "batch1_compact_graph_p95_latency_increase_vs_global_graph_at_most",
        "cuda_graph_batch1_p95_latency_increase_vs_global_graph_at_most",
    )
    throughput_limit = resolve_gate(
        "batch64_compact_graph_throughput_retention_vs_global_graph_at_least",
        "cuda_graph_batch64_throughput_retention_vs_global_graph_at_least",
    )
    reduction_limit = promotion.get(
        "batch1_compact_graph_latency_reduction_vs_compact_eager_at_least"
    )
    memory_limit = promotion.get("compact_graph_peak_memory_increase_vs_compact_eager_at_most")
    gate_results = {
        "logits_equivalent": all(
            max(row["global_logits_max_abs_difference"], row["compact_logits_max_abs_difference"])
            <= logits_tolerance
            for row in equivalence
        ),
        "parameters_equivalent": all(
            max(
                row["global_parameters_max_abs_difference"],
                row["compact_parameters_max_abs_difference"],
            ) <= parameters_tolerance
            for row in equivalence
        ),
        "batch1_median_latency": comparisons[
            "batch1_compact_graph_median_increase_vs_global_graph"
        ] <= median_limit,
        "batch1_p95_latency": comparisons[
            "batch1_compact_graph_p95_increase_vs_global_graph"
        ] <= p95_limit,
        "batch1_reduction_vs_eager": True if reduction_limit is None else comparisons[
            "batch1_compact_graph_latency_reduction_vs_compact_eager"
        ] >= reduction_limit,
        "batch64_throughput": comparisons[
            "batch64_compact_graph_throughput_retention_vs_global_graph"
        ] >= throughput_limit,
        "compact_graph_memory": True if memory_limit is None else comparisons[
            "maximum_compact_graph_memory_increase_vs_eager"
        ] <= memory_limit,
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
        "architecture": architecture,
        "compact_model": compact_model_name,
        "memory": memory,
        "equivalence": equivalence,
        "rows": rows,
        "comparisons": comparisons,
        "gate_results": gate_results,
        "promotion_passed": bool(all(gate_results.values())),
        "deployment_boundary": protocol.get(
            "deployment_boundary",
            "Fixed-shape CUDA Graph evidence requires capture buckets or eager fallback for dynamic shapes.",
        ),
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
