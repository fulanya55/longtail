"""Combine CPU and CUDA engineering receipts and apply preregistered gates."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402


def run(args):
    gpu = json.loads(args.gpu.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu.read_text(encoding="utf-8"))
    gpu_full = gpu["latency"]["complete"]["batch_64"]
    gpu_base = gpu["latency"]["baseline"]["batch_64"]
    cpu_full = cpu["latency"]["complete"]["batch_64"]
    baseline_parameters = gpu["parameters"]["neural_baseline_total"]
    increment = gpu["parameters"]["non_tree_increment_total"]
    diagnostics = {
        "gpu_batch64_samples_per_second": gpu_full["median_samples_per_second"],
        "cpu_batch64_samples_per_second": cpu_full["median_samples_per_second"],
        "gpu_complete_to_baseline_latency_ratio": gpu_full["median_ms"] / gpu_base["median_ms"],
        "gpu_peak_allocated_bytes": gpu["memory"]["peak_cuda_allocated_bytes"],
        "non_tree_increment_fraction": increment / baseline_parameters,
        "artifact_bytes": gpu["artifact_bytes"]["unique_total"],
    }
    checks = {
        "gpu_batch64_throughput_ge_250": diagnostics["gpu_batch64_samples_per_second"] >= 250.0,
        "cpu_batch64_throughput_ge_20": diagnostics["cpu_batch64_samples_per_second"] >= 20.0,
        "gpu_peak_allocated_le_2gib": diagnostics["gpu_peak_allocated_bytes"] <= 2 * 1024 ** 3,
        "gpu_latency_ratio_le_10": diagnostics["gpu_complete_to_baseline_latency_ratio"] <= 10.0,
        "non_tree_increment_fraction_le_0p05": diagnostics["non_tree_increment_fraction"] <= 0.05,
        "artifact_bytes_le_25mib": diagnostics["artifact_bytes"] <= 25 * 1024 ** 2,
    }
    result = {
        "status": "completed", "experiment": "RouteA_FinalSharedPolicy_EngineeringAudit",
        "sources": {"gpu": str(args.gpu), "gpu_sha256": _sha256(args.gpu), "cpu": str(args.cpu), "cpu_sha256": _sha256(args.cpu)},
        "diagnostics": diagnostics, "checks": checks, "passed": all(checks.values()),
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); _write_json(args.output, result)
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--gpu", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_engineering/benchmark_cuda.json"); parser.add_argument("--cpu", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_engineering/benchmark_cpu.json"); parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_engineering/engineering_audit.json"); run(parser.parse_args())
