"""P5l: full-output equivalence with fixed-minibatch guarded latency timing.

The P5k timing unit was a complete 3,780-example power pool, making a
single guarded timing run unnecessarily long.  P5l retains a full-pool,
untimed exact-output comparison and measures end-to-end latency only on the
pre-registered first 126 paired examples (one normal inference batch).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role
from cj0724.p5b3_bounded_jnr_attachment import load_base
from cj0724.p5b4_bounded_jnr_confirmation import load_head
from cj0724.p5g_shared_dense_execution import _dense_reference, _finite_values_exact, _metric_difference, _quantiles, _shared_dense
from cj0724.p5h_exclusive_gpu_guard import append_exclusive_receipt, assert_exclusive_gpu
from cj0724.p5h_shared_dense_execution import _timings


EXACT_KEYS = (
    "logits_exact", "probabilities_exact", "hard_decisions_exact",
    "base_parameters_exact", "dense_head_values_exact",
    "all_candidate_parameters_exact", "non_target_parameters_exact",
)


def fixed_benchmark_indices(pool_size: int, batch_size: int) -> np.ndarray:
    """Return the pre-registered first complete batch without data-dependent selection."""
    if batch_size <= 0 or pool_size < batch_size:
        raise ValueError("fixed benchmark requires one complete positive-size batch")
    return np.arange(batch_size, dtype=np.int64)


def _output_exact(reference: dict, candidate: dict) -> dict[str, bool]:
    return {
        "logits_exact": _finite_values_exact(reference["logits"], candidate["logits"]),
        "probabilities_exact": _finite_values_exact(reference["probability"], candidate["probability"]),
        "hard_decisions_exact": bool(np.array_equal(reference["decision"], candidate["decision"])),
        "base_parameters_exact": _finite_values_exact(reference["base_full"], candidate["base_full"]),
        "dense_head_values_exact": _finite_values_exact(reference["values"], candidate["values"]),
        "all_candidate_parameters_exact": _finite_values_exact(reference["candidate_full"], candidate["candidate_full"]),
        "non_target_parameters_exact": bool(np.array_equal(reference["candidate_full"][~reference["allowed"]], candidate["candidate_full"][~candidate["allowed"]])),
    }


def _one_model(project: Path, full_dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze: dict, source_seed: int, warmups: int, repeats: int, device: torch.device) -> dict:
    key = str(source_seed)
    model, normalizer, thresholds, source = load_base(project, source_seed, freeze["frozen_candidate"]["base_roots"][key]); model = model.to(device).eval()
    head_path = project / freeze["frozen_candidate"]["head_checkpoints"][key]; head, state = load_head(head_path, device)

    exact_receipts: list[dict[str, object]] = []
    append_exclusive_receipt(exact_receipts)
    full_reference = _dense_reference(full_dataset, full_analytic, model, normalizer, thresholds, head, state, device)
    full_candidate = _shared_dense(full_dataset, full_analytic, model, normalizer, thresholds, head, state, device)
    append_exclusive_receipt(exact_receipts)
    full_exact = _output_exact(full_reference, full_candidate)
    metric_difference, metric_rows = _metric_difference(full_reference, full_candidate, np.asarray(state["target_scale"]), thresholds)

    append_exclusive_receipt(exact_receipts)
    benchmark_reference = _dense_reference(benchmark_dataset, benchmark_analytic, model, normalizer, thresholds, head, state, device)
    benchmark_candidate = _shared_dense(benchmark_dataset, benchmark_analytic, model, normalizer, thresholds, head, state, device)
    append_exclusive_receipt(exact_receipts)
    benchmark_exact = _output_exact(benchmark_reference, benchmark_candidate)
    reference_times, reference_receipts = _timings(lambda: _dense_reference(benchmark_dataset, benchmark_analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    candidate_times, candidate_receipts = _timings(lambda: _shared_dense(benchmark_dataset, benchmark_analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    result = {
        "source": source, "head_sha256": sha256_file(head_path),
        "full_pool_exact": full_exact,
        "full_pool_metric_rows": metric_rows,
        "full_pool_metric_max_abs_difference": metric_difference,
        "benchmark_batch_exact": benchmark_exact,
        "base_forward_calls": {"dense_reference": 2, "shared_dense": 1},
        "head_rows": {"full_pool": int(len(full_reference["role"].feature)), "benchmark_batch": int(len(benchmark_reference["role"].feature))},
        "timing": {"dense_reference": _quantiles(reference_times), "shared_dense": _quantiles(candidate_times), "median_ratio_shared_dense_over_reference": float(np.median(candidate_times) / np.median(reference_times))},
        "exclusive_gpu_guard": {"equivalence_receipts": exact_receipts, "reference_receipts": reference_receipts, "shared_dense_receipts": candidate_receipts},
    }
    del model, head
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def _gates(rows: list[dict], latency_limit: float) -> dict[str, bool]:
    guard_receipts = (receipt for row in rows for phase in row["exclusive_gpu_guard"].values() for receipt in phase)
    return {
        "full_pool_outputs_exact": all(all(row["full_pool_exact"].values()) for row in rows),
        "benchmark_batch_outputs_exact": all(all(row["benchmark_batch_exact"].values()) for row in rows),
        "full_pool_metric_max_abs_difference_exactly_zero": all(row["full_pool_metric_max_abs_difference"] == 0.0 for row in rows),
        "median_latency_ratio_within_preregistered_limit": all(row["timing"]["median_ratio_shared_dense_over_reference"] <= latency_limit for row in rows),
        "full_pool_and_fixed_batch_cost_reported": all(row["head_rows"]["full_pool"] > row["head_rows"]["benchmark_batch"] for row in rows),
        "exclusive_gpu_guard_before_and_after_all_execution": all(not receipt["external_python_processes"] for receipt in guard_receipts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preregister", "p5e-freeze", "data-roots", "audits", "output-root"): parser.add_argument(f"--{name}", nargs="+" if name in ("data-roots", "audits") else None, required=True)
    args = parser.parse_args(); project = Path.cwd()
    protocol_path, freeze_path = project / args.preregister, project / args.p5e_freeze
    protocol, freeze = json.loads(protocol_path.read_text(encoding="utf-8")), json.loads(freeze_path.read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    for path_key, hash_key in ((key, f"{key}_sha256") for key in parent if key.endswith(("_result", "_preregister", "_freeze"))):
        if sha256_file(project / parent[path_key]) != parent[hash_key]: raise RuntimeError(f"P5l frozen parent hash mismatch: {path_key}")
    if sha256_file(freeze_path) != parent["P5e_candidate_freeze_sha256"]: raise RuntimeError("P5l frozen P5e candidate-freeze hash mismatch")
    if len(args.data_roots) != 3 or len(args.audits) != 3: raise RuntimeError("P5l development requires exactly three new identity pools")
    benchmark_batch_size = int(protocol["execution"]["batch_size"])
    preflight = assert_exclusive_gpu(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "exclusive_gpu_preflight": preflight, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    all_pass = True
    for root_arg, audit_arg in zip(args.data_roots, args.audits):
        audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
        if not audit["valid_for_frozen_inference"]: raise RuntimeError(f"invalid P5l audit: {audit_arg}")
        full_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
        benchmark_indices = fixed_benchmark_indices(len(full_dataset), benchmark_batch_size)
        benchmark_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), benchmark_indices)
        started = time.perf_counter(); full_analytic = collect_analytic_role(full_dataset, device); benchmark_analytic = collect_analytic_role(benchmark_dataset, device); append_exclusive_receipt([])
        rows = [_one_model(project, full_dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze, int(seed), protocol["execution"]["warmup_runs"], protocol["execution"]["timed_repetitions"], device) for seed in parent["source_model_seeds"]]
        gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"])); all_pass &= all(gates.values())
        output["development"][root_arg] = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_feature_seconds": float(time.perf_counter() - started), "rows": rows, "gates": gates}
        full_dataset.close(); benchmark_dataset.close()
    output["development_passed"] = bool(all_pass)
    root = project / args.output_root; root.mkdir(parents=True, exist_ok=True); (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": output["development_passed"], "gates": {key: value["gates"] for key, value in output["development"].items()}}, indent=2))


if __name__ == "__main__": main()
