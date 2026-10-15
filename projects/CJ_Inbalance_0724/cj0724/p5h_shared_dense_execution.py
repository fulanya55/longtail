"""P5h: P5g shared-dense execution rerun with strict exclusive-GPU guard."""
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


def _timings(function, warmups: int, repeats: int, device: torch.device) -> tuple[list[float], list[dict[str, object]]]:
    receipts: list[dict[str, object]] = []
    for _ in range(warmups):
        append_exclusive_receipt(receipts); _sync(device); function(); _sync(device); append_exclusive_receipt(receipts)
    values = []
    for _ in range(repeats):
        append_exclusive_receipt(receipts); _sync(device)
        started = time.perf_counter(); function(); _sync(device)
        values.append(float(time.perf_counter() - started)); append_exclusive_receipt(receipts)
    return values, receipts


def _one_model(project: Path, dataset, analytic, freeze: dict, source_seed: int, warmups: int, repeats: int, device: torch.device) -> dict:
    key = str(source_seed)
    model, normalizer, thresholds, source = load_base(project, source_seed, freeze["frozen_candidate"]["base_roots"][key]); model = model.to(device).eval()
    head_path = project / freeze["frozen_candidate"]["head_checkpoints"][key]; head, state = load_head(head_path, device)
    reference = _dense_reference(dataset, analytic, model, normalizer, thresholds, head, state, device)
    candidate = _shared_dense(dataset, analytic, model, normalizer, thresholds, head, state, device)
    metric_difference, metric_rows = _metric_difference(reference, candidate, np.asarray(state["target_scale"]), thresholds)
    reference_times, reference_receipts = _timings(lambda: _dense_reference(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    candidate_times, candidate_receipts = _timings(lambda: _shared_dense(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    result = {
        "source": source, "head_sha256": sha256_file(head_path),
        "logits_exact": _finite_values_exact(reference["logits"], candidate["logits"]),
        "probabilities_exact": _finite_values_exact(reference["probability"], candidate["probability"]),
        "hard_decisions_exact": bool(np.array_equal(reference["decision"], candidate["decision"])),
        "base_parameters_exact": _finite_values_exact(reference["base_full"], candidate["base_full"]),
        "dense_head_values_exact": _finite_values_exact(reference["values"], candidate["values"]),
        "all_candidate_parameters_exact": _finite_values_exact(reference["candidate_full"], candidate["candidate_full"]),
        "non_target_parameters_exact": bool(np.array_equal(reference["candidate_full"][~reference["allowed"]], candidate["candidate_full"][~candidate["allowed"]])),
        "metrics": metric_rows, "metric_max_abs_difference": metric_difference,
        "base_forward_calls": {"dense_reference": 2, "shared_dense": 1},
        "head_rows": {"dense_reference": int(len(reference["role"].feature)), "shared_dense": int(len(candidate["role"].feature))},
        "timing": {"dense_reference": _quantiles(reference_times), "shared_dense": _quantiles(candidate_times), "median_ratio_shared_dense_over_reference": float(np.median(candidate_times) / np.median(reference_times))},
        "exclusive_gpu_guard": {"reference_receipts": reference_receipts, "shared_dense_receipts": candidate_receipts},
    }
    del model, head
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def _gates(rows: list[dict], latency_limit: float) -> dict:
    ratios = [row["timing"]["median_ratio_shared_dense_over_reference"] for row in rows]
    exact = ("logits_exact", "probabilities_exact", "hard_decisions_exact", "base_parameters_exact", "dense_head_values_exact", "all_candidate_parameters_exact", "non_target_parameters_exact")
    return {
        **{name: all(bool(row[name]) for row in rows) for name in exact},
        "metric_max_abs_difference_exactly_zero": all(row["metric_max_abs_difference"] == 0.0 for row in rows),
        "median_latency_ratio_within_preregistered_limit": all(ratio <= latency_limit for ratio in ratios),
        "exclusive_gpu_guard_before_every_warmup_and_repeat": all(not receipt["external_python_processes"] for row in rows for phase in row["exclusive_gpu_guard"].values() for receipt in phase),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preregister", "p5e-freeze", "data-roots", "audits", "output-root"): parser.add_argument(f"--{name}", nargs="+" if name in ("data-roots", "audits") else None, required=True)
    args = parser.parse_args(); project = Path.cwd()
    protocol_path, freeze_path = project / args.preregister, project / args.p5e_freeze
    protocol, freeze = json.loads(protocol_path.read_text(encoding="utf-8")), json.loads(freeze_path.read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    for path_key, hash_key in (("P5e_confirmation_result", "P5e_confirmation_result_sha256"), ("P5g_preregister", "P5g_preregister_sha256"), ("P5g_outcome_freeze", "P5g_outcome_freeze_sha256")):
        if sha256_file(project / parent[path_key]) != parent[hash_key]: raise RuntimeError(f"P5h frozen parent hash mismatch: {path_key}")
    if sha256_file(freeze_path) != parent["P5e_candidate_freeze_sha256"]: raise RuntimeError("P5h frozen P5e candidate-freeze hash mismatch")
    if len(args.data_roots) != 3 or len(args.audits) != 3: raise RuntimeError("P5h development requires exactly three new identity pools")
    preflight = assert_exclusive_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "exclusive_gpu_preflight": preflight, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    all_pass = True
    for root_arg, audit_arg in zip(args.data_roots, args.audits):
        audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
        if not audit["valid_for_frozen_inference"]: raise RuntimeError(f"invalid P5h audit: {audit_arg}")
        dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
        started = time.perf_counter(); analytic = collect_analytic_role(dataset, device); _sync(device); assert_exclusive_gpu()
        rows = [_one_model(project, dataset, analytic, freeze, int(seed), protocol["execution"]["warmup_runs"], protocol["execution"]["timed_repetitions"], device) for seed in parent["source_model_seeds"]]
        gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"])); all_pass &= all(gates.values())
        output["development"][root_arg] = {"audit": audit, "analytic_feature_seconds": float(time.perf_counter() - started), "rows": rows, "gates": gates}; dataset.close()
    output["development_passed"] = bool(all_pass)
    root = project / args.output_root; root.mkdir(parents=True, exist_ok=True); (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": output["development_passed"], "gates": {key: value["gates"] for key, value in output["development"].items()}}, indent=2))


if __name__ == "__main__": main()
