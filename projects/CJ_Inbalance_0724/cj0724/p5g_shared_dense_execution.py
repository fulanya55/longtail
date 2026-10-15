"""P5g: exact dense-head execution with one shared frozen base forward.

P5f established that reducing the attachment head to a selected sub-batch is
not bitwise equivalent to dense evaluation.  This runner deliberately keeps
the head dense; its only candidate change is sharing the already-identical
classification and parameter outputs from one base-model call.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, metrics, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role, collect_frozen_posterior, with_context
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES, TARGET_NAMES, collect_base_parameter, load_base, predicted_gated_role
from cj0724.p5b4_bounded_jnr_confirmation import estimate, load_head
from cj0724.p5f_shared_lazy_execution import attach_target_values, collect_shared_base_outputs


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_values_exact(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact comparison which treats matched inactive NaNs as equal."""
    left, right = np.asarray(left), np.asarray(right)
    finite = np.isfinite(left) & np.isfinite(right)
    return bool(
        left.shape == right.shape
        and np.array_equal(np.isfinite(left), np.isfinite(right))
        and np.array_equal(left[finite], right[finite])
    )


def _dense_reference(dataset, analytic, model, normalizer, thresholds, head, state, device):
    posterior = collect_frozen_posterior(dataset, model, device)
    role = with_context(analytic, posterior)
    values = estimate(role, head, state, device)
    logits, base_full = collect_base_parameter(dataset, model, normalizer, device)
    probability = 1.0 / (1.0 + np.exp(-logits))
    decision = probability >= thresholds[None, :]
    candidate_full, allowed = attach_target_values(base_full, values, decision)
    return {
        "role": role, "values": values, "logits": logits, "probability": probability,
        "decision": decision, "base_full": base_full, "candidate_full": candidate_full,
        "allowed": allowed,
    }


def _shared_dense(dataset, analytic, model, normalizer, thresholds, head, state, device):
    logits, base_full = collect_shared_base_outputs(dataset, model, normalizer, device)
    probability = 1.0 / (1.0 + np.exp(-logits))
    decision = probability >= thresholds[None, :]
    role = with_context(analytic, probability)
    # Crucially, this is a full-batch call with exactly P5e's head batching.
    values = estimate(role, head, state, device)
    candidate_full, allowed = attach_target_values(base_full, values, decision)
    return {
        "role": role, "values": values, "logits": logits, "probability": probability,
        "decision": decision, "base_full": base_full, "candidate_full": candidate_full,
        "allowed": allowed,
    }


def _timings(function, warmups: int, repeats: int, device: torch.device) -> list[float]:
    for _ in range(warmups):
        _sync(device); function(); _sync(device)
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        function()
        _sync(device)
        values.append(float(time.perf_counter() - started))
    return values


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_seconds": float(np.quantile(array, 0.1)),
        "median_seconds": float(np.median(array)),
        "p90_seconds": float(np.quantile(array, 0.9)),
        "repetitions": int(len(array)),
    }


def _metric_difference(reference: dict, candidate: dict, scale: np.ndarray, thresholds: np.ndarray) -> tuple[float, dict]:
    result, maximum = {}, 0.0
    for target in TARGET_NAMES:
        reference_oracle = metrics(reference["candidate_full"][:, TARGET_CLASSES, 0], reference["role"], scale)["by_target"][target]["nmae"]
        candidate_oracle = metrics(candidate["candidate_full"][:, TARGET_CLASSES, 0], candidate["role"], scale)["by_target"][target]["nmae"]
        reference_gated = metrics(reference["candidate_full"][:, TARGET_CLASSES, 0], predicted_gated_role(reference["role"], reference["probability"], thresholds), scale)["by_target"][target]["nmae"]
        candidate_gated = metrics(candidate["candidate_full"][:, TARGET_CLASSES, 0], predicted_gated_role(candidate["role"], candidate["probability"], thresholds), scale)["by_target"][target]["nmae"]
        result[target] = {
            "oracle_nmae_abs_difference": float(abs(reference_oracle - candidate_oracle)),
            "predicted_gated_nmae_abs_difference": float(abs(reference_gated - candidate_gated)),
        }
        maximum = max(maximum, *result[target].values())
    return float(maximum), result


def _one_model(project: Path, dataset, analytic, freeze: dict, source_seed: int, warmups: int, repeats: int, device: torch.device) -> dict:
    key = str(source_seed)
    model, normalizer, thresholds, source = load_base(project, source_seed, freeze["frozen_candidate"]["base_roots"][key])
    model = model.to(device).eval()
    head_path = project / freeze["frozen_candidate"]["head_checkpoints"][key]
    head, state = load_head(head_path, device)
    reference = _dense_reference(dataset, analytic, model, normalizer, thresholds, head, state, device)
    candidate = _shared_dense(dataset, analytic, model, normalizer, thresholds, head, state, device)
    metric_difference, metric_rows = _metric_difference(reference, candidate, np.asarray(state["target_scale"]), thresholds)
    reference_times = _timings(lambda: _dense_reference(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    candidate_times = _timings(lambda: _shared_dense(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    result = {
        "source": source,
        "head_sha256": sha256_file(head_path),
        "logits_exact": _finite_values_exact(reference["logits"], candidate["logits"]),
        "probabilities_exact": _finite_values_exact(reference["probability"], candidate["probability"]),
        "hard_decisions_exact": bool(np.array_equal(reference["decision"], candidate["decision"])),
        "base_parameters_exact": _finite_values_exact(reference["base_full"], candidate["base_full"]),
        "dense_head_values_exact": _finite_values_exact(reference["values"], candidate["values"]),
        "all_candidate_parameters_exact": _finite_values_exact(reference["candidate_full"], candidate["candidate_full"]),
        "non_target_parameters_exact": bool(np.array_equal(reference["candidate_full"][~reference["allowed"]], candidate["candidate_full"][~candidate["allowed"]])),
        "metrics": metric_rows,
        "metric_max_abs_difference": metric_difference,
        "base_forward_calls": {"dense_reference": 2, "shared_dense": 1},
        "head_rows": {"dense_reference": int(len(reference["role"].feature)), "shared_dense": int(len(candidate["role"].feature))},
        "timing": {
            "dense_reference": _quantiles(reference_times),
            "shared_dense": _quantiles(candidate_times),
            "median_ratio_shared_dense_over_reference": float(np.median(candidate_times) / np.median(reference_times)),
        },
    }
    del model, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _gates(rows: list[dict], latency_limit: float) -> dict:
    ratios = [row["timing"]["median_ratio_shared_dense_over_reference"] for row in rows]
    exact = ("logits_exact", "probabilities_exact", "hard_decisions_exact", "base_parameters_exact", "dense_head_values_exact", "all_candidate_parameters_exact", "non_target_parameters_exact")
    return {
        **{name: all(bool(row[name]) for row in rows) for name in exact},
        "metric_max_abs_difference_exactly_zero": all(row["metric_max_abs_difference"] == 0.0 for row in rows),
        "median_latency_ratio_within_preregistered_limit": all(ratio <= latency_limit for ratio in ratios),
        "base_forward_and_dense_head_cost_reported": all("timing" in row and "head_rows" in row for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); project = Path.cwd()
    protocol_path, freeze_path = project / args.preregister, project / args.p5e_freeze
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    if sha256_file(project / parent["P5e_confirmation_result"]) != parent["P5e_confirmation_result_sha256"]:
        raise RuntimeError("P5g frozen P5e parent hash mismatch")
    if sha256_file(project / parent["P5f_stop_result"]) != parent["P5f_stop_result_sha256"]:
        raise RuntimeError("P5g frozen P5f stop-result hash mismatch")
    if sha256_file(freeze_path) != parent["P5e_candidate_freeze_sha256"]:
        raise RuntimeError("P5g frozen P5e candidate-freeze hash mismatch")
    if len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5g development requires exactly three new identity pools")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    all_pass = True
    for root_arg, audit_arg in zip(args.data_roots, args.audits):
        audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
        if not audit["valid_for_frozen_inference"]:
            raise RuntimeError(f"invalid P5g audit: {audit_arg}")
        dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
        started = time.perf_counter(); analytic = collect_analytic_role(dataset, device); _sync(device)
        rows = [_one_model(project, dataset, analytic, freeze, int(seed), protocol["benchmark"]["warmup_runs"], protocol["benchmark"]["timed_repetitions"], device) for seed in parent["source_model_seeds"]]
        gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"]))
        all_pass &= all(gates.values())
        output["development"][root_arg] = {"audit": audit, "analytic_feature_seconds": float(time.perf_counter() - started), "rows": rows, "gates": gates}
        dataset.close()
    output["development_passed"] = bool(all_pass)
    root = project / args.output_root; root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": output["development_passed"], "gates": {key: value["gates"] for key, value in output["development"].items()}}, indent=2))


if __name__ == "__main__":
    main()
