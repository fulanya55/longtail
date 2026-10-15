"""P5f frozen dense-versus-shared/lazy execution comparison."""
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
from cj0724.p5f_shared_lazy_execution import attach_target_values, collect_shared_base_outputs, lazy_estimate, union_positive_rows


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dense(dataset, analytic, model, normalizer, thresholds, head, state, device):
    posterior = collect_frozen_posterior(dataset, model, device)
    role = with_context(analytic, posterior)
    values = estimate(role, head, state, device)
    logits, base_full = collect_base_parameter(dataset, model, normalizer, device)
    probability = 1.0 / (1.0 + np.exp(-logits))
    decision = probability >= thresholds[None, :]
    candidate_full, allowed = attach_target_values(base_full, values, decision)
    return {"role": role, "logits": logits, "probability": probability, "decision": decision, "base_full": base_full, "candidate_full": candidate_full, "allowed": allowed}


def _shared_lazy(dataset, analytic, model, normalizer, thresholds, head, state, device):
    logits, base_full = collect_shared_base_outputs(dataset, model, normalizer, device)
    probability = 1.0 / (1.0 + np.exp(-logits))
    decision = probability >= thresholds[None, :]
    role = with_context(analytic, probability)
    active = union_positive_rows(decision)
    values = lazy_estimate(role, head, state, device, active)
    candidate_full, allowed = attach_target_values(base_full, values, decision)
    return {"role": role, "logits": logits, "probability": probability, "decision": decision, "base_full": base_full, "candidate_full": candidate_full, "allowed": allowed, "active": active}


def _timings(function, warmups: int, repeats: int, device: torch.device) -> list[float]:
    for _ in range(warmups):
        _sync(device); function(); _sync(device)
    values = []
    for _ in range(repeats):
        _sync(device); started = time.perf_counter(); function(); _sync(device)
        values.append(float(time.perf_counter() - started))
    return values


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"p10_seconds": float(np.quantile(array, 0.1)), "median_seconds": float(np.median(array)), "p90_seconds": float(np.quantile(array, 0.9)), "repetitions": int(len(array))}


def _metric_difference(dense: dict, candidate: dict, scale: np.ndarray, thresholds: np.ndarray) -> tuple[float, dict]:
    target_base = dense["base_full"][:, TARGET_CLASSES, 0]
    target_dense = dense["candidate_full"][:, TARGET_CLASSES, 0]
    target_candidate = candidate["candidate_full"][:, TARGET_CLASSES, 0]
    result = {}
    maximum = 0.0
    for target in TARGET_NAMES:
        dense_oracle = metrics(target_dense, dense["role"], scale)["by_target"][target]["nmae"]
        candidate_oracle = metrics(target_candidate, candidate["role"], scale)["by_target"][target]["nmae"]
        dense_predicted = metrics(target_dense, predicted_gated_role(dense["role"], dense["probability"], thresholds), scale)["by_target"][target]["nmae"]
        candidate_predicted = metrics(target_candidate, predicted_gated_role(candidate["role"], candidate["probability"], thresholds), scale)["by_target"][target]["nmae"]
        result[target] = {"oracle_nmae_abs_difference": float(abs(dense_oracle - candidate_oracle)), "predicted_gated_nmae_abs_difference": float(abs(dense_predicted - candidate_predicted))}
        maximum = max(maximum, result[target]["oracle_nmae_abs_difference"], result[target]["predicted_gated_nmae_abs_difference"])
    return maximum, result


def _one_model(project: Path, dataset, analytic, freeze: dict, source_seed: int, warmups: int, repeats: int, device: torch.device) -> dict:
    key = str(source_seed)
    model, normalizer, thresholds, source = load_base(project, source_seed, freeze["frozen_candidate"]["base_roots"][key])
    model = model.to(device).eval()
    head_path = project / freeze["frozen_candidate"]["head_checkpoints"][key]
    head, state = load_head(head_path, device)
    dense = _dense(dataset, analytic, model, normalizer, thresholds, head, state, device)
    candidate = _shared_lazy(dataset, analytic, model, normalizer, thresholds, head, state, device)
    target_difference = float(np.max(np.abs(dense["candidate_full"][:, TARGET_CLASSES, 0] - candidate["candidate_full"][:, TARGET_CLASSES, 0])))
    metric_difference, metric_rows = _metric_difference(dense, candidate, np.asarray(state["target_scale"]), thresholds)
    dense_times = _timings(lambda: _dense(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    candidate_times = _timings(lambda: _shared_lazy(dataset, analytic, model, normalizer, thresholds, head, state, device), warmups, repeats, device)
    result = {
        "source": source,
        "head_sha256": sha256_file(head_path),
        "hard_decisions_exact": bool(np.array_equal(dense["decision"], candidate["decision"])),
        "non_target_parameters_exact": bool(np.array_equal(dense["candidate_full"][~dense["allowed"]], candidate["candidate_full"][~dense["allowed"]])),
        "target_parameter_max_abs_difference": target_difference,
        "metrics": metric_rows,
        "metric_max_abs_difference": metric_difference,
        "base_forward_calls": {"dense": 2, "shared_lazy": 1},
        "head_rows": {"dense": int(len(candidate["active"])), "shared_lazy": int(candidate["active"].sum()), "union_positive_fraction": float(candidate["active"].mean())},
        "timing": {"dense": _quantiles(dense_times), "shared_lazy": _quantiles(candidate_times), "median_ratio_shared_lazy_over_dense": float(np.median(candidate_times) / np.median(dense_times))},
    }
    del model, head
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def _gates(rows: list[dict]) -> dict:
    ratios = [row["timing"]["median_ratio_shared_lazy_over_dense"] for row in rows]
    return {
        "hard_decisions_exact": all(row["hard_decisions_exact"] for row in rows),
        "target_parameter_max_abs_difference_at_most_1e6": all(row["target_parameter_max_abs_difference"] <= 1.0e-6 for row in rows),
        "metric_max_abs_difference_at_most_1e6": all(row["metric_max_abs_difference"] <= 1.0e-6 for row in rows),
        "non_target_parameters_exact": all(row["non_target_parameters_exact"] for row in rows),
        "median_latency_ratio_at_most_0_9": float(np.median(ratios)) <= 0.9,
        "timing_and_lazy_cost_reported": all("timing" in row and "head_rows" in row for row in rows),
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
    protocol, freeze = json.loads(protocol_path.read_text(encoding="utf-8")), json.loads(freeze_path.read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    if sha256_file(project / parent["P5e_confirmation_result"]) != parent["P5e_confirmation_result_sha256"] or sha256_file(freeze_path) != parent["P5e_candidate_freeze_sha256"]:
        raise RuntimeError("P5f frozen P5e parent hash mismatch")
    if len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5f development requires exactly three new identity pools")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    all_pass = True
    for root_arg, audit_arg in zip(args.data_roots, args.audits):
        audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
        if not audit["valid_for_frozen_inference"]: raise RuntimeError(f"invalid P5f audit: {audit_arg}")
        dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
        analytic_started = time.perf_counter(); analytic = collect_analytic_role(dataset, device); _sync(device)
        rows = [_one_model(project, dataset, analytic, freeze, int(seed), protocol["benchmark"]["warmup_runs"], protocol["benchmark"]["timed_repetitions"], device) for seed in parent["source_model_seeds"]]
        gates = _gates(rows); all_pass &= all(gates.values())
        output["development"][root_arg] = {"audit": audit, "analytic_feature_seconds": float(time.perf_counter() - analytic_started), "rows": rows, "gates": gates}
        dataset.close()
    output["development_passed"] = bool(all_pass)
    root = project / args.output_root; root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": output["development_passed"], "gates": {key: value["gates"] for key, value in output["development"].items()}}, indent=2))


if __name__ == "__main__": main()
