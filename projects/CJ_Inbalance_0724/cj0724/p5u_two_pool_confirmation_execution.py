"""P5u: conditional two-new-pool confirmation for the frozen cost protocol.

This module is deliberately separate from the P5s three-pool development
executor.  It reuses its frozen analytic-chunk and per-model measurement
primitives verbatim, but requires exactly the two confirmation identities
reserved in the P5u preregistration.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices
from cj0724.p5m_authorized_window_shared_dense_execution import exclusive_window_receipt, restore_default_if_changed
from cj0724.p5q_relocation_safe_checkpoint_execution import _next_attempt, _resolve
from cj0724.p5s_detached_chunked_execution import _load_or_collect_chunked, _slice_role, validate_protocol as validate_p5s_protocol


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"P5u confirmation immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise RuntimeError(f"P5u confirmation receipt appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt_base(protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    return {
        "receipt_schema": "P5u-two-pool-confirmation-v1",
        "created_utc": _utc_now(),
        "pid": os.getpid(),
        "preregister_sha256": sha256_file(protocol_path),
        "protocol_validation": validated_artifacts,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_confirmation_protocol(project: Path, protocol_path: Path, confirmation_path: Path, freeze_path: Path) -> dict:
    validated = validate_p5s_protocol(project, protocol_path, freeze_path)
    confirmation = _load_json(confirmation_path)
    if confirmation.get("schema") != "P5u-two-pool-confirmation-v1":
        raise RuntimeError("P5u confirmation preregistration schema mismatch")
    if confirmation.get("parent_p5u_preregister_sha256") != sha256_file(protocol_path):
        raise RuntimeError("P5u confirmation parent preregistration hash mismatch")
    if confirmation.get("confirmation_generator_seeds") != [3340000000, 3350000000]:
        raise RuntimeError("P5u confirmation identity contract changed")
    artifacts = confirmation.get("frozen_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("P5u confirmation frozen artifact list missing")
    for artifact in artifacts:
        path = _resolve(project, artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"P5u confirmation frozen artifact hash mismatch: {artifact['path']}")
    return {**validated, "confirmation": confirmation, "confirmation_path": confirmation_path, "confirmation_artifacts": [item["path"] for item in artifacts]}


def validate_confirmation_context(project: Path, protocol_path: Path, confirmation_path: Path, development_result: Path, child_started: Path, terminal: Path) -> dict:
    """Require a fully passing, directly supervised P5u development result."""
    protocol = _load_json(protocol_path)
    development = _load_json(development_result)
    started, terminal_receipt = _load_json(child_started), _load_json(terminal)
    if not development.get("development_passed"):
        raise RuntimeError("P5u confirmation requires a passing development result")
    if development.get("preregister_sha256") != sha256_file(protocol_path):
        raise RuntimeError("P5u development result preregistration hash mismatch")
    if development.get("execution_exception"):
        raise RuntimeError("P5u development result records an execution exception")
    rows = development.get("development", {})
    if len(rows) != 3 or not all(all(item.get("gates", {}).values()) for item in rows.values()):
        raise RuntimeError("P5u confirmation requires all three development pool gates")
    if terminal_receipt.get("stage") != "child_terminal" or terminal_receipt.get("returncode") != 0:
        raise RuntimeError("P5u development supervisor has no successful terminal receipt")
    if terminal_receipt.get("child_pid") != started.get("child_pid"):
        raise RuntimeError("P5u development terminal does not track its direct executor")
    reserved = protocol.get("confirmation_if_and_only_if_development_passes", {}).get("generator_seeds")
    if reserved != _load_json(confirmation_path).get("confirmation_generator_seeds"):
        raise RuntimeError("P5u confirmation identity contract changed")
    return {
        "development_result": str(development_result),
        "development_result_sha256": sha256_file(development_result),
        "direct_executor_pid": int(terminal_receipt["child_pid"]),
        "reserved_confirmation_seeds": reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--confirmation-preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--development-result")
    parser.add_argument("--development-child-started")
    parser.add_argument("--development-terminal")
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--data-roots", nargs="+")
    parser.add_argument("--audits", nargs="+")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    protocol_path, freeze_path = _resolve(project, args.preregister), _resolve(project, args.p5e_freeze)
    confirmation_path = _resolve(project, args.confirmation_preregister)
    validated = validate_confirmation_protocol(project, protocol_path, confirmation_path, freeze_path)
    if args.validate_protocol:
        print(json.dumps({"protocol_valid": True, "project_root": str(project), "confirmation_pool_count": 2}, indent=2))
        return
    required = (args.development_result, args.development_child_started, args.development_terminal, args.data_roots, args.audits, args.output_root)
    if not all(required) or len(args.data_roots) != 2 or len(args.audits) != 2:
        raise RuntimeError("P5u confirmation requires development receipts, exactly two data roots, two audits and one output root")
    context = validate_confirmation_context(
        project,
        protocol_path,
        confirmation_path,
        _resolve(project, args.development_result),
        _resolve(project, args.development_child_started),
        _resolve(project, args.development_terminal),
    )
    protocol, freeze = validated["protocol"], _load_json(freeze_path)
    output_root = _resolve(project, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "result.json").exists():
        raise RuntimeError("P5u confirmation final result already exists; selective rerun is prohibited")
    base = {**_receipt_base(protocol_path, validated["validated_artifacts"]), "confirmation_preregister_sha256": sha256_file(confirmation_path), "confirmation_protocol_validation": validated["confirmation_artifacts"]}
    attempt = _next_attempt(output_root)
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_launch.json", {**base, "stage": "launch", "attempt": attempt, "data_roots": args.data_roots, "audit_paths": args.audits, "development_context": context})
    environment = exclusive_window_receipt()
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_environment.json", {**base, "stage": "environment_ready", "attempt": attempt, "environment": environment})
    output: dict[str, object] = {"experiment": "P5u two-new-pool confirmation", "preregister_sha256": sha256_file(protocol_path), "confirmation_preregister_sha256": sha256_file(confirmation_path), "attempt": attempt, "development_context": context, "environment": environment, "confirmation": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    failure: Exception | None = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("P5u confirmation cost execution requires CUDA")
        all_pass = True
        for ordinal, (root_arg, audit_arg) in enumerate(zip(args.data_roots, args.audits)):
            completed_pool = output_root / f"pool_{ordinal:03d}_completed.json"
            if completed_pool.exists():
                prior = _load_json(completed_pool)
                result = prior["result"]
                if prior.get("preregister_sha256") != base["preregister_sha256"] or prior.get("data_root") != root_arg:
                    raise RuntimeError("P5u confirmation completed pool receipt mismatch")
                output["confirmation"][root_arg] = result
                all_pass &= all(result["gates"].values())
                continue
            started = output_root / f"pool_{ordinal:03d}_started.json"
            if not started.exists():
                _atomic_write_once(started, {**base, "stage": "pool_started", "pool_ordinal": ordinal, "data_root": root_arg, "audit_path": audit_arg})
            audit = _load_json(_resolve(project, audit_arg))
            if not audit.get("valid_for_frozen_inference"):
                raise RuntimeError(f"invalid P5u confirmation audit: {audit_arg}")
            dataset = PairedPowerOracleDataset(_power_files(_resolve(project, root_arg)), np.arange(3780, dtype=np.int64))
            try:
                begun = time.perf_counter()
                benchmark_indices = fixed_benchmark_indices(len(dataset), int(protocol["execution"]["batch_size"]))
                full_analytic, chunk_guards = _load_or_collect_chunked(output_root, ordinal, dataset, device, base, int(protocol["detached_chunk_contract"]["analytic_chunk_positions"]))
                benchmark_analytic = _slice_role(full_analytic, len(benchmark_indices))
                post_analytic_guard = assert_exclusive_gpu()
                rows = []
                for seed_value in protocol["base_candidate"]["source_model_seeds"]:
                    seed = int(seed_value)
                    completed = output_root / f"pool_{ordinal:03d}_model_{seed}_completed.json"
                    model_started = output_root / f"pool_{ordinal:03d}_model_{seed}_started.json"
                    if completed.exists():
                        rows.append(_load_json(completed)["result"])
                        continue
                    if not model_started.exists():
                        _atomic_write_once(model_started, {**base, "stage": "model_started", "pool_ordinal": ordinal, "source_model_seed": seed})
                    else:
                        _atomic_write_once(output_root / f"pool_{ordinal:03d}_model_{seed}_recovery_attempt_{attempt:03d}.json", {**base, "stage": "model_recompute_unobserved_unit", "pool_ordinal": ordinal, "source_model_seed": seed})
                    row = _one_model(project, dataset, full_analytic, Subset(dataset, benchmark_indices.tolist()), benchmark_analytic, freeze, seed, int(protocol["execution"]["warmup_runs"]), int(protocol["execution"]["timed_repetitions"]), device)
                    _atomic_write_once(completed, {**base, "stage": "model_completed", "pool_ordinal": ordinal, "source_model_seed": seed, "result": row})
                    rows.append(row)
                gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"]))
                gates["all_analytic_chunk_guards_clean"] = all(not guard["external_python_processes"] for guard in chunk_guards)
                gates["post_analytic_guard_clean"] = not post_analytic_guard["external_python_processes"]
                all_pass &= all(gates.values())
                result = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_chunk_positions": int(protocol["detached_chunk_contract"]["analytic_chunk_positions"]), "analytic_chunk_count": len(chunk_guards), "analytic_wall_seconds": float(time.perf_counter() - begun), "analytic_chunk_guards": chunk_guards, "post_analytic_guard": post_analytic_guard, "rows": rows, "gates": gates}
                output["confirmation"][root_arg] = result
                _atomic_write_once(completed_pool, {**base, "stage": "pool_completed", "pool_ordinal": ordinal, "data_root": root_arg, "result": result})
            finally:
                dataset.close()
        post_guard = assert_exclusive_gpu()
        output["environment"]["post_execution_guard"] = post_guard
        output["environment_window_receipt_complete"] = not environment["guard"]["external_python_processes"] and not post_guard["external_python_processes"]
        output["confirmation_passed"] = bool(all_pass and output["environment_window_receipt_complete"])
    except Exception as error:
        failure = error
        output["execution_exception"] = {"type": type(error).__name__, "message": str(error)}
        output["confirmation_passed"] = False
    finally:
        output["environment"]["restore_default"] = restore_default_if_changed(environment)
        _atomic_write_once(output_root / f"attempt_{attempt:03d}_restore.json", {**base, "stage": "restore", "attempt": attempt, "restore_default": output["environment"]["restore_default"]})
    if failure is not None:
        _atomic_write_once(output_root / f"attempt_{attempt:03d}_abort.json", output)
        raise failure
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_completed.json", {**base, "stage": "attempt_completed", "attempt": attempt, "confirmation_passed": output["confirmation_passed"]})
    _atomic_write_once(output_root / "result.json", output)
    print(json.dumps({"confirmation_passed": output["confirmation_passed"], "attempt": attempt}, indent=2))


if __name__ == "__main__":
    main()
