"""P5p: checkpointed, protocol-generic guarded shared-dense cost executor.

This executor keeps P5o's candidate and timing contract exactly unchanged.  Its
sole change is crash-auditable, immutable receipts: launch, environment,
per-pool start/completion and terminal outcome are atomically written.  It
does not resume a consumed development pool after interruption.
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

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices
from cj0724.p5m_authorized_window_shared_dense_execution import exclusive_window_receipt, restore_default_if_changed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    """Write an immutable JSON receipt; never silently replace evidence."""
    if path.exists():
        raise RuntimeError(f"P5p receipt already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise RuntimeError(f"P5p receipt appeared concurrently: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_protocol(project: Path, protocol_path: Path, freeze_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = ("frozen_artifacts", "base_candidate", "development_data", "execution", "development_gates", "checkpoint_contract")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise RuntimeError(f"P5p protocol missing required fields: {missing}")
    validated: list[str] = []
    for artifact in protocol["frozen_artifacts"]:
        path = project / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"P5p frozen artifact hash mismatch: {artifact['path']}")
        validated.append(artifact["path"])
    candidate = protocol["base_candidate"]
    if (project / candidate["freeze_path"]).resolve() != freeze_path.resolve():
        raise RuntimeError("P5p provided candidate freeze does not match protocol")
    if sha256_file(freeze_path) != candidate["freeze_sha256"]:
        raise RuntimeError("P5p candidate-freeze hash mismatch")
    if list(candidate["source_model_seeds"]) != [3041, 3042, 3043, 3044, 3045]:
        raise RuntimeError("P5p source-model seed contract changed")
    execution = protocol["execution"]
    if int(execution["batch_size"]) != 126 or int(execution["warmup_runs"]) != 5 or int(execution["timed_repetitions"]) != 15:
        raise RuntimeError("P5p fixed timing contract changed")
    contract = protocol["checkpoint_contract"]
    if contract.get("allow_resume_after_interruption") is not False:
        raise RuntimeError("P5p must prohibit resuming a consumed identity pool")
    return {"protocol": protocol, "validated_artifacts": validated}


def _receipt_base(protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    return {"receipt_schema": "P5p-v1", "created_utc": _utc_now(), "pid": os.getpid(), "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated_artifacts}


def _self_test_receipts(root: Path, protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    """Exercise receipt durability only; this opens neither data nor CUDA."""
    root.mkdir(parents=True, exist_ok=False)
    base = _receipt_base(protocol_path, validated_artifacts)
    _atomic_write_once(root / "launch_receipt.json", {**base, "stage": "launch", "self_test": True})
    _atomic_write_once(root / "pool_000_started.json", {**base, "stage": "pool_started", "pool_ordinal": 0, "self_test": True})
    inspection = {"root": str(root), "launch_exists": (root / "launch_receipt.json").is_file(), "started_exists": (root / "pool_000_started.json").is_file(), "completed_exists": (root / "pool_000_completed.json").exists(), "resume_allowed": False, "classification": "interrupted_before_pool_completion_no_resume"}
    _atomic_write_once(root / "recovery_validation.json", {**base, "stage": "recovery_validation", "inspection": inspection, "self_test": True})
    return inspection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--self-test-receipts", action="store_true")
    parser.add_argument("--receipt-root")
    parser.add_argument("--data-roots", nargs="+")
    parser.add_argument("--audits", nargs="+")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    project = Path.cwd()
    protocol_path, freeze_path = project / args.preregister, project / args.p5e_freeze
    validated = validate_protocol(project, protocol_path, freeze_path)
    if args.validate_protocol:
        print(json.dumps({"protocol_valid": True, "validated_artifacts": validated["validated_artifacts"]}, indent=2))
        return
    if args.self_test_receipts:
        if not args.receipt_root or args.data_roots or args.audits or args.output_root:
            raise RuntimeError("P5p receipt self-test needs only --receipt-root")
        inspection = _self_test_receipts(project / args.receipt_root, protocol_path, validated["validated_artifacts"])
        print(json.dumps({"receipt_self_test_valid": True, "inspection": inspection}, indent=2))
        return
    if not args.data_roots or not args.audits or not args.output_root:
        raise RuntimeError("P5p execution requires data roots, audits and output root")
    if len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5p development requires exactly three fresh identity pools")

    protocol, freeze = validated["protocol"], json.loads(freeze_path.read_text(encoding="utf-8"))
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    base = _receipt_base(protocol_path, validated["validated_artifacts"])
    _atomic_write_once(output_root / "launch_receipt.json", {**base, "stage": "launch", "data_roots": list(args.data_roots), "audit_paths": list(args.audits), "resume_allowed": False})
    environment = exclusive_window_receipt()
    _atomic_write_once(output_root / "environment_receipt.json", {**base, "stage": "environment_ready", "environment": environment})
    output: dict[str, object] = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated["validated_artifacts"], "environment": environment, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    failure: Exception | None = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("P5p cost execution requires CUDA")
        all_pass = True
        for ordinal, (root_arg, audit_arg) in enumerate(zip(args.data_roots, args.audits)):
            _atomic_write_once(output_root / f"pool_{ordinal:03d}_started.json", {**base, "stage": "pool_started", "pool_ordinal": ordinal, "data_root": root_arg, "audit_path": audit_arg})
            audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
            if not audit["valid_for_frozen_inference"]:
                raise RuntimeError(f"invalid P5p audit: {audit_arg}")
            full_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
            benchmark_indices = fixed_benchmark_indices(len(full_dataset), int(protocol["execution"]["batch_size"]))
            benchmark_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), benchmark_indices)
            started = time.perf_counter()
            full_analytic = collect_analytic_role(full_dataset, device)
            benchmark_analytic = collect_analytic_role(benchmark_dataset, device)
            post_analytic_guard = assert_exclusive_gpu()
            rows = [_one_model(project, full_dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze, int(seed), int(protocol["execution"]["warmup_runs"]), int(protocol["execution"]["timed_repetitions"]), device) for seed in protocol["base_candidate"]["source_model_seeds"]]
            gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"]))
            gates["post_analytic_guard_clean"] = not post_analytic_guard["external_python_processes"]
            all_pass &= all(gates.values())
            pool_result = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_feature_seconds": float(time.perf_counter() - started), "post_analytic_guard": post_analytic_guard, "rows": rows, "gates": gates}
            output["development"][root_arg] = pool_result
            _atomic_write_once(output_root / f"pool_{ordinal:03d}_completed.json", {**base, "stage": "pool_completed", "pool_ordinal": ordinal, "data_root": root_arg, "result": pool_result})
            full_dataset.close(); benchmark_dataset.close()
        post_guard = assert_exclusive_gpu()
        output["environment"]["post_execution_guard"] = post_guard
        output["environment_window_receipt_complete"] = not environment["guard"]["external_python_processes"] and not post_guard["external_python_processes"]
        output["development_passed"] = bool(all_pass and output["environment_window_receipt_complete"])
    except Exception as error:
        failure = error
        output["execution_exception"] = {"type": type(error).__name__, "message": str(error)}
        output["development_passed"] = False
    finally:
        output["environment"]["restore_default"] = restore_default_if_changed(environment)
        _atomic_write_once(output_root / "restore_receipt.json", {**base, "stage": "restore", "restore_default": output["environment"]["restore_default"]})
    if failure is not None:
        _atomic_write_once(output_root / "pre_result_abort.json", output)
        raise failure
    _atomic_write_once(output_root / "result.json", output)
    print(json.dumps({"development_passed": output["development_passed"], "environment": output["environment"]}, indent=2))


if __name__ == "__main__":
    main()
