"""P5q: relocation-safe, fine-grained checkpointed P5e cost executor.

Only execution durability and path portability differ from P5p. Scientific
inputs, outputs, batch size, repetitions and gates remain frozen.
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

from cj0724.p5b1_jnr_physical_oracle import FeatureRole, PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices
from cj0724.p5m_authorized_window_shared_dense_execution import exclusive_window_receipt, restore_default_if_changed


ROLE_FIELDS = ("feature", "composition", "target", "valid", "identities", "power", "combination")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(project: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"P5q immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise RuntimeError(f"P5q receipt appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_role(path: Path, full: FeatureRole, benchmark: FeatureRole) -> str:
    if path.exists():
        raise RuntimeError(f"P5q immutable analytic checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    arrays = {f"full_{field}": getattr(full, field) for field in ROLE_FIELDS}
    arrays.update({f"benchmark_{field}": getattr(benchmark, field) for field in ROLE_FIELDS})
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise RuntimeError(f"P5q analytic checkpoint appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _load_role_checkpoint(path: Path) -> tuple[FeatureRole, FeatureRole]:
    with np.load(path, allow_pickle=False) as payload:
        expected = {f"{prefix}_{field}" for prefix in ("full", "benchmark") for field in ROLE_FIELDS}
        if set(payload.files) != expected:
            raise RuntimeError("P5q analytic checkpoint schema mismatch")
        roles = []
        for prefix in ("full", "benchmark"):
            roles.append(FeatureRole(**{field: np.asarray(payload[f"{prefix}_{field}"]) for field in ROLE_FIELDS}))
    return roles[0], roles[1]


def _role_equal(left: FeatureRole, right: FeatureRole) -> bool:
    for field in ROLE_FIELDS:
        a, b = getattr(left, field), getattr(right, field)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if np.issubdtype(a.dtype, np.floating):
            if not np.array_equal(a, b, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def validate_protocol(project: Path, protocol_path: Path, freeze_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = ("frozen_artifacts", "base_candidate", "development_data", "execution", "development_gates", "checkpoint_contract")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise RuntimeError(f"P5q protocol missing required fields: {missing}")
    validated = []
    for artifact in protocol["frozen_artifacts"]:
        path = _resolve(project, artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"P5q frozen artifact hash mismatch: {artifact['path']}")
        validated.append(artifact["path"])
    candidate = protocol["base_candidate"]
    if _resolve(project, candidate["freeze_path"]) != freeze_path:
        raise RuntimeError("P5q provided candidate freeze does not match protocol")
    if sha256_file(freeze_path) != candidate["freeze_sha256"]:
        raise RuntimeError("P5q candidate-freeze hash mismatch")
    if list(candidate["source_model_seeds"]) != [3041, 3042, 3043, 3044, 3045]:
        raise RuntimeError("P5q source-model seed contract changed")
    execution = protocol["execution"]
    if (int(execution["batch_size"]), int(execution["warmup_runs"]), int(execution["timed_repetitions"])) != (126, 5, 15):
        raise RuntimeError("P5q fixed timing contract changed")
    contract = protocol["checkpoint_contract"]
    if contract.get("schema") != "P5q-v1" or "selectively rerun" not in contract.get("selection_prohibition", ""):
        raise RuntimeError("P5q checkpoint/recovery contract changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for seed in candidate["source_model_seeds"]:
        key = str(seed)
        if not _resolve(project, freeze["frozen_candidate"]["base_roots"][key]).is_dir():
            raise RuntimeError(f"P5q relocated base root missing for source {seed}")
        if not _resolve(project, freeze["frozen_candidate"]["head_checkpoints"][key]).is_file():
            raise RuntimeError(f"P5q relocated head checkpoint missing for source {seed}")
    return {"protocol": protocol, "validated_artifacts": validated}


def _receipt_base(protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    return {"receipt_schema": "P5q-v1", "created_utc": _utc_now(), "pid": os.getpid(), "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated_artifacts}


def _next_attempt(root: Path) -> int:
    existing = []
    for path in root.glob("attempt_*_launch.json"):
        try:
            existing.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(existing, default=-1) + 1


def _validated_completed_model(path: Path, base: dict[str, object], pool_ordinal: int, source_seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("preregister_sha256") != base["preregister_sha256"] or payload.get("pool_ordinal") != pool_ordinal or payload.get("source_model_seed") != source_seed:
        raise RuntimeError(f"P5q completed model receipt identity/hash mismatch: {path.name}")
    if "result" not in payload:
        raise RuntimeError(f"P5q completed model receipt lacks result: {path.name}")
    return payload["result"]


def _self_test(root: Path, protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    base = _receipt_base(protocol_path, validated_artifacts)
    full = FeatureRole(
        feature=np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        composition=np.array([[1, 0], [0, 1]], dtype=np.int64),
        target=np.array([0.5, 1.5], dtype=np.float32),
        valid=np.array([True, False]), identities=np.array([7, 8], dtype=np.uint32),
        power=np.array([10, 30], dtype=np.int16), combination=np.array(["A", "B"]),
    )
    benchmark = FeatureRole(**{field: getattr(full, field)[:1].copy() for field in ROLE_FIELDS})
    checkpoint = root / "pool_000_analytic.npz"
    digest = _atomic_save_role(checkpoint, full, benchmark)
    loaded_full, loaded_benchmark = _load_role_checkpoint(checkpoint)
    exact = _role_equal(full, loaded_full) and _role_equal(benchmark, loaded_benchmark)
    if not exact:
        raise RuntimeError("P5q synthetic analytic checkpoint did not round-trip exactly")
    _atomic_write_once(root / "pool_000_model_3041_started.json", {**base, "stage": "model_started", "pool_ordinal": 0, "source_model_seed": 3041})
    classification = "recoverable_unobserved_unit" if not (root / "pool_000_model_3041_completed.json").exists() else "invalid"
    inspection = {"analytic_roundtrip_exact": exact, "analytic_sha256": digest, "classification": classification, "data_opened": False, "cuda_opened": False}
    _atomic_write_once(root / "recovery_validation.json", {**base, "stage": "recovery_validation", "inspection": inspection})
    return inspection


def _load_or_collect_analytic(root: Path, ordinal: int, full_dataset, benchmark_dataset, device: torch.device, base: dict[str, object]) -> tuple[FeatureRole, FeatureRole, str]:
    checkpoint = root / f"pool_{ordinal:03d}_analytic.npz"
    completed = root / f"pool_{ordinal:03d}_analytic_completed.json"
    started = root / f"pool_{ordinal:03d}_analytic_started.json"
    if completed.exists():
        receipt = json.loads(completed.read_text(encoding="utf-8"))
        digest = sha256_file(checkpoint)
        if receipt.get("preregister_sha256") != base["preregister_sha256"] or receipt.get("checkpoint_sha256") != digest:
            raise RuntimeError(f"P5q analytic receipt/hash mismatch for pool {ordinal}")
        full, benchmark = _load_role_checkpoint(checkpoint)
        return full, benchmark, "validated_completed_checkpoint"
    if checkpoint.exists():
        full, benchmark = _load_role_checkpoint(checkpoint)
        digest = sha256_file(checkpoint)
        _atomic_write_once(completed, {**base, "stage": "analytic_completed_after_atomic_checkpoint_recovery", "pool_ordinal": ordinal, "checkpoint_sha256": digest})
        return full, benchmark, "recovered_atomic_checkpoint"
    if not started.exists():
        _atomic_write_once(started, {**base, "stage": "analytic_started", "pool_ordinal": ordinal})
    else:
        recovery = root / f"pool_{ordinal:03d}_analytic_recovery_{int(time.time())}.json"
        _atomic_write_once(recovery, {**base, "stage": "analytic_recompute_unobserved_unit", "pool_ordinal": ordinal})
    full = collect_analytic_role(full_dataset, device)
    benchmark = collect_analytic_role(benchmark_dataset, device)
    digest = _atomic_save_role(checkpoint, full, benchmark)
    _atomic_write_once(completed, {**base, "stage": "analytic_completed", "pool_ordinal": ordinal, "checkpoint_sha256": digest})
    return full, benchmark, "new_checkpoint"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--self-test-checkpoints", action="store_true")
    parser.add_argument("--receipt-root")
    parser.add_argument("--data-roots", nargs="+")
    parser.add_argument("--audits", nargs="+")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    protocol_path, freeze_path = _resolve(project, args.preregister), _resolve(project, args.p5e_freeze)
    validated = validate_protocol(project, protocol_path, freeze_path)
    if args.validate_protocol:
        print(json.dumps({"protocol_valid": True, "project_root": str(project), "validated_artifacts": validated["validated_artifacts"]}, indent=2))
        return
    if args.self_test_checkpoints:
        if not args.receipt_root or args.data_roots or args.audits or args.output_root:
            raise RuntimeError("P5q checkpoint self-test needs only --receipt-root")
        inspection = _self_test(_resolve(project, args.receipt_root), protocol_path, validated["validated_artifacts"])
        print(json.dumps({"checkpoint_self_test_valid": True, "inspection": inspection}, indent=2))
        return
    if not args.data_roots or not args.audits or not args.output_root or len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5q execution requires exactly three data roots, three audits and one output root")

    protocol = validated["protocol"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    output_root = _resolve(project, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "result.json").exists():
        raise RuntimeError("P5q final result already exists; selective rerun is prohibited")
    base = _receipt_base(protocol_path, validated["validated_artifacts"])
    attempt = _next_attempt(output_root)
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_launch.json", {**base, "stage": "launch", "attempt": attempt, "project_root": str(project), "data_roots": args.data_roots, "audit_paths": args.audits})
    environment = exclusive_window_receipt()
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_environment.json", {**base, "stage": "environment_ready", "attempt": attempt, "environment": environment})
    output: dict[str, object] = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated["validated_artifacts"], "project_root": str(project), "attempt": attempt, "environment": environment, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    failure: Exception | None = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("P5q cost execution requires CUDA")
        all_pass = True
        for ordinal, (root_arg, audit_arg) in enumerate(zip(args.data_roots, args.audits)):
            pool_completed = output_root / f"pool_{ordinal:03d}_completed.json"
            if pool_completed.exists():
                prior = json.loads(pool_completed.read_text(encoding="utf-8"))
                if prior.get("preregister_sha256") != base["preregister_sha256"] or prior.get("data_root") != root_arg:
                    raise RuntimeError(f"P5q completed pool receipt mismatch for pool {ordinal}")
                pool_result = prior["result"]
                output["development"][root_arg] = pool_result
                all_pass &= all(pool_result["gates"].values())
                continue
            pool_started = output_root / f"pool_{ordinal:03d}_started.json"
            if not pool_started.exists():
                _atomic_write_once(pool_started, {**base, "stage": "pool_started", "pool_ordinal": ordinal, "data_root": root_arg, "audit_path": audit_arg})
            audit = json.loads(_resolve(project, audit_arg).read_text(encoding="utf-8"))
            if not audit["valid_for_frozen_inference"]:
                raise RuntimeError(f"invalid P5q audit: {audit_arg}")
            full_dataset = PairedPowerOracleDataset(_power_files(_resolve(project, root_arg)), np.arange(3780, dtype=np.int64))
            benchmark_indices = fixed_benchmark_indices(len(full_dataset), int(protocol["execution"]["batch_size"]))
            benchmark_dataset = PairedPowerOracleDataset(_power_files(_resolve(project, root_arg)), benchmark_indices)
            try:
                started_time = time.perf_counter()
                full_analytic, benchmark_analytic, analytic_origin = _load_or_collect_analytic(output_root, ordinal, full_dataset, benchmark_dataset, device, base)
                post_analytic_guard = assert_exclusive_gpu()
                rows = []
                for seed_value in protocol["base_candidate"]["source_model_seeds"]:
                    seed = int(seed_value)
                    completed = output_root / f"pool_{ordinal:03d}_model_{seed}_completed.json"
                    started_receipt = output_root / f"pool_{ordinal:03d}_model_{seed}_started.json"
                    if completed.exists():
                        rows.append(_validated_completed_model(completed, base, ordinal, seed))
                        continue
                    if not started_receipt.exists():
                        _atomic_write_once(started_receipt, {**base, "stage": "model_started", "pool_ordinal": ordinal, "source_model_seed": seed})
                    else:
                        recovery = output_root / f"pool_{ordinal:03d}_model_{seed}_recovery_attempt_{attempt:03d}.json"
                        _atomic_write_once(recovery, {**base, "stage": "model_recompute_unobserved_unit", "pool_ordinal": ordinal, "source_model_seed": seed, "attempt": attempt})
                    row = _one_model(project, full_dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze, seed, int(protocol["execution"]["warmup_runs"]), int(protocol["execution"]["timed_repetitions"]), device)
                    _atomic_write_once(completed, {**base, "stage": "model_completed", "pool_ordinal": ordinal, "source_model_seed": seed, "result": row})
                    rows.append(row)
                gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"]))
                gates["post_analytic_guard_clean"] = not post_analytic_guard["external_python_processes"]
                all_pass &= all(gates.values())
                pool_result = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_origin": analytic_origin, "attempt_wall_seconds_after_analytic_load_or_collection": float(time.perf_counter() - started_time), "post_analytic_guard": post_analytic_guard, "rows": rows, "gates": gates}
                output["development"][root_arg] = pool_result
                _atomic_write_once(pool_completed, {**base, "stage": "pool_completed", "pool_ordinal": ordinal, "data_root": root_arg, "result": pool_result})
            finally:
                full_dataset.close()
                benchmark_dataset.close()
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
        _atomic_write_once(output_root / f"attempt_{attempt:03d}_restore.json", {**base, "stage": "restore", "attempt": attempt, "restore_default": output["environment"]["restore_default"]})
    if failure is not None:
        _atomic_write_once(output_root / f"attempt_{attempt:03d}_abort.json", output)
        raise failure
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_completed.json", {**base, "stage": "attempt_completed", "attempt": attempt, "development_passed": output["development_passed"]})
    _atomic_write_once(output_root / "result.json", output)
    print(json.dumps({"development_passed": output["development_passed"], "attempt": attempt, "environment": output["environment"]}, indent=2))


if __name__ == "__main__":
    main()
