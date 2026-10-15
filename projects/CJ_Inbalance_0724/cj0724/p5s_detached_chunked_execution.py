"""P5s: detached-supervised P5e cost executor with deterministic analytic chunks."""
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

from cj0724.p5b1_jnr_physical_oracle import FeatureRole, PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices
from cj0724.p5m_authorized_window_shared_dense_execution import exclusive_window_receipt, restore_default_if_changed
from cj0724.p5q_relocation_safe_checkpoint_execution import ROLE_FIELDS, _next_attempt, _resolve, validate_protocol as validate_p5q_protocol


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists(): raise RuntimeError(f"P5s immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
        if path.exists(): raise RuntimeError(f"P5s receipt appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _save_role(path: Path, role: FeatureRole) -> str:
    if path.exists(): raise RuntimeError(f"P5s immutable analytic chunk already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **{field: getattr(role, field) for field in ROLE_FIELDS})
            handle.flush(); os.fsync(handle.fileno())
        if path.exists(): raise RuntimeError(f"P5s analytic chunk appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return sha256_file(path)


def _load_role(path: Path) -> FeatureRole:
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != set(ROLE_FIELDS): raise RuntimeError("P5s analytic chunk schema mismatch")
        return FeatureRole(**{field: np.asarray(values[field]) for field in ROLE_FIELDS})


def _combine_roles(roles: list[FeatureRole], expected_rows: int) -> FeatureRole:
    if not roles: raise RuntimeError("P5s requires at least one analytic chunk")
    result = FeatureRole(**{field: np.concatenate([getattr(role, field) for role in roles]) for field in ROLE_FIELDS})
    if result.feature.shape[0] != expected_rows: raise RuntimeError("P5s analytic chunk row count mismatch")
    return result


def _slice_role(role: FeatureRole, rows: int) -> FeatureRole:
    return FeatureRole(**{field: getattr(role, field)[:rows].copy() for field in ROLE_FIELDS})


def validate_protocol(project: Path, protocol_path: Path, freeze_path: Path) -> dict[str, object]:
    validated = validate_p5q_protocol(project, protocol_path, freeze_path)
    protocol = validated["protocol"]
    durability = protocol.get("detached_chunk_contract", {})
    if int(durability.get("analytic_chunk_positions", -1)) != 630:
        raise RuntimeError("P5s analytic chunk contract changed")
    if durability.get("supervisor_schema") != "P5s-detached-supervisor-v1":
        raise RuntimeError("P5s detached supervisor contract changed")
    return validated


def _receipt_base(protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    return {"receipt_schema": "P5s-v1", "created_utc": _utc_now(), "pid": os.getpid(), "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated_artifacts}


def _load_or_collect_chunked(root: Path, ordinal: int, dataset, device: torch.device, base: dict[str, object], chunk_positions: int) -> tuple[FeatureRole, list[dict[str, object]]]:
    chunks: list[FeatureRole] = []
    guards: list[dict[str, object]] = []
    for chunk_ordinal, start in enumerate(range(0, len(dataset), chunk_positions)):
        stop = min(start + chunk_positions, len(dataset))
        checkpoint = root / f"pool_{ordinal:03d}_analytic_chunk_{chunk_ordinal:03d}.npz"
        completed = root / f"pool_{ordinal:03d}_analytic_chunk_{chunk_ordinal:03d}_completed.json"
        started = root / f"pool_{ordinal:03d}_analytic_chunk_{chunk_ordinal:03d}_started.json"
        if completed.exists():
            receipt = json.loads(completed.read_text(encoding="utf-8"))
            digest = sha256_file(checkpoint)
            if receipt.get("preregister_sha256") != base["preregister_sha256"] or receipt.get("checkpoint_sha256") != digest or receipt.get("positions") != [start, stop]:
                raise RuntimeError(f"P5s analytic chunk receipt/hash mismatch: {chunk_ordinal}")
            role = _load_role(checkpoint)
        elif checkpoint.exists():
            role = _load_role(checkpoint)
            digest = sha256_file(checkpoint)
            _atomic_write_once(completed, {**base, "stage": "analytic_chunk_completed_after_atomic_checkpoint_recovery", "pool_ordinal": ordinal, "chunk_ordinal": chunk_ordinal, "positions": [start, stop], "checkpoint_sha256": digest})
        else:
            if not started.exists():
                _atomic_write_once(started, {**base, "stage": "analytic_chunk_started", "pool_ordinal": ordinal, "chunk_ordinal": chunk_ordinal, "positions": [start, stop]})
            else:
                _atomic_write_once(root / f"pool_{ordinal:03d}_analytic_chunk_{chunk_ordinal:03d}_recovery_{int(time.time())}.json", {**base, "stage": "analytic_chunk_recompute_unobserved_unit", "pool_ordinal": ordinal, "chunk_ordinal": chunk_ordinal, "positions": [start, stop]})
            role = collect_analytic_role(Subset(dataset, list(range(start, stop))), device)
            digest = _save_role(checkpoint, role)
            _atomic_write_once(completed, {**base, "stage": "analytic_chunk_completed", "pool_ordinal": ordinal, "chunk_ordinal": chunk_ordinal, "positions": [start, stop], "checkpoint_sha256": digest})
        if role.feature.shape[0] != stop - start: raise RuntimeError(f"P5s chunk length mismatch: {chunk_ordinal}")
        chunks.append(role)
        guards.append(assert_exclusive_gpu())
    return _combine_roles(chunks, len(dataset)), guards


def _self_test(root: Path, protocol_path: Path, validated_artifacts: list[str]) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    base = _receipt_base(protocol_path, validated_artifacts)
    role = FeatureRole(feature=np.arange(20, dtype=np.float32).reshape(5, 4), composition=np.ones((5, 6), dtype=np.float32), target=np.zeros((5, 3), dtype=np.float32), valid=np.ones((5, 3), dtype=bool), identities=np.arange(5, dtype=np.uint32), power=np.full(5, 10, dtype=np.int16), combination=np.arange(5, dtype=np.int16))
    first, second = _slice_role(role, 2), FeatureRole(**{field: getattr(role, field)[2:].copy() for field in ROLE_FIELDS})
    p0, p1 = root / "chunk_000.npz", root / "chunk_001.npz"
    h0, h1 = _save_role(p0, first), _save_role(p1, second)
    exact = all(np.array_equal(getattr(_combine_roles([_load_role(p0), _load_role(p1)], 5), field), getattr(role, field)) for field in ROLE_FIELDS)
    if not exact: raise RuntimeError("P5s synthetic chunk round-trip mismatch")
    inspection = {"chunk_roundtrip_exact": exact, "chunk_sha256": [h0, h1], "data_opened": False, "cuda_opened": False}
    _atomic_write_once(root / "self_test.json", {**base, "stage": "self_test", "inspection": inspection})
    return inspection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True); parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--validate-protocol", action="store_true"); parser.add_argument("--self-test-chunks", action="store_true"); parser.add_argument("--receipt-root")
    parser.add_argument("--data-roots", nargs="+"); parser.add_argument("--audits", nargs="+"); parser.add_argument("--output-root")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    protocol_path, freeze_path = _resolve(project, args.preregister), _resolve(project, args.p5e_freeze)
    validated = validate_protocol(project, protocol_path, freeze_path)
    if args.validate_protocol:
        print(json.dumps({"protocol_valid": True, "project_root": str(project)}, indent=2)); return
    if args.self_test_chunks:
        if not args.receipt_root or args.data_roots or args.audits or args.output_root: raise RuntimeError("P5s chunk self-test needs only --receipt-root")
        print(json.dumps({"chunk_self_test_valid": True, "inspection": _self_test(_resolve(project, args.receipt_root), protocol_path, validated["validated_artifacts"])}, indent=2)); return
    if not args.data_roots or not args.audits or not args.output_root or len(args.data_roots) != 3 or len(args.audits) != 3: raise RuntimeError("P5s execution requires exactly three data roots, audits and output root")
    protocol, freeze = validated["protocol"], json.loads(freeze_path.read_text(encoding="utf-8"))
    output_root = _resolve(project, args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "result.json").exists(): raise RuntimeError("P5s final result already exists; selective rerun is prohibited")
    base = _receipt_base(protocol_path, validated["validated_artifacts"]); attempt = _next_attempt(output_root)
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_launch.json", {**base, "stage": "launch", "attempt": attempt, "data_roots": args.data_roots, "audit_paths": args.audits})
    environment = exclusive_window_receipt()
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_environment.json", {**base, "stage": "environment_ready", "attempt": attempt, "environment": environment})
    output: dict[str, object] = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "attempt": attempt, "environment": environment, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    failure: Exception | None = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda": raise RuntimeError("P5s cost execution requires CUDA")
        all_pass = True
        for ordinal, (root_arg, audit_arg) in enumerate(zip(args.data_roots, args.audits)):
            pool_completed = output_root / f"pool_{ordinal:03d}_completed.json"
            if pool_completed.exists():
                prior = json.loads(pool_completed.read_text(encoding="utf-8")); pool_result = prior["result"]
                if prior.get("preregister_sha256") != base["preregister_sha256"] or prior.get("data_root") != root_arg: raise RuntimeError("P5s completed pool receipt mismatch")
                output["development"][root_arg] = pool_result; all_pass &= all(pool_result["gates"].values()); continue
            started = output_root / f"pool_{ordinal:03d}_started.json"
            if not started.exists(): _atomic_write_once(started, {**base, "stage": "pool_started", "pool_ordinal": ordinal, "data_root": root_arg, "audit_path": audit_arg})
            audit = json.loads(_resolve(project, audit_arg).read_text(encoding="utf-8"))
            if not audit["valid_for_frozen_inference"]: raise RuntimeError(f"invalid P5s audit: {audit_arg}")
            dataset = PairedPowerOracleDataset(_power_files(_resolve(project, root_arg)), np.arange(3780, dtype=np.int64))
            try:
                t0 = time.perf_counter(); benchmark_indices = fixed_benchmark_indices(len(dataset), int(protocol["execution"]["batch_size"]))
                full_analytic, chunk_guards = _load_or_collect_chunked(output_root, ordinal, dataset, device, base, int(protocol["detached_chunk_contract"]["analytic_chunk_positions"]))
                benchmark_dataset = Subset(dataset, benchmark_indices.tolist()); benchmark_analytic = _slice_role(full_analytic, len(benchmark_indices)); post_analytic_guard = assert_exclusive_gpu()
                rows = []
                for seed_value in protocol["base_candidate"]["source_model_seeds"]:
                    seed = int(seed_value); completed = output_root / f"pool_{ordinal:03d}_model_{seed}_completed.json"; model_started = output_root / f"pool_{ordinal:03d}_model_{seed}_started.json"
                    if completed.exists(): rows.append(json.loads(completed.read_text(encoding="utf-8"))["result"]); continue
                    if not model_started.exists(): _atomic_write_once(model_started, {**base, "stage": "model_started", "pool_ordinal": ordinal, "source_model_seed": seed})
                    else: _atomic_write_once(output_root / f"pool_{ordinal:03d}_model_{seed}_recovery_attempt_{attempt:03d}.json", {**base, "stage": "model_recompute_unobserved_unit", "pool_ordinal": ordinal, "source_model_seed": seed})
                    row = _one_model(project, dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze, seed, int(protocol["execution"]["warmup_runs"]), int(protocol["execution"]["timed_repetitions"]), device)
                    _atomic_write_once(completed, {**base, "stage": "model_completed", "pool_ordinal": ordinal, "source_model_seed": seed, "result": row}); rows.append(row)
                gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"])); gates["all_analytic_chunk_guards_clean"] = all(not item["external_python_processes"] for item in chunk_guards); gates["post_analytic_guard_clean"] = not post_analytic_guard["external_python_processes"]
                all_pass &= all(gates.values())
                pool_result = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_chunk_positions": int(protocol["detached_chunk_contract"]["analytic_chunk_positions"]), "analytic_chunk_count": len(chunk_guards), "analytic_wall_seconds": float(time.perf_counter() - t0), "analytic_chunk_guards": chunk_guards, "post_analytic_guard": post_analytic_guard, "rows": rows, "gates": gates}
                output["development"][root_arg] = pool_result
                _atomic_write_once(pool_completed, {**base, "stage": "pool_completed", "pool_ordinal": ordinal, "data_root": root_arg, "result": pool_result})
            finally: dataset.close()
        post_guard = assert_exclusive_gpu(); output["environment"]["post_execution_guard"] = post_guard; output["environment_window_receipt_complete"] = not environment["guard"]["external_python_processes"] and not post_guard["external_python_processes"]; output["development_passed"] = bool(all_pass and output["environment_window_receipt_complete"])
    except Exception as error:
        failure = error; output["execution_exception"] = {"type": type(error).__name__, "message": str(error)}; output["development_passed"] = False
    finally:
        output["environment"]["restore_default"] = restore_default_if_changed(environment); _atomic_write_once(output_root / f"attempt_{attempt:03d}_restore.json", {**base, "stage": "restore", "attempt": attempt, "restore_default": output["environment"]["restore_default"]})
    if failure is not None:
        _atomic_write_once(output_root / f"attempt_{attempt:03d}_abort.json", output); raise failure
    _atomic_write_once(output_root / f"attempt_{attempt:03d}_completed.json", {**base, "stage": "attempt_completed", "attempt": attempt, "development_passed": output["development_passed"]}); _atomic_write_once(output_root / "result.json", output)
    print(json.dumps({"development_passed": output["development_passed"], "attempt": attempt}, indent=2))


if __name__ == "__main__": main()
