"""P5m: user-authorized-window successor to the stopped P5l cost run.

The frozen candidate and P5l exactness/latency unit are reused unchanged.
Only fresh identities and a documented authorized exclusive CUDA window differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices


PARENT_HASH_KEYS = (
    "P5e_candidate_freeze", "P5l_preregister", "P5l_outcome_freeze",
    "P5l_runner", "P5l_generator", "audit_runner",
)


def _command(args: list[str]) -> dict[str, object]:
    completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"args": args, "returncode": int(completed.returncode), "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def exclusive_window_receipt() -> dict[str, object]:
    """Record a permitted mode attempt and require a clean process-level guard."""
    before = _command(["nvidia-smi", "--query-gpu=index,name,compute_mode,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"])
    mode_attempt = _command(["nvidia-smi", "-c", "EXCLUSIVE_PROCESS"])
    after = _command(["nvidia-smi", "--query-gpu=index,name,compute_mode,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"])
    return {"before": before, "mode_attempt": mode_attempt, "after": after, "guard": assert_exclusive_gpu()}


def restore_default_if_changed(receipt: dict[str, object]) -> dict[str, object]:
    mode_attempt = dict(receipt["mode_attempt"])
    if int(mode_attempt["returncode"]) == 0:
        return _command(["nvidia-smi", "-c", "DEFAULT"])
    return {"skipped": True, "reason": "EXCLUSIVE_PROCESS mode was not changed"}


def _verify_parent(project: Path, protocol: dict, freeze_path: Path) -> dict:
    parent = protocol["frozen_parent"]
    for key in PARENT_HASH_KEYS:
        if sha256_file(project / parent[key]) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"P5m frozen parent hash mismatch: {key}")
    if sha256_file(freeze_path) != parent["P5e_candidate_freeze_sha256"]:
        raise RuntimeError("P5m frozen P5e candidate-freeze hash mismatch")
    return parent


def _window_complete(receipt: dict[str, object], post_guard: dict[str, object]) -> bool:
    return not receipt["guard"]["external_python_processes"] and not post_guard["external_python_processes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preregister", "p5e-freeze", "data-roots", "audits", "output-root"):
        parser.add_argument(f"--{name}", nargs="+" if name in ("data-roots", "audits") else None, required=True)
    args = parser.parse_args()
    project = Path.cwd()
    protocol_path, freeze_path = project / args.preregister, project / args.p5e_freeze
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    parent = _verify_parent(project, protocol, freeze_path)
    if len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5m development requires exactly three fresh identity pools")

    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    environment = exclusive_window_receipt()
    output: dict[str, object] = {
        "experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path),
        "environment": environment, "development": {},
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("P5m cost execution requires CUDA")
        all_pass = True
        for root_arg, audit_arg in zip(args.data_roots, args.audits):
            audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
            if not audit["valid_for_frozen_inference"]:
                raise RuntimeError(f"invalid P5m audit: {audit_arg}")
            full_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), np.arange(3780, dtype=np.int64))
            benchmark_indices = fixed_benchmark_indices(len(full_dataset), int(protocol["execution"]["batch_size"]))
            benchmark_dataset = PairedPowerOracleDataset(_power_files(project / root_arg), benchmark_indices)
            started = time.perf_counter()
            full_analytic = collect_analytic_role(full_dataset, device)
            benchmark_analytic = collect_analytic_role(benchmark_dataset, device)
            post_analytic_guard = assert_exclusive_gpu()
            rows = [
                _one_model(project, full_dataset, full_analytic, benchmark_dataset, benchmark_analytic, freeze, int(seed),
                           int(protocol["execution"]["warmup_runs"]), int(protocol["execution"]["timed_repetitions"]), device)
                for seed in parent["source_model_seeds"]
            ]
            gates = _gates(rows, float(protocol["development_gates"]["all_data_seeds_and_models_median_latency_ratio_shared_dense_over_reference_at_most"]))
            gates["post_analytic_guard_clean"] = not post_analytic_guard["external_python_processes"]
            all_pass &= all(gates.values())
            output["development"][root_arg] = {
                "audit": audit, "benchmark_indices": benchmark_indices.tolist(),
                "analytic_feature_seconds": float(time.perf_counter() - started), "post_analytic_guard": post_analytic_guard,
                "rows": rows, "gates": gates,
            }
            full_dataset.close(); benchmark_dataset.close()
        post_guard = assert_exclusive_gpu()
        output["environment"]["post_execution_guard"] = post_guard
        output["environment_window_receipt_complete"] = _window_complete(environment, post_guard)
        output["development_passed"] = bool(all_pass and output["environment_window_receipt_complete"])
    except Exception as error:
        output["execution_exception"] = {"type": type(error).__name__, "message": str(error)}
        output["development_passed"] = False
        (output_root / "pre_result_abort.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        raise
    finally:
        output["environment"]["restore_default"] = restore_default_if_changed(environment)
    (output_root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": output["development_passed"], "environment": output["environment"]}, indent=2))


if __name__ == "__main__":
    main()
