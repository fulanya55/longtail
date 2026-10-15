"""P5o: protocol-generic guarded shared-dense cost executor.

Unlike P5m, provenance validation is read directly from a frozen artifact
list.  ``--validate-protocol`` proves that integration without opening data,
initializing CUDA, or creating an output directory.
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
from cj0724.p5h_exclusive_gpu_guard import assert_exclusive_gpu
from cj0724.p5l_minibatch_shared_dense_execution import _gates, _one_model, fixed_benchmark_indices
from cj0724.p5m_authorized_window_shared_dense_execution import exclusive_window_receipt, restore_default_if_changed


def validate_protocol(project: Path, protocol_path: Path, freeze_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = ("frozen_artifacts", "base_candidate", "development_data", "execution", "development_gates")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise RuntimeError(f"P5o protocol missing required fields: {missing}")
    validated = []
    for artifact in protocol["frozen_artifacts"]:
        path = project / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"P5o frozen artifact hash mismatch: {artifact['path']}")
        validated.append(artifact["path"])
    candidate = protocol["base_candidate"]
    if (project / candidate["freeze_path"]).resolve() != freeze_path.resolve():
        raise RuntimeError("P5o provided candidate freeze does not match protocol")
    if sha256_file(freeze_path) != candidate["freeze_sha256"]:
        raise RuntimeError("P5o candidate-freeze hash mismatch")
    if list(candidate["source_model_seeds"]) != [3041, 3042, 3043, 3044, 3045]:
        raise RuntimeError("P5o source-model seed contract changed")
    if int(protocol["execution"]["batch_size"]) != 126 or int(protocol["execution"]["warmup_runs"]) != 5 or int(protocol["execution"]["timed_repetitions"]) != 15:
        raise RuntimeError("P5o fixed timing contract changed")
    return {"protocol": protocol, "validated_artifacts": validated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--validate-protocol", action="store_true")
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
    if not args.data_roots or not args.audits or not args.output_root:
        raise RuntimeError("P5o execution requires data roots, audits and output root")
    if len(args.data_roots) != 3 or len(args.audits) != 3:
        raise RuntimeError("P5o development requires exactly three fresh identity pools")
    protocol, freeze = validated["protocol"], json.loads(freeze_path.read_text(encoding="utf-8"))
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    environment = exclusive_window_receipt()
    output: dict[str, object] = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "protocol_validation": validated["validated_artifacts"], "environment": environment, "development": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("P5o cost execution requires CUDA")
        all_pass = True
        for root_arg, audit_arg in zip(args.data_roots, args.audits):
            audit = json.loads((project / audit_arg).read_text(encoding="utf-8"))
            if not audit["valid_for_frozen_inference"]:
                raise RuntimeError(f"invalid P5o audit: {audit_arg}")
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
            output["development"][root_arg] = {"audit": audit, "benchmark_indices": benchmark_indices.tolist(), "analytic_feature_seconds": float(time.perf_counter() - started), "post_analytic_guard": post_analytic_guard, "rows": rows, "gates": gates}
            full_dataset.close(); benchmark_dataset.close()
        post_guard = assert_exclusive_gpu()
        output["environment"]["post_execution_guard"] = post_guard
        output["environment_window_receipt_complete"] = not environment["guard"]["external_python_processes"] and not post_guard["external_python_processes"]
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
