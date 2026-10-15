"""One-cell readiness execution of the exact R2 formal cell code path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_route_a_dual_axis_sawfm_r2_formal import (
    R1_ROOT,
    VIEW_PATHS,
    read_labels,
    run_seed_cell,
    sha256,
    validate_sources,
    write_json,
)


DEFAULT_OUTPUT = ROOT / "outputs/route_a_dual_axis_next/r2_formal_readiness"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args_cli = parser.parse_args()
    started = __import__("time").time()
    data_audit, source_hashes = validate_sources()
    labels = read_labels(VIEW_PATHS["VH"])
    saved = np.load(R1_ROOT / "role_indices.npz")
    roles = {
        name: saved[name].astype(np.int64)
        for name in ("early", "calibration", "threshold", "evaluation")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal readiness requires CUDA")
    args = SimpleNamespace(
        output_root=args_cli.output_root,
        batch_size=64,
        eval_batch_size=128,
        epochs=2,
        patience=2,
        resume=False,
    )
    row = run_seed_cell(
        args,
        labels,
        roles,
        frequency="LT",
        view="VL",
        seed=260811,
        device=device,
        source_hashes=source_hashes,
    )
    checks = {
        "cell_completed": row["status"] == "completed",
        "all_variants_present": set(row["variants"]) == {
            "M0_B0_Focal", "MF_B0_DB", "MV_B0_RawSaw", "MFV_B0_DB_RawSaw"
        },
        "parameter_counts_exact": row["parameter_counts"] == {
            "M0_B0_Focal": 257511,
            "MF_B0_DB": 257511,
            "MV_B0_RawSaw": 263032,
            "MFV_B0_DB_RawSaw": 263032,
            "RawSaw_increment": 5521,
        },
        "non_saw_routing_invariant": row["non_saw_routing_bitwise_invariant"] is True,
        "paired_roles_full_size": all(
            row["role_identity_counts"][name] == 2520
            for name in ("early", "calibration", "threshold", "evaluation")
        ),
        "all_metrics_finite_where_defined": all(
            np.isfinite(row["variants"][variant]["by_view"][view]["macro_aupr"])
            for variant in row["variants"] for view in ("VH", "VL")
        ),
        "formal_run_admitted": False,
    }
    checks["formal_run_admitted"] = all(
        value for key, value in checks.items() if key != "formal_run_admitted"
    )
    payload = {
        "status": "completed" if checks["formal_run_admitted"] else "stopped",
        "evidence_level": "readiness_only_not_performance_evidence",
        "cell": "LT-VL",
        "seed": 260811,
        "epochs": 2,
        "checks": checks,
        "data_audit_sha256": sha256(R1_ROOT / "data_audit.json"),
        "formal_script_sha256": sha256(
            ROOT / "experiments/run_route_a_dual_axis_sawfm_r2_formal.py"
        ),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args_cli.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args_cli.output_root / "readiness_summary.json"
    write_json(summary_path, payload)
    write_json(
        args_cli.output_root / "run_receipt.json",
        {
            "status": payload["status"],
            "decision": "admit_formal_r2" if checks["formal_run_admitted"] else "stop_before_formal_r2",
            "summary_sha256": sha256(summary_path),
            "elapsed_seconds": float(__import__("time").time() - started),
            "model_trained": True,
            "evidence_level": payload["evidence_level"],
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not checks["formal_run_admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
