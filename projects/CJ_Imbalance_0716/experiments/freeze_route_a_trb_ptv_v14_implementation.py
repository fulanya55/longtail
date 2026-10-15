"""Freeze the complete TRB-PTV v14 implementation before data generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.route_a_v14_identity import write_preflight  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "tapc" / "route_a_trb_ptv_implementation_freeze"
FROZEN_FILES = (
    "docs/plans/route_a_trb_ptv_literature_audit_and_fresh_development_preregistration_20260801.md",
    "docs/plans/route_a_trb_ptv_implementation_manifest_20260801.json",
    "configs/student_a/six_class_development_v14.yaml",
    "experiments/route_a_trb_ptv_core.py",
    "experiments/route_a_v14_identity.py",
    "experiments/generate_route_a_v14_dataset.m",
    "experiments/run_route_a_development_v14_trb_ptv.py",
    "experiments/freeze_route_a_trb_ptv_v14_implementation.py",
    "tests/test_route_a_trb_ptv_core.py",
    "tests/test_route_a_v14_identity.py",
    "tests/test_freeze_route_a_trb_ptv_v14_implementation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "v14_identity_preflight.json"
    preflight = write_preflight(preflight_path)
    smoke = PROJECT_ROOT / "outputs" / "tapc" / "route_a_trb_ptv_core_smoke" / "pytest.xml"
    if not smoke.is_file():
        raise FileNotFoundError("synthetic TRB core smoke receipt is required before freeze")
    files = {}
    for relative in FROZEN_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    train_exists = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_v14\closedset_6class_train.mat").is_file()
    payload = {
        "status": "implementation_repair_frozen_before_first_v14_model_evaluation" if train_exists else "implementation_frozen_before_v14_data_generation",
        "method": "TRB-PTV",
        "evidence_role": "development_v14",
        "formal_seeds": [2039, 2040, 2041, 2042, 2043],
        "formal_epochs": 12,
        "variants": [
            "C6_Joint", "M5_GradientIsolated", "FullAlpha1p5", "TRB_PTV",
            "ParameterNormOnly", "ClassifierOnlyAlpha2p0", "ModelAverage0p5",
            "CAGrad0p4TwoTask", "OrthogonalTaskSubspaceFunctionalMatched",
        ],
        "q0": [1.5, 1.5], "solver": "SLSQP", "maxiter": 200, "ftol": 1e-12,
        "constraint_tolerance": 1e-7, "activation_norm_min": 0.05, "activation_min_seeds": 4,
        "implementation_files": files,
        "synthetic_core_smoke": {"path": str(smoke), "sha256": sha256(smoke)},
        "identity_preflight": {"path": str(preflight_path), "sha256": sha256(preflight_path), "passed": preflight["identity_audit_passed"]},
        "development_v14_train_exists_at_freeze": train_exists,
        "test_files_opened": False, "locked_test_used": False, "test_v2_used": False,
        "test_v3_used": False, "test_v14_used": False, "test_access_allowed": False,
        "post_freeze_rule": "Any change to listed source/config/test files invalidates this freeze before v14 generation or formal evaluation.",
    }
    target = output_dir / "implementation_freeze.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {"status": payload["status"], "implementation_freeze_sha256": sha256(target), "identity_preflight_passed": True, "test_files_opened": False, "test_access_allowed": False}
    (output_dir / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    print(json.dumps(freeze(parser.parse_args().output_dir), ensure_ascii=False, indent=2))
