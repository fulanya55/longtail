"""Freeze the preregistered FC-PTV implementation before v12 model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_implementation_freeze"
FROZEN_FILES = (
    "docs/plans/route_a_fc_ptv_literature_gap_and_fresh_development_preregistration_20260801.md",
    "configs/student_a/six_class_development_v12.yaml",
    "experiments/generate_route_a_v12_dataset.m",
    "experiments/audit_route_a_development_v12_dataset.py",
    "experiments/route_a_fc_ptv_core.py",
    "experiments/run_route_a_development_v12_fc_ptv.py",
    "tests/test_route_a_development_v12_audit.py",
    "tests/test_route_a_development_v12_fc_ptv.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(output_dir):
    files = {}
    for relative in FROZEN_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    v12_train = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_v12\closedset_6class_train.mat")
    smoke_xml = PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_core_smoke/pytest.xml"
    if not smoke_xml.is_file():
        raise FileNotFoundError("focused smoke test receipt is missing")
    end_to_end_smoke = PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv_smoke/run_receipt.json"
    if not end_to_end_smoke.is_file():
        raise FileNotFoundError("end-to-end smoke receipt is missing")
    payload = {
        "status": "implementation_frozen_before_formal_model_evaluation",
        "method": "FC-PTV",
        "evidence_role": "development_v12",
        "formal_seeds": [2039, 2040, 2041, 2042, 2043],
        "formal_epochs": 12,
        "variants": [
            "C6_Joint", "M5_GradientIsolated", "FullAlpha1p5", "FC_PTV",
            "ParameterNormOnly", "ClassifierOnlyAlpha2p0", "ModelAverage0p5",
            "CAGrad0p4TwoTask", "OrthogonalTaskSubspaceFunctionalMatched",
        ],
        "q0": [1.5, 1.5],
        "functional_radius": "C6 fit-only per-common-label and SawFM-stratum 1.4826*MAD",
        "exact_functional_tolerance": 1.05,
        "implementation_files": files,
        "focused_smoke": {"path": str(smoke_xml), "sha256": _sha256(smoke_xml)},
        "end_to_end_smoke": {"path": str(end_to_end_smoke), "sha256": _sha256(end_to_end_smoke)},
        "development_v12_train_exists_at_implementation_freeze": v12_train.exists(),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v12_used": False,
        "test_access_allowed": False,
        "post_freeze_rule": "Any implementation change invalidates this manifest and requires a new freeze before formal model evaluation.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "implementation_freeze.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    receipt = {
        "status": payload["status"],
        "implementation_freeze_sha256": _sha256(target),
        "development_v12_train_exists_at_implementation_freeze": payload["development_v12_train_exists_at_implementation_freeze"],
        "locked_test_used": False,
        "test_access_allowed": False,
    }
    receipt_path = output_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload, receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    payload, receipt = freeze(parse_args().output_dir)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
