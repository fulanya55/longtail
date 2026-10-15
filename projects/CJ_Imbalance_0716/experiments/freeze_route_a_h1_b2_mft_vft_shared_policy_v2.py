"""Freeze the revised unified candidate before development-v12 evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from freeze_route_a_h1_b2_mft_vft_unified_v1 import ROOT, record, sha


def run():
    parent_path = ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_unified_v1/frozen_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    audit_path = ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["test_files_opened"]:
        raise ValueError("development-v12 identity audit is not closed")
    seal_path = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not closed")
    payload = {
        "status": "frozen",
        "candidate_name": "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2",
        "frozen_at": "2026-08-01",
        "parent_candidate_manifest": record(parent_path),
        "construction_summary": parent["construction_summary"],
        "shared_policy_audit": record(ROOT / "outputs/tapc/route_a_mft_shared_decision_policy_audit/route_a_mft_shared_decision_policy_audit_summary.json"),
        "source_code": {
            **parent["source_code"],
            "shared_decision_policy": record(ROOT / "experiments/run_route_a_mft_shared_decision_policy_audit.py"),
        },
        "fixed_hyperparameters": {
            **parent["fixed_hyperparameters"],
            "mft_calibration_source": "Q1_H1B2",
            "mft_threshold_source": "Q1_H1B2",
            "non_mft_calibration_source": "Q4_VFTAlignedRidgeFusion",
            "non_mft_threshold_source": "Q4_VFTAlignedRidgeFusion",
            "mft_negative_to_positive_allowed": False,
        },
        "development_v3_sha256": parent["development_v3_sha256"],
        "development_v11_role": "diagnostic_reuse_not_independent_after_policy_change",
        "development_v12": audit,
        "seed_artifacts": parent["seed_artifacts"],
        "test_v3_manifest": record(seal_path),
        "test_v3": {"access_authorized": False, "analysis_allowed": False, "used": False},
    }
    output = ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "sha256": sha(output), "seed_count": len(payload["seed_artifacts"])}, indent=2))
    return payload


if __name__ == "__main__":
    run()
