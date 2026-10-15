"""Build a read-only, hash-addressed evidence bundle for frozen ISRJ B2 results.

This tool intentionally reads summary/audit JSON files and the sealed-manifest
metadata only.  It neither opens test I/Q nor trains, calibrates, thresholds,
or selects a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _external_record(name: str, summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError(f"{name} has not completed")
    if summary.get("external_refit") is not False:
        raise ValueError(f"{name} is not frozen external evaluation")
    if summary.get("locked_test_used") is not False or summary.get("test_v3_used") is not False:
        raise ValueError(f"{name} violates test boundary")
    gate = summary.get("external_gate", {})
    if gate.get("passed") is not True:
        raise ValueError(f"{name} did not pass its frozen external gate")
    delta = summary["b2_vs_b0"]["aggregate"]
    return {
        "name": name,
        "path": str(summary_path),
        "sha256": _sha256(summary_path),
        "experiment": summary["experiment"],
        "evidence_role": summary["evidence_role"],
        "external_data_sha256": summary["external_data_sha256"],
        "seeds": summary["seeds"],
        "external_refit": False,
        "locked_test_used": False,
        "test_v3_used": False,
        "gate_passed": True,
        "b2_minus_b0": {
            key: {
                "mean": delta[key]["mean"],
                "bootstrap_95_ci": delta[key]["bootstrap_95_ci"],
            }
            for key in ("isrj_aupr", "isrj_weak_auroc", "isrj_weak_recall", "isrj_fpr", "macro_aupr", "macro_f1")
        },
        "positive_weak_auroc_seeds": delta["positive_weak_auroc_seeds"],
        "positive_weak_recall_seeds": delta["positive_weak_recall_seeds"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    sealed = _read_json(args.test_v3_manifest)
    test_v3 = sealed.get("test_v3")
    if not isinstance(test_v3, dict):
        raise ValueError("test-v3 manifest lacks its required nested seal")
    if any(test_v3.get(key) is not False for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed and unused")
    if sealed.get("test_v3_used") is not False or sealed.get("historic_locked_test_used") is not False:
        raise ValueError("test-v3 manifest records prior test use")
    colabel = _read_json(args.colabel_audit)
    if colabel.get("status") != "completed_nonselection_frozen_prediction_audit":
        raise ValueError("co-label audit must be completed nonselection audit")
    if colabel.get("selection_or_refit_performed") is not False:
        raise ValueError("co-label audit must not perform selection/refit")
    if any(colabel.get(key) is not False for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("co-label audit violates test boundary")

    records = [
        _external_record("H0g_high_replica_parameter_ood", args.h0g_summary),
        _external_record("H0h_ar1_receiver_noise", args.h0h_summary),
        _external_record("H0s_joint_high_replica_ar1", args.h0s_summary),
    ]
    artifact_paths = (args.h0g_summary, args.h0h_summary, args.h0s_summary, args.colabel_audit, args.test_v3_manifest)
    payload = {
        "status": "completed_read_only_evidence_bundle",
        "experiment": "RouteA_ISRJAlignedPeriodicity_EvidenceBundle_20260803",
        "evidence_role": "paper_ready_hash_addressed_frozen_component_evidence",
        "selection_or_refit_performed": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v3_seal": {key: test_v3[key] for key in ("access_authorized", "analysis_allowed", "used")},
        "external_replications": records,
        "colabel_audit": {
            "path": str(args.colabel_audit),
            "sha256": _sha256(args.colabel_audit),
            "status": colabel["status"],
            "selection_or_refit_performed": False,
            "finding_boundary": "ranking gain is not confined to one reported MFT/VFT co-label; threshold recall/FPR remain context-dependent",
        },
        "input_artifacts": [{"path": str(path), "sha256": _sha256(path)} for path in artifact_paths],
        "supported_claim": "Frozen aligned-periodicity B2 improves ISRJ weak ranking and recall over frozen B0 under high-replica, AR(1)-noise, and joint high-replica-plus-AR(1) identity-isolated external conditions.",
        "prohibited_claim": "This bundle does not establish a promoted MFT/VFT/ISRJ unified model, universal threshold safety, or a final locked-test result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def parse_args() -> argparse.Namespace:
    output_root = PROJECT_ROOT / "outputs/tapc/route_a_isrj_evidence_bundle_20260803"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h0g-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_frozen/route_a_h0g_isrj_parameter_ood_v2_frozen_summary.json")
    parser.add_argument("--h0h-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h0h_ar1_noise_frozen/route_a_h0h_ar1_noise_frozen_summary.json")
    parser.add_argument("--h0s-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h0s_isrj_combined_frozen/route_a_h0s_isrj_combined_frozen_summary.json")
    parser.add_argument("--colabel-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_frozen_colabel_audit/route_a_isrj_frozen_colabel_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=output_root / "evidence_bundle.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"status": result["status"], "output": str(parse_args().output)}, ensure_ascii=False))
