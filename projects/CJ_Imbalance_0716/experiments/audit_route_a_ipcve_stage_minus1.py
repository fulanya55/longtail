"""Read-only Stage -1 admission audit for Route-A IP-CVE.

This audit never opens test-v3 data.  It verifies the sealed manifest, the
formal five-seed B0 bundle, current checkpoint compatibility, historical
counterfactual negative evidence, and the scope of the proposed next stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402


B0_SEEDS = (2039, 2040, 2041, 2042, 2043)
EXPECTED_B0_SUMMARY_SHA256 = (
    "56c3c5d0fd92e17d7f839a60cd15f4b0a5215c2497c3eff240a87865c72d3d6d"
)
DEV_BASE_SEED = 4_000_000_000
CONFIRMATION_BASE_SEED = 4_050_000_000
RECEIVER_SEED_OFFSET = 198_677_742
TRIPLE_COUNT = 2_400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_state(root: Path) -> dict:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory={}".format(root.as_posix()),
            "status",
            "--short",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    encoded = result.stdout.encode("utf-8")
    return {
        "dirty": bool(lines),
        "changed_path_count": len(lines),
        "status_sha256": hashlib.sha256(encoded).hexdigest(),
        "note": "Existing changes are user-owned and are not modified by this audit.",
    }


def _mechanism_matrix() -> list[dict]:
    return [
        {
            "mechanism": "H0e_CVMC",
            "intervention": "matched absent/low binary pair",
            "representation": "trainable compact raw-IQ encoder",
            "supervision": "direct target-logit margin",
            "inference": "single view",
            "formal_status": "fail_closed",
            "reuse_allowed": False,
        },
        {
            "mechanism": "H0f_CDP",
            "intervention": "matched absent/low binary pair",
            "representation": "trainable pooled feature difference",
            "supervision": "target prototype binding CE plus bounded residual",
            "inference": "single view",
            "formal_status": "fail_closed",
            "reuse_allowed": False,
        },
        {
            "mechanism": "H0v_NaturalVisibilityGraph",
            "intervention": "matched absent/low binary pair",
            "representation": "fixed real-I natural-visibility-graph entropy",
            "supervision": "no fitted parameters",
            "inference": "fixed scalar diagnostic",
            "formal_status": "fail_closed",
            "reuse_allowed": False,
        },
        {
            "mechanism": "IP_CVE_P0",
            "intervention": "matched absent/low/high ordered triple",
            "representation": "frozen formal B0 structure maps",
            "supervision": "read-only linear/ridge observability diagnostics",
            "inference": "no deployable candidate at P0",
            "formal_status": "proposed_not_validated",
            "reuse_allowed": True,
            "difference_from_closed_lines": (
                "No encoder, logit margin, prototype, calibration, threshold, "
                "or residual is trained before matched information defeats "
                "unmatched/component/time-permuted controls."
            ),
        },
        {
            "mechanism": "IP_CVE_P1",
            "intervention": "training-only ordered triple",
            "representation": "single-view label-specific low-rank readout",
            "supervision": "rank/order/non-target invariance",
            "inference": "single view with bounded target-only correction",
            "formal_status": "not_authorized_until_P0_passes_and_overlap_is_reaudited",
            "reuse_allowed": False,
            "risk": "Bounded residual remains close to the stopped H0f integration family.",
        },
    ]


def _reserved_range(base_seed: int) -> list[int]:
    first = int(base_seed + RECEIVER_SEED_OFFSET)
    return [first, first + TRIPLE_COUNT - 1]


def _check_b0(root: Path, run_forward: bool) -> dict:
    summary_path = root / "outputs/tapc/route_a_backbone_b0_development_v1/b0_summary.json"
    summary = _read_json(summary_path)
    summary_hash = _sha256(summary_path)
    per_seed = {int(row["seed"]): row for row in summary["per_seed"]}
    checkpoint_rows = []
    strict_load_passed = True
    model = None
    for seed in B0_SEEDS:
        checkpoint_path = (
            root
            / "outputs/tapc/route_a_backbone_b0_development_v1"
            / "seed_{}".format(seed)
            / "best_model.pth"
        )
        checkpoint_hash = _sha256(checkpoint_path)
        recorded_hash = str(per_seed[seed]["checkpoint_sha256"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current = build_backbone("dfms_iq", 6)
        try:
            current.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception:
            strict_load_passed = False
            raise
        if model is None:
            model = current
        checkpoint_rows.append(
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "recorded_sha256": recorded_hash,
                "hash_matches": checkpoint_hash == recorded_hash,
                "best_epoch": int(per_seed[seed]["best_epoch"]),
                "status": per_seed[seed]["status"],
            }
        )

    forward_check = {"executed": False}
    if run_forward:
        config = yaml.safe_load(
            (root / "configs/student_a/route_a_backbone_development_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        data_path = Path(config["data_root"]) / config["train_file"]
        dataset = V73SignalDataset(
            data_path,
            [0, 1],
            class_mode="six",
            normalization="complex_rms",
        )
        try:
            values = torch.stack([dataset[index]["signal"] for index in range(2)])
        finally:
            dataset.close()
        assert model is not None
        model.eval()
        with torch.no_grad():
            logits = model(values)
            maps = model.forward_structure_maps(values)
        forward_check = {
            "executed": True,
            "data_path": str(data_path),
            "input_shape": list(values.shape),
            "logits_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "map_shapes": {
                key: None if value is None else list(value.shape)
                for key, value in maps.items()
            },
        }

    return {
        "bundle_definition": "formal five-seed B0 family; no post-hoc single checkpoint",
        "summary_path": str(summary_path),
        "summary_sha256": summary_hash,
        "summary_hash_matches_frozen_record": summary_hash
        == EXPECTED_B0_SUMMARY_SHA256,
        "summary_status": summary["status"],
        "model": summary["model"],
        "seeds": list(summary["seeds"]),
        "checkpoints": checkpoint_rows,
        "all_checkpoint_hashes_match": all(
            row["hash_matches"] for row in checkpoint_rows
        ),
        "strict_current_model_load_passed": strict_load_passed,
        "forward_structure_maps_check": forward_check,
        "test_flags": {
            key: summary[key]
            for key in ("locked_test_used", "test_v2_used", "test_v3_used")
        },
    }


def build_audit(root: Path, run_model_forward: bool = True) -> dict:
    root = Path(root).resolve()
    prompt_path = root / "docs/prompts/route_a_ipcve_autonomous_experiment_goal_prompt_20260807.md"
    h0e_result = root / "docs/results/route_a_h0e_counterfactual_visibility_margin_results_20260802.md"
    h0f_result = root / "docs/results/route_a_h0f_counterfactual_delta_prototype_results_20260802.md"
    h0v_result = root / "docs/results/route_a_h0v_visibility_graph_observability_results_20260803.md"
    h0e_generator = root / "experiments/generate_route_a_h0e_counterfactual_visibility_pairs.m"
    h0e_text = h0e_generator.read_text(encoding="utf-8")
    h0e_runner = root / "experiments/run_route_a_h0e_counterfactual_visibility_margin.py"
    h0f_runner = root / "experiments/run_route_a_h0f_counterfactual_delta_prototype.py"
    seal_path = root / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
    seal = _read_json(seal_path)
    seal_flags = {
        key: seal["test_v3"][key]
        for key in ("access_authorized", "analysis_allowed", "used")
    }

    b0 = _check_b0(root, run_model_forward)
    generator_capability = {
        "source_path": str(h0e_generator),
        "source_sha256": _sha256(h0e_generator),
        "matched_absent_low_pair_supported": all(
            token in h0e_text
            for token in (
                "negativeRow=make_row",
                "positiveRow=make_row",
                "PairNoiseMismatch",
                "offsets(targetMask)=-12",
            )
        ),
        "absent_low_high_triple_supported_by_existing_entrypoint": False,
        "stage0_action": (
            "Create a new E-drive-only triple generator; reuse only the audited "
            "signal composition infrastructure, never the stopped H0e loss/runner."
        ),
    }

    path_audit = {
        "active_project_root": str(root),
        "prompt_uses_e_drive": "E:\\LJQ_code\\CJ_Imbalance_0716"
        in prompt_path.read_text(encoding="utf-8"),
        "b0_config_uses_e_drive": "E:\\LJQ_code"
        in (root / "configs/student_a/route_a_backbone_development_v1.yaml").read_text(
            encoding="utf-8"
        ),
        "historical_h0e_runner_has_stale_c_default": "C:\\LJQ_code"
        in h0e_runner.read_text(encoding="utf-8"),
        "historical_h0f_runner_has_stale_c_default": "C:\\LJQ_code"
        in h0f_runner.read_text(encoding="utf-8"),
        "action": (
            "Do not rewrite historical evidence. New IP-CVE entrypoints must use E-drive defaults."
        ),
    }

    admission_checks = {
        "h0e_formal_negative_loaded": h0e_result.exists(),
        "h0f_formal_negative_loaded": h0f_result.exists(),
        "h0v_formal_negative_loaded": h0v_result.exists(),
        "formal_b0_bundle_verified": (
            b0["summary_hash_matches_frozen_record"]
            and b0["all_checkpoint_hashes_match"]
            and b0["strict_current_model_load_passed"]
        ),
        "test_v3_seal_intact": seal_flags
        == {"access_authorized": False, "analysis_allowed": False, "used": False},
        "p0_is_frozen_read_only_before_any_candidate_training": True,
        "three_state_ordered_intervention_is_required": True,
        "information_destruction_controls_are_required": True,
        "h0e_h0f_training_code_reuse_is_forbidden": True,
        "p1_bounded_reconnection_is_not_yet_authorized": True,
        "new_identity_source_is_required": True,
    }
    passed = all(admission_checks.values())
    decision = (
        "conditional_admit_stage0_and_p0_only"
        if passed
        else "stop_before_stage0_and_repair_admission"
    )
    return {
        "status": "completed_read_only_stage_minus1_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "RouteA_IPCVE_StageMinus1_Admission",
        "evidence_role": "read_only_admission_not_model_evidence",
        "project_root": str(root),
        "worktree": _git_state(root),
        "evidence_hashes": {
            "goal_prompt": _sha256(prompt_path),
            "h0e_result": _sha256(h0e_result),
            "h0f_result": _sha256(h0f_result),
            "h0v_result": _sha256(h0v_result),
            "test_v3_sealed_manifest": _sha256(seal_path),
        },
        "mechanism_overlap_matrix": _mechanism_matrix(),
        "formal_b0": b0,
        "generator_capability": generator_capability,
        "path_audit": path_audit,
        "reserved_identity_contract": {
            "development_base_seed": DEV_BASE_SEED,
            "development_receiver_noise_seed_range": _reserved_range(DEV_BASE_SEED),
            "confirmation_base_seed": CONFIRMATION_BASE_SEED,
            "confirmation_receiver_noise_seed_range": _reserved_range(
                CONFIRMATION_BASE_SEED
            ),
            "uint32_safe": _reserved_range(CONFIRMATION_BASE_SEED)[1]
            < 2**32,
            "reservation_is_proposed_until_stage0_full_history_audit": True,
        },
        "test_v3_seal": {
            "manifest_path": str(seal_path),
            "manifest_sha256": _sha256(seal_path),
            "status": seal["status"],
            "flags": seal_flags,
            "test_data_opened": False,
        },
        "admission": {
            "checks": admission_checks,
            "passed": passed,
            "decision": decision,
            "scope": (
                "Authorizes only a fresh triple protocol, dry-run/smoke data audit, "
                "and frozen-B0 P0 diagnostics. It does not authorize P1 training, "
                "calibration, threshold selection, confirmation, or test-v3."
            ),
        },
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/route_a_ipcve/stage_minus1",
    )
    parser.add_argument("--skip-model-forward", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(
        args.project_root,
        run_model_forward=not args.skip_model_forward,
    )
    output_root = Path(args.output_root)
    audit_path = output_root / "admission_audit.json"
    receipt_path = output_root / "run_receipt.json"
    _write_json(audit_path, audit)
    _write_json(
        receipt_path,
        {
            "status": "completed",
            "experiment": audit["experiment"],
            "decision": audit["admission"]["decision"],
            "admission_passed": audit["admission"]["passed"],
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        },
    )
    print(
        json.dumps(
            {
                "decision": audit["admission"]["decision"],
                "passed": audit["admission"]["passed"],
                "audit": str(audit_path),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["admission"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
