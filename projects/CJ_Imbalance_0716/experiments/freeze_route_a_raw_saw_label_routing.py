"""Freeze and audit RawSaw+H2Others without accessing the locked test."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402


CLASS_NAMES = ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
SAW_INDEX = 1
EXPECTED_SUMMARY_SHA256 = "1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC"


def _assert_completed_unlocked(payload, name):
    if payload.get("status") != "completed":
        raise ValueError(f"{name} is not completed")
    if payload.get("locked_test_used"):
        raise ValueError(f"{name} used the locked test")


def _assert_hash(path, expected, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: {path}")
    actual = _sha256(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"{name} hash drift: expected={expected} actual={actual}")
    return actual


def _selected_raw_threshold(summary, seed):
    rows = [
        row for row in summary["confirmation"]["per_run"]
        if int(row["seed"]) == int(seed) and row["variant"] == "I1_RawPlatt"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one RawPlatt row for seed {seed}")
    return float(rows[0]["selected_threshold"])


def _validate_summary(summary):
    _assert_completed_unlocked(summary, "five-seed summary")
    confirmation = summary["confirmation"]["promotion_gate"]
    routing = summary["label_routing"]["promotion_gate"]
    if not confirmation["passed"] or int(confirmation["passed_seed_count"]) < 4:
        raise ValueError("five-seed confirmation gate is not frozen-passed")
    if not routing["passed"] or int(routing["passed_seed_count"]) != 5:
        raise ValueError("label-routing gate is not 5/5 passed")
    if summary.get("next_decision") != "freeze_raw_saw_label_routing_and_prepare_locked_test_request":
        raise ValueError("summary next decision drifted")


def _seed_bundle(args, summary, audit):
    seed = int(audit["seed"])
    if int(audit.get("identity_overlap", -1)) != 0 or audit.get("locked_test_used"):
        raise ValueError(f"seed {seed} identity/locked-test audit failed")
    f2_hash = _assert_hash(
        audit["f2_checkpoint"], audit["f2_checkpoint_sha256"], f"seed {seed} F2"
    )
    h2_hash = _assert_hash(
        audit["h2_checkpoint"], audit["h2_checkpoint_sha256"], f"seed {seed} H2"
    )
    raw_hash = _assert_hash(
        audit["frontend_checkpoint"], audit["frontend_checkpoint_sha256"], f"seed {seed} Raw"
    )
    seed_root = args.experiment_root / f"seed_{seed}"
    receipt = _read(seed_root / "run_receipt.json")
    _assert_completed_unlocked(receipt, f"seed {seed} route receipt")
    if int(receipt.get("identity_overlap", -1)) != 0:
        raise ValueError(f"seed {seed} route receipt identity overlap")
    calibration_path = seed_root / "calibration.json"
    calibration = _read(calibration_path)
    if calibration.get("locked_test_used"):
        raise ValueError(f"seed {seed} calibration used locked test")
    raw_platt = calibration.get("raw_platt", {})
    if set(raw_platt) != {"coefficient", "intercept"}:
        raise ValueError(f"seed {seed} Raw Platt is incomplete")
    threshold_search_path = seed_root / "threshold_search.json"
    threshold_search = _read(threshold_search_path)
    selected = threshold_search["I1_RawPlatt"]["selected"]
    raw_threshold = float(selected["threshold"])
    summary_threshold = _selected_raw_threshold(summary, seed)
    if abs(raw_threshold - summary_threshold) > 1e-12:
        raise ValueError(f"seed {seed} Raw threshold drift")
    h2_root = args.h2_root / f"seed_{seed}" / "H2_PosBCE"
    h2_calibration_path = h2_root / "calibration.json"
    h2_thresholds_path = h2_root / "thresholds.json"
    h2_calibration = _read(h2_calibration_path)
    h2_thresholds = _read(h2_thresholds_path)
    if h2_calibration.get("source_role") != "calibration":
        raise ValueError(f"seed {seed} H2 calibration role drift")
    primary_mode = h2_thresholds["primary_mode"]
    thresholds = h2_thresholds["modes"][primary_mode]
    if len(thresholds) != len(CLASS_NAMES):
        raise ValueError(f"seed {seed} H2 threshold count drift")
    route_metrics = _read(seed_root / "label_routing" / "metrics.json")
    if not route_metrics.get("passed") or route_metrics.get("locked_test_used"):
        raise ValueError(f"seed {seed} label routing did not pass cleanly")
    return {
        "seed": seed,
        "split_sha256": audit["split_sha256"],
        "frontend_protocol": audit["frontend_protocol"],
        "identity_overlap": 0,
        "f2": {"path": audit["f2_checkpoint"], "sha256": f2_hash},
        "h2": {
            "path": audit["h2_checkpoint"],
            "sha256": h2_hash,
            "temperature": float(h2_calibration["temperature"]),
            "temperature_file": str(h2_calibration_path.resolve()),
            "temperature_file_sha256": _sha256(h2_calibration_path),
            "threshold_mode": primary_mode,
            "thresholds": [float(value) for value in thresholds],
            "threshold_file": str(h2_thresholds_path.resolve()),
            "threshold_file_sha256": _sha256(h2_thresholds_path),
        },
        "raw_sawfm": {
            "path": audit["frontend_checkpoint"],
            "sha256": raw_hash,
            "platt_coefficient": float(raw_platt["coefficient"]),
            "platt_intercept": float(raw_platt["intercept"]),
            "calibration_file": str(calibration_path.resolve()),
            "calibration_file_sha256": _sha256(calibration_path),
            "threshold": raw_threshold,
            "threshold_file": str(threshold_search_path.resolve()),
            "threshold_file_sha256": _sha256(threshold_search_path),
        },
        "selection_hashes": {
            "calibration": audit["calibration_selection_sha256"],
            "development": audit["development_selection_sha256"],
        },
        "route_metrics_file": str((seed_root / "label_routing" / "metrics.json").resolve()),
        "route_metrics_sha256": _sha256(seed_root / "label_routing" / "metrics.json"),
        "locked_test_used": False,
    }


def run(args):
    summary_hash = _assert_hash(
        args.summary, args.expected_summary_sha256, "formal development summary"
    )
    summary = _read(args.summary)
    _validate_summary(summary)
    seeds = sorted(int(value) for value in summary["protocol"]["seeds"])
    if seeds != [2039, 2040, 2041, 2042, 2043]:
        raise ValueError(f"unexpected frozen seeds: {seeds}")
    audits = {int(audit["seed"]): audit for audit in summary["audits"]}
    if sorted(audits) != seeds:
        raise ValueError("summary audit seed set drifted")
    seed_bundles = [_seed_bundle(args, summary, audits[seed]) for seed in seeds]
    code_files = {
        "five_seed_experiment": PROJECT_ROOT / "experiments" / "run_route_a_raw_platt_five_seed_label_routing.py",
        "freeze_preflight": Path(__file__).resolve(),
        "f2_model": PROJECT_ROOT / "cjlt" / "models" / "phase_multiscale.py",
        "raw_model": PROJECT_ROOT / "cjlt" / "models" / "evidence_readout.py",
        "h2_training": PROJECT_ROOT / "experiments" / "run_route_a_dfms_iq_frozen_heads.py",
        "config": args.config,
    }
    bundle = {
        "status": "ready_for_authorization",
        "authorization_required": True,
        "locked_test_authorized": False,
        "locked_test_accessed": False,
        "candidate": "RawSaw+H2Others",
        "ensemble": False,
        "class_order": CLASS_NAMES,
        "sawfm_index": SAW_INDEX,
        "input": {"shape": ["B", 2, 3000], "normalization": "complex_rms"},
        "routing": {
            "SawFM": "RawStage1SawFMFrontend + frozen Platt + frozen unified threshold",
            "other_labels": "H2_PosBCE + frozen temperature + frozen per-label thresholds",
            "learned_fusion": False,
            "oracle_power_input": False,
        },
        "architecture": {
            "f2": {"name": "F2_DFMS_Full", "width": 16, "feature_dim": 128, "dropout": 0.2, "phase_lags": [1, 4, 16]},
            "h2": {"name": "H2_PosBCE", "input_dim": 128, "num_labels": 6},
            "raw": {"name": "RawStage1SawFMFrontend", "mode": "raw", "hidden_channels": 16, "topk": 16, "dropout": 0.1, "descriptor_mode": "full"},
        },
        "formal_summary": {"path": str(args.summary.resolve()), "sha256": summary_hash},
        "formal_gates": {
            "confirmation_passed_seed_count": int(summary["confirmation"]["promotion_gate"]["passed_seed_count"]),
            "label_routing_passed_seed_count": int(summary["label_routing"]["promotion_gate"]["passed_seed_count"]),
        },
        "seeds": seed_bundles,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in code_files.items()
        },
        "known_risks": [
            "seed 2039 ordinary early FPR is 0.2806",
            "Raw checkpoint protocols are new_split_R0 for 2039/2040 and original_N0 for 2041-2043",
            "locked test has not been accessed and requires explicit user authorization",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_root / "frozen_inference_bundle.json"
    _write(bundle_path, bundle)
    audit = {
        "status": "passed",
        "summary_sha256_unchanged": summary_hash.lower() == args.expected_summary_sha256.lower(),
        "seed_count": len(seed_bundles),
        "checkpoint_hashes_passed": True,
        "calibration_threshold_artifacts_complete": True,
        "identity_overlap": 0,
        "confirmation_gate": "4_of_5_or_better",
        "label_routing_gate": "5_of_5",
        "locked_test_accessed": False,
        "ready_for_authorization": True,
    }
    audit_path = args.output_root / "preflight_audit.json"
    _write(audit_path, audit)
    receipt = {
        "status": "completed",
        "action": "freeze_preflight_only",
        "bundle_sha256": _sha256(bundle_path),
        "audit_sha256": _sha256(audit_path),
        "formal_summary_sha256": summary_hash,
        "locked_test_authorized": False,
        "locked_test_accessed": False,
        "next_action": "wait_for_explicit_locked_test_authorization",
    }
    _write(args.output_root / "run_receipt.json", receipt)
    print(json.dumps({"bundle": bundle, "audit": audit, "receipt": receipt}, ensure_ascii=False, indent=2))
    return bundle, audit, receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing" / "route_a_raw_platt_five_seed_label_routing_summary.json")
    parser.add_argument("--experiment-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing")
    parser.add_argument("--h2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_saw_locked_test_preflight")
    parser.add_argument("--expected-summary-sha256", default=EXPECTED_SUMMARY_SHA256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
