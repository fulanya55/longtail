"""Freeze the cross-seed RawSaw threshold candidate without locked-test access."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.freeze_route_a_raw_saw_label_routing import (  # noqa: E402
    CLASS_NAMES,
    EXPECTED_SUMMARY_SHA256,
    SAW_INDEX,
    _assert_completed_unlocked,
    _assert_hash,
    _seed_bundle,
    _validate_summary,
)
from experiments.run_route_a_controlled_lt import _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402


EXPECTED_CROSS_SUMMARY_SHA256 = "5FF8EB70BA1F76A006A2F25AA244E5C33823FEA714A5360843B258D150265D61"


def _cross_rows(summary):
    rows = {
        int(row["seed"]): row
        for row in summary["per_run"]
        if row["variant"] == "C1_LOSO_FPR15_Shared"
    }
    if sorted(rows) != [2039, 2040, 2041, 2042, 2043]:
        raise ValueError("cross-seed primary row set drifted")
    return rows


def run(args):
    base_hash = _assert_hash(args.base_summary, args.expected_base_sha256, "base development summary")
    cross_hash = _assert_hash(args.cross_summary, args.expected_cross_sha256, "cross-seed threshold summary")
    base = _read(args.base_summary)
    cross = _read(args.cross_summary)
    _validate_summary(base)
    _assert_completed_unlocked(cross, "cross-seed threshold summary")
    if not cross["promotion_gate"]["passed"]:
        raise ValueError("cross-seed threshold gate did not pass")
    if cross.get("next_decision") != "refreeze_cross_seed_stable_threshold_candidate":
        raise ValueError("cross-seed next decision drifted")
    if cross["protocol"].get("evidence_scope") != "development_cross_seed_stress_test_not_independent_locked_test":
        raise ValueError("cross-seed evidence scope drifted")

    seeds = sorted(int(value) for value in base["protocol"]["seeds"])
    audits = {int(audit["seed"]): audit for audit in base["audits"]}
    cross_rows = _cross_rows(cross)
    seed_bundles = []
    for seed in seeds:
        bundle = _seed_bundle(args, base, audits[seed])
        row = cross_rows[seed]
        selection = row["selection"]
        if int(selection["held_out_seed"]) != seed or seed in selection["source_seeds"]:
            raise ValueError(f"seed {seed} was not held out during threshold selection")
        bundle["raw_sawfm"].update({
            "threshold": float(row["threshold"]),
            "threshold_rule": "LOSO shared threshold selected from the other four seeds at design FPR<=0.15",
            "threshold_source_seeds": [int(value) for value in selection["source_seeds"]],
            "threshold_file": str(args.cross_summary.resolve()),
            "threshold_file_sha256": cross_hash,
        })
        bundle["cross_seed_development_metrics"] = {
            "ordinary_recall": float(row["ordinary"]["recall"]),
            "ordinary_fpr": float(row["ordinary"]["fpr"]),
            "power_50db_recall": float(row["power"]["50"]["recall"]),
            "power_50db_fpr": float(row["power"]["50"]["fpr"]),
            "route_macro_f1": float(row["route"]["routed"]["macro_f1"]),
            "route_exact_match": float(row["route"]["routed"]["exact_match"]),
            "route_passed": bool(row["route"]["passed"]),
        }
        seed_bundles.append(bundle)

    code_files = {
        "base_experiment": PROJECT_ROOT / "experiments" / "run_route_a_raw_platt_five_seed_label_routing.py",
        "threshold_margin_experiment": PROJECT_ROOT / "experiments" / "run_route_a_raw_threshold_safety_margin.py",
        "cross_seed_experiment": PROJECT_ROOT / "experiments" / "run_route_a_raw_cross_seed_threshold.py",
        "freeze_preflight": Path(__file__).resolve(),
        "f2_model": PROJECT_ROOT / "cjlt" / "models" / "phase_multiscale.py",
        "raw_model": PROJECT_ROOT / "cjlt" / "models" / "evidence_readout.py",
        "h2_training": PROJECT_ROOT / "experiments" / "run_route_a_dfms_iq_frozen_heads.py",
        "config": args.config,
    }
    aggregate = cross["aggregate"]["C1_LOSO_FPR15_Shared"]
    candidate = {
        "status": "ready_for_authorization",
        "authorization_required": True,
        "locked_test_authorized": False,
        "locked_test_accessed": False,
        "candidate": "RawSawCrossSeedFPR15+H2Others",
        "supersedes_preflight_candidate": "RawSaw+H2Others",
        "ensemble": False,
        "class_order": CLASS_NAMES,
        "sawfm_index": SAW_INDEX,
        "input": {"shape": ["B", 2, 3000], "normalization": "complex_rms"},
        "routing": {
            "SawFM": "RawStage1SawFMFrontend + frozen Platt + frozen LOSO threshold",
            "other_labels": "H2_PosBCE + frozen temperature + frozen per-label thresholds",
            "learned_fusion": False,
            "oracle_power_input": False,
        },
        "architecture": {
            "f2": {"name": "F2_DFMS_Full", "width": 16, "feature_dim": 128, "dropout": 0.2, "phase_lags": [1, 4, 16]},
            "h2": {"name": "H2_PosBCE", "input_dim": 128, "num_labels": 6},
            "raw": {"name": "RawStage1SawFMFrontend", "mode": "raw", "hidden_channels": 16, "topk": 16, "dropout": 0.1, "descriptor_mode": "full"},
        },
        "formal_summaries": {
            "base": {"path": str(args.base_summary.resolve()), "sha256": base_hash},
            "cross_seed_threshold": {"path": str(args.cross_summary.resolve()), "sha256": cross_hash},
        },
        "development_gate": {
            "passed": True,
            "ordinary_fpr_mean": float(aggregate["ordinary"]["fpr"]["mean"]),
            "ordinary_recall_mean": float(aggregate["ordinary"]["recall"]["mean"]),
            "power_50db_recall_mean": float(aggregate["by_level"]["50"]["recall"]["mean"]),
            "route_macro_f1_mean": float(aggregate["route"]["macro_f1"]["mean"]),
            "route_passed_seed_count": int(aggregate["route_passed_seed_count"]),
        },
        "evidence_scope": "development_cross_seed_stress_test_only",
        "seeds": seed_bundles,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in code_files.items()
        },
        "known_risks": [
            "ordinary_early was already inspected during candidate development and is not an independent final test",
            "Raw checkpoint protocols are new_split_R0 for 2039/2040 and original_N0 for 2041-2043",
            "locked test has not been accessed and requires explicit user authorization",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_root / "frozen_inference_bundle.json"
    _write(bundle_path, candidate)
    audit = {
        "status": "passed",
        "base_summary_sha256_unchanged": base_hash.lower() == args.expected_base_sha256.lower(),
        "cross_summary_sha256_unchanged": cross_hash.lower() == args.expected_cross_sha256.lower(),
        "cross_seed_gate_passed": True,
        "held_out_threshold_selection_passed": True,
        "seed_count": len(seed_bundles),
        "checkpoint_hashes_passed": True,
        "identity_overlap": 0,
        "locked_test_accessed": False,
        "ready_for_authorization": True,
    }
    audit_path = args.output_root / "preflight_audit.json"
    _write(audit_path, audit)
    receipt = {
        "status": "completed",
        "action": "refreeze_cross_seed_threshold_preflight_only",
        "bundle_sha256": _sha256(bundle_path),
        "audit_sha256": _sha256(audit_path),
        "base_summary_sha256": base_hash,
        "cross_summary_sha256": cross_hash,
        "locked_test_authorized": False,
        "locked_test_accessed": False,
        "next_action": "wait_for_explicit_locked_test_authorization",
    }
    _write(args.output_root / "run_receipt.json", receipt)
    print(json.dumps({"bundle": candidate, "audit": audit, "receipt": receipt}, ensure_ascii=False, indent=2))
    return candidate, audit, receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing" / "route_a_raw_platt_five_seed_label_routing_summary.json")
    parser.add_argument("--cross-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_cross_seed_threshold" / "route_a_raw_cross_seed_threshold_summary.json")
    parser.add_argument("--experiment-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing")
    parser.add_argument("--h2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_saw_cross_seed_locked_test_preflight")
    parser.add_argument("--expected-base-sha256", default=EXPECTED_SUMMARY_SHA256)
    parser.add_argument("--expected-cross-sha256", default=EXPECTED_CROSS_SUMMARY_SHA256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

