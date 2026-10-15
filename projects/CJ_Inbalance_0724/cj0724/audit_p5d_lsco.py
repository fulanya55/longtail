"""Audit the frozen P5d LSCO screen without any threshold selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (15041, 15042, 15043)
BASELINE = "legacy_mlcnn_global"
CANDIDATE = "mlcnn_iq_grouped_task_routed"


def _required_rows(rows):
    return [
        (fold, seed, model)
        for fold in range(3)
        for seed in SEEDS
        for model in (BASELINE, CANDIDATE)
    ]


def _target_deltas(base_metrics, candidate_metrics):
    return {
        f"{b['class_name']}.{b['name']}": float(
            c["normalized_mae"] - b["normalized_mae"]
        )
        for b, c in zip(
            base_metrics["parameter"]["per_target"],
            candidate_metrics["parameter"]["per_target"],
        )
    }


def audit_rows(rows):
    lookup = {(r["fold"], r["seed"], r["model"]): r for r in rows}
    missing = [key for key in _required_rows(rows) if key not in lookup]
    paired = []
    audit_fields_missing = []
    for fold in range(3):
        for seed in SEEDS:
            if (fold, seed, BASELINE) not in lookup or (fold, seed, CANDIDATE) not in lookup:
                continue
            base = lookup[(fold, seed, BASELINE)]
            cand = lookup[(fold, seed, CANDIDATE)]
            for row in (base, cand):
                unseen = row.get("unseen", {})
                checkpoint = row.get("checkpoint")
                if not unseen.get("per_class"):
                    audit_fields_missing.append({"fold": fold, "seed": seed, "model": row["model"], "field": "unseen.per_class"})
                if not unseen.get("logits_sha256"):
                    audit_fields_missing.append({"fold": fold, "seed": seed, "model": row["model"], "field": "unseen.logits_sha256"})
                if not checkpoint or not Path(checkpoint).is_file():
                    audit_fields_missing.append({"fold": fold, "seed": seed, "model": row["model"], "field": "checkpoint"})
            bm, cm = base["unseen"]["metrics"], cand["unseen"]["metrics"]
            base_audit = base["unseen"].get("per_class", {})
            cand_audit = cand["unseen"].get("per_class", {})
            class_deltas = {
                name: {
                    "recall": float(cand_audit[name]["recall"] - base_audit[name]["recall"]),
                    "fpr": float(cand_audit[name]["fpr"] - base_audit[name]["fpr"]),
                }
                for name in base_audit.keys() & cand_audit.keys()
            }
            paired.append({
                "fold": fold,
                "seed": seed,
                "parameter_relative_reduction": 1.0 - float(cm["parameter"]["oracle_macro_target_nmae"]) / float(bm["parameter"]["oracle_macro_target_nmae"]),
                "macro_aupr_delta": float(cm["macro_aupr"] - bm["macro_aupr"]),
                "subset_delta": float(cm["subset_acc"] - bm["subset_acc"]),
                "target_deltas": _target_deltas(bm, cm),
                "class_deltas": class_deltas,
            })
    return lookup, missing, audit_fields_missing, paired


def promotion_gates(paired, protocol_complete):
    if not protocol_complete:
        return {"evaluated": False, "all_pass": False, "reason": "incomplete required audit fields"}
    reductions = [r["parameter_relative_reduction"] for r in paired]
    aupr = [r["macro_aupr_delta"] for r in paired]
    subset = [r["subset_delta"] for r in paired]
    target_gate = {}
    target_names = sorted(paired[0]["target_deltas"])
    for fold in range(3):
        fold_rows = [r for r in paired if r["fold"] == fold]
        target_gate[str(fold)] = {
            target: not all(r["target_deltas"][target] > 0.03 for r in fold_rows)
            for target in target_names
        }
    class_guard = all(
        values["fpr"] <= 0.02 or values["recall"] <= 0.0
        for r in paired for values in r["class_deltas"].values()
    )
    gates = {
        "mean_parameter_relative_reduction_at_least_0.03": float(np.mean(reductions)) >= 0.03,
        "positive_parameter_reduction_all_3_seeds_each_fold": all(
            all(r["parameter_relative_reduction"] > 0.0 for r in paired if r["fold"] == fold)
            for fold in range(3)
        ),
        "macro_aupr_drop_no_more_than_0.005_each_pair": min(aupr) >= -0.005,
        "subset_accuracy_drop_no_more_than_0.03_each_pair": min(subset) >= -0.03,
        "no_recall_gain_with_fpr_increase_over_0.02": class_guard,
        "no_target_worse_by_over_0.03_in_all_3_seeds_within_any_fold": all(
            all(values.values()) for values in target_gate.values()
        ),
    }
    return {"evaluated": True, "all_pass": all(gates.values()), "gates": gates, "per_fold_target_guard": target_gate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    doc = json.loads(Path(args.result).read_text(encoding="utf-8"))
    rows = doc["results"]
    _, missing_rows, missing_fields, paired = audit_rows(rows)
    protocol_complete = len(rows) == 18 and not missing_rows and not missing_fields
    summary = {
        "mean_parameter_relative_reduction": float(np.mean([r["parameter_relative_reduction"] for r in paired])) if paired else None,
        "mean_macro_aupr_delta": float(np.mean([r["macro_aupr_delta"] for r in paired])) if paired else None,
        "mean_subset_delta": float(np.mean([r["subset_delta"] for r in paired])) if paired else None,
    }
    gates = promotion_gates(paired, protocol_complete)
    result = {
        "runs_expected": 18,
        "runs_found": len(rows),
        "missing_rows": missing_rows,
        "missing_required_audit_fields": missing_fields,
        "paired": paired,
        "available_summary": summary,
        "protocol_complete": protocol_complete,
        "promotion_allowed": bool(gates["all_pass"]),
        "promotion_gates": gates,
        "decision": "Frozen P5d screen only: no tuning, calibration, threshold selection, or test access.",
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"protocol_complete": protocol_complete, **summary, "promotion_allowed": result["promotion_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
