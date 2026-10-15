"""Fixed confirmation of the Raw-IQ plus Stage1 SawFM evidence source."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_early_complex_sawfm_identifiability import (  # noqa: E402
    _development_groups, _evaluate, _fit_fixed_probe, _fixed_probe_scores,
    _role_groups, _wrap_base_extract, _wrap_power_extract,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


VARIANTS = ("C0_VisibilityFixed", "C1_RawPlusStage1Fixed", "C2_ShuffledControl")


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    replay_args = copy.copy(args)
    replay_args.output_root = args.output_root / "frozen_replay"
    original_base = stable._extract_base_indices
    original_power = stable._extract_power_identities
    try:
        stable._extract_base_indices = _wrap_base_extract(original_base)
        stable._extract_power_identities = _wrap_power_extract(original_power)
        _, audit = selective._run_seed(replay_args, config, labels, identities, seed, device, prior_summary)
    finally:
        stable._extract_base_indices = original_base
        stable._extract_power_identities = original_power
    payloads = selective._PAYLOADS
    fit = _role_groups(payloads, "fit", args.levels)
    scale = [group for group in _role_groups(payloads, "scale", args.levels) if group["name"] != "root"]
    development = _development_groups(payloads, args.levels)
    specifications = {
        "C0_VisibilityFixed": ("P1_VisibilityStats", 0.01, None),
        "C1_RawPlusStage1Fixed": ("P6_RawPlusStage1", 0.1, None),
        "C2_ShuffledControl": ("P6_RawPlusStage1", 0.1, 461000 + int(seed)),
    }
    rows = []
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    for variant, (source, c_value, shuffle_seed) in specifications.items():
        bundle = _fit_fixed_probe(fit, source, c_value, shuffle_seed)
        scale_scores = _fixed_probe_scores(bundle, scale, source)
        development_scores = _fixed_probe_scores(bundle, development, source)
        row = {
            "seed": int(seed), "variant": variant, "source": source,
            "c": float(c_value), "shuffle_seed": shuffle_seed,
            "scale_confirmation": _evaluate(scale, scale_scores),
            "development_reproduction": _evaluate(development, development_scores),
            "identity_overlap": int(audit["identity_overlap"]), "locked_test_used": False,
        }
        rows.append(row)
        variant_root = seed_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        _write(variant_root / "metrics.json", row)
        np.savez_compressed(
            str(variant_root / "scale_predictions.npz"),
            **{f"score_{level}": score for level, score in scale_scores.items()},
            **{f"target_{group['name']}": group["targets"] for group in scale},
        )
        _write(variant_root / "run_receipt.json", {
            "status": "completed", "seed": int(seed), "variant": variant,
            "source": source, "c": float(c_value), "shuffle_seed": shuffle_seed,
            "f2_checkpoint_sha256": audit["f2_checkpoint_sha256"],
            "identity_overlap": int(audit["identity_overlap"]), "locked_test_used": False,
        })
    _write(seed_root / "source_and_identity_audit.json", audit)
    return rows, audit


def _aggregate(rows):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            role: {
                metric: _mean_ci(
                    [row[role][metric] for row in selected],
                    471000 + variant_index * 100 + role_index * 20 + metric_index,
                )
                for metric_index, metric in enumerate(("mean_power_auroc", "worst_power_auroc", "mean_power_aupr"))
            }
            for role_index, role in enumerate(("scale_confirmation", "development_reproduction"))
        }
    return result


def _gate(rows, audits):
    indexed = {(row["seed"], row["variant"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    per_seed = []
    for seed in seeds:
        c0 = indexed[(seed, "C0_VisibilityFixed")]["scale_confirmation"]
        c1 = indexed[(seed, "C1_RawPlusStage1Fixed")]
        c2 = indexed[(seed, "C2_ShuffledControl")]["scale_confirmation"]
        scale = c1["scale_confirmation"]
        development = c1["development_reproduction"]
        checks = {
            "scale_mean_auroc_0p90": scale["mean_power_auroc"] >= 0.90,
            "scale_worst_auroc_0p85": scale["worst_power_auroc"] >= 0.85,
            "scale_50db_auroc_0p85": scale["by_level"]["50"]["auroc"] >= 0.85,
            "delta_vs_c0_0p03": scale["mean_power_auroc"] >= c0["mean_power_auroc"] + 0.03,
            "development_mean_auroc_0p90": development["mean_power_auroc"] >= 0.90,
            "shuffled_mean_auroc_0p60": c2["mean_power_auroc"] <= 0.60,
        }
        per_seed.append({"seed": int(seed), "checks": checks, "passed": all(checks.values())})
    audit_pass = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    passed_count = int(sum(item["passed"] for item in per_seed))
    return {"per_seed": per_seed, "passed_seed_count": passed_count, "audit_passed": audit_pass, "passed": passed_count >= 2 and audit_pass}


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    prior_summary = _read(args.prior_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device, prior_summary)
        rows.extend(seed_rows); audits.append(audit)
    aggregate = _aggregate(rows)
    gate = _gate(rows, audits)
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {"seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS), "fixed_hyperparameters": True},
        "audits": audits, "per_run": rows, "aggregate": aggregate, "promotion_gate": gate,
        "next_decision": "design_small_raw_stage1_sawfm_frontend_keep_h2_as_decision_baseline" if gate["passed"] else "stop_early_complex_frontend_hypothesis",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_early_complex_sawfm_confirmation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--frozen-visibility-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration")
    parser.add_argument("--prior-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration" / "route_a_stable_expert_visibility_calibration_summary.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_early_complex_sawfm_confirmation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
