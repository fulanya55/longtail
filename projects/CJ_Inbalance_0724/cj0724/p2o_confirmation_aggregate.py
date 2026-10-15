"""Aggregate the two frozen P2o physical and full-system confirmation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable


DATA_SEEDS = (2800000000, 2900000000)
REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
EXPECTED_FREEZE_SHA256 = "ab1d5efee7f0a35278a3e9e3b53d68e70d9c40adcd21a676a27195bd6a5f0f10"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gates(
    physical_summaries: list[dict[str, object]],
    system_summaries: list[dict[str, object]],
    rows: list[dict[str, object]],
    cross_overlap: int,
    freeze_unchanged: bool,
) -> dict[str, bool]:
    per_seed_regime = []
    for data_seed, summary in zip(DATA_SEEDS, system_summaries):
        for regime in REGIMES:
            value = summary["per_regime"][regime]
            per_seed_regime.append({"data_seed": data_seed, "regime": regime, **value})
    return {
        "both_physical_runs_pass_all_p2m_gates": bool(all(summary.get("promotion_passed", False) for summary in physical_summaries)),
        "both_system_runs_pass_all_p2n_gates": bool(all(summary.get("promotion_passed", False) for summary in system_summaries)),
        "all_30_full_improvement_vs_safe_at_least_0p05": bool(all(row["full_improvement_vs_safe"] >= 0.05 for row in rows)),
        "all_30_full_improvement_vs_P2k_positive": bool(all(row["full_improvement_vs_P2k"] > 0.0 for row in rows)),
        "each_seed_regime_mean_full_improvement_vs_P2k_at_least_0p002": bool(all(row["mean_full_improvement_vs_P2k"] >= 0.002 for row in per_seed_regime)),
        "each_seed_regime_mean_miss_delta_vs_P2k_at_most_zero": bool(all(row["mean_miss_delta_vs_P2k"] <= 0.0 for row in per_seed_regime)),
        "each_seed_regime_mean_worst_combo_delta_vs_P2k_at_most_zero": bool(all(row["mean_worst_combo_delta_vs_P2k"] <= 0.0 for row in per_seed_regime)),
        "each_of_10_bootstrap_full_delta_ci95_high_at_most_zero": bool(all(row["bootstrap"]["ci95_high"] <= 0.0 for row in per_seed_regime)),
        "classification_and_other_outputs_unchanged": bool(all(
            summary["promotion_gates"]["classification_arrays_unchanged"]
            and summary["promotion_gates"]["other_11_parameter_outputs_unchanged"]
            for summary in system_summaries
        )),
        "cross_data_seed_identity_overlap_zero": bool(cross_overlap == 0),
        "frozen_artifact_hashes_unchanged": bool(freeze_unchanged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/p2o_analytic_ood_confirmation_20260801")
    parser.add_argument("--data-root-template", default="generated_data/p2o_analytic_ood_confirmation_seed{seed}_ID_control_rechunked")
    parser.add_argument("--freeze", default="outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json")
    parser.add_argument("--preregister", default="configs/p2o_two_seed_analytic_ood_confirmation_preregister_20260801.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.root
    freeze_path = project / args.freeze
    preregister = project / args.preregister
    physical_summaries = [
        json.loads((root / f"data_seed_{seed}" / "physical" / "summary.json").read_text(encoding="utf-8"))
        for seed in DATA_SEEDS
    ]
    system_summaries = [
        json.loads((root / f"data_seed_{seed}" / "system" / "summary.json").read_text(encoding="utf-8"))
        for seed in DATA_SEEDS
    ]
    identity_sets = []
    for seed in DATA_SEEDS:
        data_root = project / args.data_root_template.format(seed=seed)
        identities, _ = read_identity_metadata(data_root / "level_04__jnr_25dB" / "closedset_6class_train.mat")
        identity_sets.append(np.asarray(identities, dtype=np.int64))
    cross_overlap = int(np.intersect1d(identity_sets[0], identity_sets[1]).size)
    rows = []
    for data_seed, summary in zip(DATA_SEEDS, system_summaries):
        rows.extend([{**row, "data_seed": data_seed} for row in summary["runs"]])
    freeze_unchanged = sha256_file(freeze_path) == EXPECTED_FREEZE_SHA256
    gates = build_gates(physical_summaries, system_summaries, rows, cross_overlap, freeze_unchanged)
    passed = bool(all(gates.values()))
    per_data_seed = {}
    for data_seed, summary in zip(DATA_SEEDS, system_summaries):
        subset = [row for row in rows if row["data_seed"] == data_seed]
        per_data_seed[str(data_seed)] = {
            "mean_full_improvement_vs_safe": float(np.mean([row["full_improvement_vs_safe"] for row in subset])),
            "minimum_full_improvement_vs_safe": float(min(row["full_improvement_vs_safe"] for row in subset)),
            "mean_full_improvement_vs_P2k": float(np.mean([row["full_improvement_vs_P2k"] for row in subset])),
            "minimum_full_improvement_vs_P2k": float(min(row["full_improvement_vs_P2k"] for row in subset)),
            "per_regime": summary["per_regime"],
        }
    output = {
        "experiment": "P2o two-new-identity-seed analytic SawFM OOD confirmation",
        "preregister_sha256": sha256_file(preregister),
        "p2n_method_freeze_sha256": sha256_file(freeze_path),
        "identity_audit": {
            "data_seeds": list(DATA_SEEDS),
            "identity_counts": [int(len(values)) for values in identity_sets],
            "identity_sha256": [hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest() for values in identity_sets],
            "cross_seed_overlap": cross_overlap,
        },
        "confirmation_gates": gates,
        "confirmation_passed": passed,
        "aggregate": {
            "mean_full_improvement_vs_safe": float(np.mean([row["full_improvement_vs_safe"] for row in rows])),
            "minimum_full_improvement_vs_safe": float(min(row["full_improvement_vs_safe"] for row in rows)),
            "mean_full_improvement_vs_P2k": float(np.mean([row["full_improvement_vs_P2k"] for row in rows])),
            "minimum_full_improvement_vs_P2k": float(min(row["full_improvement_vs_P2k"] for row in rows)),
            "maximum_bootstrap_ci95_high": float(max(
                value["bootstrap"]["ci95_high"]
                for summary in system_summaries for value in summary["per_regime"].values()
            )),
        },
        "per_data_seed": per_data_seed,
        "physical_summaries_sha256": [sha256_file(root / f"data_seed_{seed}" / "physical" / "summary.json") for seed in DATA_SEEDS],
        "system_summaries_sha256": [sha256_file(root / f"data_seed_{seed}" / "system" / "summary.json") for seed in DATA_SEEDS],
        "runs": rows,
        "interpretation": (
            "P2o confirmed the frozen P2n analytic SawFM attachment on two new identity seeds across all five OOD regimes."
            if passed else
            "P2o confirmation failed; retain P2n as development evidence only and do not retune on confirmation identities."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (root / "summary.json").write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(json.dumps({
        "confirmation_gates": gates, "confirmation_passed": passed,
        "aggregate": output["aggregate"], "interpretation": output["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
