"""Aggregate two frozen P2n SawFM-period OOD confirmation identity seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_INDEX, paired_identity_bootstrap, variant_record
from cj0724.p2n_analytic_saw_full_system import MODEL_SEEDS, REGIMES


DATA_SEEDS = (2800000000, 2900000000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/p2o_analytic_saw_ood_confirmation_20260801")
    parser.add_argument("--method-freeze", default="outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json")
    parser.add_argument("--preregister", default="configs/p2o_analytic_saw_ood_confirmation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2o_analytic_saw_ood_confirmation_20260801/final")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    input_root = project / args.input_root
    method_path = project / args.method_freeze
    preregister_path = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    method_hash = sha256_file(method_path)
    (output_root / "confirmation_access_receipt.json").write_text(json.dumps({
        "method_freeze": str(method_path.resolve()),
        "method_freeze_sha256": method_hash,
        "preregister": str(preregister_path.resolve()),
        "preregister_sha256": sha256_file(preregister_path),
        "access_started_unix": time.time(),
        "confirmation_fit_or_selection": False,
        "locked_test_used": False,
    }, indent=2), encoding="utf-8")

    summaries, runs, bootstrap_by_regime = {}, [], {}
    identity_hashes = []
    for data_seed in DATA_SEEDS:
        seed_root = input_root / f"seed_{data_seed}"
        system_summary_path = seed_root / "analytic_system" / "summary.json"
        physical_summary_path = seed_root / "physical" / "summary.json"
        system = json.loads(system_summary_path.read_text(encoding="utf-8"))
        physical = json.loads(physical_summary_path.read_text(encoding="utf-8"))
        if not system.get("promotion_passed", False) or not physical.get("promotion_passed", False):
            raise RuntimeError(f"Frozen P2n confirmation seed {data_seed} failed before aggregation")
        identity_hash = physical["identity_audit"]["identity_sha256"]
        identity_hashes.append(identity_hash)
        summaries[str(data_seed)] = {
            "system_summary": str(system_summary_path.resolve()),
            "system_summary_sha256": sha256_file(system_summary_path),
            "physical_summary": str(physical_summary_path.resolve()),
            "physical_summary_sha256": sha256_file(physical_summary_path),
            "identity_count": physical["identity_audit"]["identity_count"],
            "identity_sha256": identity_hash,
            "paired_across_regimes": physical["identity_audit"]["paired_across_regimes"],
            "all_excluded_overlaps_zero": all(
                exclusion["overlap_count"] == 0
                for regime in physical["identity_audit"]["regimes"]
                for exclusion in regime["excluded_prior_splits"]
            ),
        }
        runs.extend({"data_seed": data_seed, **row} for row in system["runs"])

    if len(set(identity_hashes)) != len(identity_hashes):
        raise RuntimeError("P2o confirmation data seeds are not identity distinct")

    for regime_index, regime in enumerate(REGIMES):
        bootstrap_rows = []
        for data_seed in DATA_SEEDS:
            for model_seed in MODEL_SEEDS:
                record_path = (
                    input_root / f"seed_{data_seed}" / "analytic_system" / "analytic_records"
                    / regime / f"seed_{model_seed}" / "development_per_sample.npz"
                )
                with np.load(record_path, allow_pickle=False) as saved:
                    arrays = {name: saved[name] for name in saved.files}
                system_record_path = (
                    input_root / f"seed_{data_seed}" / "system_records"
                    / regime / f"seed_{model_seed}" / "development_per_sample.npz"
                )
                with np.load(system_record_path, allow_pickle=False) as saved:
                    parameter_scale = saved["parameter_scale"]
                payload = {
                    "sample_indices": arrays["sample_index"], "power": arrays["power_db"],
                    "labels": arrays["labels"], "target": arrays["target"], "mask": arrays["mask"],
                }
                records = {
                    "P2k_learned": variant_record(
                        payload, arrays["routed_probability"], arrays["routed_thresholds"],
                        arrays["learned_prediction"], parameter_scale,
                    ),
                    "P2n_analytic": variant_record(
                        payload, arrays["routed_probability"], arrays["routed_thresholds"],
                        arrays["analytic_prediction"], parameter_scale,
                    ),
                }
                bootstrap_rows.append({"identity": arrays["receiver_noise_seed"], "records": records})
        bootstrap_by_regime[regime] = paired_identity_bootstrap(
            bootstrap_rows, "P2n_analytic", "P2k_learned",
            "predicted_gate_parameter_nmae", args.bootstrap_replicates, 20261100 + regime_index,
        )

    per_data_seed_regime = {}
    for data_seed in DATA_SEEDS:
        per_data_seed_regime[str(data_seed)] = {}
        for regime in REGIMES:
            subset = [row for row in runs if row["data_seed"] == data_seed and row["regime"] == regime]
            per_data_seed_regime[str(data_seed)][regime] = {
                "mean_full_improvement_vs_P2k": float(np.mean([row["full_improvement_vs_P2k"] for row in subset])),
                "mean_miss_delta_vs_P2k": float(np.mean([row["miss_delta_vs_P2k"] for row in subset])),
                "mean_worst_combo_delta_vs_P2k": float(np.mean([row["worst_combo_delta_vs_P2k"] for row in subset])),
            }

    classification_unchanged = all(
        json.loads((input_root / f"seed_{seed}" / "analytic_system" / "summary.json").read_text(encoding="utf-8"))
        ["promotion_gates"]["classification_arrays_unchanged"]
        for seed in DATA_SEEDS
    )
    other_parameters_unchanged = all(
        json.loads((input_root / f"seed_{seed}" / "analytic_system" / "summary.json").read_text(encoding="utf-8"))
        ["promotion_gates"]["other_11_parameter_outputs_unchanged"]
        for seed in DATA_SEEDS
    )
    gates = {
        "all_30_runs_improve_vs_P2k": bool(all(row["full_improvement_vs_P2k"] > 0 for row in runs)),
        "each_data_seed_regime_mean_improvement_at_least_0p002": bool(all(
            value["mean_full_improvement_vs_P2k"] >= 0.002
            for by_regime in per_data_seed_regime.values() for value in by_regime.values()
        )),
        "each_data_seed_regime_mean_miss_delta_nonpositive": bool(all(
            value["mean_miss_delta_vs_P2k"] <= 0
            for by_regime in per_data_seed_regime.values() for value in by_regime.values()
        )),
        "each_data_seed_regime_mean_worst_delta_nonpositive": bool(all(
            value["mean_worst_combo_delta_vs_P2k"] <= 0
            for by_regime in per_data_seed_regime.values() for value in by_regime.values()
        )),
        "each_regime_combined_bootstrap_ci95_high_nonpositive": bool(all(
            value["ci95_high"] <= 0 for value in bootstrap_by_regime.values()
        )),
        "classification_arrays_unchanged": bool(classification_unchanged),
        "other_11_parameter_outputs_unchanged": bool(other_parameters_unchanged),
        "identity_isolation_and_pairing": bool(all(
            value["paired_across_regimes"] and value["all_excluded_overlaps_zero"]
            for value in summaries.values()
        )),
    }
    summary = {
        "experiment": "P2o frozen analytic SawFM-period OOD independent confirmation",
        "method_freeze_sha256": method_hash,
        "preregister_sha256": sha256_file(preregister_path),
        "identity_audit": summaries,
        "per_data_seed_regime": per_data_seed_regime,
        "combined_identity_bootstrap": bootstrap_by_regime,
        "confirmation_gates": gates,
        "confirmation_passed": bool(all(gates.values())),
        "evidence_boundary": {
            "confirmed": "SawFM sweep-period marginal ID/near/far OOD under frozen predicted-label parameter metrics",
            "not_confirmed": [
                "VFT and ISRJ parameter OOD",
                "new parameter-correlation structures",
                "analytic false-positive candidate runtime",
                "locked test",
            ],
        },
        "confirmation_fit_or_selection": False,
        "locked_test_used": False,
        "runs": runs,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "confirmation_gates": gates,
        "confirmation_passed": summary["confirmation_passed"],
        "per_data_seed_regime": per_data_seed_regime,
        "combined_identity_bootstrap": bootstrap_by_regime,
    }, indent=2))


if __name__ == "__main__":
    main()
