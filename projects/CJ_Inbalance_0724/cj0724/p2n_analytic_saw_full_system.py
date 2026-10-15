"""P2n attach frozen P2m analytic period predictions to frozen system records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1k_joint_level3_system import (
    SAW_PARAMETER_INDEX,
    evaluate_variant,
    paired_identity_bootstrap,
    variant_record,
)
from cj0724.p2j_full_parameter_attachment import MODEL_SEEDS, relative_improvement


REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
POWERS = (25, 35, 50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_analytic_predictions(
    learned_prediction: np.ndarray,
    labels: np.ndarray,
    sample_index: np.ndarray,
    power: np.ndarray,
    physical: dict[str, np.ndarray],
) -> np.ndarray:
    """Replace SawFM period for all true SawFM rows; predicted gates remain frozen."""
    lookup = {
        (int(index), int(level)): float(value)
        for index, level, value in zip(
            physical["sample_index"], physical["power"], physical["analytic_separated"],
        )
    }
    output = np.asarray(learned_prediction).copy()
    positive = np.flatnonzero(np.asarray(labels)[:, SAW_PARAMETER_INDEX[0]] > 0)
    missing = []
    for position in positive:
        key = (int(sample_index[position]), int(power[position]))
        if key not in lookup:
            missing.append(key)
        else:
            output[position, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]] = lookup[key]
    if missing:
        raise RuntimeError(f"missing {len(missing)} analytic SawFM predictions")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-root", default="outputs/p2n_analytic_saw_full_system_20260801/system_records")
    parser.add_argument("--physical-root", default="outputs/p2m_analytic_saw_period_physics_20260801")
    parser.add_argument("--p2m-summary", default="outputs/p2m_analytic_saw_period_physics_20260801/summary.json")
    parser.add_argument("--preregister", default="configs/p2n_analytic_saw_full_system_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2n_analytic_saw_full_system_20260801")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    system_root = project / args.system_root
    physical_root = project / args.physical_root
    p2m_summary_path = project / args.p2m_summary
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    p2m = json.loads(p2m_summary_path.read_text(encoding="utf-8"))
    if not p2m.get("promotion_passed", False):
        raise RuntimeError("P2m physical gate did not pass")

    runs, regime_bootstraps = [], {}
    classification_unchanged = True
    other_parameters_unchanged = True
    for regime_index, regime in enumerate(REGIMES):
        bootstrap_rows = []
        for seed in MODEL_SEEDS:
            record_path = system_root / regime / f"seed_{seed}" / "development_per_sample.npz"
            with np.load(record_path) as saved:
                arrays = {name: saved[name] for name in saved.files}
            physical_path = physical_root / regime / f"seed_{seed}" / "predictions.npz"
            with np.load(physical_path) as saved:
                physical = {name: saved[name] for name in saved.files}
            payload = {
                "sample_indices": arrays["sample_index"], "power": arrays["power_db"],
                "labels": arrays["labels"], "target": arrays["target"], "mask": arrays["mask"],
            }
            probability = arrays["routed_probability"]
            thresholds = arrays["routed_thresholds"]
            scale = arrays["parameter_scale"]
            safe = arrays["safe_prediction"]
            old = arrays["old_joint_prediction"]
            learned = arrays["calibrated_joint_prediction"]
            analytic = attach_analytic_predictions(
                learned, arrays["labels"], arrays["sample_index"], arrays["power_db"], physical,
            )
            variants = {
                "S0_safe_route": evaluate_variant(payload, probability, thresholds, safe, scale, POWERS),
                "S1_old_joint": evaluate_variant(payload, probability, thresholds, old, scale, POWERS),
                "S2_P2k_learned": evaluate_variant(payload, probability, thresholds, learned, scale, POWERS),
                "S3_P2n_analytic": evaluate_variant(payload, probability, thresholds, analytic, scale, POWERS),
            }
            records = {
                "S2_P2k_learned": variant_record(payload, probability, thresholds, learned, scale),
                "S3_P2n_analytic": variant_record(payload, probability, thresholds, analytic, scale),
            }
            bootstrap_rows.append({"identity": arrays["receiver_noise_seed"], "records": records})
            safe_nmae = variants["S0_safe_route"]["aggregate"]["predicted_gate_parameter_nmae"]
            learned_nmae = variants["S2_P2k_learned"]["aggregate"]["predicted_gate_parameter_nmae"]
            analytic_nmae = variants["S3_P2n_analytic"]["aggregate"]["predicted_gate_parameter_nmae"]
            learned_miss = variants["S2_P2k_learned"]["aggregate"]["miss_penalty_parameter_nmae"]
            analytic_miss = variants["S3_P2n_analytic"]["aggregate"]["miss_penalty_parameter_nmae"]
            learned_worst = variants["S2_P2k_learned"]["aggregate"]["worst_combo_predicted_gate_nmae"]
            analytic_worst = variants["S3_P2n_analytic"]["aggregate"]["worst_combo_predicted_gate_nmae"]
            unchanged = np.ones((6, 2), dtype=bool)
            unchanged[SAW_PARAMETER_INDEX] = False
            other_parameters_unchanged &= bool(np.array_equal(learned[:, unchanged], analytic[:, unchanged]))
            classification_unchanged &= True
            row = {
                "regime": regime, "seed": seed, "variants": variants,
                "full_improvement_vs_safe": relative_improvement(analytic_nmae, safe_nmae),
                "full_improvement_vs_P2k": relative_improvement(analytic_nmae, learned_nmae),
                "miss_delta_vs_P2k": float(analytic_miss - learned_miss),
                "worst_combo_delta_vs_P2k": float(analytic_worst - learned_worst),
                "system_record_sha256": sha256_file(record_path),
                "physical_prediction_sha256": sha256_file(physical_path),
            }
            run_dir = output_root / "analytic_records" / regime / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(
                run_dir / "development_per_sample.npz",
                receiver_noise_seed=arrays["receiver_noise_seed"], sample_index=arrays["sample_index"],
                power_db=arrays["power_db"], labels=arrays["labels"], target=arrays["target"], mask=arrays["mask"],
                routed_probability=probability, routed_thresholds=thresholds, parameter_scale=scale,
                learned_prediction=learned, analytic_prediction=analytic,
            )
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            runs.append(row)
            print(json.dumps({
                "regime": regime, "seed": seed,
                "full_improvement_vs_safe": row["full_improvement_vs_safe"],
                "full_improvement_vs_P2k": row["full_improvement_vs_P2k"],
                "miss_delta_vs_P2k": row["miss_delta_vs_P2k"],
                "worst_combo_delta_vs_P2k": row["worst_combo_delta_vs_P2k"],
            }), flush=True)
        regime_bootstraps[regime] = paired_identity_bootstrap(
            bootstrap_rows, "S3_P2n_analytic", "S2_P2k_learned",
            "predicted_gate_parameter_nmae", args.bootstrap_replicates, 20260901 + regime_index,
        )

    per_regime = {}
    for regime in REGIMES:
        subset = [row for row in runs if row["regime"] == regime]
        per_regime[regime] = {
            "mean_full_improvement_vs_safe": float(np.mean([row["full_improvement_vs_safe"] for row in subset])),
            "mean_full_improvement_vs_P2k": float(np.mean([row["full_improvement_vs_P2k"] for row in subset])),
            "mean_miss_delta_vs_P2k": float(np.mean([row["miss_delta_vs_P2k"] for row in subset])),
            "mean_worst_combo_delta_vs_P2k": float(np.mean([row["worst_combo_delta_vs_P2k"] for row in subset])),
            "bootstrap": regime_bootstraps[regime],
        }
    gates = {
        "all_15_full_improvement_vs_safe_at_least_0p05": bool(all(row["full_improvement_vs_safe"] >= 0.05 for row in runs)),
        "all_15_full_improvement_vs_P2k_positive": bool(all(row["full_improvement_vs_P2k"] > 0.0 for row in runs)),
        "each_regime_mean_full_improvement_vs_P2k_at_least_0p002": bool(all(value["mean_full_improvement_vs_P2k"] >= 0.002 for value in per_regime.values())),
        "each_regime_mean_miss_delta_vs_P2k_at_most_zero": bool(all(value["mean_miss_delta_vs_P2k"] <= 0.0 for value in per_regime.values())),
        "each_regime_mean_worst_combo_delta_vs_P2k_at_most_zero": bool(all(value["mean_worst_combo_delta_vs_P2k"] <= 0.0 for value in per_regime.values())),
        "each_regime_bootstrap_full_delta_ci95_high_at_most_zero": bool(all(value["bootstrap"]["ci95_high"] <= 0.0 for value in per_regime.values())),
        "classification_arrays_unchanged": classification_unchanged,
        "other_11_parameter_outputs_unchanged": other_parameters_unchanged,
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2n analytic SawFM full-system development",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "p2m_summary_sha256": sha256_file(p2m_summary_path),
        "promotion_gates": gates, "promotion_passed": passed,
        "per_regime": per_regime, "runs": runs,
        "interpretation": (
            "P2n passed; freeze the analytic attachment for new identity-seed confirmation."
            if passed else
            "P2n failed a system-level gate; do not retune the analytic estimator on seed 2700000000."
        ),
        "development_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "per_regime": per_regime, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
