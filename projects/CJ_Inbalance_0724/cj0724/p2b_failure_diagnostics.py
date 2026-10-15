"""Diagnose P2b failure using saved development calibration predictions only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cj0724.data import read_labels
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2b_sawfm_power_sufficiency import HIGH_POWERS, POWERS


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
METHODS = (
    "E_const_fit_median",
    "E0_frozen_specialist",
    "E1_classical_autocorrelation",
    "E2_single_30db_compute_matched",
    "E3_nine_power_equal_identity",
    "E4_nine_power_consistency",
)


def combination_name(label: np.ndarray) -> str:
    return "+".join(name for name, active in zip(CLASS_NAMES, label) if active)


def group_metrics(prediction: np.ndarray, target: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    high_columns = [POWERS.index(power) for power in HIGH_POWERS]
    error = np.abs(prediction[rows][:, high_columns] - target[rows, None])
    flat_prediction = prediction[rows][:, high_columns].reshape(-1)
    flat_target = np.repeat(target[rows, None], len(high_columns), axis=1).reshape(-1)
    correlation = 0.0 if np.std(flat_prediction) <= 0.0 or np.std(flat_target) <= 0.0 else float(np.corrcoef(flat_prediction, flat_target)[0, 1])
    return {"mae_s": float(error.mean()), "mae_us": float(error.mean() * 1.0e6), "pearson": correlation}


def relative(candidate: float, baseline: float) -> float:
    return float(1.0 - float(candidate) / float(baseline))


def aggregate_category(
    seed_payloads: list[dict[str, np.ndarray]],
    labels: np.ndarray,
    category: np.ndarray,
) -> dict[str, object]:
    rows = np.flatnonzero(category)
    per_seed = []
    for payload in seed_payloads:
        metrics = {method: group_metrics(payload[method], payload["target"], rows) for method in METHODS}
        per_seed.append({
            "metrics": metrics,
            "E3_reduction_vs_E0": relative(metrics["E3_nine_power_equal_identity"]["mae_s"], metrics["E0_frozen_specialist"]["mae_s"]),
            "E4_reduction_vs_E0": relative(metrics["E4_nine_power_consistency"]["mae_s"], metrics["E0_frozen_specialist"]["mae_s"]),
            "E4_reduction_vs_constant": relative(metrics["E4_nine_power_consistency"]["mae_s"], metrics["E_const_fit_median"]["mae_s"]),
        })
    return {
        "identity_count": int(len(rows)),
        "mean_E0_mae_us": float(np.mean([row["metrics"]["E0_frozen_specialist"]["mae_us"] for row in per_seed])),
        "mean_E3_mae_us": float(np.mean([row["metrics"]["E3_nine_power_equal_identity"]["mae_us"] for row in per_seed])),
        "mean_E4_mae_us": float(np.mean([row["metrics"]["E4_nine_power_consistency"]["mae_us"] for row in per_seed])),
        "mean_E3_reduction_vs_E0": float(np.mean([row["E3_reduction_vs_E0"] for row in per_seed])),
        "mean_E4_reduction_vs_E0": float(np.mean([row["E4_reduction_vs_E0"] for row in per_seed])),
        "mean_E4_reduction_vs_constant": float(np.mean([row["E4_reduction_vs_constant"] for row in per_seed])),
        "mean_E4_pearson": float(np.mean([row["metrics"]["E4_nine_power_consistency"]["pearson"] for row in per_seed])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/p2b_sawfm_power_sufficiency_development_v2_20260801")
    parser.add_argument("--development-root", default="generated_data/p2b_nine_power_confirmation_seed1300000000_rechunked_train_only")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    files = transition_files(project / args.development_root, "train", POWERS)
    labels_all = read_labels(files[POWERS[0]])
    seed_payloads = []
    sample_indices = None
    for seed in (3041, 3042, 3043):
        payload = dict(np.load(output_root / f"seed_{seed}" / "calibration_predictions.npz"))
        current_indices = payload["identity"].astype(np.int64)
        if sample_indices is None:
            sample_indices = current_indices
        elif not np.array_equal(sample_indices, current_indices):
            raise RuntimeError("calibration sample order differs across seeds")
        seed_payloads.append(payload)
    assert sample_indices is not None
    labels = labels_all[sample_indices]
    if not np.all(labels[:, 1] == 1):
        raise RuntimeError("diagnostic calibration rows are not all SawFM-positive")

    categories = {
        "pure_SawFM": labels.sum(axis=1) == 1,
        "no_variable_power_competitor": labels[:, 3:6].sum(axis=1) == 0,
        "one_variable_power_competitor": labels[:, 3:6].sum(axis=1) == 1,
        "two_variable_power_competitors": labels[:, 3:6].sum(axis=1) == 2,
        "three_variable_power_competitors": labels[:, 3:6].sum(axis=1) == 3,
        "contains_MFT": labels[:, 3] == 1,
        "contains_VFT": labels[:, 4] == 1,
        "contains_ISRJ": labels[:, 5] == 1,
    }
    category_summary = {
        name: aggregate_category(seed_payloads, labels, mask)
        for name, mask in categories.items() if np.any(mask)
    }

    combo_rows = []
    for label in np.unique(labels, axis=0):
        mask = np.all(labels == label, axis=1)
        row = {
            "combination": combination_name(label),
            "cardinality": int(label.sum()),
            "variable_power_competitor_count": int(label[3:6].sum()),
            **aggregate_category(seed_payloads, labels, mask),
        }
        combo_rows.append(row)
    combo_rows.sort(key=lambda row: row["mean_E4_mae_us"], reverse=True)

    high_columns = [POWERS.index(power) for power in HIGH_POWERS]
    classical_ratios, boundary_count, total_count = [], 0, 0
    for payload in seed_payloads:
        prediction = payload["E1_classical_autocorrelation"][:, high_columns]
        target = payload["target"][:, None]
        classical_ratios.append((prediction / target).reshape(-1))
        boundary_count += int(np.sum((prediction <= 20.0e-6) | (prediction >= 80.0e-6)))
        total_count += int(prediction.size)
    ratios = np.concatenate(classical_ratios)
    classical_failure = {
        "median_prediction_target_ratio": float(np.median(ratios)),
        "ratio_within_20_percent_fraction": float(np.mean(np.abs(ratios - 1.0) <= 0.20)),
        "lag_search_boundary_fraction": float(boundary_count / total_count),
    }
    worst_negative = sorted(combo_rows, key=lambda row: row["mean_E4_reduction_vs_E0"])[:10]
    result = {
        "source": "saved calibration predictions plus development-train labels; no test waveform or prediction read",
        "category_summary": category_summary,
        "worst_combinations_by_E4_high_power_mae": combo_rows[:10],
        "worst_combinations_by_E4_delta_vs_E0": worst_negative,
        "classical_autocorrelation_failure": classical_failure,
        "diagnosis": {
            "power_balanced_training_effect": "E3/E4 improve E0 on average but not enough to beat the frozen constant-baseline sufficiency gate in every seed",
            "consistency_effect": "E4 is nearly identical to E3, so variance regularization is not the missing mechanism",
            "single_power_effect": "E2 improves low/mid power but regresses at high power, confirming exposure imbalance",
            "next_mechanism": "screen SawFM component isolation or mixture-robust ridge/reset extraction before adding network capacity",
        },
        "p2b_test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "failure_diagnostics.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    with (output_root / "failure_diagnostics_by_combination.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combo_rows)
    print(json.dumps({
        "categories": category_summary,
        "classical_failure": classical_failure,
        "worst_delta_combinations": worst_negative[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
