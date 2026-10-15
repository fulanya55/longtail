"""Create a read-only error atlas for the stopped P2o confirmation.

This analysis is diagnostic only.  It never changes the frozen P2m/P2n method
and it records that the P2o identities are confirmation-only and unavailable
for subsequent development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATA_SEEDS = (2800000000, 2900000000)
MODEL_SEEDS = (2041, 2042, 2043)
REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
POWERS = (25, 35, 50)
EXACT_PEARSON_LIMIT = 0.60
SEPARATED_PEARSON_LIMIT = {
    "ID_control": 0.55,
    "near_low": 0.50,
    "near_high": 0.15,
    "far_low": 0.85,
    "far_high": 0.25,
}


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error_us = np.abs(np.asarray(prediction) - np.asarray(target)) * 1.0e6
    ratio = np.asarray(prediction) / np.maximum(np.asarray(target), 1.0e-12)
    nontrivial = error_us >= 0.5
    harmonic = np.zeros_like(nontrivial, dtype=bool)
    for candidate in (0.25, 0.5, 2.0, 3.0, 4.0):
        harmonic |= np.abs(ratio - candidate) <= 0.08 * candidate
    return {
        "count": int(error_us.size),
        "mae_us": float(np.mean(error_us)),
        "median_ae_us": float(np.median(error_us)),
        "p90_ae_us": float(np.quantile(error_us, 0.90)),
        "p95_ae_us": float(np.quantile(error_us, 0.95)),
        "max_ae_us": float(np.max(error_us)),
        "pearson": pearson(prediction, target),
        "error_ge_0p5us_rate": float(np.mean(error_us >= 0.5)),
        "error_ge_1us_rate": float(np.mean(error_us >= 1.0)),
        "harmonic_like_error_rate": float(np.mean(harmonic & nontrivial)),
    }


def load_predictions(root: Path, data_seed: int, regime: str, model_seed: int) -> dict[str, np.ndarray]:
    path = root / f"data_seed_{data_seed}" / "physical" / regime / f"seed_{model_seed}" / "predictions.npz"
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def build_atlas(root: Path) -> dict[str, object]:
    summaries = {}
    exact_rows, separated_rows = [], []
    for data_seed in DATA_SEEDS:
        summary_path = root / f"data_seed_{data_seed}" / "physical" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[str(data_seed)] = {
            "promotion_passed": bool(summary["promotion_passed"]),
            "gates": summary["promotion_gates"],
            "identity_sha256": summary["identity_audit"]["identity_sha256"],
        }
        for regime in REGIMES:
            # Exact analytic output is deterministic and repeated for every model seed.
            exact = load_predictions(root, data_seed, regime, MODEL_SEEDS[0])
            overall = metrics(exact["analytic_exact"], exact["target"])
            exact_rows.append({"data_seed": data_seed, "regime": regime, "power": "all", **overall})
            for power in POWERS:
                selected = exact["power"] == power
                exact_rows.append({
                    "data_seed": data_seed,
                    "regime": regime,
                    "power": power,
                    **metrics(exact["analytic_exact"][selected], exact["target"][selected]),
                })
            for model_seed in MODEL_SEEDS:
                separated = load_predictions(root, data_seed, regime, model_seed)
                overall = metrics(separated["analytic_separated"], separated["target"])
                separated_rows.append({
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "regime": regime,
                    "power": "all",
                    **overall,
                })
                for power in POWERS:
                    selected = separated["power"] == power
                    separated_rows.append({
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "regime": regime,
                        "power": power,
                        **metrics(
                            separated["analytic_separated"][selected],
                            separated["target"][selected],
                        ),
                    })

    exact_failures = [
        row for row in exact_rows
        if row["power"] == "all" and row["pearson"] < EXACT_PEARSON_LIMIT
    ]
    exact_failure_outliers = []
    for failure in exact_failures:
        prediction = load_predictions(
            root, int(failure["data_seed"]), str(failure["regime"]), MODEL_SEEDS[0]
        )
        error_us = np.abs(prediction["analytic_exact"] - prediction["target"]) * 1.0e6
        # Power views repeat the same exact component; report each identity once.
        first_view = prediction["power"] == POWERS[0]
        positions = np.flatnonzero(first_view)
        ranked = positions[np.argsort(error_us[positions])[::-1]]
        for position in ranked[:10]:
            if error_us[position] < 0.5:
                break
            exact_failure_outliers.append({
                "data_seed": int(failure["data_seed"]),
                "regime": str(failure["regime"]),
                "sample_index": int(prediction["sample_index"][position]),
                "target_us": float(prediction["target"][position] * 1.0e6),
                "prediction_us": float(prediction["analytic_exact"][position] * 1.0e6),
                "prediction_target_ratio": float(
                    prediction["analytic_exact"][position] / prediction["target"][position]
                ),
                "absolute_error_us": float(error_us[position]),
                "selected_peak_score": float(prediction["exact_score"][position]),
            })
    separated_means = []
    for data_seed in DATA_SEEDS:
        for regime in REGIMES:
            selected = [
                row for row in separated_rows
                if row["data_seed"] == data_seed and row["regime"] == regime and row["power"] == "all"
            ]
            correlations = [row["pearson"] for row in selected]
            separated_means.append({
                "data_seed": data_seed,
                "regime": regime,
                "pearson_mean": float(np.mean(correlations)),
                "pearson_min": float(np.min(correlations)),
                "pearson_max": float(np.max(correlations)),
                "model_seed_range": float(np.max(correlations) - np.min(correlations)),
                "limit": SEPARATED_PEARSON_LIMIT[regime],
                "margin": float(np.mean(correlations) - SEPARATED_PEARSON_LIMIT[regime]),
                "model_seed_members": [
                    {
                        "model_seed": int(row["model_seed"]),
                        "pearson": float(row["pearson"]),
                        "mae_us": float(row["mae_us"]),
                        "p95_ae_us": float(row["p95_ae_us"]),
                    }
                    for row in selected
                ],
            })
    separated_failures = [row for row in separated_means if row["margin"] < 0.0]

    worst_exact_power = min(
        (row for row in exact_rows if row["power"] != "all"),
        key=lambda row: row["pearson"],
    )
    worst_separated_power = min(
        (row for row in separated_rows if row["power"] != "all"),
        key=lambda row: row["pearson"],
    )
    return {
        "experiment": "P2o stopped-confirmation failure atlas",
        "decision": "STOPPED_AT_PHYSICAL_GATE",
        "system_attachment_executed": False,
        "summaries": summaries,
        "failed_exact_gates": exact_failures,
        "exact_failure_outliers": exact_failure_outliers,
        "failed_separated_mean_gates": separated_failures,
        "worst_exact_power_slice": worst_exact_power,
        "worst_separated_power_slice": worst_separated_power,
        "exact_rows": exact_rows,
        "separated_rows": separated_rows,
        "diagnosis": {
            "exact_peak_rule_instability": bool(exact_failures),
            "separator_seed_instability": bool(any(row["model_seed_range"] >= 0.20 for row in separated_failures)),
            "failure_is_not_global_mae_regression": bool(all(row["mae_us"] < 1.0 for row in exact_failures)),
            "interpretation": (
                "The frozen peak rule remains low-MAE but is not correlation-stable on every fresh narrow-range seed; "
                "the separated far-low result is additionally dominated by model-seed variation. A revised method "
                "must use a fresh development seed and target harmonic/peak consistency plus separator uncertainty, "
                "not relax the confirmation thresholds."
            ),
        },
        "development_prohibition": {
            "excluded_data_seeds": list(DATA_SEEDS),
            "may_tune_on_p2o_predictions": False,
            "locked_test_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/p2o_analytic_ood_confirmation_20260801")
    parser.add_argument(
        "--output",
        default="outputs/p2o_analytic_ood_confirmation_20260801/failure_atlas.json",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = build_atlas(project / args.root)
    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "failed_exact_gates": result["failed_exact_gates"],
        "failed_separated_mean_gates": result["failed_separated_mean_gates"],
        "worst_exact_power_slice": result["worst_exact_power_slice"],
        "worst_separated_power_slice": result["worst_separated_power_slice"],
        "diagnosis": result["diagnosis"],
    }, indent=2))


if __name__ == "__main__":
    main()
