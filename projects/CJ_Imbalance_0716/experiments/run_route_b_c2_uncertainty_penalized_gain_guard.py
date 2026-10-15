"""Uncertainty-penalized gain guard built from cross-seed mean and Q30 gain.

This post-lock exploratory experiment uses only saved development predictions.
No model is fitted here and no locked-test file is read.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_b_c2_cross_seed_expected_gain_guard import (  # noqa: E402
    SEEDS,
    TARGETS,
    _evaluate_target,
    _load_seed,
    _mean_std,
    _read_labels,
    _sha256,
    _write_json,
)


RISK_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20)
PRIMARY_RISK_WEIGHT = 0.10


def _risk_adjusted_score(expected_gain, lower_gain, risk_weight):
    weight = float(risk_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("risk_weight must be in [0, 1]")
    mean = np.asarray(expected_gain, dtype=np.float64)
    lower = np.asarray(lower_gain, dtype=np.float64)
    if mean.shape != lower.shape:
        raise ValueError("expected and lower gain arrays must match")
    return ((1.0 - weight) * mean + weight * lower).astype(np.float32)


def _gain_prediction_paths(source_root, seed):
    base = Path(source_root)
    return (
        base / "E1_ExpectedGain" / "seed_{}".format(seed) / "guard_predictions.npz",
        base / "Q30_LowerGain" / "seed_{}".format(seed) / "guard_predictions.npz",
    )


def _aggregate(rows, risk_weight):
    target_rows = []
    seed_combined = []
    for class_name in TARGETS:
        selected = [row["targets"][class_name] for row in rows]
        target_rows.append(
            {
                "class": class_name,
                "relative_reduction": _mean_std(
                    [row["relative_reduction"] for row in selected]
                ),
                "positive_seed_count": int(
                    sum(row["relative_reduction"] > 0.0 for row in selected)
                ),
                "route_coverage": _mean_std(
                    [row["route_coverage"] for row in selected]
                ),
                "negative_combination_count_total": int(
                    sum(row["negative_combination_count"] for row in selected)
                ),
                "minimum_combination_reduction": float(
                    min(row["minimum_combination_reduction"] for row in selected)
                ),
            }
        )
    for row in rows:
        c2 = np.mean([row["targets"][name]["c2_nmae"] for name in TARGETS])
        final = np.mean([row["targets"][name]["final_nmae"] for name in TARGETS])
        seed_combined.append(1.0 - final / c2)
    seed_target = [
        row["targets"][name]["relative_reduction"]
        for row in rows
        for name in TARGETS
    ]
    return {
        "risk_weight": float(risk_weight),
        "targets": target_rows,
        "mean_seed_target_relative_reduction": float(np.mean(seed_target)),
        "minimum_seed_target_relative_reduction": float(min(seed_target)),
        "two_target_combined_relative_reduction": _mean_std(seed_combined),
        "two_target_positive_seed_count": int(sum(value > 0.0 for value in seed_combined)),
        "negative_combination_count_total": int(
            sum(
                row["targets"][name]["negative_combination_count"]
                for row in rows
                for name in TARGETS
            )
        ),
        "minimum_combination_reduction": float(
            min(
                row["targets"][name]["minimum_combination_reduction"]
                for row in rows
                for name in TARGETS
            )
        ),
    }


def run(args):
    if "test" in str(args.data_path).lower():
        raise ValueError("this exploratory runner must never read a test file")
    labels = _read_labels(args.data_path)
    data = {seed: _load_seed(seed) for seed in SEEDS}
    variants = {}
    for risk_weight in RISK_WEIGHTS:
        name = "R{:02d}_UncertaintyPenalty".format(int(round(risk_weight * 100)))
        rows = []
        for seed in SEEDS:
            expected_path, lower_path = _gain_prediction_paths(args.source_root, seed)
            if not expected_path.is_file() or not lower_path.is_file():
                raise FileNotFoundError("cross-seed gain predictions are missing")
            with np.load(expected_path) as expected_saved, np.load(lower_path) as lower_saved:
                targets = {}
                arrays = {}
                for class_name in TARGETS:
                    expected_gain = expected_saved[class_name + "_predicted_gain"]
                    lower_gain = lower_saved[class_name + "_predicted_gain"]
                    score = _risk_adjusted_score(
                        expected_gain, lower_gain, risk_weight
                    )
                    metrics = _evaluate_target(data[seed][class_name], score, labels)
                    arrays[class_name + "_risk_adjusted_score"] = score
                    arrays[class_name + "_final_gate"] = metrics.pop(
                        "final_gate"
                    ).astype(np.uint8)
                    arrays[class_name + "_final_prediction"] = metrics.pop(
                        "final_prediction"
                    )
                    arrays[class_name + "_base_indices"] = data[seed][class_name][
                        "base_indices"
                    ]
                    targets[class_name] = metrics
            seed_root = args.output_root / name / "seed_{}".format(seed)
            seed_root.mkdir(parents=True, exist_ok=True)
            prediction_path = seed_root / "risk_guard_predictions.npz"
            np.savez_compressed(str(prediction_path), **arrays)
            row = {
                "status": "completed",
                "seed": int(seed),
                "variant": name,
                "risk_weight": float(risk_weight),
                "targets": targets,
                "source_hashes": {
                    "expected_gain_predictions": _sha256(expected_path),
                    "lower_gain_predictions": _sha256(lower_path),
                },
                "predictions_sha256": _sha256(prediction_path),
                "locked_test_used": False,
                "post_locked_test_exploratory": True,
            }
            _write_json(seed_root / "run_receipt.json", row)
            rows.append(row)
        variants[name] = {"aggregate": _aggregate(rows, risk_weight), "runs": rows}
    baseline = variants["R00_UncertaintyPenalty"]["aggregate"]
    primary = variants["R10_UncertaintyPenalty"]["aggregate"]
    promotion = {
        "mean_seed_target_not_lower": bool(
            primary["mean_seed_target_relative_reduction"]
            >= baseline["mean_seed_target_relative_reduction"]
        ),
        "combined_gain_not_lower": bool(
            primary["two_target_combined_relative_reduction"]["mean"]
            >= baseline["two_target_combined_relative_reduction"]["mean"]
        ),
        "minimum_seed_target_improved": bool(
            primary["minimum_seed_target_relative_reduction"]
            > baseline["minimum_seed_target_relative_reduction"]
        ),
        "negative_combinations_reduced": bool(
            primary["negative_combination_count_total"]
            < baseline["negative_combination_count_total"]
        ),
        "worst_combination_improved": bool(
            primary["minimum_combination_reduction"]
            > baseline["minimum_combination_reduction"]
        ),
    }
    promotion["passed"] = bool(all(promotion.values()))
    source_summary = (
        Path(args.source_root)
        / "route_b_c2_cross_seed_expected_gain_guard_summary.json"
    )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "post_locked_test_exploratory": True,
        "eligible_for_existing_locked_test_claim": False,
        "protocol": {
            "risk_score": "(1-lambda)*expected_gain + lambda*Q30_lower_gain",
            "risk_weights": list(RISK_WEIGHTS),
            "primary_risk_weight": PRIMARY_RISK_WEIGHT,
            "base_candidate_gate_unchanged": True,
            "no_additional_fitting": True,
            "source_summary_sha256": _sha256(source_summary),
        },
        "variants": {name: value["aggregate"] for name, value in variants.items()},
        "primary_promotion_gate": promotion,
        "runs": variants,
    }
    summary_path = (
        args.output_root
        / "route_b_c2_uncertainty_penalized_gain_guard_summary.json"
    )
    _write_json(summary_path, payload)
    print(
        json.dumps(
            {"variants": payload["variants"], "promotion": promotion},
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_cross_seed_expected_gain_guard_exploratory_20260726",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_uncertainty_penalized_gain_guard_exploratory_20260726",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
