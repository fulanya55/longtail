"""Run the five-seed Route-A DB calibration experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("FMB", "SinFM", "MFT", "ISRJ")
LOSSES = ("asl", "db")
THRESHOLD_MODES = (
    "fixed_0p5",
    "global",
    "per_label_f1",
    "shrinkage_per_label_f1",
    "per_label_youden",
)
PRIMARY_MODE = "shrinkage_per_label_f1"


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _tail_order(tail_index):
    return [index for index in range(len(CLASS_NAMES)) if index != tail_index] + [tail_index]


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _run_tapc(args, directory, tail_index, loss, seed):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", loss,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-train-samples", str(args.max_train_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", str(args.label_imbalance_factor),
        "--label-count-tolerance", str(args.label_count_tolerance),
        "--label-rank-order", *[str(value) for value in _tail_order(tail_index)],
        "--selection-metric", "macro_aupr",
        "--output-dir", str(directory),
        "--amp",
    ]
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _collect_run(directory, tail_name, loss, seed):
    metrics_by_mode = _read(directory / "metrics_threshold_modes.json")
    calibration = _read(directory / "calibration.json")
    thresholds = _read(directory / "thresholds.json")
    profile = _read(directory / "multi_axis_profile.json")
    receipt = _read(directory / "run_receipt.json")
    fixed_per_label = {
        item["label"]: item
        for item in metrics_by_mode["fixed_0p5"]["per_label"]
    }
    compact_modes = {}
    for mode in THRESHOLD_MODES:
        metrics = metrics_by_mode[mode]
        per_label = {item["label"]: item for item in metrics["per_label"]}
        compact_modes[mode] = {
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "hamming_loss": metrics["hamming_loss"],
            "subset_acc": metrics["subset_acc"],
            "cardinality_mae": metrics["cardinality_mae"],
            "tail_recall": per_label[tail_name]["recall"],
            "tail_precision": per_label[tail_name]["precision"],
            "tail_f1": per_label[tail_name]["f1"],
        }
    return {
        "tail_label": tail_name,
        "loss": loss,
        "seed": int(seed),
        "tail_count": profile["label_counts"][tail_name],
        "macro_aupr": metrics_by_mode["fixed_0p5"]["macro_aupr"],
        "tail_aupr": fixed_per_label[tail_name]["aupr"],
        "temperature": calibration["temperature"],
        "calibration_loss_before": calibration["loss_before"],
        "calibration_loss_after": calibration["loss_after"],
        "thresholds": thresholds["modes"],
        "threshold_metrics": compact_modes,
        "best_epoch": receipt["best_epoch"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt.get("manifest_sha256"),
        "config_sha256": receipt["config_sha256"],
        "locked_test_used": False,
    }


def _aggregate(rows):
    aggregate = []
    metric_names = (
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "hamming_loss",
        "subset_acc",
        "cardinality_mae",
        "tail_recall",
        "tail_precision",
        "tail_f1",
    )
    for loss in LOSSES:
        selected_loss = [row for row in rows if row["loss"] == loss]
        macro_aupr_mean, macro_aupr_std = _mean_std(
            [row["macro_aupr"] for row in selected_loss]
        )
        tail_aupr_mean, tail_aupr_std = _mean_std(
            [row["tail_aupr"] for row in selected_loss]
        )
        for mode in THRESHOLD_MODES:
            entry = {
                "loss": loss,
                "threshold_mode": mode,
                "run_count": len(selected_loss),
                "macro_aupr_mean": macro_aupr_mean,
                "macro_aupr_std": macro_aupr_std,
                "tail_aupr_mean": tail_aupr_mean,
                "tail_aupr_std": tail_aupr_std,
            }
            for metric in metric_names:
                mean, std = _mean_std(
                    [row["threshold_metrics"][mode][metric] for row in selected_loss]
                )
                entry[metric + "_mean"] = mean
                entry[metric + "_std"] = std
            aggregate.append(entry)
    return aggregate


def _tail_aggregate(rows):
    result = []
    for tail_name in CLASS_NAMES:
        for loss in LOSSES:
            selected = [
                row for row in rows
                if row["tail_label"] == tail_name and row["loss"] == loss
            ]
            result.append(
                {
                    "tail_label": tail_name,
                    "loss": loss,
                    "run_count": len(selected),
                    "macro_aupr_mean": _mean_std([row["macro_aupr"] for row in selected])[0],
                    "tail_aupr_mean": _mean_std([row["tail_aupr"] for row in selected])[0],
                    "primary_macro_f1_mean": _mean_std(
                        [row["threshold_metrics"][PRIMARY_MODE]["macro_f1"] for row in selected]
                    )[0],
                    "primary_tail_recall_mean": _mean_std(
                        [row["threshold_metrics"][PRIMARY_MODE]["tail_recall"] for row in selected]
                    )[0],
                    "fixed_tail_recall_mean": _mean_std(
                        [row["threshold_metrics"]["fixed_0p5"]["tail_recall"] for row in selected]
                    )[0],
                }
            )
    return result


def _paired_differences(rows):
    pairs = []
    for tail_name in CLASS_NAMES:
        seeds = sorted({row["seed"] for row in rows if row["tail_label"] == tail_name})
        for seed in seeds:
            asl = next(
                row for row in rows
                if row["tail_label"] == tail_name and row["seed"] == seed and row["loss"] == "asl"
            )
            db = next(
                row for row in rows
                if row["tail_label"] == tail_name and row["seed"] == seed and row["loss"] == "db"
            )
            item = {
                "tail_label": tail_name,
                "seed": seed,
                "delta_macro_aupr": db["macro_aupr"] - asl["macro_aupr"],
                "delta_tail_aupr": db["tail_aupr"] - asl["tail_aupr"],
                "threshold_modes": {},
            }
            for mode in THRESHOLD_MODES:
                item["threshold_modes"][mode] = {
                    metric: db["threshold_metrics"][mode][metric]
                    - asl["threshold_metrics"][mode][metric]
                    for metric in (
                        "macro_f1",
                        "hamming_loss",
                        "tail_recall",
                        "tail_precision",
                        "tail_f1",
                    )
                }
            pairs.append(item)
    return pairs


def _gate_summary(rows, tail_aggregate, pairs):
    db_rows = [row for row in rows if row["loss"] == "db"]
    primary_recall_gain = _mean_std(
        [
            row["threshold_metrics"][PRIMARY_MODE]["tail_recall"]
            - row["threshold_metrics"]["fixed_0p5"]["tail_recall"]
            for row in db_rows
        ]
    )[0]
    primary_hamming_change = _mean_std(
        [
            row["threshold_metrics"][PRIMARY_MODE]["hamming_loss"]
            - row["threshold_metrics"]["fixed_0p5"]["hamming_loss"]
            for row in db_rows
        ]
    )[0]
    db_asl_macro_f1 = _mean_std(
        [pair["threshold_modes"][PRIMARY_MODE]["macro_f1"] for pair in pairs]
    )[0]
    db_asl_macro_aupr = _mean_std([pair["delta_macro_aupr"] for pair in pairs])[0]
    tail_wins = 0
    for tail_name in CLASS_NAMES:
        asl = next(
            row for row in tail_aggregate
            if row["tail_label"] == tail_name and row["loss"] == "asl"
        )
        db = next(
            row for row in tail_aggregate
            if row["tail_label"] == tail_name and row["loss"] == "db"
        )
        tail_wins += int(db["tail_aupr_mean"] >= asl["tail_aupr_mean"])
    gates = {
        "db_primary_tail_recall_gain_vs_fixed_mean": primary_recall_gain,
        "db_primary_hamming_change_vs_fixed_mean": primary_hamming_change,
        "db_minus_asl_primary_macro_f1_mean": db_asl_macro_f1,
        "db_minus_asl_macro_aupr_mean": db_asl_macro_aupr,
        "tail_aupr_winning_labels": tail_wins,
        "tail_recall_gate": primary_recall_gain >= 0.10,
        "hamming_gate": primary_hamming_change <= 0.01,
        "macro_f1_gate": db_asl_macro_f1 >= -0.01,
        "macro_aupr_gate": db_asl_macro_aupr > 0.0,
        "tail_aupr_gate": tail_wins >= 3,
    }
    gates["all_gates_passed"] = all(
        gates[name]
        for name in (
            "tail_recall_gate",
            "hamming_gate",
            "macro_f1_gate",
            "macro_aupr_gate",
            "tail_aupr_gate",
        )
    )
    return gates


def run(args):
    root = Path(args.output_root)
    rows = []
    for tail_index, tail_name in enumerate(CLASS_NAMES):
        for seed in args.seeds:
            for loss in LOSSES:
                directory = root / tail_name / loss / "seed_{}".format(seed)
                result_path = directory / "metrics_threshold_modes.json"
                if not result_path.exists():
                    _run_tapc(args, directory, tail_index, loss, seed)
                rows.append(_collect_run(directory, tail_name, loss, seed))
    aggregate = _aggregate(rows)
    tail_aggregate = _tail_aggregate(rows)
    pairs = _paired_differences(rows)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "primary_threshold_mode": PRIMARY_MODE,
        "protocol": vars(args),
        "gate_summary": _gate_summary(rows, tail_aggregate, pairs),
        "aggregate": aggregate,
        "tail_aggregate": tail_aggregate,
        "paired_db_minus_asl": pairs,
        "runs": rows,
    }
    _write(root / "route_a_db_calibration_summary.json", payload)
    print(json.dumps(payload["gate_summary"], ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_db_calibration",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--label-imbalance-factor", type=float, default=3.0)
    parser.add_argument("--label-count-tolerance", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
