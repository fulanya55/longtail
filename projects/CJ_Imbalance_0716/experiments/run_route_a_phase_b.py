"""Run Route-A Phase-B validation and low-weight ordering screen."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_VARIANTS = ("T0P", "T2D")
ORDERING_WEIGHTS = (0.02, 0.05, 0.10)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history_row(path, epoch):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return next(
            row
            for row in csv.DictReader(handle)
            if int(float(row["epoch"])) == int(epoch)
        )


def _float(row, key):
    value = row.get(key)
    return float("nan") if value in (None, "", "nan", "NaN") else float(value)


def _run(args, variant, seed, output_dir, power_weight, ordering_mode):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(PROJECT_ROOT / "configs" / "student_a" / "tapc_four_class.yaml"),
        "--variant", variant,
        "--protocol", "power_lt",
        "--loss", "cb_asl",
        "--conditioning-mode", "independent",
        "--ordering-mode", ordering_mode,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-train-samples", str(args.max_train_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--power-levels", *[str(value) for value in args.power_levels],
        "--reference-level", str(args.reference_level),
        "--selection-metric", "macro_aupr",
        "--power-weight", str(power_weight),
        "--output-dir", str(output_dir),
        "--amp",
    ]
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _result_row(directory, group, variant, seed, power_weight):
    receipt = _load_json(directory / "run_receipt.json")
    fixed = _load_json(directory / "metrics_fixed_0p5.json")
    selected = _load_json(directory / "metrics.json")
    history = _history_row(directory / "history.csv", receipt["best_epoch"])
    return {
        "group": group,
        "variant": variant,
        "seed": int(seed),
        "power_weight": float(power_weight),
        "config_sha256": receipt["config_sha256"],
        "best_epoch": int(receipt["best_epoch"]),
        "macro_aupr": float(fixed["macro_aupr"]),
        "macro_f1_fixed_0p5": float(fixed["macro_f1"]),
        "macro_f1_selected": float(selected["macro_f1"]),
        "subset_acc_fixed_0p5": float(fixed["subset_acc"]),
        "hamming_fixed_0p5": float(fixed["hamming_loss"]),
        "weak_recall_fixed_0p5": float(fixed["weak_component_recall"]),
        "weak_component_count": int(fixed["weak_component_count"]),
        "cardinality_mae_fixed_0p5": float(fixed["cardinality_mae"]),
        "directional_margin_mean": _float(history, "directional_margin_mean"),
        "directional_margin_p10": _float(history, "directional_margin_p10"),
        "directional_satisfaction_rate": _float(history, "directional_satisfaction_rate"),
        "unchanged_drift_p90": _float(history, "unchanged_drift_p90"),
        "power_ordering_loss": _float(history, "power_ordering_loss"),
        "power_distillation_loss": _float(history, "power_distillation_loss"),
        "train_seconds": float(receipt["train_seconds"]),
    }


def _mean_std(rows, variant):
    selected = [row for row in rows if row["group"] == "primary" and row["variant"] == variant]
    result = {"variant": variant, "seed_count": len(selected)}
    for key in (
        "macro_aupr",
        "macro_f1_fixed_0p5",
        "hamming_fixed_0p5",
        "weak_recall_fixed_0p5",
        "cardinality_mae_fixed_0p5",
    ):
        values = [float(row[key]) for row in selected]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        result[key + "_mean"] = mean
        result[key + "_std"] = variance ** 0.5
    return result


def run(args):
    output_root = Path(args.output_root)
    rows = []
    for seed in args.seeds:
        for variant in PRIMARY_VARIANTS:
            directory = output_root / "primary" / variant / "seed_{}".format(seed)
            _run(args, variant, seed, directory, args.distillation_weight, "strong_only")
            rows.append(
                _result_row(
                    directory, "primary", variant, seed, args.distillation_weight
                )
            )
    for weight in ORDERING_WEIGHTS:
        tag = "weight_{:.2f}".format(weight).replace(".", "p")
        directory = output_root / "ordering_screen" / tag / "seed_{}".format(args.ordering_seed)
        _run(args, "T2O_V2", args.ordering_seed, directory, weight, "strong_only")
        rows.append(
            _result_row(
                directory, "ordering_screen", "T2O_V2", args.ordering_seed, weight
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "route_a_phase_b_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": args.seeds,
            "ordering_seed": args.ordering_seed,
            "epochs": args.epochs,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "power_levels": args.power_levels,
            "primary_variants": list(PRIMARY_VARIANTS),
            "ordering_weights": list(ORDERING_WEIGHTS),
        },
        "primary_aggregate": [
            _mean_std(rows, variant) for variant in PRIMARY_VARIANTS
        ],
        "runs": rows,
    }
    with (output_root / "route_a_phase_b_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_phase_b",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--ordering-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[25, 30, 35])
    parser.add_argument("--reference-level", type=int, default=30)
    parser.add_argument("--distillation-weight", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
