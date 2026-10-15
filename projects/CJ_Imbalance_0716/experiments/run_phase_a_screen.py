"""Run and aggregate the preregistered Phase-A power-mechanism screen."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("T0", "T0P", "T2", "T2S")


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_history_row(path, epoch):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if int(float(row["epoch"])) == int(epoch))


def _float(row, key):
    value = row.get(key)
    if value in (None, "", "nan", "NaN"):
        return float("nan")
    return float(value)


def run_matrix(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        output_dir = output_root / variant
        command = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
            "--config",
            str(PROJECT_ROOT / "configs" / "student_a" / "tapc_four_class.yaml"),
            "--variant",
            variant,
            "--protocol",
            "power_lt",
            "--loss",
            "cb_asl",
            "--conditioning-mode",
            "independent",
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-eval-samples",
            str(args.max_eval_samples),
            "--power-levels",
            *[str(value) for value in args.power_levels],
            "--reference-level",
            str(args.reference_level),
            "--selection-metric",
            "macro_aupr",
            "--output-dir",
            str(output_dir),
            "--amp",
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return aggregate(output_root, args)


def aggregate(output_root, args):
    rows = []
    for variant in VARIANTS:
        directory = Path(output_root) / variant
        receipt = _load_json(directory / "run_receipt.json")
        fixed = _load_json(directory / "metrics_fixed_0p5.json")
        selected = _load_json(directory / "metrics.json")
        history = _best_history_row(directory / "history.csv", receipt["best_epoch"])
        rows.append(
            {
                "variant": variant,
                "config_sha256": receipt["config_sha256"],
                "best_epoch": int(receipt["best_epoch"]),
                "train_seconds": float(receipt["train_seconds"]),
                "macro_aupr": float(fixed["macro_aupr"]),
                "macro_f1_fixed_0p5": float(fixed["macro_f1"]),
                "macro_f1_selected": float(selected["macro_f1"]),
                "subset_acc_fixed_0p5": float(fixed["subset_acc"]),
                "hamming_fixed_0p5": float(fixed["hamming_loss"]),
                "tail_recall_fixed_0p5": float(fixed["tail_label_recall"]),
                "tail_positive_count": int(fixed["tail_positive_count"]),
                "weak_recall_fixed_0p5": float(fixed["weak_component_recall"]),
                "weak_component_count": int(fixed["weak_component_count"]),
                "cardinality_mae_fixed_0p5": float(fixed["cardinality_mae"]),
                "directional_margin_mean": _float(history, "directional_margin_mean"),
                "directional_margin_p10": _float(history, "directional_margin_p10"),
                "directional_satisfaction_rate": _float(history, "directional_satisfaction_rate"),
                "unchanged_drift_p90": _float(history, "unchanged_drift_p90"),
                "power_ordering_loss": _float(history, "power_ordering_loss"),
                "power_distillation_loss": _float(history, "power_distillation_loss"),
                "stability_loss": _float(history, "stability_loss"),
                "prototype_positive_cosine_max_offdiag": _float(
                    history, "prototype_positive_cosine_max_offdiag"
                ),
            }
        )

    csv_path = Path(output_root) / "phase_a_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "power_levels": [int(value) for value in args.power_levels],
            "reference_level": int(args.reference_level),
            "variants": list(VARIANTS),
        },
        "rows": rows,
    }
    with (Path(output_root) / "phase_a_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "phase_a_screen_v2_1_macro_aupr" / "seed_2026",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[25, 30, 35])
    parser.add_argument("--reference-level", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_matrix(parse_args())
