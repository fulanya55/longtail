"""Run Route-A tail rotation and ordering-gradient screens."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("FMB", "SinFM", "MFT", "ISRJ")


def _json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _tapc(args, output, extra):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-train-samples", str(args.max_train_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--selection-metric", "macro_aupr",
        "--output-dir", str(output),
        "--amp",
    ]
    command.extend(extra)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _tail_order(tail_index):
    return [index for index in range(4) if index != tail_index] + [tail_index]


def run(args):
    root = Path(args.output_root)
    tail_rows = []
    for tail_index, tail_name in enumerate(CLASS_NAMES):
        order = _tail_order(tail_index)
        for seed in args.tail_seeds:
            for loss in ("asl", "db"):
                directory = root / "tail_rotation" / tail_name / loss / "seed_{}".format(seed)
                _tapc(
                    args,
                    directory,
                    [
                        "--variant", "T0",
                        "--protocol", "label_lt",
                        "--loss", loss,
                        "--seed", str(seed),
                        "--label-imbalance-factor", "3",
                        "--label-count-tolerance", "0.20",
                        "--label-rank-order", *[str(value) for value in order],
                    ],
                )
                metrics = _json(directory / "metrics_fixed_0p5.json")
                profile = _json(directory / "multi_axis_profile.json")
                receipt = _json(directory / "run_receipt.json")
                per_label = {item["label"]: item for item in metrics["per_label"]}
                tail_rows.append(
                    {
                        "tail_label": tail_name,
                        "loss": loss,
                        "seed": seed,
                        "tail_count": profile["label_counts"][tail_name],
                        "macro_aupr": metrics["macro_aupr"],
                        "tail_aupr": per_label[tail_name]["aupr"],
                        "tail_recall_fixed_0p5": per_label[tail_name]["recall"],
                        "hamming_fixed_0p5": metrics["hamming_loss"],
                        "best_epoch": receipt["best_epoch"],
                    }
                )
    tail_summary = []
    for tail_name in CLASS_NAMES:
        row = {"tail_label": tail_name}
        for loss in ("asl", "db"):
            selected = [
                item for item in tail_rows
                if item["tail_label"] == tail_name and item["loss"] == loss
            ]
            row[loss + "_macro_aupr"] = sum(item["macro_aupr"] for item in selected) / len(selected)
            row[loss + "_tail_aupr"] = sum(item["tail_aupr"] for item in selected) / len(selected)
        row["delta_macro_aupr"] = row["db_macro_aupr"] - row["asl_macro_aupr"]
        row["delta_tail_aupr"] = row["db_tail_aupr"] - row["asl_tail_aupr"]
        tail_summary.append(row)

    ordering_rows = []
    ordering_variants = (
        ("T0P", "T0P", "shared", 0.0, 0.0),
        ("Shared", "T2O_V2", "shared", 0.0, args.ordering_weight),
        ("StopGrad", "T2O_V2", "backbone_stop_gradient", 0.0, args.ordering_weight),
        ("DelayedStopGrad", "T2O_V2", "backbone_stop_gradient", 0.5, args.ordering_weight),
    )
    for name, variant, gradient_mode, start_fraction, weight in ordering_variants:
        directory = root / "ordering_gradient" / name / "seed_{}".format(args.ordering_seed)
        _tapc(
            args,
            directory,
            [
                "--variant", variant,
                "--protocol", "power_lt",
                "--loss", "cb_asl",
                "--seed", str(args.ordering_seed),
                "--power-levels", "25", "30", "35",
                "--reference-level", "30",
                "--power-weight", str(weight),
                "--ordering-mode", "strong_only",
                "--ordering-gradient-mode", gradient_mode,
                "--ordering-start-fraction", str(start_fraction),
            ],
        )
        receipt = _json(directory / "run_receipt.json")
        metrics = _json(directory / "metrics_fixed_0p5.json")
        history = _history(directory / "history.csv")
        active = [row for row in history if float(row.get("ordering_active", 0.0)) > 0]
        ordering_rows.append(
            {
                "variant": name,
                "macro_aupr": receipt["macro_aupr"],
                "macro_f1_fixed_0p5": receipt["macro_f1_fixed_0p5"],
                "hamming_fixed_0p5": metrics["hamming_loss"],
                "weak_component_recall": receipt["weak_component_recall"],
                "directional_margin_mean": receipt["directional_margin_mean"],
                "directional_satisfaction_rate": receipt["directional_satisfaction_rate"],
                "gradient_cosine_mean": (
                    sum(float(row["ordering_gradient_cosine"]) for row in active if row["ordering_gradient_cosine"] not in ("nan", "NaN"))
                    / max(sum(row["ordering_gradient_cosine"] not in ("nan", "NaN") for row in active), 1)
                ),
                "classification_gradient_norm_mean": (
                    sum(float(row["classification_gradient_norm"]) for row in active if row["classification_gradient_norm"] not in ("nan", "NaN"))
                    / max(sum(row["classification_gradient_norm"] not in ("nan", "NaN") for row in active), 1)
                ),
                "ordering_gradient_norm_mean": (
                    sum(float(row["ordering_gradient_norm"]) for row in active if row["ordering_gradient_norm"] not in ("nan", "NaN"))
                    / max(sum(row["ordering_gradient_norm"] not in ("nan", "NaN") for row in active), 1)
                ),
            }
        )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "tail_rotation_summary": tail_summary,
        "tail_rotation_runs": tail_rows,
        "ordering_gradient_screen": ordering_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "route_a_next_stage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_next_stage")
    parser.add_argument("--tail-seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--ordering-seed", type=int, default=2026)
    parser.add_argument("--ordering-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
