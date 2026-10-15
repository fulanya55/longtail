"""Compare supervised multi-label losses on an achievable label long tail."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("asl", "cb_asl", "db")


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args):
    root = Path(args.output_root)
    rows = []
    for seed in args.seeds:
        for loss in LOSSES:
            directory = root / loss / "seed_{}".format(seed)
            subprocess.run(
                [
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
                    "--label-rank-order", "0", "1", "2", "3",
                    "--selection-metric", "macro_aupr",
                    "--output-dir", str(directory),
                    "--amp",
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            metrics = _read(directory / "metrics_fixed_0p5.json")
            receipt = _read(directory / "run_receipt.json")
            profile = _read(directory / "multi_axis_profile.json")
            counts = profile["label_counts"]
            tail_name = min(counts, key=counts.get)
            per_label = {item["label"]: item for item in metrics["per_label"]}
            rows.append(
                {
                    "loss": loss,
                    "seed": seed,
                    "macro_aupr": metrics["macro_aupr"],
                    "macro_f1_fixed_0p5": metrics["macro_f1"],
                    "hamming_fixed_0p5": metrics["hamming_loss"],
                    "tail_label": tail_name,
                    "tail_count": counts[tail_name],
                    "tail_aupr": per_label[tail_name]["aupr"],
                    "tail_recall_fixed_0p5": per_label[tail_name]["recall"],
                    "per_label_aupr": {
                        name: per_label[name]["aupr"] for name in counts
                    },
                    "train_samples": sum(profile["combination_counts"].values()),
                    "best_epoch": receipt["best_epoch"],
                    "locked_test_used": False,
                }
            )
    aggregate = []
    for loss in LOSSES:
        selected = [row for row in rows if row["loss"] == loss]
        aggregate.append(
            {
                "loss": loss,
                "seed_count": len(selected),
                "macro_aupr_mean": sum(row["macro_aupr"] for row in selected) / len(selected),
                "tail_aupr_mean": sum(row["tail_aupr"] for row in selected) / len(selected),
                "macro_f1_fixed_0p5_mean": sum(row["macro_f1_fixed_0p5"] for row in selected) / len(selected),
                "hamming_fixed_0p5_mean": sum(row["hamming_fixed_0p5"] for row in selected) / len(selected),
            }
        )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "aggregate": aggregate,
        "runs": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "route_a_label_lt_loss_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_label_lt_losses")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--label-imbalance-factor", type=float, default=3.0)
    parser.add_argument("--label-count-tolerance", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
