"""Run Route-B v2 screen with fair labeled exposure and teacher warm-up."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("B0", "B1", "B2", "B3")


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, key):
    value = row.get(key)
    return float("nan") if value in (None, "", "nan", "NaN") else float(value)


def aggregate(output_root, args):
    rows = []
    for variant in VARIANTS:
        directory = Path(output_root) / variant
        receipt = _load_json(directory / "run_receipt.json")
        history = _history(directory / "history.csv")
        best = next(
            row for row in history if int(float(row["epoch"])) == int(receipt["best_epoch"])
        )
        active = [row for row in history if _float(row, "pseudo_coverage") > 0.0]
        rows.append(
            {
                "variant": variant,
                "config_sha256": receipt["config_sha256"],
                "best_epoch": int(receipt["best_epoch"]),
                "best_macro_aupr": _float(best, "macro_aupr"),
                "best_macro_f1": _float(best, "macro_f1"),
                "best_subset_acc": _float(best, "subset_acc"),
                "best_hamming_loss": _float(best, "hamming_loss"),
                "max_pseudo_coverage": max(
                    (_float(row, "pseudo_coverage") for row in history), default=0.0
                ),
                "mean_active_pseudo_coverage": (
                    sum(_float(row, "pseudo_coverage") for row in active) / len(active)
                    if active else 0.0
                ),
                "active_pseudo_epochs": len(active),
                "best_unsupervised_classification_loss": _float(
                    best, "unsupervised_classification_loss"
                ),
                "best_feature_consistency_loss": _float(
                    best, "feature_consistency_loss"
                ),
                "labeled_examples_per_epoch": int(float(best["labeled_examples"])),
                "unlabeled_examples_at_best": int(float(best["unlabeled_examples"])),
                "train_seconds": float(receipt["train_seconds"]),
            }
        )
    with (Path(output_root) / "route_b_v2_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seed": args.seed,
            "epochs": args.epochs,
            "labeled_ratio": args.labeled_ratio,
            "max_labeled_samples": args.max_labeled_samples,
            "max_unlabeled_samples": args.max_unlabeled_samples,
            "max_eval_samples": args.max_eval_samples,
            "teacher_warmup_epochs": args.teacher_warmup_epochs,
            "tau_pos": args.tau_pos,
            "tau_neg": args.tau_neg,
            "selection_metric": "macro_aupr",
            "variants": list(VARIANTS),
        },
        "rows": rows,
    }
    with (Path(output_root) / "route_b_v2_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return payload


def run_matrix(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_student_b.py"),
            "--config", str(PROJECT_ROOT / "configs" / "student_b" / "daps_no_param.yaml"),
            "--variant", variant,
            "--labeled-ratio", str(args.labeled_ratio),
            "--loss", "cb_asl",
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--max-labeled-samples", str(args.max_labeled_samples),
            "--max-unlabeled-samples", str(args.max_unlabeled_samples),
            "--max-eval-samples", str(args.max_eval_samples),
            "--teacher-warmup-epochs", str(args.teacher_warmup_epochs),
            "--unsupervised-warmup-fraction", str(args.unsupervised_warmup_fraction),
            "--tau-pos", str(args.tau_pos),
            "--tau-neg", str(args.tau_neg),
            "--reliability-threshold", "0.0",
            "--stability-ceiling", str(args.stability_ceiling),
            "--threshold-update-interval", str(args.threshold_update_interval),
            "--selection-metric", "macro_aupr",
            "--output-dir", str(output_root / variant),
            "--amp",
        ]
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return aggregate(output_root, args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "student_b" / "route_b_v2_screen" / "seed_2026",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--max-labeled-samples", type=int, default=512)
    parser.add_argument("--max-unlabeled-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=2)
    parser.add_argument("--unsupervised-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--tau-pos", type=float, default=0.80)
    parser.add_argument("--tau-neg", type=float, default=0.20)
    parser.add_argument("--stability-ceiling", type=float, default=0.10)
    parser.add_argument("--threshold-update-interval", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_matrix(parse_args())
