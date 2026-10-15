"""Run pretrained Route-B comparison with quantile pseudo-label thresholds."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_VARIANTS = ("B0_CONT", "B1", "B2")


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, key):
    value = row.get(key)
    return float("nan") if value in (None, "", "nan", "NaN") else float(value)


def _base_command(args, variant, output_dir):
    actual_variant = "B0" if variant in ("B0_PRETRAIN", "B0_CONT") else variant
    return [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_student_b.py"),
        "--config", str(PROJECT_ROOT / "configs" / "student_b" / "daps_no_param.yaml"),
        "--variant", actual_variant,
        "--labeled-ratio", str(args.labeled_ratio),
        "--loss", "cb_asl",
        "--seed", str(args.seed),
        "--epochs", str(args.continuation_epochs if variant != "B0_PRETRAIN" else args.pretrain_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-labeled-samples", "0",
        "--max-unlabeled-samples", str(args.max_unlabeled_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--teacher-warmup-epochs", "0",
        "--unsupervised-warmup-fraction", str(args.unsupervised_warmup_fraction),
        "--reliability-threshold", "0.0",
        "--stability-ceiling", str(args.stability_ceiling),
        "--selection-metric", "macro_aupr",
        "--output-dir", str(output_dir),
        "--amp",
    ]


def _row(directory, name):
    receipt = _load_json(directory / "run_receipt.json")
    history = _history(directory / "history.csv")
    best = next(row for row in history if int(float(row["epoch"])) == int(receipt["best_epoch"]))
    active = [row for row in history if _float(row, "pseudo_coverage") > 0.0]
    pseudo = _load_json(directory / "pseudo_label_statistics.json")
    return {
        "variant": name,
        "config_sha256": receipt["config_sha256"],
        "best_epoch": int(receipt["best_epoch"]),
        "macro_aupr": _float(best, "macro_aupr"),
        "macro_f1": _float(best, "macro_f1"),
        "subset_acc": _float(best, "subset_acc"),
        "hamming_loss": _float(best, "hamming_loss"),
        "max_pseudo_coverage": max((_float(row, "pseudo_coverage") for row in history), default=0.0),
        "mean_active_pseudo_coverage": (
            sum(_float(row, "pseudo_coverage") for row in active) / len(active)
            if active else 0.0
        ),
        "active_pseudo_epochs": len(active),
        "last_positive_count": sum(pseudo[-1]["positive_count"]),
        "last_negative_count": sum(pseudo[-1]["negative_count"]),
        "last_stability_rejected_count": sum(pseudo[-1]["stability_rejected_count"]),
        "labeled_examples_per_epoch": int(float(best["labeled_examples"])),
        "train_seconds": float(receipt["train_seconds"]),
    }


def run(args):
    output_root = Path(args.output_root)
    pretrain_dir = output_root / "B0_PRETRAIN"
    subprocess.run(
        _base_command(args, "B0_PRETRAIN", pretrain_dir),
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    threshold_path = output_root / "quantile_thresholds.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "build_student_b_quantile_thresholds.py"),
            "--checkpoint", str(pretrain_dir / "best_model.pth"),
            "--output", str(threshold_path),
            "--labeled-ratio", str(args.labeled_ratio),
            "--seed", str(args.seed),
            "--max-eval-samples", str(args.max_eval_samples),
            "--batch-size", str(args.batch_size),
            "--positive-quantile", str(args.positive_quantile),
            "--negative-quantile", str(args.negative_quantile),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    rows = [_row(pretrain_dir, "B0_PRETRAIN")]
    for name in CONTINUATION_VARIANTS:
        directory = output_root / name
        command = _base_command(args, name, directory)
        command.extend(["--initial-checkpoint", str(pretrain_dir / "best_model.pth")])
        if name != "B0_CONT":
            command.extend(["--thresholds-file", str(threshold_path)])
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        rows.append(_row(directory, name))
    with (output_root / "route_b_phase_b_runs.csv").open(
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
            "labeled_ratio": args.labeled_ratio,
            "pretrain_epochs": args.pretrain_epochs,
            "continuation_epochs": args.continuation_epochs,
            "max_unlabeled_samples": args.max_unlabeled_samples,
            "max_eval_samples": args.max_eval_samples,
            "positive_quantile": args.positive_quantile,
            "negative_quantile": args.negative_quantile,
            "variants": ["B0_PRETRAIN", *CONTINUATION_VARIANTS],
        },
        "thresholds": _load_json(threshold_path),
        "runs": rows,
    }
    with (output_root / "route_b_phase_b_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "student_b" / "route_b_phase_b" / "seed_2026",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--continuation-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-unlabeled-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--negative-quantile", type=float, default=0.10)
    parser.add_argument("--unsupervised-warmup-fraction", type=float, default=0.20)
    parser.add_argument("--stability-ceiling", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
