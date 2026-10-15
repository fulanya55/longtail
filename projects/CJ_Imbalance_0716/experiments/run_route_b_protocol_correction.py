"""Run the corrected low-weight Route-B pseudo-label experiment."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("B0_CONT", None), ("B1_U005", 0.05), ("B1_U010", 0.10))


def _json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row, key):
    return float(row[key])


def _train_command(args, name, output_dir, checkpoint=None, thresholds=None, lambda_u=0.0):
    variant = "B0" if name in ("B0_PRETRAIN", "B0_CONT") else "B1"
    epochs = args.pretrain_epochs if name == "B0_PRETRAIN" else args.continuation_epochs
    command = [
        sys.executable, str(PROJECT_ROOT / "experiments" / "run_student_b.py"),
        "--config", str(PROJECT_ROOT / "configs" / "student_b" / "daps_no_param.yaml"),
        "--variant", variant,
        "--labeled-ratio", str(args.labeled_ratio),
        "--loss", "cb_asl",
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--patience", str(epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-labeled-samples", "0",
        "--max-unlabeled-samples", str(args.max_unlabeled_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--teacher-warmup-epochs", "0",
        "--unsupervised-warmup-fraction", str(args.warmup_epochs / args.continuation_epochs),
        "--reliability-threshold", "0.0",
        "--stability-ceiling", str(args.stability_ceiling),
        "--max-pseudo-side-fraction", str(args.max_pseudo_side_fraction),
        "--lambda-u", str(lambda_u),
        "--lambda-feature", "0.0",
        "--selection-metric", "macro_aupr",
        "--output-dir", str(output_dir),
        "--balance-pseudo-sides",
        "--amp",
    ]
    if checkpoint is not None:
        command.extend(["--initial-checkpoint", str(checkpoint)])
    if thresholds is not None:
        command.extend(["--thresholds-file", str(thresholds)])
    return command


def _audit_command(args, checkpoint, thresholds, output, state):
    return [
        sys.executable, str(PROJECT_ROOT / "experiments" / "audit_student_b_pseudo_labels.py"),
        "--checkpoint", str(checkpoint),
        "--thresholds-file", str(thresholds),
        "--output", str(output),
        "--state", state,
        "--labeled-ratio", str(args.labeled_ratio),
        "--seed", str(args.seed),
        "--max-unlabeled-samples", str(args.max_unlabeled_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--stability-ceiling", str(args.stability_ceiling),
        "--max-pseudo-side-fraction", str(args.max_pseudo_side_fraction),
        "--augmentation-seed", str(args.seed),
    ]


def _quality_gate(audit, minimum_precision):
    precisions = []
    for prefix in ("positive", "negative"):
        for count, precision in zip(audit[prefix + "_count"], audit[prefix + "_precision"]):
            if count:
                precisions.append(float(precision))
    return bool(precisions) and min(precisions) >= float(minimum_precision)


def _result_row(directory, name, audit=None):
    receipt = _json(directory / "run_receipt.json")
    rows = _history(directory / "history.csv")
    best = next(row for row in rows if int(float(row["epoch"])) == int(receipt["best_epoch"]))
    pseudo = _json(directory / "pseudo_label_statistics.json")
    result = {
        "variant": name,
        "best_epoch": int(receipt["best_epoch"]),
        "macro_aupr": _number(best, "macro_aupr"),
        "macro_f1": _number(best, "macro_f1"),
        "subset_acc": _number(best, "subset_acc"),
        "hamming_loss": _number(best, "hamming_loss"),
        "max_pseudo_coverage": max(float(row["pseudo_coverage"]) for row in rows),
        "best_unsupervised_loss": _number(best, "unsupervised_classification_loss"),
        "train_seconds": float(receipt["train_seconds"]),
        "last_positive_count": sum(pseudo[-1]["positive_count"]),
        "last_negative_count": sum(pseudo[-1]["negative_count"]),
    }
    if audit is not None:
        result.update(
            {
                "audit_positive_precision_min": min(audit["positive_precision"]),
                "audit_negative_precision_min": min(audit["negative_precision"]),
                "audit_accepted_coverage": audit["accepted_coverage_total"],
            }
        )
    return result


def run(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    pretrain = root / "B0_PRETRAIN"
    subprocess.run(_train_command(args, "B0_PRETRAIN", pretrain), cwd=str(PROJECT_ROOT), check=True)
    threshold_path = root / "weak_view_quantile_thresholds.json"
    subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "experiments" / "build_student_b_quantile_thresholds.py"),
            "--checkpoint", str(pretrain / "best_model.pth"),
            "--output", str(threshold_path),
            "--labeled-ratio", str(args.labeled_ratio),
            "--seed", str(args.seed),
            "--max-eval-samples", str(args.max_eval_samples),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--positive-quantile", str(args.positive_quantile),
            "--negative-quantile", str(args.negative_quantile),
            "--teacher-view", "weak",
            "--augmentation-seed", str(args.seed),
        ],
        cwd=str(PROJECT_ROOT), check=True,
    )
    initial_audit_path = root / "initial_pseudo_audit.json"
    subprocess.run(
        _audit_command(args, pretrain / "best_model.pth", threshold_path, initial_audit_path, "model"),
        cwd=str(PROJECT_ROOT), check=True,
    )
    initial_audit = _json(initial_audit_path)
    if not _quality_gate(initial_audit, args.minimum_precision):
        payload = {
            "status": "stopped_by_initial_pseudo_quality_gate",
            "locked_test_used": False,
            "minimum_precision": args.minimum_precision,
            "initial_audit": initial_audit,
        }
        with (root / "route_b_protocol_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload
    results = [_result_row(pretrain, "B0_PRETRAIN", initial_audit)]
    for name, lambda_u in VARIANTS:
        directory = root / name
        subprocess.run(
            _train_command(
                args, name, directory,
                checkpoint=pretrain / "best_model.pth",
                thresholds=threshold_path if lambda_u is not None else None,
                lambda_u=lambda_u or 0.0,
            ),
            cwd=str(PROJECT_ROOT), check=True,
        )
        audit_payload = None
        if lambda_u is not None:
            audit_path = directory / "best_teacher_pseudo_audit.json"
            subprocess.run(
                _audit_command(args, directory / "best_model.pth", threshold_path, audit_path, "teacher"),
                cwd=str(PROJECT_ROOT), check=True,
            )
            audit_payload = _json(audit_path)
        results.append(_result_row(directory, name, audit_payload))
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "thresholds": _json(threshold_path),
        "initial_quality_gate_passed": True,
        "runs": results,
    }
    with (root / "route_b_protocol_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "student_b" / "route_b_protocol_correction" / "seed_2026")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--continuation-epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-unlabeled-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--negative-quantile", type=float, default=0.10)
    parser.add_argument("--stability-ceiling", type=float, default=0.10)
    parser.add_argument("--max-pseudo-side-fraction", type=float, default=0.30)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
