"""Run Route-A IF=2/4 confirmatory experiments on new seeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("FMB", "SinFM", "MFT", "ISRJ")
LOSSES = ("asl", "db")
PRIMARY_MODE = "per_label_f1"
SECONDARY_MODE = "shrinkage_per_label_f1"


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _tail_order(tail_index):
    return [index for index in range(4) if index != tail_index] + [tail_index]


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _train(args, directory, factor, tail_index, loss, seed):
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
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", str(factor),
        "--label-count-tolerance", str(args.label_count_tolerance),
        "--label-rank-order", *[str(value) for value in _tail_order(tail_index)],
        "--selection-metric", "macro_aupr",
        "--output-dir", str(directory),
        "--amp",
    ]
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(
            "TAPC run failed for IF={} tail={} loss={} seed={}: {}".format(
                factor, CLASS_NAMES[tail_index], loss, seed, result.stderr[-2000:]
            )
        )


def _collect(directory, factor, tail_name, loss, seed):
    metrics = _read(directory / "metrics_threshold_modes.json")
    profile = _read(directory / "multi_axis_profile.json")
    receipt = _read(directory / "run_receipt.json")
    calibration = _read(directory / "calibration.json")
    thresholds = _read(directory / "thresholds.json")
    rows = {}
    for mode in ("fixed_0p5", PRIMARY_MODE, SECONDARY_MODE):
        values = metrics[mode]
        tail = next(item for item in values["per_label"] if item["label"] == tail_name)
        rows[mode] = {
            "macro_f1": values["macro_f1"],
            "hamming_loss": values["hamming_loss"],
            "cardinality_mae": values["cardinality_mae"],
            "tail_precision": tail["precision"],
            "tail_recall": tail["recall"],
            "tail_f1": tail["f1"],
        }
    fixed_tail = next(
        item for item in metrics["fixed_0p5"]["per_label"]
        if item["label"] == tail_name
    )
    return {
        "imbalance_factor": float(factor),
        "tail_label": tail_name,
        "loss": loss,
        "seed": int(seed),
        "train_samples": int(sum(profile["combination_counts"].values())),
        "label_counts": profile["label_counts"],
        "macro_aupr": metrics["fixed_0p5"]["macro_aupr"],
        "tail_aupr": fixed_tail["aupr"],
        "threshold_metrics": rows,
        "temperature": calibration["temperature"],
        "thresholds": thresholds["modes"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "config_sha256": receipt["config_sha256"],
        "locked_test_used": False,
    }


def _paired(rows):
    result = []
    for factor in sorted({row["imbalance_factor"] for row in rows}):
        for tail_name in CLASS_NAMES:
            for seed in sorted({row["seed"] for row in rows}):
                asl = next(
                    row for row in rows
                    if row["imbalance_factor"] == factor
                    and row["tail_label"] == tail_name
                    and row["seed"] == seed
                    and row["loss"] == "asl"
                )
                db = next(
                    row for row in rows
                    if row["imbalance_factor"] == factor
                    and row["tail_label"] == tail_name
                    and row["seed"] == seed
                    and row["loss"] == "db"
                )
                item = {
                    "imbalance_factor": factor,
                    "tail_label": tail_name,
                    "seed": seed,
                    "delta_macro_aupr": db["macro_aupr"] - asl["macro_aupr"],
                    "delta_tail_aupr": db["tail_aupr"] - asl["tail_aupr"],
                    "threshold_modes": {},
                }
                for mode in (PRIMARY_MODE, SECONDARY_MODE):
                    item["threshold_modes"][mode] = {
                        key: db["threshold_metrics"][mode][key]
                        - asl["threshold_metrics"][mode][key]
                        for key in (
                            "macro_f1", "hamming_loss", "tail_recall", "tail_f1"
                        )
                    }
                result.append(item)
    return result


def _aggregate(pairs):
    result = []
    for factor in sorted({row["imbalance_factor"] for row in pairs}):
        selected = [row for row in pairs if row["imbalance_factor"] == factor]
        entry = {
            "imbalance_factor": factor,
            "pair_count": len(selected),
            "delta_macro_aupr": _mean_std([row["delta_macro_aupr"] for row in selected]),
            "delta_tail_aupr": _mean_std([row["delta_tail_aupr"] for row in selected]),
            "macro_aupr_wins": sum(row["delta_macro_aupr"] > 0 for row in selected),
            "tail_aupr_wins": sum(row["delta_tail_aupr"] > 0 for row in selected),
        }
        for mode in (PRIMARY_MODE, SECONDARY_MODE):
            entry[mode] = {
                key: _mean_std([row["threshold_modes"][mode][key] for row in selected])
                for key in ("macro_f1", "hamming_loss", "tail_recall", "tail_f1")
            }
        result.append(entry)
    return result


def run(args):
    root = Path(args.output_root)
    rows = []
    for factor in args.imbalance_factors:
        for tail_index, tail_name in enumerate(CLASS_NAMES):
            for seed in args.seeds:
                for loss in LOSSES:
                    directory = (
                        root / "if_{}".format(str(factor).replace(".", "p"))
                        / tail_name / loss / "seed_{}".format(seed)
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    if not (directory / "metrics_threshold_modes.json").exists():
                        _train(args, directory, factor, tail_index, loss, seed)
                    rows.append(_collect(directory, factor, tail_name, loss, seed))
    pairs = _paired(rows)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "primary_threshold_mode": PRIMARY_MODE,
        "protocol": vars(args),
        "aggregate": _aggregate(pairs),
        "paired_db_minus_asl": pairs,
        "runs": rows,
    }
    _write(root / "route_a_if_confirmatory_summary.json", payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if_confirmatory",
    )
    parser.add_argument("--imbalance-factors", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2031, 2032, 2033])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--label-count-tolerance", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
