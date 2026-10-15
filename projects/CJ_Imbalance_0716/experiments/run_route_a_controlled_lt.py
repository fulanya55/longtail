"""Run Route-A controlled Label-LT experiments with fixed N and tail positives."""

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


def _mean_ci(values, seed=20260723, bootstrap_samples=10000):
    array = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "count": int(len(array)),
    }
    if len(array) <= 1:
        result["bootstrap_95_ci"] = [float(array[0]), float(array[0])]
        return result
    rng = np.random.RandomState(int(seed))
    draws = rng.choice(array, size=(int(bootstrap_samples), len(array)), replace=True)
    interval = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    result["bootstrap_95_ci"] = [float(interval[0]), float(interval[1])]
    return result


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
        "--label-tail-count-anchor", str(args.tail_count_anchor),
        "--label-target-samples", str(args.target_samples),
        "--label-sample-count-tolerance", "0",
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
                factor, CLASS_NAMES[tail_index], loss, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, factor, tail_index, loss, seed, expected_samples):
    tail_name = CLASS_NAMES[tail_index]
    head_index = _tail_order(tail_index)[0]
    metrics = _read(directory / "metrics_threshold_modes.json")
    profile = _read(directory / "multi_axis_profile.json")
    manifest = _read(directory / "longtail_manifest.json")
    receipt = _read(directory / "run_receipt.json")
    calibration = _read(directory / "calibration.json")
    thresholds = _read(directory / "thresholds.json")
    train_samples = int(sum(profile["combination_counts"].values()))
    if train_samples != int(expected_samples):
        raise ValueError(
            "Controlled manifest sample count drifted: expected {}, got {}".format(
                expected_samples, train_samples
            )
        )
    if manifest["target_sample_count"] != int(expected_samples):
        raise ValueError("Manifest did not record the frozen target sample count")
    if manifest["achieved_sample_count"] != int(expected_samples):
        raise ValueError("Manifest did not achieve the frozen sample count")
    achieved = np.asarray(manifest["achieved_label_counts"], dtype=np.int64)
    target = np.asarray(manifest["target_label_counts"], dtype=np.int64)
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
        "head_label": CLASS_NAMES[head_index],
        "loss": loss,
        "seed": int(seed),
        "train_samples": train_samples,
        "target_label_counts": target.tolist(),
        "achieved_label_counts": achieved.tolist(),
        "max_label_relative_error": float(
            np.max(np.asarray(manifest["label_count_error"], dtype=np.float64))
        ),
        "realized_head_tail_ratio": float(
            achieved[head_index] / max(achieved[tail_index], 1)
        ),
        "macro_aupr": metrics["fixed_0p5"]["macro_aupr"],
        "tail_aupr": fixed_tail["aupr"],
        "threshold_metrics": rows,
        "temperature": calibration["temperature"],
        "thresholds": thresholds["modes"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "config_sha256": receipt["config_sha256"],
        "train_seconds": receipt["train_seconds"],
        "locked_test_used": bool(receipt["locked_test_used"]),
    }


def _paired(rows):
    result = []
    for factor in sorted({row["imbalance_factor"] for row in rows}):
        for tail_name in CLASS_NAMES:
            for seed in sorted({row["seed"] for row in rows}):
                selected = [
                    row for row in rows
                    if row["imbalance_factor"] == factor
                    and row["tail_label"] == tail_name
                    and row["seed"] == seed
                ]
                asl = next(row for row in selected if row["loss"] == "asl")
                db = next(row for row in selected if row["loss"] == "db")
                if asl["manifest_sha256"] != db["manifest_sha256"]:
                    raise ValueError("ASL and DB must share the identical manifest")
                item = {
                    "imbalance_factor": factor,
                    "tail_label": tail_name,
                    "seed": seed,
                    "train_samples": asl["train_samples"],
                    "achieved_label_counts": asl["achieved_label_counts"],
                    "realized_head_tail_ratio": asl["realized_head_tail_ratio"],
                    "manifest_sha256": asl["manifest_sha256"],
                    "delta_macro_aupr": db["macro_aupr"] - asl["macro_aupr"],
                    "delta_tail_aupr": db["tail_aupr"] - asl["tail_aupr"],
                    "threshold_modes": {},
                }
                for mode in (PRIMARY_MODE, SECONDARY_MODE):
                    item["threshold_modes"][mode] = {
                        key: db["threshold_metrics"][mode][key]
                        - asl["threshold_metrics"][mode][key]
                        for key in (
                            "macro_f1",
                            "hamming_loss",
                            "cardinality_mae",
                            "tail_precision",
                            "tail_recall",
                            "tail_f1",
                        )
                    }
                result.append(item)
    return result


def _aggregate_group(selected, bootstrap_seed):
    entry = {
        "pair_count": len(selected),
        "delta_macro_aupr": _mean_ci(
            [row["delta_macro_aupr"] for row in selected], bootstrap_seed
        ),
        "delta_tail_aupr": _mean_ci(
            [row["delta_tail_aupr"] for row in selected], bootstrap_seed + 1
        ),
        "macro_aupr_wins": sum(row["delta_macro_aupr"] > 0 for row in selected),
        "tail_aupr_wins": sum(row["delta_tail_aupr"] > 0 for row in selected),
        "realized_head_tail_ratio": _mean_ci(
            [row["realized_head_tail_ratio"] for row in selected],
            bootstrap_seed + 2,
        ),
    }
    for mode_index, mode in enumerate((PRIMARY_MODE, SECONDARY_MODE)):
        entry[mode] = {
            key: _mean_ci(
                [row["threshold_modes"][mode][key] for row in selected],
                bootstrap_seed + 10 + mode_index * 10 + key_index,
            )
            for key_index, key in enumerate(
                (
                    "macro_f1",
                    "hamming_loss",
                    "cardinality_mae",
                    "tail_precision",
                    "tail_recall",
                    "tail_f1",
                )
            )
        }
    return entry


def _aggregate(pairs):
    by_factor = []
    by_factor_tail = []
    for factor_index, factor in enumerate(
        sorted({row["imbalance_factor"] for row in pairs})
    ):
        selected = [row for row in pairs if row["imbalance_factor"] == factor]
        entry = {"imbalance_factor": factor}
        entry.update(_aggregate_group(selected, 20260723 + factor_index * 100))
        by_factor.append(entry)
        for tail_index, tail_name in enumerate(CLASS_NAMES):
            tail_rows = [row for row in selected if row["tail_label"] == tail_name]
            tail_entry = {
                "imbalance_factor": factor,
                "tail_label": tail_name,
            }
            tail_entry.update(
                _aggregate_group(
                    tail_rows, 20261723 + factor_index * 100 + tail_index * 10
                )
            )
            by_factor_tail.append(tail_entry)
    return by_factor, by_factor_tail


def _gate(by_factor, by_factor_tail):
    checks = []
    for entry in by_factor:
        factor = entry["imbalance_factor"]
        values = {
            "macro_aupr_positive": entry["delta_macro_aupr"]["mean"] > 0.0,
            "tail_aupr_positive": entry["delta_tail_aupr"]["mean"] > 0.0,
            "macro_f1_noninferior": (
                entry[PRIMARY_MODE]["macro_f1"]["mean"] >= -0.002
            ),
            "hamming_noninferior": (
                entry[PRIMARY_MODE]["hamming_loss"]["mean"] <= 0.002
            ),
        }
        checks.append(
            {
                "scope": "IF={}".format(factor),
                "checks": values,
                "passed": all(values.values()),
            }
        )
    sinfm_if4 = next(
        row for row in by_factor_tail
        if row["imbalance_factor"] == 4.0 and row["tail_label"] == "SinFM"
    )
    sinfm_checks = {
        "mean_macro_aupr_not_below_minus_0p005": (
            sinfm_if4["delta_macro_aupr"]["mean"] >= -0.005
        ),
        "at_least_two_thirds_macro_aupr_wins": (
            sinfm_if4["macro_aupr_wins"]
            >= int(np.ceil(2.0 * sinfm_if4["pair_count"] / 3.0))
        ),
    }
    checks.append(
        {
            "scope": "IF=4 SinFM tail",
            "checks": sinfm_checks,
            "passed": all(sinfm_checks.values()),
        }
    )
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
        "decision": (
            "expand_to_five_seeds" if passed else "do_not_expand_analyze_failure"
        ),
    }


def run(args):
    root = Path(args.output_root)
    rows = []
    total = (
        len(args.imbalance_factors) * len(CLASS_NAMES) * len(args.seeds) * len(LOSSES)
    )
    completed = 0
    for factor in args.imbalance_factors:
        for tail_index, tail_name in enumerate(CLASS_NAMES):
            for seed in args.seeds:
                for loss in LOSSES:
                    completed += 1
                    directory = (
                        root / "if_{}".format(str(factor).replace(".", "p"))
                        / tail_name / loss / "seed_{}".format(seed)
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    if not (directory / "metrics_threshold_modes.json").exists():
                        print(
                            "[{}/{}] train IF={} tail={} loss={} seed={}".format(
                                completed, total, factor, tail_name, loss, seed
                            ),
                            flush=True,
                        )
                        _train(args, directory, factor, tail_index, loss, seed)
                    else:
                        print(
                            "[{}/{}] reuse IF={} tail={} loss={} seed={}".format(
                                completed, total, factor, tail_name, loss, seed
                            ),
                            flush=True,
                        )
                    rows.append(
                        _collect(
                            directory,
                            factor,
                            tail_index,
                            loss,
                            seed,
                            args.target_samples,
                        )
                    )
    if any(row["locked_test_used"] for row in rows):
        raise ValueError("Locked test usage detected in controlled experiment")
    pairs = _paired(rows)
    by_factor, by_factor_tail = _aggregate(pairs)
    gate = _gate(by_factor, by_factor_tail)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "primary_threshold_mode": PRIMARY_MODE,
        "scientific_question": (
            "DB vs ASL under fixed total N and fixed tail-positive anchor"
        ),
        "protocol": vars(args),
        "gate": gate,
        "aggregate_by_factor": by_factor,
        "aggregate_by_factor_tail": by_factor_tail,
        "paired_db_minus_asl": pairs,
        "runs": rows,
    }
    _write(root / "route_a_controlled_lt_summary.json", payload)
    print(json.dumps({"gate": gate, "aggregate": by_factor}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_controlled_lt",
    )
    parser.add_argument(
        "--imbalance-factors", type=float, nargs="+", default=[2.0, 3.0, 4.0]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2036, 2037, 2038])
    parser.add_argument("--target-samples", type=int, default=1600)
    parser.add_argument("--tail-count-anchor", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--label-count-tolerance", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
