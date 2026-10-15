"""Screen constrained partial adaptation for difficult six-class tails."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import (  # noqa: E402
    PRIMARY_MODE,
    _mean_ci,
    _read,
    _write,
)


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
METHODS = {
    "A6-PU": {"loss": "db", "starting_point_weight": 0.0},
    "A6-PU-SP": {"loss": "db", "starting_point_weight": 0.1},
    "A6-TailMix": {
        "loss": "tail_asl_db",
        "starting_point_weight": 0.1,
    },
}


def _tail_index(name):
    return CLASS_NAMES.index(str(name))


def _tail_order(tail_index):
    return [index for index in range(6) if index != tail_index] + [tail_index]


def _candidate_dir(args, tail_name, method, seed):
    return Path(args.output_root) / "runs" / tail_name / method / "seed_{}".format(seed)


def _baseline_dir(args, tail_name, variant, seed):
    if variant == "HeadDB20":
        return (
            Path(args.head20_root)
            / "runs" / tail_name / "HeadDB20" / "seed_{}".format(seed)
        )
    return (
        Path(args.baseline_root)
        / "runs" / tail_name / variant / "seed_{}".format(seed)
    )


def _command(args, directory, tail_name, method, seed):
    method_config = METHODS[method]
    tail_index = _tail_index(tail_name)
    checkpoint = _baseline_dir(args, tail_name, "BaseASL", seed) / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Missing six-class Base-ASL checkpoint: {}".format(checkpoint))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", method_config["loss"],
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--learning-rate", str(args.learning_rate),
        "--representation-learning-rate-scale", str(args.representation_lr_scale),
        "--starting-point-weight", str(method_config["starting_point_weight"]),
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", *[str(value) for value in _tail_order(tail_index)],
        "--selection-metric", "macro_aupr",
        "--initial-checkpoint", str(checkpoint),
        "--trainable-scope", "last_block_classifier",
        "--output-dir", str(directory),
        "--amp",
    ]
    if method == "A6-TailMix":
        command.extend(
            [
                "--tail-label-index", str(tail_index),
                "--tail-loss-weight", str(args.tail_loss_weight),
            ]
        )
    return command


def _train(args, directory, tail_name, method, seed):
    receipt_path = directory / "run_receipt.json"
    metrics_path = directory / "metrics_threshold_modes.json"
    if receipt_path.exists() and metrics_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") == "completed":
            print("reuse tail={} method={} seed={}".format(tail_name, method, seed))
            return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, tail_name, method, seed),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Partial adaptation failed tail={} method={} seed={}: {}".format(
                tail_name, method, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, tail_name, variant, seed):
    metrics = _read(directory / "metrics_threshold_modes.json")
    receipt = _read(directory / "run_receipt.json")
    manifest = _read(directory / "longtail_manifest.json")
    fixed = metrics["fixed_0p5"]
    primary = metrics[PRIMARY_MODE]
    tail_fixed = next(row for row in fixed["per_label"] if row["label"] == tail_name)
    tail_primary = next(row for row in primary["per_label"] if row["label"] == tail_name)
    return {
        "variant": variant,
        "tail_label": tail_name,
        "seed": int(seed),
        "macro_aupr": fixed["macro_aupr"],
        "tail_aupr": tail_fixed["aupr"],
        "macro_f1": primary["macro_f1"],
        "hamming_loss": primary["hamming_loss"],
        "tail_f1": tail_primary["f1"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "trainable_scope": receipt.get("trainable_scope", "all"),
        "training_parameters": receipt["training_parameters"],
        "locked_test_used": bool(receipt["locked_test_used"]),
        "achieved_sample_count": manifest["achieved_sample_count"],
        "label_count_error": manifest["label_count_error"],
        "best_epoch": receipt["best_epoch"],
        "train_seconds": receipt["train_seconds"],
    }


def _difference(candidate, baseline):
    return {
        "delta_macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "delta_tail_aupr": candidate["tail_aupr"] - baseline["tail_aupr"],
        "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "delta_hamming_loss": candidate["hamming_loss"] - baseline["hamming_loss"],
        "delta_tail_f1": candidate["tail_f1"] - baseline["tail_f1"],
    }


def _aggregate_difference(rows, key, seed):
    result = {}
    for index, metric in enumerate(
        (
            "delta_macro_aupr",
            "delta_tail_aupr",
            "delta_macro_f1",
            "delta_hamming_loss",
            "delta_tail_f1",
        )
    ):
        result[metric] = _mean_ci([row[key][metric] for row in rows], seed + index)
    result["macro_wins"] = sum(row[key]["delta_macro_aupr"] > 0 for row in rows)
    result["tail_wins"] = sum(row[key]["delta_tail_aupr"] > 0 for row in rows)
    return result


def _aggregate(comparisons, tails):
    payload = {}
    for method_index, method in enumerate(METHODS):
        selected_method = [row for row in comparisons if row["method"] == method]
        by_tail = []
        for tail_index, tail_name in enumerate(tails):
            selected = [row for row in selected_method if row["tail_label"] == tail_name]
            by_tail.append(
                {
                    "tail_label": tail_name,
                    "pair_count": len(selected),
                    "vs_asl": _aggregate_difference(
                        selected, "vs_asl", 6100 + 100 * method_index + 10 * tail_index
                    ),
                    "vs_db": _aggregate_difference(
                        selected, "vs_db", 7100 + 100 * method_index + 10 * tail_index
                    ),
                    "vs_head20": _aggregate_difference(
                        selected, "vs_head20", 8100 + 100 * method_index + 10 * tail_index
                    ),
                }
            )
        payload[method] = {
            "overall": {
                "pair_count": len(selected_method),
                "vs_asl": _aggregate_difference(
                    selected_method, "vs_asl", 9100 + 100 * method_index
                ),
                "vs_db": _aggregate_difference(
                    selected_method, "vs_db", 10100 + 100 * method_index
                ),
                "vs_head20": _aggregate_difference(
                    selected_method, "vs_head20", 11100 + 100 * method_index
                ),
            },
            "by_tail": by_tail,
        }
    return payload


def _gate(aggregate):
    checks = {}
    passed = []
    for method, values in aggregate.items():
        overall = values["overall"]
        tail_checks = {
            row["tail_label"]: {
                "tail_positive_vs_asl": row["vs_asl"]["delta_tail_aupr"]["mean"] > 0.0,
                "macro_noninferior_vs_asl": row["vs_asl"]["delta_macro_aupr"]["mean"] >= -0.005,
            }
            for row in values["by_tail"]
        }
        method_checks = {
            "tail_positive_vs_asl": overall["vs_asl"]["delta_tail_aupr"]["mean"] > 0.0,
            "tail_positive_vs_head20": overall["vs_head20"]["delta_tail_aupr"]["mean"] > 0.0,
            "macro_noninferior_vs_asl": overall["vs_asl"]["delta_macro_aupr"]["mean"] >= -0.005,
            "macro_wins_at_least_6_of_9": overall["vs_asl"]["macro_wins"] >= 6,
            "tail_wins_vs_head20_at_least_6_of_9": overall["vs_head20"]["tail_wins"] >= 6,
            "all_tail_means_positive_vs_asl": all(
                item["tail_positive_vs_asl"] for item in tail_checks.values()
            ),
            "all_tail_macro_noninferior_vs_asl": all(
                item["macro_noninferior_vs_asl"] for item in tail_checks.values()
            ),
        }
        is_passed = all(method_checks.values())
        if is_passed:
            passed.append(method)
        checks[method] = {
            "passed": is_passed,
            "overall_checks": method_checks,
            "per_tail_checks": tail_checks,
        }
    promoted = None
    if passed:
        promoted = max(
            passed,
            key=lambda name: aggregate[name]["overall"]["vs_asl"][
                "delta_tail_aupr"
            ]["mean"],
        )
    return {
        "passed": bool(passed),
        "promoted_method": promoted,
        "decision": (
            "expand_promoted_partial_adaptation"
            if promoted is not None
            else "diagnose_partial_adaptation_failure"
        ),
        "method_checks": checks,
    }


def run(args):
    runs = []
    comparisons = []
    total = len(args.tails) * len(args.seeds) * len(METHODS)
    current = 0
    for tail_name in args.tails:
        if tail_name not in CLASS_NAMES:
            raise ValueError("Unknown tail label: {}".format(tail_name))
        for seed in args.seeds:
            baselines = {
                variant: _collect(
                    _baseline_dir(args, tail_name, variant, seed),
                    tail_name,
                    variant,
                    seed,
                )
                for variant in ("BaseASL", "BaseDB", "HeadDB20")
            }
            for method in METHODS:
                current += 1
                directory = _candidate_dir(args, tail_name, method, seed)
                print(
                    "[{}/{}] train tail={} method={} seed={}".format(
                        current, total, tail_name, method, seed
                    )
                )
                _train(args, directory, tail_name, method, seed)
                candidate = _collect(directory, tail_name, method, seed)
                if candidate["split_sha256"] != baselines["BaseASL"]["split_sha256"]:
                    raise ValueError("Candidate split differs from Base-ASL")
                if candidate["manifest_sha256"] != baselines["BaseASL"]["manifest_sha256"]:
                    raise ValueError("Candidate manifest differs from Base-ASL")
                if candidate["trainable_scope"] != "last_block_classifier":
                    raise ValueError("Candidate did not use partial adaptation scope")
                if candidate["training_parameters"] != 12806:
                    raise ValueError("Unexpected partial adaptation parameter count")
                if candidate["locked_test_used"]:
                    raise ValueError("Locked test was used")
                runs.append(candidate)
                comparisons.append(
                    {
                        "method": method,
                        "tail_label": tail_name,
                        "seed": int(seed),
                        "manifest_sha256": candidate["manifest_sha256"],
                        "vs_asl": _difference(candidate, baselines["BaseASL"]),
                        "vs_db": _difference(candidate, baselines["BaseDB"]),
                        "vs_head20": _difference(candidate, baselines["HeadDB20"]),
                    }
                )
    aggregate = _aggregate(comparisons, args.tails)
    gate = _gate(aggregate)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "class_mode": "six",
            "tails": list(args.tails),
            "seeds": list(args.seeds),
            "label_target_samples": 2400,
            "label_tail_count_anchor": 400,
            "label_imbalance_factor": 4.0,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "representation_learning_rate_scale": args.representation_lr_scale,
            "tail_loss_weight": args.tail_loss_weight,
            "methods": METHODS,
        },
        "gate": gate,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "runs": runs,
    }
    summary = Path(args.output_root) / "route_a_six_class_partial_adaptation_summary.json"
    _write(summary, payload)
    concise = {
        method: {
            "macro_vs_asl": values["overall"]["vs_asl"]["delta_macro_aupr"]["mean"],
            "tail_vs_asl": values["overall"]["vs_asl"]["delta_tail_aupr"]["mean"],
            "tail_vs_head20": values["overall"]["vs_head20"]["delta_tail_aupr"]["mean"],
            "tail_vs_db": values["overall"]["vs_db"]["delta_tail_aupr"]["mean"],
        }
        for method, values in aggregate.items()
    }
    print(json.dumps({"gate": gate, "aggregate": concise}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--head20-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_epoch20_diagnostic",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_partial_adaptation",
    )
    parser.add_argument("--tails", nargs="+", default=["SawFM", "VFT", "ISRJ"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--representation-lr-scale", type=float, default=0.1)
    parser.add_argument("--tail-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
