"""Run six-class controlled IF=4 Base-ASL/Base-DB/Head-DB screening."""

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
VARIANTS = ("BaseASL", "BaseDB", "HeadDB")


def _tail_index(name):
    return CLASS_NAMES.index(str(name))


def _tail_order(tail_index):
    return [index for index in range(6) if index != tail_index] + [tail_index]


def _directory(args, tail_name, variant, seed):
    return Path(args.output_root) / "runs" / tail_name / variant / "seed_{}".format(seed)


def _base_command(args, directory, tail_name, seed, loss):
    return [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", loss,
        "--seed", str(seed),
        "--epochs", str(args.base_epochs),
        "--patience", str(args.base_epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", *[str(value) for value in _tail_order(_tail_index(tail_name))],
        "--selection-metric", "macro_aupr",
        "--output-dir", str(directory),
        "--amp",
    ]


def _head_command(args, directory, tail_name, seed, checkpoint):
    command = _base_command(args, directory, tail_name, seed, "db")
    command[command.index("--epochs") + 1] = str(args.head_epochs)
    command[command.index("--patience") + 1] = str(args.head_epochs)
    command.extend(
        [
            "--learning-rate", str(args.head_learning_rate),
            "--initial-checkpoint", str(checkpoint),
            "--trainable-scope", "classifier",
        ]
    )
    return command


def _train(args, directory, tail_name, variant, seed):
    directory.mkdir(parents=True, exist_ok=True)
    if variant in ("BaseASL", "BaseDB"):
        loss = "asl" if variant == "BaseASL" else "db"
        command = _base_command(args, directory, tail_name, seed, loss)
    else:
        checkpoint = _directory(args, tail_name, "BaseASL", seed) / "best_model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError("Missing six-class Base-ASL checkpoint: {}".format(checkpoint))
        command = _head_command(args, directory, tail_name, seed, checkpoint)
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Six-class run failed tail={} variant={} seed={}: {}".format(
                tail_name, variant, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, tail_name, variant, seed):
    metrics = _read(directory / "metrics_threshold_modes.json")
    receipt = _read(directory / "run_receipt.json")
    manifest = _read(directory / "longtail_manifest.json")
    tail_fixed = next(
        row for row in metrics["fixed_0p5"]["per_label"] if row["label"] == tail_name
    )
    primary = metrics[PRIMARY_MODE]
    tail_primary = next(row for row in primary["per_label"] if row["label"] == tail_name)
    return {
        "variant": variant,
        "tail_label": tail_name,
        "seed": int(seed),
        "macro_aupr": metrics["fixed_0p5"]["macro_aupr"],
        "tail_aupr": tail_fixed["aupr"],
        "macro_f1": primary["macro_f1"],
        "hamming_loss": primary["hamming_loss"],
        "tail_f1": tail_primary["f1"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "trainable_scope": receipt.get("trainable_scope", "all"),
        "training_parameters": receipt["training_parameters"],
        "initial_checkpoint_sha256": receipt.get("initial_checkpoint_sha256"),
        "train_seconds": receipt["train_seconds"],
        "target_label_counts": manifest["target_label_counts"],
        "achieved_label_counts": manifest["achieved_label_counts"],
        "label_count_error": manifest["label_count_error"],
        "achieved_sample_count": manifest["achieved_sample_count"],
        "locked_test_used": bool(receipt["locked_test_used"]),
    }


def _difference(candidate, baseline):
    return {
        "delta_macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "delta_tail_aupr": candidate["tail_aupr"] - baseline["tail_aupr"],
        "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "delta_hamming_loss": candidate["hamming_loss"] - baseline["hamming_loss"],
        "delta_tail_f1": candidate["tail_f1"] - baseline["tail_f1"],
    }


def _paired(rows, tails, seeds):
    pairs = []
    for tail_name in tails:
        for seed in seeds:
            selected = [
                row for row in rows
                if row["tail_label"] == tail_name and row["seed"] == seed
            ]
            values = {row["variant"]: row for row in selected}
            if set(values) != set(VARIANTS):
                raise ValueError("Incomplete six-class triplet")
            if len({row["manifest_sha256"] for row in selected}) != 1:
                raise ValueError("Six-class triplet must share one manifest")
            if len({row["split_sha256"] for row in selected}) != 1:
                raise ValueError("Six-class triplet must share one split")
            if values["HeadDB"]["trainable_scope"] != "classifier":
                raise ValueError("HeadDB did not freeze the six-class representation")
            pairs.append(
                {
                    "tail_label": tail_name,
                    "seed": seed,
                    "manifest_sha256": values["HeadDB"]["manifest_sha256"],
                    "head_db_vs_asl": _difference(values["HeadDB"], values["BaseASL"]),
                    "head_db_vs_db": _difference(values["HeadDB"], values["BaseDB"]),
                }
            )
    return pairs


def _aggregate_difference(rows, key, seed):
    result = {}
    for index, metric in enumerate(
        ("delta_macro_aupr", "delta_tail_aupr", "delta_macro_f1", "delta_hamming_loss", "delta_tail_f1")
    ):
        result[metric] = _mean_ci([row[key][metric] for row in rows], seed + index)
    result["macro_wins"] = sum(row[key]["delta_macro_aupr"] > 0 for row in rows)
    result["tail_wins"] = sum(row[key]["delta_tail_aupr"] > 0 for row in rows)
    return result


def _aggregate(pairs, tails):
    by_tail = []
    for tail_index, tail_name in enumerate(tails):
        selected = [row for row in pairs if row["tail_label"] == tail_name]
        by_tail.append(
            {
                "tail_label": tail_name,
                "pair_count": len(selected),
                "vs_asl": _aggregate_difference(selected, "head_db_vs_asl", 20260723 + tail_index * 30),
                "vs_db": _aggregate_difference(selected, "head_db_vs_db", 20261723 + tail_index * 30),
            }
        )
    overall = {
        "pair_count": len(pairs),
        "vs_asl": _aggregate_difference(pairs, "head_db_vs_asl", 20262723),
        "vs_db": _aggregate_difference(pairs, "head_db_vs_db", 20263723),
    }
    return by_tail, overall


def _gate(by_tail, overall):
    per_tail = {}
    for row in by_tail:
        values = row["vs_asl"]
        per_tail[row["tail_label"]] = {
            "macro_noninferior": values["delta_macro_aupr"]["mean"] >= -0.005,
            "tail_noninferior": values["delta_tail_aupr"]["mean"] >= -0.005,
            "macro_f1_noninferior": values["delta_macro_f1"]["mean"] >= -0.005,
            "hamming_noninferior": values["delta_hamming_loss"]["mean"] <= 0.005,
        }
    positive_tail_count = sum(
        row["vs_asl"]["delta_macro_aupr"]["mean"] > 0 for row in by_tail
    )
    values = overall["vs_asl"]
    overall_checks = {
        "macro_mean_positive": values["delta_macro_aupr"]["mean"] > 0,
        "tail_mean_nonnegative": values["delta_tail_aupr"]["mean"] >= 0,
        "macro_wins_at_least_12_of_18": values["macro_wins"] >= 12,
        "tail_wins_at_least_9_of_18": values["tail_wins"] >= 9,
        "at_least_4_of_6_tail_macro_means_positive": positive_tail_count >= 4,
        "tail_recovery_vs_end_to_end_db": overall["vs_db"]["delta_tail_aupr"]["mean"] > 0,
    }
    passed = all(all(item.values()) for item in per_tail.values()) and all(overall_checks.values())
    return {
        "passed": passed,
        "decision": "expand_six_class_to_five_seeds" if passed else "diagnose_six_class_failure",
        "per_tail_checks": per_tail,
        "overall_checks": overall_checks,
    }


def run(args):
    rows = []
    total = len(args.tails) * len(args.seeds) * len(VARIANTS)
    completed = 0
    for tail_name in args.tails:
        if tail_name not in CLASS_NAMES:
            raise ValueError("Unknown six-class tail: {}".format(tail_name))
        for seed in args.seeds:
            for variant in VARIANTS:
                completed += 1
                directory = _directory(args, tail_name, variant, seed)
                directory.mkdir(parents=True, exist_ok=True)
                if not (directory / "metrics_threshold_modes.json").exists():
                    print(
                        "[{}/{}] train tail={} variant={} seed={}".format(
                            completed, total, tail_name, variant, seed
                        ),
                        flush=True,
                    )
                    _train(args, directory, tail_name, variant, seed)
                else:
                    print(
                        "[{}/{}] reuse tail={} variant={} seed={}".format(
                            completed, total, tail_name, variant, seed
                        ),
                        flush=True,
                    )
                rows.append(_collect(directory, tail_name, variant, seed))
    if any(row["locked_test_used"] for row in rows):
        raise ValueError("Locked test usage detected")
    pairs = _paired(rows, args.tails, args.seeds)
    by_tail, overall = _aggregate(pairs, args.tails)
    gate = _gate(by_tail, overall)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "gate": gate,
        "aggregate_by_tail": by_tail,
        "aggregate_overall": overall,
        "paired": pairs,
        "runs": rows,
    }
    _write(Path(args.output_root) / "route_a_six_class_head_db_screen_summary.json", payload)
    concise = [
        {
            "tail": row["tail_label"],
            "macro": row["vs_asl"]["delta_macro_aupr"]["mean"],
            "tail_aupr": row["vs_asl"]["delta_tail_aupr"]["mean"],
            "tail_recovery_vs_db": row["vs_db"]["delta_tail_aupr"]["mean"],
        }
        for row in by_tail
    ]
    print(json.dumps({"gate": gate, "aggregate": concise}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument("--tails", nargs="+", default=list(CLASS_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--base-epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
