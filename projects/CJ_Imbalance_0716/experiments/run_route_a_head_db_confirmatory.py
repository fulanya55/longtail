"""Confirm frozen-ASL representation plus DB head on four tails and five seeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import (  # noqa: E402
    CLASS_NAMES,
    PRIMARY_MODE,
    _mean_ci,
    _read,
    _tail_order,
    _write,
)


VARIANTS = ("BaseASL", "BaseDB", "HeadDB")
PRIOR_SEEDS = (2036, 2037, 2038)


def _tail_index(name):
    return CLASS_NAMES.index(str(name))


def _prior_directory(args, tail_name, variant, seed):
    if variant == "BaseASL":
        return Path(args.controlled_root) / "if_4p0" / tail_name / "asl" / "seed_{}".format(seed)
    if variant == "BaseDB":
        return Path(args.controlled_root) / "if_4p0" / tail_name / "db" / "seed_{}".format(seed)
    if variant == "HeadDB" and tail_name != "SinFM":
        return Path(args.decoupled_root) / tail_name / "HeadDB" / "seed_{}".format(seed)
    return None


def _new_directory(args, tail_name, variant, seed):
    return Path(args.output_root) / "runs" / tail_name / variant / "seed_{}".format(seed)


def _directory(args, tail_name, variant, seed):
    if int(seed) in PRIOR_SEEDS:
        prior = _prior_directory(args, tail_name, variant, seed)
        if prior is not None:
            return prior, True
    return _new_directory(args, tail_name, variant, seed), False


def _base_command(args, directory, tail_name, seed, loss):
    tail_index = _tail_index(tail_name)
    return [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
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
        "--label-target-samples", "1600",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", *[str(value) for value in _tail_order(tail_index)],
        "--selection-metric", "macro_aupr",
        "--output-dir", str(directory),
        "--amp",
    ]


def _head_command(args, directory, tail_name, seed, checkpoint):
    command = _base_command(args, directory, tail_name, seed, "db")
    epoch_index = command.index("--epochs") + 1
    patience_index = command.index("--patience") + 1
    command[epoch_index] = str(args.head_epochs)
    command[patience_index] = str(args.head_epochs)
    command.extend(
        [
            "--learning-rate", str(args.head_learning_rate),
            "--initial-checkpoint", str(checkpoint),
            "--trainable-scope", "classifier",
        ]
    )
    return command


def _run_command(command, directory, description):
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError("{} failed: {}".format(description, result.stderr[-2500:]))


def _train(args, directory, tail_name, variant, seed):
    directory.mkdir(parents=True, exist_ok=True)
    if variant in ("BaseASL", "BaseDB"):
        loss = "asl" if variant == "BaseASL" else "db"
        command = _base_command(args, directory, tail_name, seed, loss)
    else:
        base_directory, _ = _directory(args, tail_name, "BaseASL", seed)
        checkpoint = base_directory / "best_model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError("Missing Base-ASL checkpoint: {}".format(checkpoint))
        command = _head_command(args, directory, tail_name, seed, checkpoint)
    _run_command(command, directory, "{} {} seed {}".format(tail_name, variant, seed))


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
        "achieved_sample_count": manifest.get("achieved_sample_count"),
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
                raise ValueError("Incomplete confirmatory triplet")
            if len({row["manifest_sha256"] for row in selected}) != 1:
                raise ValueError("Confirmatory triplet must share the manifest")
            if len({row["split_sha256"] for row in selected}) != 1:
                raise ValueError("Confirmatory triplet must share the split")
            if values["HeadDB"]["trainable_scope"] != "classifier":
                raise ValueError("HeadDB did not use classifier-only scope")
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
    result = []
    for tail_index, tail_name in enumerate(tails):
        selected = [row for row in pairs if row["tail_label"] == tail_name]
        result.append(
            {
                "tail_label": tail_name,
                "pair_count": len(selected),
                "vs_asl": _aggregate_difference(selected, "head_db_vs_asl", 20260723 + tail_index * 30),
                "vs_db": _aggregate_difference(selected, "head_db_vs_db", 20261723 + tail_index * 30),
            }
        )
    all_vs_asl = _aggregate_difference(pairs, "head_db_vs_asl", 20262723)
    all_vs_db = _aggregate_difference(pairs, "head_db_vs_db", 20263723)
    return result, {"pair_count": len(pairs), "vs_asl": all_vs_asl, "vs_db": all_vs_db}


def _gate(by_tail):
    checks = {}
    for row in by_tail:
        values = row["vs_asl"]
        tail_name = row["tail_label"]
        checks[tail_name] = {
            "macro_mean_positive": values["delta_macro_aupr"]["mean"] > 0,
            "macro_wins_at_least_3_of_5": values["macro_wins"] >= 3,
            "tail_noninferior": values["delta_tail_aupr"]["mean"] >= -0.005,
            "tail_wins_at_least_2_of_5": values["tail_wins"] >= 2,
            "macro_f1_noninferior": values["delta_macro_f1"]["mean"] >= -0.005,
            "hamming_noninferior": values["delta_hamming_loss"]["mean"] <= 0.005,
        }
    lookup = {row["tail_label"]: row for row in by_tail}
    mechanism = {
        name: lookup[name]["vs_db"]["delta_tail_aupr"]["mean"] > 0
        for name in ("MFT", "ISRJ")
    }
    passed = all(all(item.values()) for item in checks.values()) and all(mechanism.values())
    return {
        "passed": passed,
        "decision": "promote_four_class_head_db" if passed else "retain_conditional_status",
        "tail_checks": checks,
        "mechanism_checks": mechanism,
    }


def run(args):
    rows = []
    jobs = []
    for tail_name in args.tails:
        for seed in args.seeds:
            for variant in VARIANTS:
                directory, reused = _directory(args, tail_name, variant, seed)
                jobs.append((tail_name, seed, variant, directory, reused))
    new_jobs = [job for job in jobs if not job[4]]
    completed = 0
    for tail_name, seed, variant, directory, reused in jobs:
        if reused:
            if not (directory / "metrics_threshold_modes.json").exists():
                raise FileNotFoundError("Missing prior run: {}".format(directory))
        else:
            completed += 1
            if not (directory / "metrics_threshold_modes.json").exists():
                print(
                    "[{}/{}] train tail={} variant={} seed={}".format(
                        completed, len(new_jobs), tail_name, variant, seed
                    ),
                    flush=True,
                )
                _train(args, directory, tail_name, variant, seed)
            else:
                print(
                    "[{}/{}] reuse-new tail={} variant={} seed={}".format(
                        completed, len(new_jobs), tail_name, variant, seed
                    ),
                    flush=True,
                )
        rows.append(_collect(directory, tail_name, variant, seed))
    if any(row["locked_test_used"] for row in rows):
        raise ValueError("Locked test usage detected")
    pairs = _paired(rows, args.tails, args.seeds)
    by_tail, overall = _aggregate(pairs, args.tails)
    gate = _gate(by_tail)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "new_training_count": len(new_jobs),
        "gate": gate,
        "aggregate_by_tail": by_tail,
        "aggregate_overall": overall,
        "paired": pairs,
        "runs": rows,
    }
    _write(Path(args.output_root) / "route_a_head_db_confirmatory_summary.json", payload)
    concise = [
        {
            "tail": row["tail_label"],
            "macro": row["vs_asl"]["delta_macro_aupr"]["mean"],
            "tail_aupr": row["vs_asl"]["delta_tail_aupr"]["mean"],
            "macro_wins": row["vs_asl"]["macro_wins"],
            "tail_wins": row["vs_asl"]["tail_wins"],
            "tail_recovery_vs_db": row["vs_db"]["delta_tail_aupr"]["mean"],
        }
        for row in by_tail
    ]
    print(json.dumps({"gate": gate, "aggregate": concise}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_controlled_lt",
    )
    parser.add_argument(
        "--decoupled-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if4_decoupled_head",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if4_head_db_confirmatory",
    )
    parser.add_argument("--tails", nargs="+", default=["FMB", "SinFM", "MFT", "ISRJ"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2036, 2037, 2038, 2039, 2040])
    parser.add_argument("--base-epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
