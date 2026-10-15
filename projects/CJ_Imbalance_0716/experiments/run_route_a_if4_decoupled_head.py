"""Run classifier-only Route-A IF=4 experiments from frozen ASL checkpoints."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import (  # noqa: E402
    CLASS_NAMES,
    PRIMARY_MODE,
    SECONDARY_MODE,
    _mean_ci,
    _read,
    _tail_order,
    _write,
)


HEAD_VARIANTS = {
    "HeadASL": {"loss": "asl", "db_mix_weight": 0.5},
    "HeadMix25": {"loss": "asl_db_mix", "db_mix_weight": 0.25},
    "HeadDB": {"loss": "db", "db_mix_weight": 0.5},
}


def _tail_index(name):
    return CLASS_NAMES.index(str(name))


def _base_directory(args, tail_name, variant, seed):
    if variant == "BaseASL":
        return Path(args.controlled_root) / "if_4p0" / tail_name / "asl" / "seed_{}".format(seed)
    if variant == "BaseDB":
        return Path(args.controlled_root) / "if_4p0" / tail_name / "db" / "seed_{}".format(seed)
    if variant == "E2EMix25":
        return Path(args.mix_root) / tail_name / "Mix25" / "seed_{}".format(seed)
    raise ValueError("Unknown base variant: {}".format(variant))


def _head_directory(args, tail_name, variant, seed):
    return Path(args.output_root) / tail_name / variant / "seed_{}".format(seed)


def _asl_checkpoint(args, tail_name, seed):
    return _base_directory(args, tail_name, "BaseASL", seed) / "best_model.pth"


def _train(args, directory, tail_name, variant, seed):
    spec = HEAD_VARIANTS[variant]
    checkpoint = _asl_checkpoint(args, tail_name, seed)
    if not checkpoint.exists():
        raise FileNotFoundError("Missing ASL initialization checkpoint: {}".format(checkpoint))
    tail_index = _tail_index(tail_name)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", spec["loss"],
        "--db-mix-weight", str(spec["db_mix_weight"]),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--learning-rate", str(args.learning_rate),
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "1600",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", *[str(value) for value in _tail_order(tail_index)],
        "--selection-metric", "macro_aupr",
        "--initial-checkpoint", str(checkpoint),
        "--trainable-scope", "classifier",
        "--output-dir", str(directory),
        "--amp",
    ]
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Head run failed tail={} variant={} seed={}: {}".format(
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
    threshold = {}
    for mode in (PRIMARY_MODE, SECONDARY_MODE):
        values = metrics[mode]
        tail = next(row for row in values["per_label"] if row["label"] == tail_name)
        threshold[mode] = {
            "macro_f1": values["macro_f1"],
            "hamming_loss": values["hamming_loss"],
            "cardinality_mae": values["cardinality_mae"],
            "tail_f1": tail["f1"],
        }
    return {
        "variant": variant,
        "tail_label": tail_name,
        "seed": int(seed),
        "macro_aupr": metrics["fixed_0p5"]["macro_aupr"],
        "tail_aupr": tail_fixed["aupr"],
        "threshold_metrics": threshold,
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "initial_checkpoint_sha256": receipt.get("initial_checkpoint_sha256"),
        "trainable_scope": receipt.get("trainable_scope", "all"),
        "training_parameters": receipt["training_parameters"],
        "train_seconds": receipt["train_seconds"],
        "achieved_sample_count": manifest.get("achieved_sample_count"),
        "locked_test_used": bool(receipt["locked_test_used"]),
    }


def _difference(candidate, baseline):
    result = {
        "delta_macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "delta_tail_aupr": candidate["tail_aupr"] - baseline["tail_aupr"],
        "threshold_modes": {},
    }
    for mode in (PRIMARY_MODE, SECONDARY_MODE):
        result["threshold_modes"][mode] = {
            key: candidate["threshold_metrics"][mode][key]
            - baseline["threshold_metrics"][mode][key]
            for key in candidate["threshold_metrics"][mode]
        }
    return result


def _paired(rows, tails, seeds):
    pairs = []
    comparisons = ("BaseASL", "BaseDB", "E2EMix25", "HeadASL")
    for variant in HEAD_VARIANTS:
        for tail_name in tails:
            for seed in seeds:
                selected = [
                    row for row in rows
                    if row["tail_label"] == tail_name and row["seed"] == seed
                ]
                values = {row["variant"]: row for row in selected}
                if len({row["manifest_sha256"] for row in selected}) != 1:
                    raise ValueError("All decoupled comparisons must share a manifest")
                if len({row["split_sha256"] for row in selected}) != 1:
                    raise ValueError("All decoupled comparisons must share a split")
                head = values[variant]
                if head["trainable_scope"] != "classifier":
                    raise ValueError("Head run did not use classifier-only scope")
                if head["initial_checkpoint_sha256"] is None:
                    raise ValueError("Head run did not record ASL initialization hash")
                pairs.append(
                    {
                        "variant": variant,
                        "tail_label": tail_name,
                        "seed": seed,
                        "manifest_sha256": head["manifest_sha256"],
                        "comparisons": {
                            name: _difference(head, values[name]) for name in comparisons
                        },
                    }
                )
    return pairs


def _aggregate_comparison(rows, baseline, seed):
    result = {
        "delta_macro_aupr": _mean_ci(
            [row["comparisons"][baseline]["delta_macro_aupr"] for row in rows], seed
        ),
        "delta_tail_aupr": _mean_ci(
            [row["comparisons"][baseline]["delta_tail_aupr"] for row in rows], seed + 1
        ),
        "macro_wins": sum(
            row["comparisons"][baseline]["delta_macro_aupr"] > 0 for row in rows
        ),
        "tail_wins": sum(
            row["comparisons"][baseline]["delta_tail_aupr"] > 0 for row in rows
        ),
        PRIMARY_MODE: {},
    }
    for index, metric in enumerate(("macro_f1", "hamming_loss", "cardinality_mae", "tail_f1")):
        result[PRIMARY_MODE][metric] = _mean_ci(
            [
                row["comparisons"][baseline]["threshold_modes"][PRIMARY_MODE][metric]
                for row in rows
            ],
            seed + 10 + index,
        )
    return result


def _aggregate(pairs):
    result = []
    for variant_index, variant in enumerate(HEAD_VARIANTS):
        for tail_index, tail_name in enumerate(("FMB", "MFT", "ISRJ")):
            selected = [
                row for row in pairs
                if row["variant"] == variant and row["tail_label"] == tail_name
            ]
            entry = {"variant": variant, "tail_label": tail_name, "pair_count": len(selected)}
            base_seed = 20260723 + variant_index * 200 + tail_index * 30
            for baseline_index, baseline in enumerate(
                ("BaseASL", "BaseDB", "E2EMix25", "HeadASL")
            ):
                entry["vs_{}".format(baseline)] = _aggregate_comparison(
                    selected, baseline, base_seed + baseline_index * 40
                )
            result.append(entry)
    return result


def _gate(aggregate):
    entries = {
        row["tail_label"]: row
        for row in aggregate
        if row["variant"] == "HeadMix25"
    }
    checks = {}
    for tail_name in ("MFT", "ISRJ"):
        row = entries[tail_name]
        base = row["vs_BaseASL"]
        checks[tail_name] = {
            "tail_better_than_head_asl": row["vs_HeadASL"]["delta_tail_aupr"]["mean"] > 0,
            "tail_noninferior_to_base_asl": base["delta_tail_aupr"]["mean"] >= -0.005,
            "macro_noninferior_to_base_asl": base["delta_macro_aupr"]["mean"] >= -0.005,
            "tail_noninferior_to_e2e_mix25": row["vs_E2EMix25"]["delta_tail_aupr"]["mean"] >= -0.005,
            "macro_f1_noninferior": base[PRIMARY_MODE]["macro_f1"]["mean"] >= -0.005,
            "hamming_noninferior": base[PRIMARY_MODE]["hamming_loss"]["mean"] <= 0.005,
        }
    fmb = entries["FMB"]["vs_BaseASL"]
    checks["FMB_sanity"] = {
        "macro_noninferior": fmb["delta_macro_aupr"]["mean"] >= -0.005,
        "tail_noninferior": fmb["delta_tail_aupr"]["mean"] >= -0.005,
    }
    head_mix25_passed = all(all(values.values()) for values in checks.values())
    head_db = {
        row["tail_label"]: row
        for row in aggregate
        if row["variant"] == "HeadDB"
    }
    representation_damage_supported = all(
        head_db[name]["vs_BaseDB"]["delta_tail_aupr"]["mean"] > 0
        for name in ("MFT", "ISRJ")
    )
    head_db_checks = {}
    for tail_name in ("MFT", "ISRJ"):
        row = head_db[tail_name]
        base = row["vs_BaseASL"]
        head_db_checks[tail_name] = {
            "macro_better_than_base_asl": base["delta_macro_aupr"]["mean"] > 0,
            "tail_noninferior_to_base_asl": base["delta_tail_aupr"]["mean"] >= -0.005,
            "macro_f1_noninferior": base[PRIMARY_MODE]["macro_f1"]["mean"] >= -0.005,
            "hamming_noninferior": base[PRIMARY_MODE]["hamming_loss"]["mean"] <= 0.005,
            "tail_better_than_end_to_end_db": row["vs_BaseDB"]["delta_tail_aupr"]["mean"] > 0,
        }
    head_db_fmb = head_db["FMB"]["vs_BaseASL"]
    head_db_checks["FMB_sanity"] = {
        "macro_better_than_base_asl": head_db_fmb["delta_macro_aupr"]["mean"] > 0,
        "tail_noninferior_to_base_asl": head_db_fmb["delta_tail_aupr"]["mean"] >= -0.005,
    }
    head_db_diagnostic_passed = all(
        all(values.values()) for values in head_db_checks.values()
    )
    return {
        "passed": head_mix25_passed,
        "decision": (
            "promote_decoupled_head_mix25"
            if head_mix25_passed
            else "do_not_promote_head_mix25"
        ),
        "head_mix25_preregistered_checks": checks,
        "head_db_diagnostic": {
            "passed": head_db_diagnostic_passed,
            "status": (
                "promising_confirm_on_new_seeds"
                if head_db_diagnostic_passed
                else "failed_diagnostic"
            ),
            "checks": head_db_checks,
        },
        "mechanism_diagnostic": {
            "head_db_improves_both_failed_tails_vs_base_db": representation_damage_supported,
            "interpretation": (
                "end_to_end_db_representation_damage_supported"
                if representation_damage_supported
                else "classification_boundary_or_mixed_mechanism"
            ),
        },
    }


def run(args):
    rows = []
    total = len(args.tails) * len(args.seeds) * len(HEAD_VARIANTS)
    completed = 0
    for tail_name in args.tails:
        if tail_name not in ("FMB", "MFT", "ISRJ"):
            raise ValueError("Frozen stage permits FMB, MFT, and ISRJ only")
        for seed in args.seeds:
            for base_variant in ("BaseASL", "BaseDB", "E2EMix25"):
                directory = _base_directory(args, tail_name, base_variant, seed)
                if not (directory / "metrics_threshold_modes.json").exists():
                    raise FileNotFoundError("Missing baseline: {}".format(directory))
                rows.append(_collect(directory, tail_name, base_variant, seed))
            for variant in HEAD_VARIANTS:
                completed += 1
                directory = _head_directory(args, tail_name, variant, seed)
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
    aggregate = _aggregate(pairs)
    gate = _gate(aggregate)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "gate": gate,
        "aggregate": aggregate,
        "paired": pairs,
        "runs": rows,
    }
    _write(Path(args.output_root) / "route_a_if4_decoupled_head_summary.json", payload)
    concise = [
        {
            "variant": row["variant"],
            "tail_label": row["tail_label"],
            "vs_BaseASL_macro": row["vs_BaseASL"]["delta_macro_aupr"]["mean"],
            "vs_BaseASL_tail": row["vs_BaseASL"]["delta_tail_aupr"]["mean"],
            "vs_HeadASL_tail": row["vs_HeadASL"]["delta_tail_aupr"]["mean"],
        }
        for row in aggregate
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
        "--mix-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if4_mix",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if4_decoupled_head",
    )
    parser.add_argument("--tails", nargs="+", default=["FMB", "MFT", "ISRJ"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2036, 2037, 2038])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
