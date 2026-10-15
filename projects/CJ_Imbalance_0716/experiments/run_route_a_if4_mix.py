"""Evaluate ASL-DB interpolation on controlled IF=4 Route-A failures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import (
    CLASS_NAMES,
    PRIMARY_MODE,
    SECONDARY_MODE,
    _mean_ci,
    _read,
    _tail_order,
    _write,
)


VARIANTS = {
    "ASL": {"loss": "asl", "db_mix_weight": None, "reuse": True},
    "DB": {"loss": "db", "db_mix_weight": None, "reuse": True},
    "Mix25": {"loss": "asl_db_mix", "db_mix_weight": 0.25, "reuse": False},
    "Mix50": {"loss": "asl_db_mix", "db_mix_weight": 0.50, "reuse": False},
}


def _tail_index(name):
    return CLASS_NAMES.index(str(name))


def _baseline_directory(args, tail_name, variant, seed):
    loss = VARIANTS[variant]["loss"]
    return (
        Path(args.baseline_root)
        / "if_4p0"
        / tail_name
        / loss
        / "seed_{}".format(seed)
    )


def _new_directory(args, tail_name, variant, seed):
    return Path(args.output_root) / tail_name / variant / "seed_{}".format(seed)


def _train(args, directory, tail_name, variant, seed):
    spec = VARIANTS[variant]
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
            "Mix run failed tail={} variant={} seed={}: {}".format(
                tail_name, variant, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, tail_name, variant, seed):
    metrics = _read(directory / "metrics_threshold_modes.json")
    manifest = _read(directory / "longtail_manifest.json")
    receipt = _read(directory / "run_receipt.json")
    if manifest["achieved_sample_count"] != 1600:
        raise ValueError("Controlled sample count drifted")
    tail_fixed = next(
        row for row in metrics["fixed_0p5"]["per_label"]
        if row["label"] == tail_name
    )
    threshold = {}
    for mode in (PRIMARY_MODE, SECONDARY_MODE):
        values = metrics[mode]
        tail = next(row for row in values["per_label"] if row["label"] == tail_name)
        threshold[mode] = {
            "macro_f1": values["macro_f1"],
            "hamming_loss": values["hamming_loss"],
            "cardinality_mae": values["cardinality_mae"],
            "tail_precision": tail["precision"],
            "tail_recall": tail["recall"],
            "tail_f1": tail["f1"],
        }
    return {
        "tail_label": tail_name,
        "variant": variant,
        "seed": int(seed),
        "macro_aupr": metrics["fixed_0p5"]["macro_aupr"],
        "tail_aupr": tail_fixed["aupr"],
        "threshold_metrics": threshold,
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "config_sha256": receipt["config_sha256"],
        "train_seconds": receipt["train_seconds"],
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
    for variant in ("Mix25", "Mix50"):
        for tail_name in tails:
            for seed in seeds:
                selected = [
                    row for row in rows
                    if row["tail_label"] == tail_name and row["seed"] == seed
                ]
                values = {row["variant"]: row for row in selected}
                manifest_hashes = {row["manifest_sha256"] for row in selected}
                split_hashes = {row["split_sha256"] for row in selected}
                if len(manifest_hashes) != 1 or len(split_hashes) != 1:
                    raise ValueError("Variant pairing did not preserve split/manifest")
                pairs.append(
                    {
                        "variant": variant,
                        "tail_label": tail_name,
                        "seed": seed,
                        "manifest_sha256": values[variant]["manifest_sha256"],
                        "vs_asl": _difference(values[variant], values["ASL"]),
                        "vs_db": _difference(values[variant], values["DB"]),
                    }
                )
    return pairs


def _aggregate_difference(rows, comparison, seed):
    result = {
        "delta_macro_aupr": _mean_ci(
            [row[comparison]["delta_macro_aupr"] for row in rows], seed
        ),
        "delta_tail_aupr": _mean_ci(
            [row[comparison]["delta_tail_aupr"] for row in rows], seed + 1
        ),
        "macro_aupr_wins": sum(
            row[comparison]["delta_macro_aupr"] > 0 for row in rows
        ),
        "tail_aupr_wins": sum(
            row[comparison]["delta_tail_aupr"] > 0 for row in rows
        ),
        PRIMARY_MODE: {},
        SECONDARY_MODE: {},
    }
    for mode_index, mode in enumerate((PRIMARY_MODE, SECONDARY_MODE)):
        for metric_index, metric in enumerate(
            ("macro_f1", "hamming_loss", "cardinality_mae", "tail_f1")
        ):
            result[mode][metric] = _mean_ci(
                [row[comparison]["threshold_modes"][mode][metric] for row in rows],
                seed + 10 + mode_index * 10 + metric_index,
            )
    return result


def _aggregate(pairs):
    result = []
    for variant_index, variant in enumerate(("Mix25", "Mix50")):
        for tail_index, tail_name in enumerate(("FMB", "MFT", "ISRJ")):
            selected = [
                row for row in pairs
                if row["variant"] == variant and row["tail_label"] == tail_name
            ]
            entry = {
                "variant": variant,
                "tail_label": tail_name,
                "pair_count": len(selected),
            }
            base_seed = 20260723 + variant_index * 100 + tail_index * 20
            entry["vs_asl"] = _aggregate_difference(selected, "vs_asl", base_seed)
            entry["vs_db"] = _aggregate_difference(selected, "vs_db", base_seed + 500)
            result.append(entry)
    return result


def _gate(aggregate):
    candidates = []
    for variant in ("Mix25", "Mix50"):
        entries = {
            row["tail_label"]: row for row in aggregate if row["variant"] == variant
        }
        checks = {}
        for tail_name in ("MFT", "ISRJ"):
            row = entries[tail_name]
            checks[tail_name] = {
                "tail_aupr_better_than_db": row["vs_db"]["delta_tail_aupr"]["mean"] > 0,
                "tail_aupr_noninferior_to_asl": row["vs_asl"]["delta_tail_aupr"]["mean"] >= -0.005,
                "macro_aupr_better_than_asl": row["vs_asl"]["delta_macro_aupr"]["mean"] > 0,
                "macro_f1_noninferior_to_asl": row["vs_asl"][PRIMARY_MODE]["macro_f1"]["mean"] >= -0.005,
                "hamming_noninferior_to_asl": row["vs_asl"][PRIMARY_MODE]["hamming_loss"]["mean"] <= 0.005,
            }
        fmb = entries["FMB"]
        checks["FMB_sanity"] = {
            "macro_aupr_better_than_asl": fmb["vs_asl"]["delta_macro_aupr"]["mean"] > 0,
            "tail_aupr_noninferior_to_asl": fmb["vs_asl"]["delta_tail_aupr"]["mean"] >= -0.005,
        }
        passed = all(all(values.values()) for values in checks.values())
        worst_failure_tail = min(
            entries[name]["vs_asl"]["delta_tail_aupr"]["mean"]
            for name in ("MFT", "ISRJ")
        )
        mean_failure_macro = float(
            np.mean(
                [
                    entries[name]["vs_asl"]["delta_macro_aupr"]["mean"]
                    for name in ("MFT", "ISRJ")
                ]
            )
        )
        candidates.append(
            {
                "variant": variant,
                "passed": passed,
                "checks": checks,
                "selection_score": {
                    "worst_failure_tail_delta_vs_asl": worst_failure_tail,
                    "mean_failure_macro_delta_vs_asl": mean_failure_macro,
                },
            }
        )
    passing = [row for row in candidates if row["passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                row["selection_score"]["worst_failure_tail_delta_vs_asl"],
                row["selection_score"]["mean_failure_macro_delta_vs_asl"],
            ),
        )["variant"]
    return {
        "passed": bool(passing),
        "selected_variant": selected,
        "decision": (
            "promote_selected_mix" if passing else "stop_loss_interpolation"
        ),
        "candidates": candidates,
    }


def run(args):
    rows = []
    total_new = len(args.tails) * len(args.seeds) * 2
    completed_new = 0
    for tail_name in args.tails:
        if tail_name not in ("FMB", "MFT", "ISRJ"):
            raise ValueError("This frozen stage only permits FMB, MFT, and ISRJ")
        for seed in args.seeds:
            for variant in VARIANTS:
                if VARIANTS[variant]["reuse"]:
                    directory = _baseline_directory(args, tail_name, variant, seed)
                    if not (directory / "metrics_threshold_modes.json").exists():
                        raise FileNotFoundError("Missing frozen baseline: {}".format(directory))
                else:
                    completed_new += 1
                    directory = _new_directory(args, tail_name, variant, seed)
                    directory.mkdir(parents=True, exist_ok=True)
                    if not (directory / "metrics_threshold_modes.json").exists():
                        print(
                            "[{}/{}] train tail={} variant={} seed={}".format(
                                completed_new, total_new, tail_name, variant, seed
                            ),
                            flush=True,
                        )
                        _train(args, directory, tail_name, variant, seed)
                    else:
                        print(
                            "[{}/{}] reuse tail={} variant={} seed={}".format(
                                completed_new, total_new, tail_name, variant, seed
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
    output = Path(args.output_root) / "route_a_if4_mix_summary.json"
    _write(output, payload)
    print(json.dumps({"gate": gate, "aggregate": aggregate}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_controlled_lt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_if4_mix",
    )
    parser.add_argument("--tails", nargs="+", default=["FMB", "MFT", "ISRJ"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2036, 2037, 2038])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
