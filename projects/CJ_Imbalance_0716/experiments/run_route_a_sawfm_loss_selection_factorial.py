"""Run a 2x2 SawFM loss-by-checkpoint-selection factorial experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402


TAIL_NAME = "SawFM"
TAIL_INDEX = 1
CELLS = {
    "DB-Macro": {"loss": "db", "selection": "macro_aupr", "reuse": "db"},
    "DB-Tail": {"loss": "db", "selection": "tail_aupr", "reuse": None},
    "Mix90-Macro": {
        "loss": "asl_db_mix",
        "selection": "macro_aupr",
        "reuse": None,
    },
    "Mix90-Tail": {
        "loss": "asl_db_mix",
        "selection": "tail_aupr",
        "reuse": "mix90",
    },
}
METRICS = (
    "macro_aupr",
    "tail_aupr",
    "macro_f1",
    "hamming_loss",
    "tail_f1",
)


def _existing_dir(args, cell, seed):
    if CELLS[cell]["reuse"] == "db":
        return (
            Path(args.db_root)
            / "runs" / TAIL_NAME / "BaseDB" / "seed_{}".format(seed)
        )
    if CELLS[cell]["reuse"] == "mix90":
        return (
            Path(args.mix_root)
            / "runs" / TAIL_NAME / "Mix90" / "seed_{}".format(seed)
        )
    return None


def _run_dir(args, cell, seed):
    return Path(args.output_root) / "runs" / TAIL_NAME / cell / "seed_{}".format(seed)


def _command(args, directory, cell, seed):
    spec = CELLS[cell]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", spec["loss"],
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", "64",
        "--workers", "0",
        "--learning-rate", "0.0003",
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", "0", "2", "3", "4", "5", "1",
        "--selection-metric", spec["selection"],
        "--output-dir", str(directory),
        "--amp",
    ]
    if spec["selection"] == "tail_aupr":
        command.extend(["--selection-tail-index", str(TAIL_INDEX)])
    if spec["loss"] == "asl_db_mix":
        command.extend(["--db-mix-weight", "0.9"])
    return command


def _train(args, directory, cell, seed):
    receipt_path = directory / "run_receipt.json"
    metrics_path = directory / "metrics_threshold_modes.json"
    if receipt_path.exists() and metrics_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") == "completed":
            print("reuse-new cell={} seed={}".format(cell, seed))
            return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, cell, seed),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Factorial run failed cell={} seed={}: {}".format(
                cell, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, cell, seed):
    modes = _read(directory / "metrics_threshold_modes.json")
    metrics = modes["fixed_0p5"]
    receipt = _read(directory / "run_receipt.json")
    manifest = _read(directory / "longtail_manifest.json")
    tail = next(row for row in metrics["per_label"] if row["label"] == TAIL_NAME)
    return {
        "cell": cell,
        "loss": CELLS[cell]["loss"],
        "selection_metric": receipt["selection_metric"],
        "seed": int(seed),
        "macro_aupr": metrics["macro_aupr"],
        "tail_aupr": tail["aupr"],
        "macro_f1": metrics["macro_f1"],
        "hamming_loss": metrics["hamming_loss"],
        "tail_f1": tail["f1"],
        "best_epoch": receipt["best_epoch"],
        "epochs_ran": receipt["epochs_ran"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "training_parameters": receipt["training_parameters"],
        "trainable_scope": receipt["trainable_scope"],
        "locked_test_used": bool(receipt["locked_test_used"]),
        "sample_count": manifest["achieved_sample_count"],
        "max_abs_label_error": max(
            abs(value) for value in manifest["label_count_error"]
        ),
        "source_directory": str(Path(directory).resolve()),
    }


def _delta(left, right):
    return {"delta_{}".format(metric): left[metric] - right[metric] for metric in METRICS}


def _aggregate_effect(rows, effect_name, seed):
    output = {"effect": effect_name}
    for index, metric in enumerate(METRICS):
        output["delta_{}".format(metric)] = _mean_ci(
            [row["delta_{}".format(metric)] for row in rows], seed + index
        )
    return output


def run(args):
    runs = []
    new_cells = [name for name, spec in CELLS.items() if spec["reuse"] is None]
    total = len(new_cells) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        seed_rows = []
        for cell in CELLS:
            directory = _existing_dir(args, cell, seed)
            if directory is None:
                directory = _run_dir(args, cell, seed)
                current += 1
                print("[{}/{}] cell={} seed={}".format(current, total, cell, seed))
                _train(args, directory, cell, seed)
            else:
                print("reuse-existing cell={} seed={}".format(cell, seed))
            row = _collect(directory, cell, seed)
            expected_selection = CELLS[cell]["selection"]
            if row["selection_metric"] != expected_selection:
                raise ValueError(
                    "{} used selection {}, expected {}".format(
                        cell, row["selection_metric"], expected_selection
                    )
                )
            if row["trainable_scope"] != "all" or row["training_parameters"] != 47718:
                raise ValueError("{} did not use full from-scratch training".format(cell))
            if row["locked_test_used"]:
                raise ValueError("Locked test was used")
            if row["sample_count"] != 2400 or row["max_abs_label_error"] != 0:
                raise ValueError("Controlled Label-LT manifest drifted")
            seed_rows.append(row)
            runs.append(row)
        split_hashes = {row["split_sha256"] for row in seed_rows}
        manifest_hashes = {row["manifest_sha256"] for row in seed_rows}
        if len(split_hashes) != 1 or len(manifest_hashes) != 1:
            raise ValueError("Factorial cells do not share split/manifest for seed {}".format(seed))

    effects = []
    effect_rows = {
        "loss_at_macro_selection": [],
        "loss_at_tail_selection": [],
        "selection_under_db": [],
        "selection_under_mix90": [],
        "loss_selection_interaction": [],
    }
    for seed in args.seeds:
        values = {row["cell"]: row for row in runs if row["seed"] == seed}
        loss_macro = _delta(values["Mix90-Macro"], values["DB-Macro"])
        loss_tail = _delta(values["Mix90-Tail"], values["DB-Tail"])
        selection_db = _delta(values["DB-Tail"], values["DB-Macro"])
        selection_mix = _delta(values["Mix90-Tail"], values["Mix90-Macro"])
        interaction = {
            key: loss_tail[key] - loss_macro[key]
            for key in loss_macro
        }
        for name, delta in (
            ("loss_at_macro_selection", loss_macro),
            ("loss_at_tail_selection", loss_tail),
            ("selection_under_db", selection_db),
            ("selection_under_mix90", selection_mix),
            ("loss_selection_interaction", interaction),
        ):
            effect_rows[name].append({"seed": int(seed), **delta})

    for index, (name, rows) in enumerate(effect_rows.items()):
        effects.append(
            {
                **_aggregate_effect(rows, name, 19000 + index * 100),
                "per_seed": rows,
            }
        )

    cell_aggregate = {}
    for index, cell in enumerate(CELLS):
        selected = [row for row in runs if row["cell"] == cell]
        cell_aggregate[cell] = {
            metric: _mean_ci(
                [row[metric] for row in selected], 20000 + index * 100 + offset
            )
            for offset, metric in enumerate(METRICS)
        }

    effect_by_name = {row["effect"]: row for row in effects}
    macro_check = effect_by_name["loss_at_macro_selection"]
    tail_check = effect_by_name["loss_at_tail_selection"]
    gate = {
        "mix90_passes_under_macro_selection": (
            macro_check["delta_macro_aupr"]["mean"] >= -0.005
            and macro_check["delta_tail_aupr"]["mean"] >= 0.005
        ),
        "mix90_passes_under_tail_selection": (
            tail_check["delta_macro_aupr"]["mean"] >= -0.005
            and tail_check["delta_tail_aupr"]["mean"] >= 0.005
        ),
    }
    gate["passed"] = any(
        value for key, value in gate.items() if key.startswith("mix90_passes")
    )
    gate["decision"] = (
        "mix90_has_selection_controlled_value"
        if gate["passed"]
        else "retain_basedb_and_stop_mix90_expansion"
    )

    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "tail_label": TAIL_NAME,
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "cells": CELLS,
            "new_training_runs": int(total),
        },
        "gate": gate,
        "cell_aggregate": cell_aggregate,
        "effects": effects,
        "runs": runs,
    }
    output = Path(args.output_root) / "route_a_sawfm_loss_selection_factorial_summary.json"
    _write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--db-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--mix-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_db_path_screen",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_loss_selection_factorial",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
