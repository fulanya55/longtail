"""Run SawFM A6-D2 with two-block discriminative fine-tuning."""

import argparse
import csv
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


def _baseline_dir(args, variant, seed):
    if variant == "A6-PU-TailSelect":
        return (
            Path(args.tail_selection_root)
            / "runs" / TAIL_NAME / variant / "seed_{}".format(seed)
        )
    return (
        Path(args.baseline_root)
        / "runs" / TAIL_NAME / variant / "seed_{}".format(seed)
    )


def _run_dir(args, seed):
    return Path(args.output_root) / "runs" / TAIL_NAME / "A6-D2" / "seed_{}".format(seed)


def _command(args, directory, seed):
    checkpoint = _baseline_dir(args, "BaseASL", seed) / "best_model.pth"
    return [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", "db",
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", "64",
        "--workers", "0",
        "--learning-rate", "0.001",
        "--representation-learning-rate-scale", "0.1",
        "--penultimate-representation-learning-rate-scale", "0.03",
        "--max-train-samples", "0",
        "--max-eval-samples", "300",
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", "0", "2", "3", "4", "5", "1",
        "--selection-metric", "tail_aupr",
        "--selection-tail-index", str(TAIL_INDEX),
        "--initial-checkpoint", str(checkpoint),
        "--trainable-scope", "last_two_blocks_classifier",
        "--output-dir", str(directory),
        "--amp",
    ]


def _train(args, directory, seed):
    if (directory / "run_receipt.json").exists() and (
        directory / "metrics_threshold_modes.json"
    ).exists():
        receipt = _read(directory / "run_receipt.json")
        if receipt.get("status") == "completed":
            print("reuse seed={}".format(seed))
            return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, seed),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError("SawFM A6-D2 failed: {}".format(result.stderr[-2500:]))


def _metrics(directory):
    metrics = _read(directory / "metrics_threshold_modes.json")["fixed_0p5"]
    tail = next(row for row in metrics["per_label"] if row["label"] == TAIL_NAME)
    return {"macro_aupr": metrics["macro_aupr"], "tail_aupr": tail["aupr"]}


def run(args):
    rows = []
    for index, seed in enumerate(args.seeds, start=1):
        directory = _run_dir(args, seed)
        print("[{}/{}] SawFM A6-D2 seed={}".format(index, len(args.seeds), seed))
        _train(args, directory, seed)
        receipt = _read(directory / "run_receipt.json")
        manifest = _read(directory / "longtail_manifest.json")
        baseline_receipt = _read(_baseline_dir(args, "BaseASL", seed) / "run_receipt.json")
        if receipt["split_sha256"] != baseline_receipt["split_sha256"]:
            raise ValueError("A6-D2 split differs from Base-ASL")
        if receipt["manifest_sha256"] != baseline_receipt["manifest_sha256"]:
            raise ValueError("A6-D2 manifest differs from Base-ASL")
        if receipt["trainable_scope"] != "last_two_blocks_classifier":
            raise ValueError("A6-D2 used the wrong trainable scope")
        if receipt["training_parameters"] != 19078:
            raise ValueError("Unexpected A6-D2 trainable parameter count")
        if receipt["selection_metric"] != "tail_aupr":
            raise ValueError("A6-D2 used the wrong selection metric")
        if receipt["locked_test_used"]:
            raise ValueError("Locked test was used")
        selected = _metrics(directory)
        asl = _metrics(_baseline_dir(args, "BaseASL", seed))
        db = _metrics(_baseline_dir(args, "BaseDB", seed))
        d1 = _metrics(_baseline_dir(args, "A6-PU-TailSelect", seed))
        with (directory / "history.csv").open("r", encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        rows.append(
            {
                "seed": int(seed),
                "best_epoch": receipt["best_epoch"],
                "best_early_tail_aupr": receipt["best_selection_value"],
                "history_tail_aupr": [float(row["tail_aupr"]) for row in history],
                "delta_macro_aupr_vs_asl": selected["macro_aupr"] - asl["macro_aupr"],
                "delta_tail_aupr_vs_asl": selected["tail_aupr"] - asl["tail_aupr"],
                "delta_tail_aupr_vs_d1": selected["tail_aupr"] - d1["tail_aupr"],
                "delta_tail_aupr_vs_db": selected["tail_aupr"] - db["tail_aupr"],
                "training_parameters": receipt["training_parameters"],
                "sample_count": manifest["achieved_sample_count"],
                "max_abs_label_error": max(abs(value) for value in manifest["label_count_error"]),
            }
        )
    aggregate = {
        "delta_macro_aupr_vs_asl": _mean_ci(
            [row["delta_macro_aupr_vs_asl"] for row in rows], 13001
        ),
        "delta_tail_aupr_vs_asl": _mean_ci(
            [row["delta_tail_aupr_vs_asl"] for row in rows], 13002
        ),
        "delta_tail_aupr_vs_d1": _mean_ci(
            [row["delta_tail_aupr_vs_d1"] for row in rows], 13003
        ),
        "delta_tail_aupr_vs_db": _mean_ci(
            [row["delta_tail_aupr_vs_db"] for row in rows], 13004
        ),
    }
    gate = {
        "passed": bool(
            aggregate["delta_tail_aupr_vs_asl"]["mean"] > 0.0
            and aggregate["delta_macro_aupr_vs_asl"]["mean"] >= -0.005
            and sum(row["delta_tail_aupr_vs_asl"] > 0.0 for row in rows) >= 2
        ),
        "tail_positive_vs_asl": aggregate["delta_tail_aupr_vs_asl"]["mean"] > 0.0,
        "macro_noninferior_vs_asl": aggregate["delta_macro_aupr_vs_asl"]["mean"] >= -0.005,
        "tail_wins_vs_asl": sum(row["delta_tail_aupr_vs_asl"] > 0.0 for row in rows),
    }
    gate["decision"] = (
        "validate_a6_d2_on_vft_isrj"
        if gate["passed"]
        else "sawfm_requires_deeper_or_multiscale_adaptation"
    )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "tail_label": TAIL_NAME,
            "seeds": list(args.seeds),
            "selection_metric": "tail_aupr",
            "trainable_scope": "last_two_blocks_classifier",
            "epochs": args.epochs,
            "learning_rates": {
                "classifier": 1e-3,
                "final_temporal_block": 1e-4,
                "penultimate_temporal_block": 3e-5,
            },
        },
        "gate": gate,
        "aggregate": aggregate,
        "runs": rows,
    }
    summary = Path(args.output_root) / "route_a_sawfm_d2_summary.json"
    _write(summary, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--tail-selection-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_tail_selection",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_d2",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
