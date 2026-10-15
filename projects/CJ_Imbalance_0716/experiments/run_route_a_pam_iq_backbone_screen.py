"""Capacity-matched Route-A screen for the PAM-IQ feature backbone."""

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


TAIL_INDEX = 1
TAIL_NAME = "SawFM"
MODELS = {
    "B0_BaseDB": {"backbone": "complex_1d_true", "width": 16},
    "B1_WideCapacity": {"backbone": "complex_1d_true", "width": 56},
    "P0_PAMIQ": {"backbone": "phase_multiscale", "width": 16},
}
METRICS = (
    "macro_aupr",
    "tail_aupr",
    "macro_f1",
    "hamming_loss",
    "weak_component_recall",
    "forgetting",
)


def _run_dir(args, model_name, seed):
    return Path(args.output_root) / "runs" / model_name / "seed_{}".format(seed)


def _command(args, directory, model_name, seed):
    model = MODELS[model_name]
    return [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", "db",
        "--backbone", model["backbone"],
        "--width", str(model["width"]),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", "64",
        "--workers", "0",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", "0", "2", "3", "4", "5", "1",
        "--selection-metric", "macro_aupr",
        "--selection-tail-index", str(TAIL_INDEX),
        "--output-dir", str(directory),
        "--amp",
    ]


def _train(args, directory, model_name, seed):
    receipt_path = directory / "run_receipt.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") == "completed":
            print("reuse model={} seed={}".format(model_name, seed), flush=True)
            return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, model_name, seed),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "Training failed model={} seed={}: {}".format(
                model_name, seed, (result.stdout + result.stderr)[-3000:]
            )
        )


def _history_summary(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tail_rows = [row for row in rows if row.get("tail_aupr", "nan") != "nan"]
    if not tail_rows:
        raise ValueError("tail_aupr was not recorded in {}".format(path))
    values = [float(row["tail_aupr"]) for row in tail_rows]
    peak_index = max(range(len(values)), key=values.__getitem__)
    peak = values[peak_index]
    final = values[-1]
    return {
        "epochs": len(rows),
        "tail_peak_epoch": int(float(tail_rows[peak_index]["epoch"])),
        "tail_peak_aupr": peak,
        "tail_final_aupr": final,
        "forgetting": peak - final,
    }


def _metric_row(args, model_name, seed):
    directory = _run_dir(args, model_name, seed)
    receipt = _read(directory / "run_receipt.json")
    metrics = _read(directory / "metrics.json")
    manifest = _read(directory / "longtail_manifest.json")
    if receipt.get("status") != "completed":
        raise ValueError("Run did not complete")
    if receipt.get("locked_test_used"):
        raise ValueError("Locked test was used")
    if manifest.get("achieved_sample_count") != 2400:
        raise ValueError("Long-tail sample count drifted")
    tail_metrics = next(
        item for item in metrics["per_label"] if item["label_index"] == TAIL_INDEX
    )
    return {
        "model": model_name,
        "seed": int(seed),
        "backbone": MODELS[model_name]["backbone"],
        "width": MODELS[model_name]["width"],
        "deployment_parameters": int(receipt["deployment_parameters"]),
        "training_parameters": int(receipt["training_parameters"]),
        "best_epoch": int(receipt["best_epoch"]),
        "macro_aupr": float(metrics["macro_aupr"]),
        "tail_aupr": float(tail_metrics["aupr"]),
        "macro_f1": float(metrics["macro_f1"]),
        "hamming_loss": float(metrics["hamming_loss"]),
        "weak_component_recall": float(metrics["weak_component_recall"]),
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        **_history_summary(directory / "history.csv"),
    }


def _audit_pairs(rows, seeds):
    audit = []
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if len(seed_rows) != len(MODELS):
            raise ValueError("Missing model for seed {}".format(seed))
        split_hashes = {row["split_sha256"] for row in seed_rows}
        manifest_hashes = {row["manifest_sha256"] for row in seed_rows}
        if len(split_hashes) != 1 or len(manifest_hashes) != 1:
            raise ValueError("Split or manifest mismatch for seed {}".format(seed))
        audit.append(
            {
                "seed": int(seed),
                "split_sha256": next(iter(split_hashes)),
                "manifest_sha256": next(iter(manifest_hashes)),
                "models": [row["model"] for row in seed_rows],
            }
        )
    return audit


def _comparison(rows, candidate, baseline, seed_offset):
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate_row = next(
            row for row in rows if row["seed"] == seed and row["model"] == candidate
        )
        baseline_row = next(
            row for row in rows if row["seed"] == seed and row["model"] == baseline
        )
        delta = {
            "delta_{}".format(metric): candidate_row[metric] - baseline_row[metric]
            for metric in METRICS
        }
        paired.append(
            {
                "seed": int(seed),
                "candidate": candidate,
                "baseline": baseline,
                "delta": delta,
            }
        )
    aggregate = {}
    for index, metric in enumerate(METRICS):
        key = "delta_{}".format(metric)
        aggregate[key] = _mean_ci(
            [row["delta"][key] for row in paired], seed_offset + index
        )
    aggregate["macro_noninferior_seeds"] = sum(
        row["delta"]["delta_macro_aupr"] >= -0.005 for row in paired
    )
    aggregate["tail_positive_seeds"] = sum(
        row["delta"]["delta_tail_aupr"] > 0 for row in paired
    )
    return {"per_seed": paired, "aggregate": aggregate}


def run(args):
    total = len(MODELS) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        for model_name in MODELS:
            current += 1
            print(
                "[{}/{}] model={} seed={}".format(
                    current, total, model_name, seed
                ),
                flush=True,
            )
            _train(args, _run_dir(args, model_name, seed), model_name, seed)

    rows = [
        _metric_row(args, model_name, seed)
        for seed in args.seeds
        for model_name in MODELS
    ]
    audit = _audit_pairs(rows, args.seeds)
    comparisons = {
        "wide_vs_base": _comparison(rows, "B1_WideCapacity", "B0_BaseDB", 27000),
        "pam_vs_base": _comparison(rows, "P0_PAMIQ", "B0_BaseDB", 27100),
        "pam_vs_wide": _comparison(rows, "P0_PAMIQ", "B1_WideCapacity", 27200),
    }
    primary = comparisons["pam_vs_wide"]["aggregate"]
    gate_checks = {
        "macro_mean_noninferior": primary["delta_macro_aupr"]["mean"] >= -0.005,
        "tail_mean_gain": primary["delta_tail_aupr"]["mean"] >= 0.010,
        "tail_positive_at_least_two": primary["tail_positive_seeds"] >= 2,
        "weak_recall_gain": primary["delta_weak_component_recall"]["mean"] >= 0.020,
        "hamming_noninferior": primary["delta_hamming_loss"]["mean"] <= 0.005,
    }
    passed = all(gate_checks.values())
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "sample_count": 2400,
            "tail_name": TAIL_NAME,
            "tail_count": 400,
            "imbalance_factor": 4,
            "loss": "db",
            "selection_metric": "macro_aupr",
            "tail_diagnostic_index": TAIL_INDEX,
            "max_eval_samples": int(args.max_eval_samples),
        },
        "models": MODELS,
        "audit": audit,
        "per_run": rows,
        "comparisons": comparisons,
        "gate": {
            "checks": gate_checks,
            "passed": passed,
            "decision": (
                "advance_pam_iq_to_dual_axis_visibility_screen"
                if passed
                else "stop_or_revise_pam_iq_before_dual_axis_integration"
            ),
        },
    }
    output = Path(args.output_root) / "route_a_pam_iq_backbone_screen_summary.json"
    _write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_pam_iq_backbone_screen",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

