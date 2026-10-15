"""Matched Route-A structure screen for delayed-fusion multi-scale complex IQ."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch  # Load PyTorch DLLs before sklearn/OpenMP on Windows.
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


TAIL_INDEX = 1
TAIL_NAME = "SawFM"
CLASS_NAMES = ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
MODELS = {
    "B1_WideCapacity": {"backbone": "complex_1d_true", "width": 56},
    "P0_PAMIQ": {"backbone": "phase_multiscale", "width": 16},
    "F1_DFMS_NoPhase": {"backbone": "dfms_iq_no_phase", "width": 16},
    "F2_DFMS_Full": {"backbone": "dfms_iq", "width": 16},
}
METRICS = (
    "macro_aupr",
    "tail_aupr",
    "macro_f1",
    "hamming_loss",
    "weak_component_recall",
    "weak_vs_negative_auroc",
    "weak_vs_negative_aupr",
    "primary_negative_fpr",
    "f2_macro_f1",
    "f2_hamming_loss",
    "f2_weak_component_recall",
    "f2_negative_fpr",
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
        "--dropout", "0.20",
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
                model_name, seed, (result.stdout + result.stderr)[-4000:]
            )
        )


def _history_summary(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tail_rows = [row for row in rows if row.get("tail_aupr", "nan") != "nan"]
    values = [float(row["tail_aupr"]) for row in tail_rows]
    peak_index = max(range(len(values)), key=values.__getitem__)
    return {
        "epochs": len(rows),
        "tail_peak_epoch": int(float(tail_rows[peak_index]["epoch"])),
        "tail_peak_aupr": values[peak_index],
        "tail_final_aupr": values[-1],
        "forgetting": values[peak_index] - values[-1],
    }


def _component_power(values79):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in CLASS_NAMES],
        axis=1,
    )


def _score_context(config_path, weak_threshold_db):
    config = load_config(config_path)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    weak_mask = compute_weak_component_mask(
        labels, _component_power(values79), weak_threshold_db
    )
    return {"labels": labels, "weak_mask": weak_mask}


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _weak_score_summary(directory, context):
    saved = np.load(directory / "predictions.npz")
    probabilities = saved["probabilities"].astype(np.float64)
    targets = saved["targets"].astype(np.uint8)
    sample_ids = saved["sample_ids"].astype(np.int64)
    if not np.array_equal(targets, context["labels"][sample_ids]):
        raise ValueError("Prediction target/sample alignment failed")
    weak = context["weak_mask"][sample_ids]
    positive = targets.astype(bool)
    weak_positive = positive & weak
    strong_positive = positive & ~weak
    negative = ~positive
    weak_scores = probabilities[weak_positive]
    negative_scores = probabilities[negative]
    binary_scores = np.concatenate([weak_scores, negative_scores])
    binary_targets = np.concatenate(
        [np.ones(weak_scores.size), np.zeros(negative_scores.size)]
    )
    thresholds = _read(directory / "thresholds.json")
    primary = np.asarray(thresholds["per_label"], dtype=np.float64)
    f2 = np.asarray(
        thresholds["modes"]["per_label_f2"], dtype=np.float64
    )
    per_label = {}
    for label_index, name in enumerate(CLASS_NAMES):
        label_weak = weak_positive[:, label_index]
        label_negative = negative[:, label_index]
        label_weak_scores = probabilities[label_weak, label_index]
        label_negative_scores = probabilities[label_negative, label_index]
        combined_scores = np.concatenate(
            [label_weak_scores, label_negative_scores]
        )
        combined_targets = np.concatenate(
            [
                np.ones(label_weak_scores.size),
                np.zeros(label_negative_scores.size),
            ]
        )
        per_label[name] = {
            "weak_vs_negative_auroc": float(
                roc_auc_score(combined_targets, combined_scores)
            ),
            "weak_vs_negative_aupr": float(
                average_precision_score(combined_targets, combined_scores)
            ),
            "weak_positive": _distribution(label_weak_scores),
            "negative": _distribution(label_negative_scores),
            "primary_negative_fpr": float(
                (label_negative_scores >= primary[label_index]).mean()
            ),
            "f2_negative_fpr": float(
                (label_negative_scores >= f2[label_index]).mean()
            ),
        }
    return {
        "weak_vs_negative_auroc": float(
            roc_auc_score(binary_targets, binary_scores)
        ),
        "weak_vs_negative_aupr": float(
            average_precision_score(binary_targets, binary_scores)
        ),
        "weak_positive": _distribution(weak_scores),
        "strong_positive": _distribution(probabilities[strong_positive]),
        "negative": _distribution(negative_scores),
        "primary_negative_fpr": float(
            (probabilities >= primary)[negative].mean()
        ),
        "f2_negative_fpr": float((probabilities >= f2)[negative].mean()),
        "per_label": per_label,
    }


def _metric_row(args, model_name, seed, context):
    directory = _run_dir(args, model_name, seed)
    receipt = _read(directory / "run_receipt.json")
    metrics = _read(directory / "metrics.json")
    modes = _read(directory / "metrics_threshold_modes.json")
    manifest = _read(directory / "longtail_manifest.json")
    if receipt.get("status") != "completed":
        raise ValueError("Run did not complete")
    if receipt.get("locked_test_used"):
        raise ValueError("Locked test was used")
    if manifest.get("achieved_sample_count") != 2400:
        raise ValueError("Long-tail sample count drifted")
    if "per_label_f2" not in modes:
        raise ValueError("per_label_f2 metrics were not recorded")
    tail_metrics = metrics["per_label"][TAIL_INDEX]
    f2_metrics = modes["per_label_f2"]
    score_summary = _weak_score_summary(directory, context)
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
        "weak_vs_negative_auroc": score_summary["weak_vs_negative_auroc"],
        "weak_vs_negative_aupr": score_summary["weak_vs_negative_aupr"],
        "primary_negative_fpr": score_summary["primary_negative_fpr"],
        "f2_macro_f1": float(f2_metrics["macro_f1"]),
        "f2_hamming_loss": float(f2_metrics["hamming_loss"]),
        "f2_weak_component_recall": float(
            f2_metrics["weak_component_recall"]
        ),
        "f2_negative_fpr": score_summary["f2_negative_fpr"],
        "score_distributions": {
            key: score_summary[key]
            for key in ("weak_positive", "strong_positive", "negative")
        },
        "per_label_weak_scores": score_summary["per_label"],
        "per_label_aupr": {
            item["label"]: float(item["aupr"]) for item in metrics["per_label"]
        },
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
        delta["per_label_aupr"] = {
            name: candidate_row["per_label_aupr"][name]
            - baseline_row["per_label_aupr"][name]
            for name in CLASS_NAMES
        }
        delta["per_label_weak_auroc"] = {
            name: candidate_row["per_label_weak_scores"][name][
                "weak_vs_negative_auroc"
            ]
            - baseline_row["per_label_weak_scores"][name][
                "weak_vs_negative_auroc"
            ]
            for name in CLASS_NAMES
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
    aggregate["per_label_aupr"] = {
        name: _mean_ci(
            [row["delta"]["per_label_aupr"][name] for row in paired],
            seed_offset + 100 + index,
        )
        for index, name in enumerate(CLASS_NAMES)
    }
    aggregate["per_label_weak_auroc"] = {
        name: _mean_ci(
            [row["delta"]["per_label_weak_auroc"][name] for row in paired],
            seed_offset + 200 + index,
        )
        for index, name in enumerate(CLASS_NAMES)
    }
    aggregate["weak_positive_seeds"] = sum(
        row["delta"]["delta_weak_component_recall"] > 0 for row in paired
    )
    aggregate["weak_auroc_positive_seeds"] = sum(
        row["delta"]["delta_weak_vs_negative_auroc"] > 0 for row in paired
    )
    return {"per_seed": paired, "aggregate": aggregate}


def _gate(aggregate):
    protected_labels = ["SinFM", "MFT", "VFT"]
    checks = {
        "macro_mean_noninferior": aggregate["delta_macro_aupr"]["mean"] >= -0.005,
        "tail_mean_noninferior": aggregate["delta_tail_aupr"]["mean"] >= 0.0,
        "weak_mean_gain": aggregate["delta_weak_component_recall"]["mean"] >= 0.030,
        "weak_auroc_gain": aggregate["delta_weak_vs_negative_auroc"]["mean"] >= 0.020,
        "hamming_noninferior": aggregate["delta_hamming_loss"]["mean"] <= 0.005,
        "protected_labels": all(
            aggregate["per_label_aupr"][name]["mean"] >= -0.030
            for name in protected_labels
        ),
        "weak_auroc_positive_at_least_two": (
            aggregate["weak_auroc_positive_seeds"] >= 2
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


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

    context = _score_context(args.config, args.weak_threshold_db)
    rows = [
        _metric_row(args, model_name, seed, context)
        for seed in args.seeds
        for model_name in MODELS
    ]
    audit = _audit_pairs(rows, args.seeds)
    comparisons = {
        "p0_vs_b1": _comparison(rows, "P0_PAMIQ", "B1_WideCapacity", 31000),
        "f1_vs_b1": _comparison(rows, "F1_DFMS_NoPhase", "B1_WideCapacity", 31100),
        "f2_vs_b1": _comparison(rows, "F2_DFMS_Full", "B1_WideCapacity", 31200),
        "f1_vs_p0": _comparison(rows, "F1_DFMS_NoPhase", "P0_PAMIQ", 31300),
        "f2_vs_p0": _comparison(rows, "F2_DFMS_Full", "P0_PAMIQ", 31400),
        "f2_vs_f1": _comparison(rows, "F2_DFMS_Full", "F1_DFMS_NoPhase", 31500),
    }
    f1_gate = _gate(comparisons["f1_vs_b1"]["aggregate"])
    f2_gate = _gate(comparisons["f2_vs_b1"]["aggregate"])
    if f2_gate["passed"]:
        decision = "advance_f2_full_dfms"
    elif f1_gate["passed"]:
        decision = "advance_f1_delayed_fusion_and_reject_phase_branch"
    else:
        decision = "stop_or_revise_dfms_before_weak_objective"
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
            "primary_threshold_mode": "shrinkage_per_label_f1",
            "diagnostic_threshold_mode": "per_label_f2",
            "weak_threshold_db": float(args.weak_threshold_db),
            "max_eval_samples": int(args.max_eval_samples),
        },
        "models": MODELS,
        "audit": audit,
        "per_run": rows,
        "comparisons": comparisons,
        "gate": {"f1": f1_gate, "f2": f2_gate, "decision": decision},
    }
    output = Path(args.output_root) / "route_a_dfms_iq_structure_screen_summary.json"
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
        default=(
            PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
