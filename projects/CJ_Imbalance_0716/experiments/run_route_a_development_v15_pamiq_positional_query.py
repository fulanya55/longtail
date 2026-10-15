"""Five-seed, train-only test of position-aware versus content-only PAM-IQ queries.

The sole changed mechanism is deterministic absolute temporal position in each
label query key.  Both models retain the same PAM-IQ trunk, global safety
path, query width, classifier, loss, long-tail manifest, and threshold budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# On this Windows PY10 build, importing sklearn before torch can make c10.dll
# fail to initialize.  Load torch first; the experiment mechanics are unchanged.
import torch  # noqa: F401
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_development_v15_pamiq_positional_query import (  # noqa: E402
    audit_train,
)
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _component_power  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGET_INDICES = (4, 5)
TARGET_NAMES = tuple(CLASS_NAMES[index] for index in TARGET_INDICES)
MODELS = {
    "Q0_ContentQuery": {"backbone": "pam_iq_dual_query", "width": 16},
    "Q1_PositionalQuery": {"backbone": "pam_iq_positional_query", "width": 16},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: dict) -> None:
    if config.get("evidence_role") != "development_v15_pamiq_positional_query":
        raise ValueError("requires the fresh development-v15 configuration")
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v15 forbids locked-test access")
    forbidden = ("test_v2_used", "test_v3_used", "test_v15_used")
    if any(bool(config.get(name, False)) for name in forbidden):
        raise ValueError("a forbidden test role is marked used")
    if not bool(config.get("test_v3_sealed", False)):
        raise ValueError("test-v3 must remain sealed")
    if bool(config.get("test_v15_generated", True)) or bool(
        config.get("test_v15_access_authorized", True)
    ):
        raise ValueError("development-v15 may generate and use train only")


def _run_dir(args, model_name: str, seed: int) -> Path:
    return Path(args.output_root) / "runs" / model_name / f"seed_{seed}"


def _command(args, directory: Path, model_name: str, seed: int) -> list[str]:
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
        "--selection-tail-index", "1",
        "--output-dir", str(directory),
        "--amp",
    ]


def _train(args, directory: Path, model_name: str, seed: int) -> None:
    receipt_path = directory / "run_receipt.json"
    if receipt_path.is_file() and _read(receipt_path).get("status") == "completed":
        print(f"reuse model={model_name} seed={seed}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, model_name, seed),
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            f"training failed model={model_name} seed={seed}: "
            f"{(result.stdout + result.stderr)[-4000:]}"
        )


def _target_diagnostics(predictions: np.lib.npyio.NpzFile, weak_all: np.ndarray) -> dict:
    probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
    hard = np.asarray(predictions["predictions"], dtype=bool)
    targets = np.asarray(predictions["targets"], dtype=bool)
    sample_ids = np.asarray(predictions["sample_ids"], dtype=np.int64)
    weak = np.asarray(weak_all[sample_ids], dtype=bool)
    result = {}
    for index, name in zip(TARGET_INDICES, TARGET_NAMES):
        weak_positive = weak[:, index] & targets[:, index]
        negative = ~targets[:, index]
        rank_targets = np.concatenate((np.ones(weak_positive.sum()), np.zeros(negative.sum())))
        rank_scores = np.concatenate((probabilities[weak_positive, index], probabilities[negative, index]))
        if weak_positive.sum() == 0 or negative.sum() == 0:
            raise ValueError(f"{name} lacks weak positives or negatives")
        result[name] = {
            "weak_positive_count": int(weak_positive.sum()),
            "negative_count": int(negative.sum()),
            "weak_recall": float(hard[weak_positive, index].mean()),
            "negative_fpr": float(hard[negative, index].mean()),
            "weak_vs_negative_auroc": float(roc_auc_score(rank_targets, rank_scores)),
            "aupr": float(average_precision_score(targets[:, index], probabilities[:, index])),
        }
    result["target_mean_weak_recall"] = float(
        np.mean([result[name]["weak_recall"] for name in TARGET_NAMES])
    )
    result["target_mean_negative_fpr"] = float(
        np.mean([result[name]["negative_fpr"] for name in TARGET_NAMES])
    )
    result["target_weak_vs_negative_auroc"] = float(
        np.mean([result[name]["weak_vs_negative_auroc"] for name in TARGET_NAMES])
    )
    return result


def _metric_row(args, model_name: str, seed: int, weak_all: np.ndarray) -> dict:
    directory = _run_dir(args, model_name, seed)
    receipt = _read(directory / "run_receipt.json")
    metrics = _read(directory / "metrics.json")
    manifest = _read(directory / "longtail_manifest.json")
    if receipt.get("status") != "completed" or receipt.get("locked_test_used"):
        raise ValueError("incomplete run or locked-test use")
    if manifest.get("achieved_sample_count") != 2400:
        raise ValueError("long-tail sample count drifted")
    with np.load(directory / "predictions.npz") as predictions:
        target = _target_diagnostics(predictions, weak_all)
    return {
        "model": model_name,
        "seed": int(seed),
        "backbone": MODELS[model_name]["backbone"],
        "width": MODELS[model_name]["width"],
        "deployment_parameters": int(receipt["deployment_parameters"]),
        "training_parameters": int(receipt["training_parameters"]),
        "best_epoch": int(receipt["best_epoch"]),
        "macro_aupr": float(metrics["macro_aupr"]),
        "macro_f1": float(metrics["macro_f1"]),
        "hamming_loss": float(metrics["hamming_loss"]),
        "per_label_aupr": {
            item["label"]: float(item["aupr"]) for item in metrics["per_label"]
        },
        "target": target,
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v15_used": False,
    }


def _comparison(rows: list[dict]) -> dict:
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        left = next(row for row in rows if row["seed"] == seed and row["model"] == "Q1_PositionalQuery")
        right = next(row for row in rows if row["seed"] == seed and row["model"] == "Q0_ContentQuery")
        if left["split_sha256"] != right["split_sha256"] or left["manifest_sha256"] != right["manifest_sha256"]:
            raise ValueError(f"paired role mismatch at seed {seed}")
        paired.append({
            "seed": seed,
            "delta_macro_aupr": left["macro_aupr"] - right["macro_aupr"],
            "delta_macro_f1": left["macro_f1"] - right["macro_f1"],
            "delta_hamming_loss": left["hamming_loss"] - right["hamming_loss"],
            "delta_target_mean_weak_recall": left["target"]["target_mean_weak_recall"] - right["target"]["target_mean_weak_recall"],
            "delta_target_weak_vs_negative_auroc": left["target"]["target_weak_vs_negative_auroc"] - right["target"]["target_weak_vs_negative_auroc"],
            "delta_target_mean_negative_fpr": left["target"]["target_mean_negative_fpr"] - right["target"]["target_mean_negative_fpr"],
            "delta_per_label_aupr": {name: left["per_label_aupr"][name] - right["per_label_aupr"][name] for name in CLASS_NAMES},
            "delta_target_per_label": {
                name: {metric: left["target"][name][metric] - right["target"][name][metric] for metric in ("weak_recall", "negative_fpr", "weak_vs_negative_auroc", "aupr")}
                for name in TARGET_NAMES
            },
        })
    aggregate = {
        key: _mean_ci([row[key] for row in paired], 20260820 + offset)
        for offset, key in enumerate((
            "delta_macro_aupr", "delta_macro_f1", "delta_hamming_loss",
            "delta_target_mean_weak_recall", "delta_target_weak_vs_negative_auroc",
            "delta_target_mean_negative_fpr",
        ))
    }
    aggregate["per_label_aupr"] = {
        name: _mean_ci([row["delta_per_label_aupr"][name] for row in paired], 20260900 + index)
        for index, name in enumerate(CLASS_NAMES)
    }
    aggregate["target_per_label"] = {
        name: {
            metric: _mean_ci([row["delta_target_per_label"][name][metric] for row in paired], 20261000 + 10 * index + offset)
            for offset, metric in enumerate(("weak_recall", "negative_fpr", "weak_vs_negative_auroc", "aupr"))
        }
        for index, name in enumerate(TARGET_NAMES)
    }
    aggregate["positive_target_weak_auroc_seeds"] = int(sum(
        row["delta_target_weak_vs_negative_auroc"] > 0 for row in paired
    ))
    return {"per_seed": paired, "aggregate": aggregate}


def _gate(rows: list[dict], comparison: dict) -> dict:
    delta = comparison["aggregate"]
    positional = [row for row in rows if row["model"] == "Q1_PositionalQuery"]
    checks = {
        "same_parameter_count_5_of_5": all(
            next(row for row in rows if row["seed"] == item["seed"] and row["model"] == "Q0_ContentQuery")["deployment_parameters"] == item["deployment_parameters"]
            for item in positional
        ),
        "target_weak_auroc_gain_0p015": delta["delta_target_weak_vs_negative_auroc"]["mean"] >= 0.015,
        "target_weak_auroc_ci_positive": delta["delta_target_weak_vs_negative_auroc"]["bootstrap_95_ci"][0] > 0.0,
        "target_weak_auroc_positive_4_of_5": delta["positive_target_weak_auroc_seeds"] >= 4,
        "macro_aupr_noninferior_minus_0p005": delta["delta_macro_aupr"]["mean"] >= -0.005,
        "each_target_aupr_noninferior_minus_0p010": all(delta["per_label_aupr"][name]["mean"] >= -0.010 for name in TARGET_NAMES),
        "target_fpr_noninferior_plus_0p005": delta["delta_target_mean_negative_fpr"]["mean"] <= 0.005,
        "target_fpr_at_most_0p20": float(np.mean([row["target"]["target_mean_negative_fpr"] for row in positional])) <= 0.20,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "advance_to_fresh_external_confirmation" if all(checks.values()) else "stop_position_query_and_do_not_tune_position_encoding",
    }


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    identity = audit_train(data_path)
    if not identity["identity_audit_passed"] or identity["test_files_opened"]:
        raise ValueError("fresh train identity audit failed")
    data_sha256 = _sha256(data_path)
    if args.smoke:
        args.epochs = int(args.smoke_epochs)
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak_all = compute_weak_component_mask(
        labels, _component_power(parameters), args.weak_threshold_db
    )
    for seed in args.seeds:
        for model_name in MODELS:
            print(f"model={model_name} seed={seed}", flush=True)
            _train(args, _run_dir(args, model_name, int(seed)), model_name, int(seed))
    rows = [_metric_row(args, model_name, int(seed), weak_all) for seed in args.seeds for model_name in MODELS]
    comparison = _comparison(rows)
    gate = {"passed": False, "decision": "smoke_only_no_method_claim"} if args.smoke else _gate(rows, comparison)
    payload = {
        "status": "completed", "experiment": "route_a_development_v15_pamiq_positional_query",
        "evidence_level": "smoke" if args.smoke else "development",
        "protocol": {"seeds": [int(seed) for seed in args.seeds], "models": MODELS, "loss": "db", "epochs": int(args.epochs), "longtail": {"sample_count": 2400, "tail_label": "SawFM", "tail_positive_count": 400, "imbalance_factor": 4}, "weak_threshold_db": float(args.weak_threshold_db), "threshold_rule": "shrinkage_per_label_f1 fitted on role-isolated threshold split", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v15_used": False},
        "data": {"path": str(data_path), "sha256": data_sha256, "identity_audit": identity},
        "per_run": rows, "comparison_q1_minus_q0": comparison, "gate": gate,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v15_used": False,
    }
    _write(Path(args.output_root) / "route_a_development_v15_pamiq_positional_query_summary.json", payload)
    _write(Path(args.output_root) / "run_receipt.json", {"status": "completed", "evidence_level": payload["evidence_level"], "data_sha256": data_sha256, "identity_audit_passed": True, "gate_passed": gate["passed"], "next_decision": gate["decision"], "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v15_used": False})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v15_pamiq_positional_query.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v15_pamiq_positional_query")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
