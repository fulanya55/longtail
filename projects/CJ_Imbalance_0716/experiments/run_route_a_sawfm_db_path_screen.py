"""Compare from-scratch ASL-DB mixes and reverse DB-representation decoupling."""

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
METHODS = {
    "Mix75": {"loss": "asl_db_mix", "db_weight": 0.75, "scope": "all"},
    "Mix90": {"loss": "asl_db_mix", "db_weight": 0.90, "scope": "all"},
    "DBRep-ASLHead": {"loss": "asl", "db_weight": None, "scope": "classifier"},
}


def _baseline_dir(args, variant, seed):
    return (
        Path(args.baseline_root)
        / "runs" / TAIL_NAME / variant / "seed_{}".format(seed)
    )


def _run_dir(args, method, seed):
    return Path(args.output_root) / "runs" / TAIL_NAME / method / "seed_{}".format(seed)


def _command(args, directory, method, seed):
    config = METHODS[method]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_tapc.py"),
        "--config", str(args.config),
        "--variant", "T0",
        "--protocol", "label_lt",
        "--loss", config["loss"],
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--batch-size", "64",
        "--workers", "0",
        "--learning-rate", str(1e-3 if config["scope"] == "classifier" else 3e-4),
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", "0", "2", "3", "4", "5", "1",
        "--selection-metric", "tail_aupr",
        "--selection-tail-index", str(TAIL_INDEX),
        "--output-dir", str(directory),
        "--amp",
    ]
    if config["db_weight"] is not None:
        command.extend(["--db-mix-weight", str(config["db_weight"])])
    if config["scope"] == "classifier":
        checkpoint = _baseline_dir(args, "BaseDB", seed) / "best_model.pth"
        command.extend(
            [
                "--initial-checkpoint", str(checkpoint),
                "--trainable-scope", "classifier",
            ]
        )
    return command


def _train(args, directory, method, seed):
    if (directory / "run_receipt.json").exists() and (
        directory / "metrics_threshold_modes.json"
    ).exists():
        receipt = _read(directory / "run_receipt.json")
        if receipt.get("status") == "completed":
            print("reuse method={} seed={}".format(method, seed))
            return
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _command(args, directory, method, seed),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    (directory / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            "DB-path run failed method={} seed={}: {}".format(
                method, seed, result.stderr[-2500:]
            )
        )


def _collect(directory, variant, seed):
    metrics = _read(directory / "metrics_threshold_modes.json")["fixed_0p5"]
    receipt = _read(directory / "run_receipt.json")
    manifest = _read(directory / "longtail_manifest.json")
    tail = next(row for row in metrics["per_label"] if row["label"] == TAIL_NAME)
    return {
        "variant": variant,
        "seed": int(seed),
        "macro_aupr": metrics["macro_aupr"],
        "tail_aupr": tail["aupr"],
        "macro_f1": metrics["macro_f1"],
        "hamming_loss": metrics["hamming_loss"],
        "tail_f1": tail["f1"],
        "split_sha256": receipt["split_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "selection_metric": receipt["selection_metric"],
        "trainable_scope": receipt["trainable_scope"],
        "training_parameters": receipt["training_parameters"],
        "best_epoch": receipt["best_epoch"],
        "locked_test_used": bool(receipt["locked_test_used"]),
        "sample_count": manifest["achieved_sample_count"],
        "max_abs_label_error": max(abs(value) for value in manifest["label_count_error"]),
    }


def _difference(candidate, baseline):
    return {
        "delta_macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "delta_tail_aupr": candidate["tail_aupr"] - baseline["tail_aupr"],
        "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "delta_hamming_loss": candidate["hamming_loss"] - baseline["hamming_loss"],
        "delta_tail_f1": candidate["tail_f1"] - baseline["tail_f1"],
    }


def _aggregate(rows, key, seed):
    result = {}
    for index, metric in enumerate(
        ("delta_macro_aupr", "delta_tail_aupr", "delta_macro_f1", "delta_hamming_loss", "delta_tail_f1")
    ):
        result[metric] = _mean_ci([row[key][metric] for row in rows], seed + index)
    result["macro_wins"] = sum(row[key]["delta_macro_aupr"] > 0 for row in rows)
    result["tail_wins"] = sum(row[key]["delta_tail_aupr"] > 0 for row in rows)
    result["tail_noninferior"] = sum(row[key]["delta_tail_aupr"] >= -0.005 for row in rows)
    return result


def run(args):
    runs = []
    comparisons = []
    total = len(METHODS) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        base_asl = _collect(_baseline_dir(args, "BaseASL", seed), "BaseASL", seed)
        base_db = _collect(_baseline_dir(args, "BaseDB", seed), "BaseDB", seed)
        for method in METHODS:
            current += 1
            directory = _run_dir(args, method, seed)
            print("[{}/{}] method={} seed={}".format(current, total, method, seed))
            _train(args, directory, method, seed)
            candidate = _collect(directory, method, seed)
            if candidate["split_sha256"] != base_db["split_sha256"]:
                raise ValueError("Candidate split differs from Base-DB")
            if candidate["manifest_sha256"] != base_db["manifest_sha256"]:
                raise ValueError("Candidate manifest differs from Base-DB")
            if candidate["selection_metric"] != "tail_aupr":
                raise ValueError("Candidate used the wrong selection metric")
            if method == "DBRep-ASLHead":
                if candidate["trainable_scope"] != "classifier" or candidate["training_parameters"] != 390:
                    raise ValueError("Reverse decoupling did not stay classifier-only")
            else:
                if candidate["trainable_scope"] != "all" or candidate["training_parameters"] != base_db["training_parameters"]:
                    raise ValueError("Mix candidate did not use full from-scratch training")
            if candidate["locked_test_used"]:
                raise ValueError("Locked test was used")
            runs.append(candidate)
            comparisons.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "vs_db": _difference(candidate, base_db),
                    "vs_asl": _difference(candidate, base_asl),
                }
            )
    aggregate = {}
    checks = {}
    passed = []
    for index, method in enumerate(METHODS):
        selected = [row for row in comparisons if row["method"] == method]
        aggregate[method] = {
            "vs_db": _aggregate(selected, "vs_db", 16000 + index * 100),
            "vs_asl": _aggregate(selected, "vs_asl", 17000 + index * 100),
        }
        db = aggregate[method]["vs_db"]
        method_checks = {
            "macro_noninferior_vs_db": db["delta_macro_aupr"]["mean"] >= -0.005,
            "tail_noninferior_vs_db": db["delta_tail_aupr"]["mean"] >= -0.005,
            "tail_noninferior_at_least_2_of_3": db["tail_noninferior"] >= 2,
        }
        is_passed = all(method_checks.values())
        checks[method] = {"passed": is_passed, **method_checks}
        if is_passed:
            passed.append(method)
    promoted = None
    if passed:
        promoted = max(
            passed,
            key=lambda method: aggregate[method]["vs_db"]["delta_tail_aupr"]["mean"],
        )
    gate = {
        "passed": bool(passed),
        "promoted_method": promoted,
        "method_checks": checks,
        "decision": (
            "validate_candidate_beyond_sawfm"
            if promoted is not None
            else "retain_basedb_as_primary_baseline"
        ),
    }
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "tail_label": TAIL_NAME,
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "methods": METHODS,
        },
        "gate": gate,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "runs": runs,
    }
    output = Path(args.output_root) / "route_a_sawfm_db_path_screen_summary.json"
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
        "--baseline-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_db_path_screen",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
