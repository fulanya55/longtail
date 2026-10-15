"""Train new SawFM DB endpoints and confirm fixed checkpoint-fusion weights."""

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
CELLS = {
    "BaseDB": "macro_aupr",
    "DB-Tail": "tail_aupr",
}
METRICS = ("macro_aupr", "tail_aupr", "macro_f1", "hamming_loss", "tail_f1")


def _endpoint_root(args):
    return Path(args.output_root) / "endpoints"


def _run_dir(args, cell, seed):
    return (
        _endpoint_root(args)
        / "runs" / TAIL_NAME / cell / "seed_{}".format(seed)
    )


def _command(args, directory, cell, seed):
    selection = CELLS[cell]
    command = [
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
        "--learning-rate", "0.0003",
        "--max-train-samples", "0",
        "--max-eval-samples", str(args.max_eval_samples),
        "--label-imbalance-factor", "4",
        "--label-count-tolerance", "0.05",
        "--label-tail-count-anchor", "400",
        "--label-target-samples", "2400",
        "--label-sample-count-tolerance", "0",
        "--label-rank-order", "0", "2", "3", "4", "5", "1",
        "--selection-metric", selection,
        "--output-dir", str(directory),
        "--amp",
    ]
    if selection == "tail_aupr":
        command.extend(["--selection-tail-index", str(TAIL_INDEX)])
    return command


def _train(args, directory, cell, seed):
    receipt_path = directory / "run_receipt.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") == "completed":
            print("reuse cell={} seed={}".format(cell, seed))
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
            "Endpoint training failed cell={} seed={}: {}".format(
                cell, seed, result.stderr[-2500:]
            )
        )


def _history_drift(left_path, right_path):
    with Path(left_path).open("r", encoding="utf-8-sig", newline="") as handle:
        left = list(csv.DictReader(handle))
    with Path(right_path).open("r", encoding="utf-8-sig", newline="") as handle:
        right = list(csv.DictReader(handle))
    if len(left) != len(right):
        raise ValueError("Endpoint histories have different lengths")
    result = {}
    for key in ("train_loss", "macro_f1"):
        result["max_{}_difference".format(key)] = max(
            abs(float(a[key]) - float(b[key])) for a, b in zip(left, right)
        )
    return result


def _audit_endpoints(args):
    rows = []
    for seed in args.seeds:
        macro_dir = _run_dir(args, "BaseDB", seed)
        tail_dir = _run_dir(args, "DB-Tail", seed)
        macro_receipt = _read(macro_dir / "run_receipt.json")
        tail_receipt = _read(tail_dir / "run_receipt.json")
        macro_manifest = _read(macro_dir / "longtail_manifest.json")
        tail_manifest = _read(tail_dir / "longtail_manifest.json")
        if macro_receipt["split_sha256"] != tail_receipt["split_sha256"]:
            raise ValueError("Endpoint split mismatch")
        if macro_receipt["manifest_sha256"] != tail_receipt["manifest_sha256"]:
            raise ValueError("Endpoint manifest mismatch")
        if macro_receipt["locked_test_used"] or tail_receipt["locked_test_used"]:
            raise ValueError("Locked test was used")
        if macro_manifest["achieved_sample_count"] != 2400:
            raise ValueError("Macro endpoint manifest drifted")
        if tail_manifest["achieved_sample_count"] != 2400:
            raise ValueError("Tail endpoint manifest drifted")
        drift = _history_drift(
            macro_dir / "history.csv", tail_dir / "history.csv"
        )
        if any(value != 0.0 for value in drift.values()):
            raise ValueError("Endpoint training trajectories differ")
        rows.append(
            {
                "seed": int(seed),
                "macro_best_epoch": int(macro_receipt["best_epoch"]),
                "tail_best_epoch": int(tail_receipt["best_epoch"]),
                "split_sha256": macro_receipt["split_sha256"],
                "manifest_sha256": macro_receipt["manifest_sha256"],
                **drift,
            }
        )
    return rows


def _run_fixed_fusion(args):
    fusion_root = Path(args.output_root) / "fusion"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_route_a_sawfm_checkpoint_fusion.py"),
        "--config", str(args.config),
        "--db-macro-root", str(_endpoint_root(args)),
        "--db-tail-root", str(_endpoint_root(args)),
        "--output-root", str(fusion_root),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--alphas", "0", "0.2", "0.6", "1",
        "--fixed-weight-alpha", "0.2",
        "--fixed-logit-alpha", "0.6",
        "--max-eval-samples", str(args.max_eval_samples),
    ]
    result = subprocess.run(
        command, cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    (fusion_root / "subprocess_output.txt").parent.mkdir(parents=True, exist_ok=True)
    (fusion_root / "subprocess_output.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError("Fixed fusion failed: {}".format(result.stderr[-3000:]))
    return _read(fusion_root / "route_a_sawfm_checkpoint_fusion_summary.json")


def _aggregate_rows(rows, seed_offset):
    result = {}
    for index, metric in enumerate(METRICS):
        key = "delta_{}".format(metric)
        result[key] = _mean_ci(
            [row["delta"][key] for row in rows], seed_offset + index
        )
    result["macro_noninferior_seeds"] = sum(
        row["delta"]["delta_macro_aupr"] >= -0.005 for row in rows
    )
    result["tail_positive_seeds"] = sum(
        row["delta"]["delta_tail_aupr"] > 0 for row in rows
    )
    return result


def run(args):
    total = len(CELLS) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        for cell in CELLS:
            current += 1
            print("[{}/{}] cell={} seed={}".format(current, total, cell, seed))
            _train(args, _run_dir(args, cell, seed), cell, seed)
    endpoint_audit = _audit_endpoints(args)
    confirmation = _run_fixed_fusion(args)
    if confirmation["protocol"]["alpha_selection_role"] != "fixed_from_prior_discovery":
        raise ValueError("Fusion alpha was not fixed")
    discovery = _read(args.discovery_summary)
    combined = {}
    confirmation_gate = {}
    for index, method in enumerate(("weight", "logit")):
        confirm_rows = confirmation["validation"][method]["per_seed"]
        discovery_rows = discovery["validation"][method]["per_seed"]
        combined_rows = discovery_rows + confirm_rows
        confirm_aggregate = _aggregate_rows(confirm_rows, 24000 + index * 100)
        combined_aggregate = _aggregate_rows(combined_rows, 25000 + index * 100)
        combined[method] = {
            "discovery_seeds": [row["seed"] for row in discovery_rows],
            "confirmation_seeds": [row["seed"] for row in confirm_rows],
            "confirmation": {
                "per_seed": confirm_rows,
                "aggregate": confirm_aggregate,
            },
            "five_seed_descriptive": {
                "per_seed": combined_rows,
                "aggregate": combined_aggregate,
            },
        }
        confirmation_gate[method] = (
            confirm_aggregate["delta_macro_aupr"]["mean"] >= -0.005
            and confirm_aggregate["delta_tail_aupr"]["mean"] >= 0.005
            and confirm_aggregate["macro_noninferior_seeds"] == len(confirm_rows)
            and confirm_aggregate["tail_positive_seeds"] == len(confirm_rows)
        )
    gate = {
        "weight_primary_passed": confirmation_gate["weight"],
        "logit_upper_bound_passed": confirmation_gate["logit"],
    }
    gate["passed"] = gate["weight_primary_passed"]
    gate["decision"] = (
        "advance_fixed_weight_fusion_to_cross_tail_validation"
        if gate["passed"]
        else "stop_static_weight_fusion"
    )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "confirmation_seeds": list(args.seeds),
            "weight_alpha": 0.2,
            "logit_alpha": 0.6,
            "alpha_retuned": False,
            "epochs": int(args.epochs),
            "max_eval_samples": int(args.max_eval_samples),
        },
        "gate": gate,
        "endpoint_audit": endpoint_audit,
        "confirmation_fusion_summary": str(
            (Path(args.output_root) / "fusion" / "route_a_sawfm_checkpoint_fusion_summary.json").resolve()
        ),
        "methods": combined,
    }
    output = Path(args.output_root) / "route_a_sawfm_weight_fusion_confirmatory_summary.json"
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
        "--discovery-summary", type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_checkpoint_fusion"
            / "route_a_sawfm_checkpoint_fusion_summary.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "tapc"
            / "route_a_sawfm_weight_fusion_confirmatory"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
