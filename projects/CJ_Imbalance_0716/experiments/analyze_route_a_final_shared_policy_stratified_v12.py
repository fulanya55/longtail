"""Stratified v12 robustness audit for the frozen final shared-policy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch  # Load Windows DLLs before scipy/sklearn.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.analyze_route_a_fullalpha_periodicity_v7_stratified import (  # noqa: E402
    METRICS, TARGETS, _groups, _subset_metrics,
)
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v12_frozen import SHARED_VARIANT  # noqa: E402


BASELINE = "Q0_H1"
SEEDS = (2039, 2040, 2041, 2042, 2043)


def _load_seed(root, seed):
    loaded = {}
    for variant in (BASELINE, SHARED_VARIANT):
        path = root / f"seed_{seed}" / variant / "predictions.npz"
        loaded[variant] = {key: np.asarray(value) for key, value in np.load(path).items()}
        loaded[variant]["_path"] = path
    reference = loaded[BASELINE]
    for key in ("receiver_noise_seed", "targets", "weak_mask"):
        if not np.array_equal(reference[key], loaded[SHARED_VARIANT][key]):
            raise AssertionError(f"candidate {key} differs from baseline")
    return loaded


def _run_seed(args, labels, powers, values, parameter_mask, receiver, seed):
    loaded = _load_seed(args.prediction_root, seed)
    output = {}
    for target_name, target_index in TARGETS.items():
        groups = _groups(labels, target_index, powers, values, parameter_mask, receiver)
        output[target_name] = {}
        for group_name, group_mask in groups.items():
            output[target_name][group_name] = {}
            for variant in (BASELINE, SHARED_VARIANT):
                item = loaded[variant]
                output[target_name][group_name][variant] = _subset_metrics(
                    labels[:, target_index], item["weak_mask"][:, target_index],
                    item["probabilities"][:, target_index], item["predictions"][:, target_index],
                    group_mask, args.min_positive, args.min_negative, args.min_weak_positive,
                )
    saw = 1
    preservation = {
        "probability_max_abs": float(np.max(np.abs(loaded[SHARED_VARIANT]["probabilities"][:, saw] - loaded[BASELINE]["probabilities"][:, saw]))),
        "prediction_mismatch": int(np.sum(loaded[SHARED_VARIANT]["predictions"][:, saw] != loaded[BASELINE]["predictions"][:, saw])),
    }
    return {
        "seed": seed, "metrics": output, "sawfm_preservation": preservation,
        "input_sha256": {name: _sha256(item["_path"]) for name, item in loaded.items()},
        "external_refit": False, "test_v3_used": False,
    }


def _aggregate(rows):
    aggregate = {}
    for target in TARGETS:
        aggregate[target] = {}
        group_names = rows[0]["metrics"][target].keys()
        for group in group_names:
            aggregate[target][group] = {}
            for metric in METRICS:
                values = []
                for row in rows:
                    base = row["metrics"][target][group][BASELINE]
                    candidate = row["metrics"][target][group][SHARED_VARIANT]
                    if base[metric] is not None and candidate[metric] is not None:
                        values.append(float(candidate[metric] - base[metric]))
                if values:
                    aggregate[target][group][metric] = _mean_ci(values, 20429000 + len(aggregate[target]) * 17 + len(metric))
            aggregate[target][group]["candidate"] = rows[0]["metrics"][target][group][SHARED_VARIANT]
    return aggregate


def _diagnostics(rows, aggregate):
    supported_aupr = []
    weak_positive = {"VFT": [], "ISRJ": []}
    mft_fpr = []
    receiver_weak = {index: [] for index in range(5)}
    worst_isrj = {"fpr": -1.0, "group": None}
    for target, groups in aggregate.items():
        for group, metrics in groups.items():
            if group != "overall/all" and "aupr" in metrics:
                supported_aupr.append((target, group, metrics["aupr"]["mean"]))
            if target in weak_positive and group != "overall/all" and "weak_recall" in metrics:
                weak_positive[target].append(metrics["weak_recall"]["mean"])
            if target == "MFT" and group != "overall/all" and "negative_fpr" in metrics:
                mft_fpr.append(metrics["negative_fpr"]["mean"])
            if group.startswith("receiver_cohort/") and "weak_recall" in metrics:
                receiver_weak[int(group.rsplit("/", 1)[1])].append(metrics["weak_recall"]["mean"])
            if target == "ISRJ":
                candidate = metrics["candidate"]
                if candidate["supported"] and candidate["negative_fpr"] is not None and candidate["negative_fpr"] > worst_isrj["fpr"]:
                    worst_isrj = {"fpr": float(candidate["negative_fpr"]), "group": group}
    severe = [(target, group, value) for target, group, value in supported_aupr if value < -0.01]
    weak_fraction = {
        target: float(np.mean(np.asarray(values) > 0.0)) if values else 0.0
        for target, values in weak_positive.items()
    }
    receiver_mean = {str(key): float(np.mean(values)) if values else None for key, values in receiver_weak.items()}
    diagnostics = {
        "supported_aupr_strata": len(supported_aupr), "severe_aupr_regressions": severe,
        "severe_aupr_regression_fraction": len(severe) / max(len(supported_aupr), 1),
        "positive_weak_recall_fraction": weak_fraction,
        "mft_nonincreased_fpr_fraction": float(np.mean(np.asarray(mft_fpr) <= 0.0)) if mft_fpr else 0.0,
        "receiver_cohort_mean_target_weak_recall_delta": receiver_mean,
        "worst_supported_isrj_fpr": worst_isrj,
        "sawfm_probability_max_abs": max(row["sawfm_preservation"]["probability_max_abs"] for row in rows),
        "sawfm_prediction_mismatch": sum(row["sawfm_preservation"]["prediction_mismatch"] for row in rows),
        "unseen_label_combination": "not_available_all_63_nonempty_six_label_combinations_are_present",
    }
    checks = {
        "severe_aupr_regression_fraction_le_0p20": diagnostics["severe_aupr_regression_fraction"] <= 0.20,
        "vft_positive_weak_strata_ge_0p70": weak_fraction["VFT"] >= 0.70,
        "isrj_positive_weak_strata_ge_0p70": weak_fraction["ISRJ"] >= 0.70,
        "mft_nonincreased_fpr_strata_ge_0p70": diagnostics["mft_nonincreased_fpr_fraction"] >= 0.70,
        "all_receiver_cohorts_positive_target_weak_recall": all(value is not None and value > 0.0 for value in receiver_mean.values()),
        "worst_isrj_fpr_le_0p35": worst_isrj["fpr"] <= 0.35,
        "sawfm_probability_exact": diagnostics["sawfm_probability_max_abs"] == 0.0,
        "sawfm_prediction_exact": diagnostics["sawfm_prediction_mismatch"] == 0,
    }
    return diagnostics, {"checks": checks, "passed": all(checks.values())}


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    audit = json.loads(args.identity_audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["sha256"] != args.expected_v12_sha256:
        raise ValueError("development-v12 identity audit failed")
    data_path = Path(audit["development_v12_path"])
    labels = read_labels(data_path)
    values, parameter_mask = read_parameters(data_path)
    powers = _component_power(values)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    rows = [_run_seed(args, labels, powers, values, parameter_mask, receiver, seed) for seed in args.seeds]
    aggregate = _aggregate(rows)
    diagnostics, gate = _diagnostics(rows, aggregate)
    result = {
        "status": "completed", "experiment": "RouteA_FinalSharedPolicy_StratifiedV12",
        "evidence_role": "fresh_identity_frozen_stratified_audit", "development_v12_sha256": audit["sha256"],
        "rows": rows, "aggregate": aggregate, "diagnostics": diagnostics, "gate": gate,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"gate": gate, "diagnostics": diagnostics, "output": str(args.output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json")
    parser.add_argument("--prediction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--expected-v12-sha256", default="358e08e6eecbf2c866a137ebeed847475ce1ea2efe2c399b6789a0a8a5bbffe9")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--min-positive", type=int, default=100)
    parser.add_argument("--min-negative", type=int, default=100)
    parser.add_argument("--min-weak-positive", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_evidence/stratified_v12.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

