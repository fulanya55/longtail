"""Aggregate P2k two-data-seed frozen SawFM attachment confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable


MODEL_SEEDS = (2041, 2042, 2043)
DATA_SEEDS = (2400000000, 2500000000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sufficient_statistics(arrays: dict[str, np.ndarray], prediction_key: str) -> dict[str, np.ndarray]:
    identities = np.asarray(arrays["receiver_noise_seed"], dtype=np.int64)
    unique = np.unique(identities)
    labels = np.asarray(arrays["labels"], dtype=bool)
    mask = np.asarray(arrays["mask"], dtype=bool)
    probability = np.asarray(arrays["routed_probability"])
    thresholds = np.asarray(arrays["routed_thresholds"])
    predicted = probability >= thresholds[None, :]
    target = np.asarray(arrays["target"])
    prediction = np.asarray(arrays[prediction_key])
    # Scale cancels between paired candidates only if omitted, so recover it from
    # the stored old/candidate normalized ratios is not valid. The caller injects
    # a flattened scale before using this helper.
    scale = np.asarray(arrays["parameter_scale"], dtype=np.float64)
    normalized = np.abs(prediction - target) / scale[None, :, :]
    oracle_valid = mask & labels[:, :, None] & np.isfinite(normalized)
    gated_valid = oracle_valid & predicted[:, :, None]
    penalized = np.where(predicted[:, :, None], normalized, 1.0)
    gated_sum = np.zeros((len(unique), 12), dtype=np.float64)
    gated_count = np.zeros((len(unique), 12), dtype=np.int64)
    miss_sum = np.zeros((len(unique), 12), dtype=np.float64)
    miss_count = np.zeros((len(unique), 12), dtype=np.int64)
    for identity_index, identity in enumerate(unique):
        rows = identities == identity
        gated = gated_valid[rows].reshape(-1, 12)
        oracle = oracle_valid[rows].reshape(-1, 12)
        error = normalized[rows].reshape(-1, 12)
        miss = penalized[rows].reshape(-1, 12)
        gated_sum[identity_index] = np.where(gated, error, 0.0).sum(axis=0)
        gated_count[identity_index] = gated.sum(axis=0)
        miss_sum[identity_index] = np.where(oracle, miss, 0.0).sum(axis=0)
        miss_count[identity_index] = oracle.sum(axis=0)
    return {
        "identity": unique,
        "gated_sum": gated_sum, "gated_count": gated_count,
        "miss_sum": miss_sum, "miss_count": miss_count,
    }


def macro_from_stats(total: np.ndarray, count: np.ndarray) -> float:
    valid = count > 0
    return float(np.mean(total[valid] / count[valid]))


def stats_metric(stats: dict[str, np.ndarray], indices: np.ndarray, metric: str) -> float:
    if metric == "predicted_gate_parameter_nmae":
        return macro_from_stats(stats["gated_sum"][indices].sum(axis=0), stats["gated_count"][indices].sum(axis=0))
    if metric == "miss_penalty_parameter_nmae":
        return macro_from_stats(stats["miss_sum"][indices].sum(axis=0), stats["miss_count"][indices].sum(axis=0))
    raise KeyError(metric)


def combined_bootstrap(
    groups: list[dict[str, dict[str, np.ndarray]]], metric: str, replicates: int, seed: int,
) -> dict[str, float]:
    observed = []
    for group in groups:
        size = len(group["candidate"]["identity"])
        indices = np.arange(size, dtype=np.int64)
        observed.append(stats_metric(group["candidate"], indices, metric) - stats_metric(group["old"], indices, metric))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        deltas = []
        for group in groups:
            size = len(group["candidate"]["identity"])
            sampled = rng.integers(0, size, size=size)
            deltas.append(stats_metric(group["candidate"], sampled, metric) - stats_metric(group["old"], sampled, metric))
        samples.append(float(np.mean(deltas)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_delta_positive": float((values > 0.0).mean()),
        "replicates": int(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs=2, default=[
        "outputs/p2k_full_parameter_confirmation_20260801/data_seed_2400000000",
        "outputs/p2k_full_parameter_confirmation_20260801/data_seed_2500000000",
    ])
    parser.add_argument("--data-roots", nargs=2, default=[
        "generated_data/p2k_full_parameter_confirmation_seed2400000000_rechunked",
        "generated_data/p2k_full_parameter_confirmation_seed2500000000_rechunked",
    ])
    parser.add_argument("--p2j-summary", default="outputs/p2j_full_parameter_predicted_gate_attachment_20260801/summary.json")
    parser.add_argument("--preregister", default="configs/p2k_two_seed_full_parameter_confirmation_preregister_20260801.json")
    parser.add_argument("--output", default="outputs/p2k_full_parameter_confirmation_20260801/summary.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_roots = [project / path for path in args.run_roots]
    data_roots = [project / path for path in args.data_roots]
    p2j_summary_path = project / args.p2j_summary
    preregister = project / args.preregister
    output = project / args.output

    p2j_hash = sha256_file(p2j_summary_path)
    summaries = [json.loads((root / "summary.json").read_text(encoding="utf-8")) for root in run_roots]
    identity_sets = []
    for root in data_roots:
        identities, _ = read_identity_metadata(root / "level_01__jnr_10dB" / "closedset_6class_train.mat")
        identity_sets.append(np.asarray(identities, dtype=np.int64))
    cross_overlap = int(np.intersect1d(identity_sets[0], identity_sets[1]).size)

    groups = []
    all_rows = []
    required_hashes_match = True
    for data_seed, root, summary in zip(DATA_SEEDS, run_roots, summaries):
        required = summary.get("required_promotion") or {}
        required_hashes_match &= required.get("sha256") == p2j_hash
        for model_seed in MODEL_SEEDS:
            with np.load(root / f"seed_{model_seed}" / "development_per_sample.npz") as saved:
                arrays = {name: saved[name] for name in saved.files}
            all_rows.append(next(row for row in summary["runs"] if int(row["seed"]) == model_seed) | {"data_seed": data_seed})
            groups.append({"arrays": arrays, "data_seed": data_seed, "model_seed": model_seed})

    # Parameter scale is invariant across runs. It is stored explicitly by the
    # P2j evaluator in confirmation mode; fail closed if an older artifact lacks it.
    statistics_groups = []
    for group in groups:
        arrays = group["arrays"]
        if "parameter_scale" not in arrays:
            raise RuntimeError("confirmation prediction artifact lacks parameter_scale")
        statistics_groups.append({
            "candidate": identity_sufficient_statistics(arrays, "calibrated_joint_prediction"),
            "old": identity_sufficient_statistics(arrays, "old_joint_prediction"),
        })
    bootstrap_full = combined_bootstrap(
        statistics_groups, "predicted_gate_parameter_nmae", args.bootstrap_replicates, 20260831,
    )
    bootstrap_miss = combined_bootstrap(
        statistics_groups, "miss_penalty_parameter_nmae", args.bootstrap_replicates, 20260832,
    )

    per_data_high = {
        str(seed): float(np.mean([row["high_power_saw_reduction_vs_old"] for row in all_rows if row["data_seed"] == seed]))
        for seed in DATA_SEEDS
    }
    gates = {
        "both_data_seed_runs_pass_all_p2j_gates": bool(all(summary.get("promotion_passed", False) for summary in summaries)),
        "required_p2j_promotion_hashes_match": bool(required_hashes_match),
        "cross_data_seed_identity_overlap_zero": bool(cross_overlap == 0),
        "all_six_full_improvement_vs_safe_at_least_0p05": bool(all(row["full_improvement_vs_safe"] >= 0.05 for row in all_rows)),
        "all_six_full_improvement_vs_old_positive": bool(all(row["full_improvement_vs_old"] > 0.0 for row in all_rows)),
        "each_data_seed_mean_high_power_saw_reduction_at_least_0p30": bool(all(value >= 0.30 for value in per_data_high.values())),
        "all_six_high_power_saw_reduction_positive": bool(all(row["high_power_saw_reduction_vs_old"] > 0.0 for row in all_rows)),
        "each_low_power_saw_ratio_at_most_1p05": bool(all(value <= 1.05 for row in all_rows for value in row["low_power_saw_ratios_vs_old"])),
        "combined_bootstrap_full_delta_ci95_high_at_most_zero": bool(bootstrap_full["ci95_high"] <= 0.0),
        "combined_bootstrap_miss_delta_ci95_high_at_most_zero": bool(bootstrap_miss["ci95_high"] <= 0.0),
        "classification_and_other_outputs_unchanged": bool(all(
            summary["promotion_gates"]["classification_arrays_unchanged"]
            and summary["promotion_gates"]["other_11_parameter_outputs_unchanged"]
            for summary in summaries
        )),
    }
    passed = bool(all(gates.values()))
    result = {
        "experiment": "P2k two-new-identity-seed full-parameter confirmation",
        "preregister_sha256": sha256_file(preregister),
        "p2j_summary_sha256": p2j_hash,
        "identity_audit": {
            "data_seeds": list(DATA_SEEDS),
            "identity_counts": [int(len(values)) for values in identity_sets],
            "identity_sha256": [hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest() for values in identity_sets],
            "cross_seed_overlap": cross_overlap,
        },
        "confirmation_gates": gates,
        "confirmation_passed": passed,
        "aggregate": {
            "mean_full_improvement_vs_safe": float(np.mean([row["full_improvement_vs_safe"] for row in all_rows])),
            "minimum_full_improvement_vs_safe": float(min(row["full_improvement_vs_safe"] for row in all_rows)),
            "mean_full_improvement_vs_old": float(np.mean([row["full_improvement_vs_old"] for row in all_rows])),
            "minimum_full_improvement_vs_old": float(min(row["full_improvement_vs_old"] for row in all_rows)),
            "per_data_seed_mean_high_power_saw_reduction": per_data_high,
            "maximum_low_power_saw_ratio": float(max(value for row in all_rows for value in row["low_power_saw_ratios_vs_old"])),
            "mean_candidate_coverage": float(np.mean([row["candidate_coverage"] for row in all_rows])),
            "mean_candidate_latency_seconds": float(np.mean([row["timing"]["candidate_separator_seconds"] for row in all_rows])),
        },
        "combined_identity_bootstrap": {
            "predicted_gate_parameter_nmae": bootstrap_full,
            "miss_penalty_parameter_nmae": bootstrap_miss,
        },
        "runs": all_rows,
        "interpretation": (
            "P2k confirmed the frozen calibrated SawFM attachment on two new identity seeds; freeze it as the new full-parameter candidate."
            if passed else
            "P2k confirmation failed; retain P2j as development evidence only and do not select seeds or refit coefficients."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({
        "confirmation_gates": gates, "confirmation_passed": passed,
        "aggregate": result["aggregate"], "combined_identity_bootstrap": result["combined_identity_bootstrap"],
        "interpretation": result["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
