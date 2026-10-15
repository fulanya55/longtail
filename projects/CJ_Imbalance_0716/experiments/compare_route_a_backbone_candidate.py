"""Apply the preregistered Route-A backbone promotion gates to five paired seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_LABELS = ("MFT", "VFT", "ISRJ")
SAFETY_LABELS = ("FMB", "SawFM", "SinFM")
PAIR_METRICS = (
    "macro_aupr",
    "target_mean_weak_auroc",
    "target_mean_weak_recall",
    "target_mean_strong_recall",
    "target_mean_negative_fpr",
)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paired_ci(deltas, seed, samples=200_000):
    values = np.asarray(deltas, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "positive_seed_count": int((values > 0).sum()),
        "negative_seed_count": int((values < 0).sum()),
        "zero_seed_count": int((values == 0).sum()),
        "per_seed_delta": [float(value) for value in values],
    }


def compare(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _read(baseline_path)
    candidate = _read(candidate_path)
    if baseline.get("formal_protocol") is not True or candidate.get("formal_protocol") is not True:
        raise ValueError("both summaries must be formal five-seed development evidence")
    if baseline["data_sha256"] != candidate["data_sha256"]:
        raise ValueError("candidate and baseline data SHA differ")
    baseline_rows = {int(row["seed"]): row for row in baseline["per_seed"]}
    candidate_rows = {int(row["seed"]): row for row in candidate["per_seed"]}
    seeds = sorted(baseline_rows)
    if seeds != sorted(candidate_rows) or len(seeds) != 5:
        raise ValueError("candidate and baseline must contain the same five paired seeds")

    paired = {}
    for index, metric in enumerate(PAIR_METRICS):
        paired[metric] = _paired_ci(
            [
                candidate_rows[seed]["metrics"][metric]
                - baseline_rows[seed]["metrics"][metric]
                for seed in seeds
            ],
            20808000 + index,
        )

    per_label = {}
    for label_index, label in enumerate((*SAFETY_LABELS, *TARGET_LABELS)):
        per_label[label] = {}
        for metric_index, metric in enumerate(
            ("aupr", "weak_vs_negative_auroc", "weak_recall", "strong_recall", "negative_fpr")
        ):
            per_label[label][metric] = _paired_ci(
                [
                    candidate_rows[seed]["per_label_diagnostics"][label][metric]
                    - baseline_rows[seed]["per_label_diagnostics"][label][metric]
                    for seed in seeds
                ],
                20809000 + label_index * 100 + metric_index,
            )

    candidate_aggregate = candidate["aggregate"]
    target_aupr_drops = {
        label: per_label[label]["aupr"]["mean_delta"] for label in TARGET_LABELS
    }
    safety_systematic = {
        label: bool(
            per_label[label]["aupr"]["mean_delta"] < -0.01
            or per_label[label]["aupr"]["negative_seed_count"] >= 4
        )
        for label in SAFETY_LABELS
    }
    gates = {
        "macro_aupr_gain_ge_0p005_and_ci_lower_gt_0": bool(
            paired["macro_aupr"]["mean_delta"] >= 0.005
            and paired["macro_aupr"]["paired_bootstrap_95_ci"][0] > 0
        ),
        "target_weak_auroc_gain_ge_0p020_and_ci_lower_gt_0": bool(
            paired["target_mean_weak_auroc"]["mean_delta"] >= 0.020
            and paired["target_mean_weak_auroc"]["paired_bootstrap_95_ci"][0] > 0
        ),
        "target_weak_recall_gain_ge_0p025": bool(
            paired["target_mean_weak_recall"]["mean_delta"] >= 0.025
        ),
        "at_least_4_of_5_positive_weak_auroc": bool(
            paired["target_mean_weak_auroc"]["positive_seed_count"] >= 4
        ),
        "at_least_4_of_5_positive_weak_recall": bool(
            paired["target_mean_weak_recall"]["positive_seed_count"] >= 4
        ),
        "absolute_target_mean_fpr_le_0p20": bool(
            candidate_aggregate["target_mean_negative_fpr"]["mean"] <= 0.20
        ),
        "target_strong_recall_drop_no_more_than_0p01": bool(
            paired["target_mean_strong_recall"]["mean_delta"] >= -0.01
        ),
        "no_target_label_aupr_drop_below_minus_0p01": bool(
            min(target_aupr_drops.values()) >= -0.01
        ),
        "no_systematic_safety_label_aupr_regression": bool(
            not any(safety_systematic.values())
        ),
        "cost_receipts_complete": bool(
            all(
                row.get("parameter_count") is not None
                and row.get("conv_linear_macs_per_sample") is not None
                and row.get("cuda_batch1_latency_ms") is not None
                and row.get("peak_cuda_memory_bytes") is not None
                for row in candidate["per_seed"]
            )
        ),
        "evidence_boundary_intact": bool(
            candidate.get("test_v2_used") is False
            and candidate.get("test_v3_used") is False
            and candidate.get("locked_test_used") is False
            and candidate.get("test_files_opened") is False
        ),
    }
    gates["all_stage2_promotion_gates_passed"] = bool(all(gates.values()))
    return {
        "status": "completed",
        "decision": "pass_stage2" if gates["all_stage2_promotion_gates_passed"] else "fail_stage2_close_candidate",
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "data_sha256": candidate["data_sha256"],
        "seeds": seeds,
        "baseline_summary_sha256": _sha256(baseline_path),
        "candidate_summary_sha256": _sha256(candidate_path),
        "paired_metrics": paired,
        "per_label_deltas": per_label,
        "target_label_aupr_mean_deltas": target_aupr_drops,
        "safety_label_systematic_regression_flags": safety_systematic,
        "gate_definition_note": (
            "Safety-label systematic regression is flagged when mean AUPR delta is below -0.01 "
            "or at least four of five paired seeds have negative AUPR delta."
        ),
        "gates": gates,
        "test_v3_used": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = compare(args.baseline, args.candidate)
    _write(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
