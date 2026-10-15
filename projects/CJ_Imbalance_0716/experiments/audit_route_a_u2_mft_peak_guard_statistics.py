"""Statistical replication audit for the U2 MFT peak-count guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _write_json  # noqa: E402


BENEFIT_METRICS = ("macro_aupr", "target_weak_auroc", "hard_fpr_reduction", "hard_auroc")


def _holm(values):
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, float(value) * (total - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _audit(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    statistic, pvalue = wilcoxon(values, alternative="greater", zero_method="wilcox")
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": std,
        "cohen_dz": float(values.mean() / std) if std > 0 else float("inf"),
        "positive_count": int((values > 0).sum()),
        "exact_one_sided_wilcoxon_statistic": float(statistic),
        "exact_one_sided_wilcoxon_p": float(pvalue),
    }


def _extract_v3(payload):
    rows = payload["construction_guard"]["per_seed"]
    return {
        "macro_aupr": [row["macro_aupr"] for row in rows],
        "target_weak_auroc": [row["target_weak_auroc"] for row in rows],
        "hard_fpr_reduction": [-row["hard_fpr"] for row in rows],
        "hard_auroc": [row["hard_auroc"] for row in rows],
        "target_weak_recall": [row["target_weak_recall"] for row in rows],
    }


def _extract_v10(payload):
    rows = payload["comparison"]["per_seed"]
    return {
        "macro_aupr": [row["macro_aupr"] for row in rows],
        "target_weak_auroc": [row["target_weak_auroc"] for row in rows],
        "hard_fpr_reduction": [-row["hard_fpr"] for row in rows],
        "hard_auroc": [row["hard_auroc"] for row in rows],
        "target_weak_recall": [row["target_weak_recall"] for row in rows],
    }


def run(args):
    v3 = _extract_v3(json.loads(args.v3_summary.read_text(encoding="utf-8")))
    v10 = _extract_v10(json.loads(args.v10_summary.read_text(encoding="utf-8")))
    datasets = {}
    for name, values in (("development_v3", v3), ("development_v10", v10)):
        audits = {metric: _audit(values[metric]) for metric in BENEFIT_METRICS}
        adjusted = _holm({metric: audits[metric]["exact_one_sided_wilcoxon_p"] for metric in BENEFIT_METRICS})
        for metric in BENEFIT_METRICS:
            audits[metric]["holm_adjusted_p_within_four_primary"] = float(adjusted[metric])
        recall = np.asarray(values["target_weak_recall"], dtype=np.float64)
        datasets[name] = {
            "benefit_metrics": audits,
            "weak_recall_cost": {
                "values": recall.tolist(),
                "mean": float(recall.mean()),
                "worst_seed": float(recall.min()),
                "all_above_preregistered_minus0p010": bool(np.all(recall >= -0.010)),
            },
        }
    replication = {
        metric: {
            "positive_in_both_dataset_means": bool(
                datasets["development_v3"]["benefit_metrics"][metric]["mean"] > 0
                and datasets["development_v10"]["benefit_metrics"][metric]["mean"] > 0
            ),
            "ten_of_ten_direction_count": int(
                datasets["development_v3"]["benefit_metrics"][metric]["positive_count"]
                + datasets["development_v10"]["benefit_metrics"][metric]["positive_count"]
            ),
            "v10_minus_v3_mean": float(
                datasets["development_v10"]["benefit_metrics"][metric]["mean"]
                - datasets["development_v3"]["benefit_metrics"][metric]["mean"]
            ),
        }
        for metric in BENEFIT_METRICS
    }
    result = {
        "status": "completed", "experiment": "RouteA_U2_MFT_PeakGuard_StatisticalAudit",
        "datasets": datasets, "replication": replication,
        "interpretation": "Large and direction-consistent effects are primary at n=5; exact p=0.03125 per all-positive metric does not survive four-metric Holm correction (0.125).",
        "independence_note": "The two datasets are independent receiver identities but reuse the same five fitted model seeds; do not pool them as ten independent models.",
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"status": "completed", "replication": replication, "output": str(args.output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3/route_a_u2_mft_peak_count_negative_guard_v3_summary.json")
    parser.add_argument("--v10-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_development_v10_frozen/route_a_u2_mft_peak_guard_development_v10_frozen_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_evidence_bundle/statistical_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

