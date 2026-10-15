"""Paired statistical audit of U2 versus U0 on development-v3 and development-v7."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # Load Windows DLLs before scipy.
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS

PRIMARY = ("macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc")
LABELS = ("MFT", "VFT", "ISRJ")


def _holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _audit(values, lower_is_better=False):
    values = np.asarray(values, dtype=np.float64)
    directed = -values if lower_is_better else values
    std = float(values.std(ddof=1))
    statistic, pvalue = wilcoxon(directed, alternative="greater", method="exact")
    return {
        "per_seed_delta": values.tolist(), "mean_delta": float(values.mean()),
        "std_delta": std, "paired_cohen_dz": float(directed.mean() / std) if std > 0 else float("inf"),
        "favorable_seed_count": int((directed > 0).sum()),
        "exact_one_sided_wilcoxon_statistic": float(statistic),
        "exact_one_sided_wilcoxon_p": float(pvalue),
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _values(payload):
    output = {name: [] for name in PRIMARY}
    for label in LABELS:
        for metric in ("aupr", "weak_recall", "weak_vs_negative_auroc", "strong_recall", "negative_fpr"):
            output[f"{label}__{metric}"] = []
    for row in payload["rows"]:
        variants = _variant_map(row)
        base = variants[VARIANTS[0]]["metrics"]
        candidate = variants[VARIANTS[2]]["metrics"]
        output["macro_aupr"].append(candidate["macro_aupr"] - base["macro_aupr"])
        output["macro_f1"].append(candidate["macro_f1"] - base["macro_f1"])
        output["target_weak_recall"].append(candidate["target_mean_weak_recall"] - base["target_mean_weak_recall"])
        output["target_weak_auroc"].append(candidate["target_weak_vs_negative_auroc"] - base["target_weak_vs_negative_auroc"])
        for label in LABELS:
            for metric in ("aupr", "weak_recall", "weak_vs_negative_auroc", "strong_recall", "negative_fpr"):
                output[f"{label}__{metric}"].append(candidate["per_label"][label][metric] - base["per_label"][label][metric])
    return output


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    datasets = {}
    for dataset_name, path in (("development_v3", args.development_v3), ("development_v7", args.development_v7)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["test_v3_used"] or payload.get("locked_test_used", False):
            raise ValueError(f"{dataset_name} boundary failed")
        values = _values(payload)
        audits = {
            name: _audit(items, lower_is_better=name.endswith("negative_fpr"))
            for name, items in values.items()
        }
        adjusted = _holm({name: audits[name]["exact_one_sided_wilcoxon_p"] for name in PRIMARY})
        for name, value in adjusted.items():
            audits[name]["holm_adjusted_p_within_primary_family"] = float(value)
        datasets[dataset_name] = {
            "source": str(path), "source_sha256": _sha256(path), "metrics": audits,
            "source_bootstrap_ci": payload["comparisons"]["U2_vs_U0"]["aggregate"],
        }
    replicated = {
        metric: {
            "positive_mean_on_both_datasets": all(datasets[name]["metrics"][metric]["mean_delta"] > 0 for name in datasets),
            "five_of_five_on_both_datasets": all(datasets[name]["metrics"][metric]["favorable_seed_count"] == 5 for name in datasets),
        }
        for metric in PRIMARY
    }
    result = {
        "status": "completed", "experiment": "RouteA_U2_StatisticalAudit",
        "datasets": datasets, "primary_metrics": list(PRIMARY), "replication": replicated,
        "seed_count_per_dataset": 5,
        "interpretation_rule": "Report exact p, Holm p, paired effect size, direction consistency, and source bootstrap CI; n=5 limits familywise significance.",
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_u2_statistical_audit.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "replication": replicated, "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-v3", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3/route_a_fullalpha_periodicity_unified_v3_summary.json")
    parser.add_argument("--development-v7", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_development_v7_frozen/route_a_fullalpha_periodicity_development_v7_frozen_summary.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_statistical_audit")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
