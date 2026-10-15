"""Paired five-seed statistical audit for the final frozen H1+B2 candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _sha256, _write_json

METRICS = ("isrj_aupr", "isrj_weak_recall", "isrj_weak_auroc", "macro_aupr", "macro_f1", "isrj_strong_recall", "fpr")
PRIMARY = ("isrj_aupr", "isrj_weak_recall", "isrj_weak_auroc")


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
    x = np.asarray(values, dtype=np.float64)
    directed = -x if lower_is_better else x
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    statistic, pvalue = wilcoxon(directed, alternative="greater", method="exact")
    return {
        "per_seed_delta": x.tolist(),
        "mean_delta": float(x.mean()),
        "std_delta": std,
        "paired_cohen_dz": float(((-x.mean()) if lower_is_better else x.mean()) / std) if std > 0 else float("inf"),
        "favorable_seed_count": int((directed > 0).sum()),
        "exact_one_sided_wilcoxon_statistic": float(statistic),
        "exact_one_sided_wilcoxon_p": float(pvalue),
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def _dataset_rows(payload, source):
    if source == "development_v3":
        rows = payload["comparisons"]["B2_vs_B0"]["per_seed"]
        return {metric: [row["isrj_negative_fpr" if metric == "fpr" else metric] for row in rows] for metric in METRICS}
    rows = payload["b2_vs_b0"]["per_seed"]
    return {metric: [row["isrj_fpr" if metric == "fpr" else metric] for row in rows] for metric in METRICS}


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    inputs = {"development_v3": args.development_summary, "parameter_ood_v1": args.ood_summary}
    datasets = {}
    for name, path in inputs.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = _dataset_rows(payload, name)
        audits = {metric: _audit(values[metric], lower_is_better=(metric == "fpr")) for metric in METRICS}
        adjusted = _holm({metric: audits[metric]["exact_one_sided_wilcoxon_p"] for metric in PRIMARY})
        for metric, value in adjusted.items():
            audits[metric]["holm_adjusted_p_within_primary_family"] = float(value)
        datasets[name] = {"source": str(path), "source_sha256": _sha256(path), "metrics": audits}
    result = {"status": "completed", "experiment": "RouteA_Final_H1B2_StatisticalAudit", "datasets": datasets, "primary_metrics": list(PRIMARY), "seed_count": 5, "locked_test_used": False, "test_v3_used": False, "interpretation_rule": "effect size + CI + direction consistency; do not rely on p alone at n=5"}
    args.output_root.mkdir(parents=True, exist_ok=True); out=args.output_root/"route_a_final_h1_b2_statistical_audit.json"; _write_json(out,result); print(json.dumps({"status":"completed","datasets":datasets,"summary":str(out)},ensure_ascii=False,indent=2)); return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--development-summary",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_h1_periodicity_increment/route_a_development_v3_h1_periodicity_increment_summary.json"); p.add_argument("--ood-summary",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_parameter_ood_v1_frozen/route_a_isrj_parameter_ood_v1_frozen_summary.json"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_final_h1_b2_statistical_audit"); run(p.parse_args())

