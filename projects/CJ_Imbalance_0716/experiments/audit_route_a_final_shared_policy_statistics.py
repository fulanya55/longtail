"""Paired statistical audit for the final shared-policy candidate on v12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v12_frozen import SHARED_VARIANT  # noqa: E402


BASELINE = "Q0_H1"
PRIMARY = ("macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc")


def _variant(row, name):
    return next(item for item in row["variants"] if item["variant"] == name)


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


def _audit(values, seed_offset, lower_is_better=False):
    values = np.asarray(values, dtype=np.float64)
    directed = -values if lower_is_better else values
    std = float(directed.std(ddof=1))
    if np.all(directed == 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        t_statistic, t_p = 0.0, 1.0
    else:
        wilcoxon_result = wilcoxon(directed, alternative="greater", method="exact")
        t_result = ttest_1samp(directed, popmean=0.0, alternative="greater")
        wilcoxon_statistic, wilcoxon_p = float(wilcoxon_result.statistic), float(wilcoxon_result.pvalue)
        t_statistic, t_p = float(t_result.statistic), float(t_result.pvalue)
    return {
        "per_seed_delta_candidate_minus_baseline": values.tolist(),
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "mean_delta": float(values.mean()), "std_delta": float(values.std(ddof=1)),
        "paired_cohen_dz_favorable_direction": float(directed.mean() / std) if std > 0 else float("inf"),
        "favorable_seed_count": int((directed > 0).sum()),
        "bootstrap_95_ci": _mean_ci(values.tolist(), seed_offset)["bootstrap_95_ci"],
        "exact_one_sided_wilcoxon_statistic": wilcoxon_statistic,
        "exact_one_sided_wilcoxon_p": wilcoxon_p,
        "one_sided_paired_t_statistic": t_statistic,
        "one_sided_paired_t_p": t_p,
    }


def _deltas(rows):
    output = {
        "macro_aupr": [], "macro_f1": [], "target_weak_recall": [], "target_weak_auroc": [],
        "target_fpr": [], "target_strong_recall": [], "mft_aupr": [], "vft_aupr": [],
        "isrj_aupr": [], "sawfm_aupr": [], "sawfm_weak_recall": [],
    }
    for row in rows:
        base = _variant(row, BASELINE)["metrics"]
        candidate = _variant(row, SHARED_VARIANT)["metrics"]
        output["macro_aupr"].append(candidate["macro_aupr"] - base["macro_aupr"])
        output["macro_f1"].append(candidate["macro_f1"] - base["macro_f1"])
        output["target_weak_recall"].append(candidate["target_mean_weak_recall"] - base["target_mean_weak_recall"])
        output["target_weak_auroc"].append(candidate["target_weak_vs_negative_auroc"] - base["target_weak_vs_negative_auroc"])
        output["target_fpr"].append(candidate["target_mean_negative_fpr"] - base["target_mean_negative_fpr"])
        output["target_strong_recall"].append(candidate["target_mean_strong_recall"] - base["target_mean_strong_recall"])
        for key, label in (("mft_aupr", "MFT"), ("vft_aupr", "VFT"), ("isrj_aupr", "ISRJ"), ("sawfm_aupr", "SawFM")):
            output[key].append(candidate["per_label"][label]["aupr"] - base["per_label"][label]["aupr"])
        output["sawfm_weak_recall"].append(candidate["per_label"]["SawFM"]["weak_recall"] - base["per_label"]["SawFM"]["weak_recall"])
    return output


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if not payload["confirmation_passed"] or payload["development_v12_sha256"] != args.expected_v12_sha256:
        raise ValueError("development-v12 confirmation contract failed")
    values = _deltas(payload["rows"])
    audits = {
        name: _audit(items, 20428000 + index, lower_is_better=(name == "target_fpr"))
        for index, (name, items) in enumerate(values.items())
    }
    t_adjusted = _holm({name: audits[name]["one_sided_paired_t_p"] for name in PRIMARY})
    w_adjusted = _holm({name: audits[name]["exact_one_sided_wilcoxon_p"] for name in PRIMARY})
    for name in PRIMARY:
        audits[name]["holm_adjusted_one_sided_paired_t_p"] = float(t_adjusted[name])
        audits[name]["holm_adjusted_exact_wilcoxon_p"] = float(w_adjusted[name])
    checks = {
        "primary_bootstrap_ci_positive": all(audits[name]["bootstrap_95_ci"][0] > 0.0 for name in PRIMARY),
        "primary_all_five_seeds_favorable": all(audits[name]["favorable_seed_count"] == 5 for name in PRIMARY),
        "primary_holm_paired_t_lt_0p05": all(audits[name]["holm_adjusted_one_sided_paired_t_p"] < 0.05 for name in PRIMARY),
        "macro_aupr_effect_ge_0p005": audits["macro_aupr"]["mean_delta"] >= 0.005,
        "macro_f1_effect_ge_0p010": audits["macro_f1"]["mean_delta"] >= 0.010,
        "target_weak_recall_effect_ge_0p030": audits["target_weak_recall"]["mean_delta"] >= 0.030,
        "target_weak_auroc_effect_ge_0p015": audits["target_weak_auroc"]["mean_delta"] >= 0.015,
        "sawfm_metrics_exact": max(abs(value) for value in values["sawfm_aupr"] + values["sawfm_weak_recall"]) <= 1e-12,
    }
    result = {
        "status": "completed", "experiment": "RouteA_FinalSharedPolicy_StatisticalAudit",
        "source": str(args.summary), "source_sha256": _sha256(args.summary), "seed_count": len(payload["rows"]),
        "primary_metrics": list(PRIMARY), "metrics": audits, "checks": checks, "passed": all(checks.values()),
        "interpretation": "Use paired effect sizes and bootstrap intervals as primary evidence. Exact Wilcoxon has limited resolution at five seeds; paired t results are reported with Holm correction and are not used alone.",
        "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "checks": checks, "primary": {name: audits[name] for name in PRIMARY}, "output": str(args.output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen/route_a_h1_b2_mft_vft_development_v12_frozen_summary.json")
    parser.add_argument("--expected-v12-sha256", default="358e08e6eecbf2c866a137ebeed847475ce1ea2efe2c399b6789a0a8a5bbffe9")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_evidence/statistical_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
