"""Re-score frozen v12 predictions under weak-component thresholds 5/10/15 dB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402


TARGETS = ("MFT", "VFT", "ISRJ")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(candidate, baseline):
    values = {
        "macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "target_weak_recall": candidate["target_mean_weak_recall"] - baseline["target_mean_weak_recall"],
        "target_weak_auroc": candidate["target_weak_vs_negative_auroc"] - baseline["target_weak_vs_negative_auroc"],
        "target_fpr": candidate["target_mean_negative_fpr"] - baseline["target_mean_negative_fpr"],
    }
    for label in TARGETS:
        values[f"{label.lower()}_weak_recall"] = candidate["per_label"][label]["weak_recall"] - baseline["per_label"][label]["weak_recall"]
        values[f"{label.lower()}_weak_auroc"] = candidate["per_label"][label]["weak_vs_negative_auroc"] - baseline["per_label"][label]["weak_vs_negative_auroc"]
    return values


def run(args):
    seal = _read_json(args.test_v3_manifest)["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    audit = _read_json(args.v12_audit)
    if not audit["identity_audit_passed"] or audit["test_files_opened"]:
        raise ValueError("v12 identity audit failed")
    data_path = Path(audit["development_v12_path"])
    if _sha256(data_path).lower() != audit["sha256"].lower():
        raise ValueError("development-v12 hash changed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    if len(labels) != audit["sample_count"]:
        raise ValueError("unexpected v12 sample count")
    summaries = {}
    for threshold_db in args.weak_thresholds:
        weak = compute_weak_component_mask(labels, _component_power(parameters), float(threshold_db))
        rows = []
        for seed in args.seeds:
            root = args.prediction_root / f"seed_{seed}"
            q0 = np.load(root / "Q0_H1" / "predictions.npz")
            q6 = np.load(root / "Q6_SharedMFTDecision_VFTAligned" / "predictions.npz")
            targets = np.asarray(q6["targets"], dtype=np.uint8)
            if not np.array_equal(targets, labels):
                raise ValueError(f"seed {seed} labels do not match audited v12")
            baseline = _metrics(targets, weak, np.asarray(q0["probabilities"], dtype=np.float32), np.asarray(q0["predictions"], dtype=bool))
            candidate = _metrics(targets, weak, np.asarray(q6["probabilities"], dtype=np.float32), np.asarray(q6["predictions"], dtype=bool))
            rows.append({"seed": int(seed), "weak_counts": {label: int(weak[:, index].sum()) for index, label in zip((3, 4, 5), TARGETS)}, "delta": _delta(candidate, baseline)})
        keys = tuple(rows[0]["delta"])
        aggregate = {key: _mean_ci([row["delta"][key] for row in rows], 20451000 + int(threshold_db) * 100 + index) for index, key in enumerate(keys)}
        summaries[str(int(threshold_db))] = {"threshold_db": int(threshold_db), "rows": rows, "aggregate": aggregate}
    checks = {"all_five_seeds": len(args.seeds) == 5, "thresholds_exact": list(args.weak_thresholds) == [5, 10, 15], "test_v3_unused": True}
    result = {"status": "completed", "experiment": "RouteA_FinalCandidate_WeakThresholdSensitivity", "evidence_role": "development_v12_posthoc_frozen_sensitivity", "v12_sha256": audit["sha256"], "summaries": summaries, "checks": checks, "passed": bool(all(checks.values())), "locked_test_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_final_candidate_weak_threshold_sensitivity_summary.json", result)
    print(json.dumps({"output": str(args.output_root), "checks": checks, "test_v3_used": False}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
    parser.add_argument("--v12-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_candidate_weak_threshold_sensitivity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--weak-thresholds", type=int, nargs="+", default=[5, 10, 15])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
