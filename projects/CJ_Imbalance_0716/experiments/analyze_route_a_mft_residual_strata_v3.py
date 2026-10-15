"""Read-only stratified diagnosis of the frozen MFT residual on early roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MFT_INDEX = 3
CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean_ci(values, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    if len(array) == 1:
        interval = [float(array[0]), float(array[0])]
    else:
        indices = rng.integers(0, len(array), size=(10000, len(array)))
        means = array[indices].mean(axis=1)
        interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "count": int(len(array)), "bootstrap_95_ci": interval}


def _group_metrics(mask, weak_positive, negative, base_probability, candidate_probability,
                   base_prediction, candidate_prediction):
    positive_mask = mask & weak_positive
    negative_mask = mask & negative
    if not positive_mask.any() or not negative_mask.any():
        auroc_base = auroc_candidate = float("nan")
    else:
        labels = np.concatenate([np.ones(positive_mask.sum()), np.zeros(negative_mask.sum())])
        base_score = np.concatenate([base_probability[positive_mask], base_probability[negative_mask]])
        candidate_score = np.concatenate([candidate_probability[positive_mask], candidate_probability[negative_mask]])
        auroc_base = float(roc_auc_score(labels, base_score))
        auroc_candidate = float(roc_auc_score(labels, candidate_score))
    base_recall = float(base_prediction[positive_mask].mean()) if positive_mask.any() else float("nan")
    candidate_recall = float(candidate_prediction[positive_mask].mean()) if positive_mask.any() else float("nan")
    base_fpr = float(base_prediction[negative_mask].mean()) if negative_mask.any() else float("nan")
    candidate_fpr = float(candidate_prediction[negative_mask].mean()) if negative_mask.any() else float("nan")
    return {
        "weak_positive_count": int(positive_mask.sum()), "negative_count": int(negative_mask.sum()),
        "base_weak_recall": base_recall, "candidate_weak_recall": candidate_recall,
        "weak_recall_delta": candidate_recall - base_recall,
        "base_fpr": base_fpr, "candidate_fpr": candidate_fpr, "fpr_delta": candidate_fpr - base_fpr,
        "base_weak_auroc": auroc_base, "candidate_weak_auroc": auroc_candidate,
        "weak_auroc_delta": auroc_candidate - auroc_base,
        "weak_fixed_count": int((positive_mask & ~base_prediction & candidate_prediction).sum()),
        "weak_broken_count": int((positive_mask & base_prediction & ~candidate_prediction).sum()),
        "negative_new_fp_count": int((negative_mask & ~base_prediction & candidate_prediction).sum()),
        "negative_fixed_fp_count": int((negative_mask & base_prediction & ~candidate_prediction).sum()),
    }


def _definitions(targets, uncertainty, expert_z):
    other = [index for index in range(6) if index != MFT_INDEX]
    groups = {}
    bounds = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.000001))
    for low, high in bounds:
        groups[f"uncertainty_{low:.2f}_{min(high,1.0):.2f}"] = (uncertainty >= low) & (uncertainty < high)
    z_bounds = ((-np.inf, -1.0, "lt_m1"), (-1.0, 0.0, "m1_0"),
                (0.0, 1.0, "0_1"), (1.0, np.inf, "ge_1"))
    for low, high, name in z_bounds:
        groups[f"expert_z_{name}"] = (expert_z >= low) & (expert_z < high)
    count = targets[:, other].sum(axis=1)
    groups["other_label_count_0"] = count == 0
    groups["other_label_count_1"] = count == 1
    groups["other_label_count_2"] = count == 2
    groups["other_label_count_ge_3"] = count >= 3
    for index in other:
        groups[f"colabel_{CLASS_NAMES[index]}_absent"] = targets[:, index] == 0
        groups[f"colabel_{CLASS_NAMES[index]}_present"] = targets[:, index] == 1
    return groups


def run(args):
    rows = []
    for seed in args.seeds:
        seed_root = args.source_root / f"seed_{seed}"
        base = np.load(seed_root / "M0_C4_Reconstructed" / "predictions.npz")
        candidate = np.load(seed_root / "M1_MFT_AlignedBoundedResidual" / "predictions.npz")
        expert = np.load(seed_root / "expert_residuals.npz")
        np.testing.assert_array_equal(base["sample_ids"], candidate["sample_ids"])
        np.testing.assert_array_equal(base["sample_ids"], expert["early_sample_ids"])
        targets = base["targets"].astype(bool)
        weak = base["weak_mask"].astype(bool)
        weak_positive = targets[:, MFT_INDEX] & weak[:, MFT_INDEX]
        negative = ~targets[:, MFT_INDEX]
        groups = _definitions(targets, expert["early_uncertainty"], expert["early_expert_z"])
        result = {name: _group_metrics(
            mask, weak_positive, negative,
            base["probabilities"][:, MFT_INDEX], candidate["probabilities"][:, MFT_INDEX],
            base["predictions"][:, MFT_INDEX].astype(bool), candidate["predictions"][:, MFT_INDEX].astype(bool),
        ) for name, mask in groups.items()}
        rows.append({"seed": int(seed), "strata": result})

    names = list(rows[0]["strata"])
    aggregate = {}
    for gi, name in enumerate(names):
        aggregate[name] = {}
        for mi, metric in enumerate(("weak_recall_delta", "fpr_delta", "weak_auroc_delta")):
            values = [row["strata"][name][metric] for row in rows]
            finite = [value for value in values if np.isfinite(value)]
            aggregate[name][metric] = _mean_ci(finite, 20290000 + gi * 10 + mi) if finite else None
            aggregate[name][f"positive_{metric}_seeds"] = int(sum(value > 0 for value in finite))
        aggregate[name]["mean_weak_positive_count"] = float(np.mean([
            row["strata"][name]["weak_positive_count"] for row in rows
        ]))
        aggregate[name]["mean_negative_count"] = float(np.mean([
            row["strata"][name]["negative_count"] for row in rows
        ]))
        aggregate[name]["total_weak_fixed"] = int(sum(row["strata"][name]["weak_fixed_count"] for row in rows))
        aggregate[name]["total_weak_broken"] = int(sum(row["strata"][name]["weak_broken_count"] for row in rows))
        aggregate[name]["total_negative_new_fp"] = int(sum(row["strata"][name]["negative_new_fp_count"] for row in rows))
        aggregate[name]["total_negative_fixed_fp"] = int(sum(row["strata"][name]["negative_fixed_fp_count"] for row in rows))
    payload = {
        "status": "completed", "experiment": "RouteA_MFT_Residual_StratifiedDiagnostic_v3",
        "seeds": list(args.seeds), "source_summary_sha256": _sha256(
            args.source_root / "route_a_mft_frozen_expert_bounded_residual_v3_summary.json"
        ),
        "aggregate": aggregate, "per_seed": rows,
        "diagnostic_only": True, "model_selection_allowed": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "aggregate": aggregate,
                      "test_v3_used": False}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_mft_frozen_expert_bounded_residual_v3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_mft_residual_stratified_diagnostic_v3" / "route_a_mft_residual_stratified_diagnostic_v3_summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
