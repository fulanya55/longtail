"""Read-only stratified postmortem for the completed H5o frozen OOD run.

The script does not fit a model, calibrator, threshold, or selector.  It reads
the already saved Q0/Q6 predictions and documents where frozen-decision FPR
transport fails under the H5o source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2039, 2040, 2041, 2042, 2043)
LABELS = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGETS = ("MFT", "VFT", "ISRJ")
Q0 = "Q0_H1"
Q6 = "Q6_SharedMFTDecision_VFTAligned"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(values: np.ndarray) -> float | None:
    return None if not len(values) else float(np.mean(values))


def _label_metrics(item: dict[str, np.ndarray], index: int) -> dict[str, float | None]:
    target = item["targets"][:, index].astype(bool)
    weak = item["weak_mask"][:, index].astype(bool)
    prediction = item["predictions"][:, index].astype(bool)
    probability = item["probabilities"][:, index]
    negative = ~target
    weak_positive = target & weak
    strong_positive = target & ~weak
    ranking = weak_positive | negative
    return {
        "aupr": float(average_precision_score(target, probability)),
        "weak_auroc": float(roc_auc_score(weak_positive[ranking], probability[ranking])) if weak_positive.any() and negative.any() else None,
        "negative_fpr": _rate(prediction[negative]),
        "weak_recall": _rate(prediction[weak_positive]),
        "strong_recall": _rate(prediction[strong_positive]),
        "threshold": float(item["thresholds"][index]),
        "negative_count": int(negative.sum()),
        "weak_positive_count": int(weak_positive.sum()),
        "strong_positive_count": int(strong_positive.sum()),
    }


def _mean_ci(values: list[float], seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(array, size=(20_000, len(array)), replace=True), axis=1)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "count": int(len(array)),
            "bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def _load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {name: np.asarray(payload[name]) for name in payload.files}


def _stratum_fpr(item0: dict[str, np.ndarray], item6: dict[str, np.ndarray], target_index: int, mask: np.ndarray) -> dict[str, float | int | None]:
    negative = ~item0["targets"][:, target_index].astype(bool)
    chosen = negative & mask
    return {"count": int(chosen.sum()), "q0_fpr": _rate(item0["predictions"][:, target_index].astype(bool)[chosen]),
            "q6_fpr": _rate(item6["predictions"][:, target_index].astype(bool)[chosen])}


def analyze(root: Path = ROOT) -> dict[str, object]:
    output_root = root / "outputs/tapc/route_a_h5o_frozen_unified_ood"
    rows, file_hashes = [], {}
    for seed in SEEDS:
        seed_root = output_root / f"seed_{seed}"
        q0_path, q6_path = seed_root / Q0 / "predictions.npz", seed_root / Q6 / "predictions.npz"
        item0, item6 = _load(q0_path), _load(q6_path)
        for key in ("targets", "weak_mask", "receiver_noise_seed"):
            if not np.array_equal(item0[key], item6[key]):
                raise ValueError(f"Q0/Q6 immutable evaluation target mismatch for seed {seed}: {key}")
        labels = {name: {"q0": _label_metrics(item0, index), "q6": _label_metrics(item6, index)} for index, name in enumerate(LABELS)}
        cardinality = item0["targets"].sum(axis=1)
        strata = {}
        for target in TARGETS:
            ti = LABELS.index(target)
            target_strata = {}
            for other in TARGETS:
                if other == target:
                    continue
                oi = LABELS.index(other)
                target_strata[f"{other}_present"] = _stratum_fpr(item0, item6, ti, item0["targets"][:, oi].astype(bool))
                target_strata[f"{other}_absent"] = _stratum_fpr(item0, item6, ti, ~item0["targets"][:, oi].astype(bool))
            for order in range(1, 7):
                target_strata[f"combination_order_{order}"] = _stratum_fpr(item0, item6, ti, cardinality == order)
            strata[target] = target_strata
        rows.append({"seed": seed, "labels": labels, "target_negative_fpr_strata": strata,
                     "receiver_sha256": hashlib.sha256(item0["receiver_noise_seed"].tobytes()).hexdigest()})
        file_hashes[str(seed)] = {"q0_predictions": _sha256(q0_path), "q6_predictions": _sha256(q6_path)}

    aggregate = {}
    for label in LABELS:
        aggregate[label] = {}
        for metric in ("aupr", "weak_auroc", "negative_fpr", "weak_recall", "strong_recall"):
            delta = [float(row["labels"][label]["q6"][metric]) - float(row["labels"][label]["q0"][metric]) for row in rows]
            aggregate[label][f"q6_minus_q0_{metric}"] = _mean_ci(delta, 20500000 + LABELS.index(label) * 100 + len(metric))
        aggregate[label]["q0_thresholds"] = [row["labels"][label]["q0"]["threshold"] for row in rows]
        aggregate[label]["q6_thresholds"] = [row["labels"][label]["q6"]["threshold"] for row in rows]

    fpr_strata = {}
    for target in TARGETS:
        keys = rows[0]["target_negative_fpr_strata"][target]
        fpr_strata[target] = {}
        for key in keys:
            counts = [row["target_negative_fpr_strata"][target][key]["count"] for row in rows]
            q0 = [row["target_negative_fpr_strata"][target][key]["q0_fpr"] for row in rows]
            q6 = [row["target_negative_fpr_strata"][target][key]["q6_fpr"] for row in rows]
            if any(value is None for value in q0 + q6):
                continue
            fpr_strata[target][key] = {"negative_count_per_seed": counts, "q0_fpr": _mean_ci(q0, 20510000 + len(key)),
                                       "q6_fpr": _mean_ci(q6, 20520000 + len(key)),
                                       "q6_minus_q0_fpr": _mean_ci([b - a for a, b in zip(q0, q6)], 20530000 + len(key))}

    return {
        "status": "completed_read_only_failure_postmortem",
        "experiment": "RouteA_H5o_FrozenUnifiedCombinedOOD_Postmortem",
        "evidence_role": "nonselection_read_only_stratified_analysis",
        "source_summary_sha256": _sha256(output_root / "summary.json"),
        "prediction_sha256": file_hashes,
        "seeds": list(SEEDS), "per_seed": rows, "label_delta_aggregate": aggregate,
        "target_negative_fpr_strata": fpr_strata,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        "selection_or_refit_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood/postmortem.json")
    args = parser.parse_args()
    payload = analyze()
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "test_v3_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
