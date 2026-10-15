"""Ablation, five-seed statistics, and inference-cost audit for Route B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.route_b_mechanism_router import SEEDS, fusion_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    outcomes = []
    for mask in range(1 << len(values)):
        signs = np.asarray([1 if mask & (1 << index) else -1 for index in range(len(values))])
        outcomes.append(abs(float((values * signs).mean())))
    return float(np.mean(np.asarray(outcomes) >= observed - 1e-15))


def _compact(metrics: dict[str, object]) -> dict[str, object]:
    return {key: metrics[key] for key in (
        "macro_aupr", "macro_f1", "hamming_loss", "weak_component_recall",
        "per_label_aupr", "per_label_fpr",
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, default=Path("outputs/route_b_mechanism_router_20260725"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/route_b_candidate_audit_20260725"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    variants = {"E5": [], "E5_plus_MFT": [], "E5_plus_ISRJ": [], "B_final": []}
    costs = []
    artifact_hashes = {}
    for seed in SEEDS:
        route_dir = args.route_root / f"seed_{seed}"
        route = np.load(route_dir / "development_predictions.npz")
        route_result = json.loads((route_dir / "result.json").read_text(encoding="utf-8"))
        fusion_dir = fusion_root(seed)
        e5 = np.load(fusion_dir / "development_predictions.npz")
        e5_protocol = json.loads((fusion_dir / "protocol.json").read_text(encoding="utf-8"))
        route_lookup = {int(index): position for position, index in enumerate(route["sample_indices"])}
        route_order = np.asarray([route_lookup[int(index)] for index in e5["sample_indices"]])
        route_probability = route["probabilities"][route_order]
        route_thresholds = route["thresholds"]
        e5_probability = e5["probabilities"].copy()
        e5_thresholds = np.asarray(e5_protocol["thresholds"], dtype=np.float64)
        candidates = {
            "E5": (e5_probability, e5_thresholds),
            "E5_plus_MFT": (e5_probability.copy(), e5_thresholds.copy()),
            "E5_plus_ISRJ": (e5_probability.copy(), e5_thresholds.copy()),
            "B_final": (route_probability, route_thresholds),
        }
        candidates["E5_plus_MFT"][0][:, 3] = route_probability[:, 3]
        candidates["E5_plus_MFT"][1][3] = route_thresholds[3]
        candidates["E5_plus_ISRJ"][0][:, 5] = route_probability[:, 5]
        candidates["E5_plus_ISRJ"][1][5] = route_thresholds[5]
        for name, (probability, thresholds) in candidates.items():
            metric = metrics_from_probabilities(e5["labels"], probability, thresholds, e5["weak_mask"])
            variants[name].append({"seed": seed, **_compact(metric)})
        dev_seconds = float(route_result["cost"]["feature_seconds"]["development"])
        classifier_sizes = sum((route_dir / f"{target}_physics_classifier.joblib").stat().st_size
                               for target in ("mft", "isrj"))
        costs.append({
            "seed": seed, "development_samples": int(len(route["labels"])),
            "physics_feature_seconds": dev_seconds,
            "physics_feature_samples_per_second": len(route["labels"]) / dev_seconds,
            "two_expert_model_bytes": classifier_sizes,
        })
        for path in route_dir.glob("*.joblib"):
            artifact_hashes[str(path.resolve())] = _sha256(path)
    statistics = {}
    for name in ("E5_plus_MFT", "E5_plus_ISRJ", "B_final"):
        statistics[name] = {}
        for metric in ("macro_aupr", "macro_f1", "hamming_loss", "weak_component_recall"):
            candidate = np.asarray([row[metric] for row in variants[name]], dtype=np.float64)
            reference = np.asarray([row[metric] for row in variants["E5"]], dtype=np.float64)
            delta = candidate - reference
            test = ttest_rel(candidate, reference)
            statistics[name][metric] = {
                "mean_delta": float(delta.mean()), "std_delta": float(delta.std(ddof=0)),
                "paired_deltas": delta.tolist(), "paired_t_p_two_sided": float(test.pvalue),
                "exact_sign_flip_p_two_sided": _sign_flip(delta),
            }
    e5_audit = json.loads(Path("outputs/vft_physics_e5_frozen_candidate_20260725/audit.json").read_text())
    e5_seconds = float(e5_audit["benchmark"]["e5_total_seconds"])
    mean_physics_seconds = float(np.mean([row["physics_feature_seconds"] for row in costs]))
    samples = costs[0]["development_samples"]
    cost_summary = {
        "e5_total_seconds_reference_seed20260724": e5_seconds,
        "route_b_extra_physics_seconds_mean_five_seed": mean_physics_seconds,
        "route_b_estimated_total_seconds": e5_seconds + mean_physics_seconds,
        "route_b_estimated_samples_per_second": samples / (e5_seconds + mean_physics_seconds),
        "latency_multiplier_vs_e5": (e5_seconds + mean_physics_seconds) / e5_seconds,
        "two_expert_model_bytes_mean": float(np.mean([row["two_expert_model_bytes"] for row in costs])),
        "note": "conservative end-to-end estimate; neural E5 and CPU physics feature extraction were not overlapped",
    }
    result = {
        "variants": variants, "paired_statistics_vs_e5": statistics,
        "cost_runs": costs, "cost_summary": cost_summary,
        "artifact_sha256": artifact_hashes,
        "decision": {
            "MFT_role": "secondary FPR-control expert; weaker AUPR evidence",
            "ISRJ_role": "primary physical expert; stable AUPR and FPR evidence",
            "B_final": "E5 plus both fixed class-specific experts",
            "locked_test_used": False,
        },
        "locked_test_used": False,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"statistics": statistics, "cost_summary": cost_summary,
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
