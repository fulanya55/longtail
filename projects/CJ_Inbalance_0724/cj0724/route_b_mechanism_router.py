"""Five-seed minimal mechanism router: frozen E5 + MFT/ISRJ physics experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import HistGradientBoostingClassifier

from cj0724.b1_physics_sufficiency import (
    TARGETS,
    best_threshold,
    extract_physics_features,
    metrics_from_probabilities,
)


SEEDS = (20260724, 20260725, 20260726, 271828, 314159)
FEATURES = {"MFT": "combined", "ISRJ": "isrj_periodicity"}


def baseline_root(seed: int) -> Path:
    base = "vft_d11_backbone_loss_20260725" if seed in SEEDS[:3] else "vft_d11_e5_confirmation_20260725"
    return Path(f"outputs/{base}/VFT/D11/b1_bce/seed_{seed}")


def fusion_root(seed: int) -> Path:
    base = "vft_physics_fusion_20260725" if seed in SEEDS[:3] else "vft_physics_confirmation_fusion_20260725"
    return Path(f"outputs/{base}/seed_{seed}")


def _metric_subset(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
        "hamming": metrics["hamming_loss"], "weak_recall": metrics["weak_component_recall"],
        "per_label_aupr": metrics["per_label_aupr"], "per_label_fpr": metrics["per_label_fpr"],
    }


def _sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    possible = []
    for mask in range(1 << len(values)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(values))])
        possible.append(abs(float((values * signs).mean())))
    return float(np.mean(np.asarray(possible) >= observed - 1e-15))


def _paired_summary(rows: list[dict[str, object]], metric: str, reference: str = "e5") -> dict[str, object]:
    values = np.asarray([float(row["route_b"][metric]) - float(row[reference][metric]) for row in rows])
    reference_values = np.asarray([float(row[reference][metric]) for row in rows])
    route_values = np.asarray([float(row["route_b"][metric]) for row in rows])
    test = ttest_rel(route_values, reference_values)
    return {
        "metric": metric, "reference": reference, "paired_deltas": values.tolist(),
        "mean_delta": float(values.mean()), "std_delta": float(values.std(ddof=0)),
        "improved_runs": int((values > 0).sum()) if metric != "hamming" else int((values < 0).sum()),
        "paired_t_p_two_sided": float(test.pvalue), "exact_sign_flip_p_two_sided": _sign_flip_p(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/route_b_mechanism_router_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in SEEDS:
        run_output = args.output_root / f"seed_{seed}"
        run_output.mkdir()
        baseline = baseline_root(seed)
        fusion = fusion_root(seed)
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        mat_path = Path(audit["levels"]["10"]["path"])
        split = np.load(baseline / "split_indices.npz")
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        roles = {
            "train": np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64)),
            "early": np.sort(split["early_stop"].astype(np.int64)),
            "development": np.sort(split["development"].astype(np.int64)),
        }
        with h5py.File(mat_path, "r") as handle:
            labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        matrices = {}
        feature_seconds = {}
        for role, indices in roles.items():
            started = time.perf_counter()
            matrices[role] = extract_physics_features(mat_path, indices, args.batch_size)
            feature_seconds[role] = time.perf_counter() - started
        fusion_payload = np.load(fusion / "development_predictions.npz")
        fusion_protocol = json.loads((fusion / "protocol.json").read_text(encoding="utf-8"))
        lookup = {int(index): position for position, index in enumerate(fusion_payload["sample_indices"])}
        order = np.asarray([lookup[int(index)] for index in roles["development"]], dtype=np.int64)
        e5_probability = fusion_payload["probabilities"][order].copy()
        e0_probability = fusion_payload["neural_probabilities"][order].copy()
        development_labels = fusion_payload["labels"][order]
        development_weak = fusion_payload["weak_mask"][order]
        e0_thresholds = np.asarray(json.loads((baseline / "thresholds.json").read_text(encoding="utf-8"))["thresholds"])
        e5_thresholds = np.asarray(fusion_protocol["thresholds"], dtype=np.float64)
        route_probability = e5_probability.copy()
        route_thresholds = e5_thresholds.copy()
        expert_protocol = {}
        classifier_seconds = 0.0
        for target, target_index in TARGETS.items():
            feature_name = FEATURES[target]
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                l2_regularization=1.0, random_state=seed,
            )
            started = time.perf_counter()
            model.fit(matrices["train"][feature_name], labels[roles["train"], target_index])
            early_probability = model.predict_proba(matrices["early"][feature_name])[:, 1]
            development_probability = model.predict_proba(matrices["development"][feature_name])[:, 1]
            classifier_seconds += time.perf_counter() - started
            threshold = best_threshold(labels[roles["early"], target_index], early_probability)
            route_probability[:, target_index] = development_probability
            route_thresholds[target_index] = threshold
            joblib.dump(model, run_output / f"{target.lower()}_physics_classifier.joblib")
            expert_protocol[target] = {
                "target_index": target_index, "feature": feature_name,
                "feature_dimension": int(matrices["train"][feature_name].shape[1]),
                "threshold": threshold, "router": f"replace only class {target_index}",
            }
        e0_metrics = metrics_from_probabilities(development_labels, e0_probability, e0_thresholds, development_weak)
        e5_metrics = metrics_from_probabilities(development_labels, e5_probability, e5_thresholds, development_weak)
        route_metrics = metrics_from_probabilities(development_labels, route_probability, route_thresholds, development_weak)
        row = {
            "seed": seed, "e0": _metric_subset(e0_metrics), "e5": _metric_subset(e5_metrics),
            "route_b": _metric_subset(route_metrics),
            "delta_vs_e5": {
                "macro_aupr": float(route_metrics["macro_aupr"] - e5_metrics["macro_aupr"]),
                "macro_f1": float(route_metrics["macro_f1"] - e5_metrics["macro_f1"]),
                "hamming": float(route_metrics["hamming_loss"] - e5_metrics["hamming_loss"]),
                "weak_recall": float(route_metrics["weak_component_recall"] - e5_metrics["weak_component_recall"]),
            },
            "cost": {"feature_seconds": feature_seconds, "classifier_fit_and_predict_seconds": classifier_seconds},
            "experts": expert_protocol,
        }
        np.savez_compressed(
            run_output / "development_predictions.npz", labels=development_labels,
            probabilities=route_probability.astype(np.float32), thresholds=route_thresholds.astype(np.float32),
            sample_indices=roles["development"], weak_mask=development_weak,
        )
        (run_output / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(json.dumps({"seed": seed, "delta_vs_e5": row["delta_vs_e5"]}), flush=True)
    statistics = {
        metric: _paired_summary(rows, metric) for metric in ("macro_aupr", "macro_f1", "hamming", "weak_recall")
    }
    result = {
        "method": "frozen E5 VFT route plus class-specific MFT and ISRJ physical experts",
        "router": "fixed class-index routing; no learned gate; no cross-class replacement",
        "seeds": list(SEEDS), "features": FEATURES, "runs": rows, "paired_statistics_vs_e5": statistics,
        "gates": {
            "macro_aupr_improved_4_of_5": statistics["macro_aupr"]["improved_runs"] >= 4,
            "hamming_improved_4_of_5": statistics["hamming"]["improved_runs"] >= 4,
            "no_locked_test": True,
        },
        "locked_test_used": False,
    }
    result["route_b_candidate_promoted"] = bool(all(result["gates"].values()))
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"statistics": statistics, "gates": result["gates"],
                      "route_b_candidate_promoted": result["route_b_candidate_promoted"]}, indent=2))


if __name__ == "__main__":
    main()
