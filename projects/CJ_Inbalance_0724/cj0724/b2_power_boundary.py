"""B2: five-seed, nine-power boundary audit for E0, frozen E5 and B2-Hard."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from cj0724.b1_physics_sufficiency import extract_physics_features, metrics_from_probabilities
from cj0724.dual_axis import CLASS_NAMES, POWER_LEVELS
from cj0724.route_b_mechanism_router import SEEDS, baseline_root, fusion_root
from cj0724.vft_physics_screen import extract_features


TARGETS = {"MFT": 3, "VFT": 4, "ISRJ": 5}


def _align(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {int(index): position for position, index in enumerate(source)}
    return values[[lookup[int(index)] for index in target]]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _combination_name(label: np.ndarray) -> str:
    return "+".join(name for name, value in zip(CLASS_NAMES, label) if value)


def _metric_row(
    seed: int, power: int, variant: str, labels: np.ndarray, probabilities: np.ndarray,
    thresholds: np.ndarray, weak: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    metrics = metrics_from_probabilities(labels, probabilities, thresholds, weak)
    predicted = probabilities >= thresholds[None, :]
    row = {
        "seed": seed, "power_dB": power, "variant": variant,
        "macro_aupr": float(metrics["macro_aupr"]), "macro_f1": float(metrics["macro_f1"]),
        "hamming": float(metrics["hamming_loss"]),
        "subset_accuracy": float(np.all(predicted == labels.astype(bool), axis=1).mean()),
        "weak_recall": float(metrics["weak_component_recall"]),
    }
    for name, index in TARGETS.items():
        row[f"{name}_aupr"] = float(metrics["per_label_aupr"][index])
        row[f"{name}_fpr"] = float(metrics["per_label_fpr"][index])
        positive = labels[:, index].astype(bool)
        row[f"{name}_recall"] = float(predicted[positive, index].mean())
    return row, predicted, metrics


def summarize_gates(power_rows: list[dict[str, object]]) -> dict[str, object]:
    aggregate = []
    for power in POWER_LEVELS:
        selected = [row for row in power_rows if int(row["power_dB"]) == power]
        entry: dict[str, object] = {"power_dB": power}
        for variant in ("E5", "B2-Hard"):
            rows = [row for row in selected if row["variant"] == variant]
            for metric in ("macro_aupr", "hamming", "weak_recall", "MFT_fpr", "ISRJ_fpr"):
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                entry[f"{variant}_{metric}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        for metric in ("macro_aupr", "hamming", "weak_recall", "MFT_fpr", "ISRJ_fpr"):
            entry[f"delta_{metric}"] = float(entry[f"B2-Hard_{metric}"] - entry[f"E5_{metric}"])
        aggregate.append(entry)
    macro_delta = np.asarray([row["delta_macro_aupr"] for row in aggregate], dtype=np.float64)
    hamming_delta = np.asarray([row["delta_hamming"] for row in aggregate], dtype=np.float64)
    weak_delta = np.asarray([row["delta_weak_recall"] for row in aggregate], dtype=np.float64)
    gates = {
        "mean_nine_power_macro_aupr_gain_vs_E5": float(np.mean(macro_delta)),
        "mean_nine_power_hamming_reduction_vs_E5": float(-np.mean(hamming_delta)),
        "MFT_FPR_reduced_levels": int(sum(row["delta_MFT_fpr"] < 0 for row in aggregate)),
        "ISRJ_FPR_reduced_levels": int(sum(row["delta_ISRJ_fpr"] < 0 for row in aggregate)),
        "worst_power_macro_aupr_delta": float(np.min(macro_delta)),
        "mean_weak_recall_delta": float(np.nanmean(weak_delta)),
    }
    passed = {
        "mean_macro_aupr_gain_at_least_0p015": gates["mean_nine_power_macro_aupr_gain_vs_E5"] >= 0.015,
        "mean_hamming_reduction_at_least_0p015": gates["mean_nine_power_hamming_reduction_vs_E5"] >= 0.015,
        "MFT_FPR_reduced_at_least_7_levels": gates["MFT_FPR_reduced_levels"] >= 7,
        "ISRJ_FPR_reduced_at_least_7_levels": gates["ISRJ_FPR_reduced_levels"] >= 7,
        "worst_power_macro_aupr_delta_at_least_minus_0p01": gates["worst_power_macro_aupr_delta"] >= -0.01,
        "mean_weak_recall_delta_at_least_minus_0p01": gates["mean_weak_recall_delta"] >= -0.01,
    }
    return {"power_aggregates": aggregate, "gate_values": gates, "gates": passed,
            "B2_passed": bool(all(passed.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b2_power_boundary_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vft-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    power_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    cardinality_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        run_output = args.output_root / f"seed_{seed}"
        run_output.mkdir()
        baseline = baseline_root(seed)
        fusion = fusion_root(seed)
        route = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        split = np.load(baseline / "split_indices.npz")
        indices = np.sort(split["development"].astype(np.int64))
        baseline_predictions = np.load(baseline / "development_predictions.npz")
        e0_thresholds = np.asarray(json.loads((baseline / "thresholds.json").read_text(encoding="utf-8"))["thresholds"])
        e5_protocol = json.loads((fusion / "protocol.json").read_text(encoding="utf-8"))
        e5_thresholds = np.asarray(e5_protocol["thresholds"], dtype=np.float64)
        route_10 = np.load(route / "development_predictions.npz")
        b2_thresholds = route_10["thresholds"].astype(np.float64)
        vft_model = joblib.load(Path("outputs/vft_physics_e5_frozen_candidate_20260725") /
                                 f"seed_{seed}/vft_physics_classifier.joblib")
        mft_model = joblib.load(route / "mft_physics_classifier.joblib")
        isrj_model = joblib.load(route / "isrj_physics_classifier.joblib")
        for power in POWER_LEVELS:
            mat_path = Path(audit["levels"][str(power)]["path"])
            started = time.perf_counter()
            vft_features = extract_features(mat_path, indices, frequencies, device, args.vft_batch_size)
            vft_seconds = time.perf_counter() - started
            started = time.perf_counter()
            physics_features = extract_physics_features(mat_path, indices, args.batch_size)
            shared_seconds = time.perf_counter() - started
            started = time.perf_counter()
            vft_probability = vft_model.predict_proba(vft_features)[:, 1]
            mft_probability = mft_model.predict_proba(physics_features["combined"])[:, 1]
            isrj_probability = isrj_model.predict_proba(physics_features["isrj_periodicity"])[:, 1]
            classifier_seconds = time.perf_counter() - started
            source = baseline_predictions[f"sample_indices_{power}"].astype(np.int64)
            labels = _align(baseline_predictions[f"labels_{power}"], source, indices)
            weak = _align(baseline_predictions[f"weak_mask_{power}"], source, indices)
            e0_probability = _align(baseline_predictions[f"probabilities_{power}"], source, indices)
            e5_probability = e0_probability.copy()
            e5_probability[:, 4] = vft_probability
            b2_probability = e5_probability.copy()
            b2_probability[:, 3] = mft_probability
            b2_probability[:, 5] = isrj_probability
            variants = {
                "E0": (e0_probability, e0_thresholds),
                "E5": (e5_probability, e5_thresholds),
                "B2-Hard": (b2_probability, b2_thresholds),
            }
            saved = {"labels": labels, "weak_mask": weak, "sample_indices": indices}
            for variant, (probability, thresholds) in variants.items():
                row, predicted, metrics = _metric_row(seed, power, variant, labels, probability, thresholds, weak)
                power_rows.append(row)
                saved[f"probabilities_{variant}"] = probability.astype(np.float32)
                saved[f"thresholds_{variant}"] = thresholds.astype(np.float32)
                for index, name in enumerate(CLASS_NAMES):
                    positive = labels[:, index].astype(bool)
                    negative = ~positive
                    label_rows.append({
                        "seed": seed, "power_dB": power, "variant": variant, "label": name,
                        "aupr": float(metrics["per_label_aupr"][index]),
                        "fpr": float(metrics["per_label_fpr"][index]),
                        "recall": float(predicted[positive, index].mean()),
                        "positive_support": int(positive.sum()), "negative_support": int(negative.sum()),
                    })
                for bits in np.unique(labels, axis=0):
                    mask = np.all(labels == bits, axis=1)
                    combination_rows.append({
                        "seed": seed, "power_dB": power, "variant": variant,
                        "combination": _combination_name(bits), "cardinality": int(bits.sum()),
                        "support": int(mask.sum()),
                        "exact_accuracy": float(np.all(predicted[mask] == labels[mask].astype(bool), axis=1).mean()),
                        "mean_label_accuracy": float((predicted[mask] == labels[mask].astype(bool)).mean()),
                        "MFT_predicted_positive_rate": float(predicted[mask, 3].mean()),
                        "ISRJ_predicted_positive_rate": float(predicted[mask, 5].mean()),
                    })
                cardinality = labels.sum(axis=1)
                for count in sorted(np.unique(cardinality)):
                    mask = cardinality == count
                    cardinality_rows.append({
                        "seed": seed, "power_dB": power, "variant": variant,
                        "cardinality": int(count), "support": int(mask.sum()),
                        "exact_accuracy": float(np.all(predicted[mask] == labels[mask].astype(bool), axis=1).mean()),
                        "hamming": float((predicted[mask] != labels[mask].astype(bool)).mean()),
                    })
            np.savez_compressed(run_output / f"power_{power}dB.npz", **saved)
            cost_rows.append({"seed": seed, "power_dB": power, "samples": int(len(indices)),
                              "vft_feature_seconds": vft_seconds, "shared_MFT_ISRJ_feature_seconds": shared_seconds,
                              "classifier_seconds": classifier_seconds,
                              "total_physics_seconds": vft_seconds + shared_seconds + classifier_seconds})
            print(json.dumps({"seed": seed, "power_dB": power,
                              "E5_macro_aupr": power_rows[-2]["macro_aupr"],
                              "B2_macro_aupr": power_rows[-1]["macro_aupr"],
                              "delta": power_rows[-1]["macro_aupr"] - power_rows[-2]["macro_aupr"]}), flush=True)
    summary = summarize_gates(power_rows)
    result = {
        "protocol": {
            "seeds": list(SEEDS), "powers_dB": list(POWER_LEVELS),
            "identity_rule": "same development indices and receiver_noise_seed-aligned views",
            "threshold_rule": "fixed from 10 dB early-stop/calibration; no per-power adjustment",
            "locked_test_used": False,
        },
        **summary,
        "mean_cost_seconds": float(np.mean([row["total_physics_seconds"] for row in cost_rows])),
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "power_runs.csv", power_rows)
    _write_csv(args.output_root / "per_label.csv", label_rows)
    _write_csv(args.output_root / "per_combination.csv", combination_rows)
    _write_csv(args.output_root / "per_cardinality.csv", cardinality_rows)
    _write_csv(args.output_root / "cost.csv", cost_rows)
    _write_csv(args.output_root / "power_aggregates.csv", summary["power_aggregates"])
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gate_values": result["gate_values"], "gates": result["gates"],
                      "B2_passed": result["B2_passed"]}, indent=2))


if __name__ == "__main__":
    main()
