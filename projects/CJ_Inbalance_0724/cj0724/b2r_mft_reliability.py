"""B2-R: bounded MFT reliability repair with the frozen cross-power ISRJ branch."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from cj0724.b1_physics_sufficiency import best_threshold, extract_physics_features, metrics_from_probabilities
from cj0724.b2_power_boundary import summarize_gates
from cj0724.dual_axis import POWER_LEVELS, PowerViewDataset
from cj0724.dual_axis_train import make_loader, predict
from cj0724.models import build_benchmark_model
from cj0724.route_b_mechanism_router import SEEDS, baseline_root


MFT_INDEX = 3


def robust_reference(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    return median.astype(np.float32), np.maximum(1.4826 * mad, 1e-4).astype(np.float32)


def reliability_features(
    e5_probability: np.ndarray,
    combined_probability: np.ndarray,
    delay_probability: np.ndarray,
    combined_features: np.ndarray,
    reference_median: np.ndarray,
    reference_scale: np.ndarray,
) -> np.ndarray:
    combined_features = np.asarray(combined_features, dtype=np.float32)
    standardized = np.abs((combined_features - reference_median) / reference_scale)
    complex_bins = combined_features[:, :120]
    spectrum = np.clip(combined_features[:, 240:300], 1e-8, None)
    spectrum_distribution = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-8)
    spectrum_entropy = -(spectrum_distribution * np.log(spectrum_distribution)).sum(axis=1)
    delay_peak = combined_features[:, 300]
    delay_prominence = delay_peak - np.median(complex_bins, axis=1)
    columns = np.column_stack((
        e5_probability,
        combined_probability,
        delay_probability,
        np.abs(combined_probability - e5_probability),
        np.abs(combined_probability - delay_probability),
        combined_probability * (1.0 - combined_probability),
        e5_probability * (1.0 - e5_probability),
        delay_peak,
        delay_prominence,
        spectrum.max(axis=1),
        spectrum_entropy,
        np.sqrt(np.mean(np.minimum(standardized, 20.0) ** 2, axis=1)),
        np.max(np.minimum(standardized, 20.0), axis=1),
    ))
    return columns.astype(np.float32)


def action_targets(labels: np.ndarray, e5: np.ndarray, physics: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(bool)
    target = np.zeros(len(labels), dtype=np.uint8)
    target[labels & (physics > e5)] = 1  # positive rescue
    target[(~labels) & (physics < e5)] = 2  # negative protection
    return target


def bounded_three_state(
    e5: np.ndarray, physics: np.ndarray, action_probability: np.ndarray,
    alpha: float, confidence: float,
) -> np.ndarray:
    action_probability = np.asarray(action_probability, dtype=np.float64)
    positive = action_probability[:, 1]
    negative = action_probability[:, 2]
    positive = np.where(positive >= confidence, positive, 0.0)
    negative = np.where(negative >= confidence, negative, 0.0)
    output = (
        e5
        + float(alpha) * positive * np.maximum(physics - e5, 0.0)
        - float(alpha) * negative * np.maximum(e5 - physics, 0.0)
    )
    return np.clip(output, np.minimum(e5, physics), np.maximum(e5, physics))


def _mean_power_aupr(labels: np.ndarray, probabilities: np.ndarray, powers: np.ndarray) -> float:
    return float(np.mean([
        average_precision_score(labels[powers == power], probabilities[powers == power])
        for power in POWER_LEVELS
    ]))


def select_router_hyperparameters(
    features: np.ndarray, labels: np.ndarray, e5: np.ndarray, physics: np.ndarray,
    powers: np.ndarray, groups: np.ndarray, seed: int,
    decision_threshold: float | None = None,
    max_recall_drop: float | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    candidates = {(alpha, confidence): [] for alpha in (0.25, 0.5, 0.75, 1.0)
                  for confidence in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    recall_drops = {key: [] for key in candidates}
    splitter = GroupKFold(n_splits=3)
    target = action_targets(labels, e5, physics)
    for train, validation in splitter.split(features, target, groups):
        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=2.0, random_state=seed,
        )
        model.fit(features[train], target[train])
        raw = model.predict_proba(features[validation])
        probabilities = np.zeros((len(validation), 3), dtype=np.float64)
        probabilities[:, model.classes_.astype(int)] = raw
        for key in candidates:
            candidate = bounded_three_state(e5[validation], physics[validation], probabilities, *key)
            candidates[key].append(_mean_power_aupr(labels[validation], candidate, powers[validation]))
            if decision_threshold is not None:
                drops = []
                for power in POWER_LEVELS:
                    mask = (powers[validation] == power) & labels[validation].astype(bool)
                    if np.any(mask):
                        base_recall = float((e5[validation][mask] >= decision_threshold).mean())
                        candidate_recall = float((candidate[mask] >= decision_threshold).mean())
                        drops.append(base_recall - candidate_recall)
                recall_drops[key].append(max(drops, default=0.0))
    rows = [
        {"alpha": alpha, "confidence": confidence,
         "cv_mean_power_aupr": float(np.mean(scores)),
         "cv_max_power_recall_drop": float(max(recall_drops[(alpha, confidence)], default=0.0))}
        for (alpha, confidence), scores in candidates.items()
    ]
    eligible = rows
    if max_recall_drop is not None:
        eligible = [row for row in rows if row["cv_max_power_recall_drop"] <= max_recall_drop]
    if not eligible:
        return 0.0, 1.0, rows
    selected = max(eligible, key=lambda row: (row["cv_mean_power_aupr"], row["confidence"], -row["alpha"]))
    return float(selected["alpha"]), float(selected["confidence"]), rows


def _best_mean_power_f1_threshold(labels: np.ndarray, probabilities: np.ndarray, powers: np.ndarray) -> float:
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        scores = [
            f1_score(labels[powers == power], probabilities[powers == power] >= threshold, zero_division=0)
            for power in POWER_LEVELS
        ]
        candidates.append((float(np.mean(scores)), -abs(float(threshold) - 0.5), float(threshold)))
    return max(candidates)[2]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _predict_neural_mft(
    model: torch.nn.Module, files: dict[int, Path], labels: np.ndarray,
    indices: np.ndarray, seed: int, batch_size: int, device: torch.device,
) -> dict[int, np.ndarray]:
    result = {}
    for power in POWER_LEVELS:
        dataset = PowerViewDataset(files, labels, indices, (power,), seed)
        loader = make_loader(dataset, batch_size, False, seed)
        _, probability, _, sample_indices, _ = predict(model, loader, device)
        dataset.close()
        lookup = {int(index): position for position, index in enumerate(sample_indices)}
        result[power] = probability[[lookup[int(index)] for index in indices], MFT_INDEX]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b2r_mft_reliability_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recall-constrained", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    all_power_rows = []
    router_rows = []
    state_rows = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir()
        baseline = baseline_root(seed)
        route = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        b2_seed_root = Path(f"outputs/b2_power_boundary_20260725/seed_{seed}")
        frozen_e5_mft_threshold = float(np.load(b2_seed_root / "power_10dB.npz")["thresholds_E5"][MFT_INDEX])
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        files = {power: Path(audit["levels"][str(power)]["path"]) for power in POWER_LEVELS}
        split = np.load(baseline / "split_indices.npz")
        early = np.sort(split["early_stop"].astype(np.int64))
        router_identity = early[:len(early) // 2]
        calibration_identity = early[len(early) // 2:]
        development_identity = np.sort(split["development"].astype(np.int64))
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        train_identity = np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64))
        with h5py.File(files[10], "r") as handle:
            labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
        checkpoint = torch.load(baseline / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        early_neural = _predict_neural_mft(model, files, labels, early, seed, args.neural_batch_size, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        combined_model = joblib.load(route / "mft_physics_classifier.joblib")
        train_matrix = extract_physics_features(files[10], train_identity, args.batch_size)
        delay_model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=seed,
        )
        delay_model.fit(train_matrix["mft_delay"], labels[train_identity, MFT_INDEX])
        reference_median, reference_scale = robust_reference(train_matrix["combined"])
        early_parts = {key: [] for key in ("features", "labels", "e5", "physics", "delay", "power", "identity")}
        for power in POWER_LEVELS:
            matrices = extract_physics_features(files[power], early, args.batch_size)
            combined_probability = combined_model.predict_proba(matrices["combined"])[:, 1]
            delay_probability = delay_model.predict_proba(matrices["mft_delay"])[:, 1]
            built = reliability_features(
                early_neural[power], combined_probability, delay_probability,
                matrices["combined"], reference_median, reference_scale,
            )
            early_parts["features"].append(built)
            early_parts["labels"].append(labels[early, MFT_INDEX])
            early_parts["e5"].append(early_neural[power])
            early_parts["physics"].append(combined_probability)
            early_parts["delay"].append(delay_probability)
            early_parts["power"].append(np.full(len(early), power, dtype=np.int16))
            early_parts["identity"].append(early.copy())
        early_parts = {key: np.concatenate(value) for key, value in early_parts.items()}
        router_mask = np.isin(early_parts["identity"], router_identity)
        calibration_mask = np.isin(early_parts["identity"], calibration_identity)
        alpha, confidence, cv_rows = select_router_hyperparameters(
            early_parts["features"][router_mask], early_parts["labels"][router_mask],
            early_parts["e5"][router_mask], early_parts["physics"][router_mask],
            early_parts["power"][router_mask], early_parts["identity"][router_mask], seed,
            decision_threshold=frozen_e5_mft_threshold if args.recall_constrained else None,
            max_recall_drop=0.005 if args.recall_constrained else None,
        )
        action = action_targets(
            early_parts["labels"][router_mask], early_parts["e5"][router_mask],
            early_parts["physics"][router_mask],
        )
        router = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=2.0, random_state=seed,
        )
        router.fit(early_parts["features"][router_mask], action)
        raw = router.predict_proba(early_parts["features"][calibration_mask])
        action_probability = np.zeros((int(calibration_mask.sum()), 3), dtype=np.float64)
        action_probability[:, router.classes_.astype(int)] = raw
        three_state_calibration = bounded_three_state(
            early_parts["e5"][calibration_mask], early_parts["physics"][calibration_mask],
            action_probability, alpha, confidence,
        )
        combined_threshold = float(json.loads((route / "result.json").read_text(encoding="utf-8"))["experts"]["MFT"]["threshold"])
        delay_threshold = best_threshold(
            early_parts["labels"][router_mask], early_parts["delay"][router_mask]
        )
        agreement_calibration = np.where(
            (early_parts["physics"][calibration_mask] >= combined_threshold)
            == (early_parts["delay"][calibration_mask] >= delay_threshold),
            early_parts["physics"][calibration_mask], early_parts["e5"][calibration_mask],
        )
        ood_threshold = float(np.quantile(early_parts["features"][router_mask, -2], 0.95))
        ood_calibration = np.where(
            ((early_parts["features"][calibration_mask, -2] <= ood_threshold)
             & ((early_parts["physics"][calibration_mask] >= combined_threshold)
                == (early_parts["delay"][calibration_mask] >= delay_threshold))),
            early_parts["physics"][calibration_mask], early_parts["e5"][calibration_mask],
        )
        calibration_candidates = {
            "agreement": agreement_calibration,
            "ood_keep_e5": ood_calibration,
            "three_state": three_state_calibration,
        }
        calibration_protocol = []
        for name, probability in calibration_candidates.items():
            threshold = frozen_e5_mft_threshold if args.recall_constrained else _best_mean_power_f1_threshold(
                early_parts["labels"][calibration_mask], probability, early_parts["power"][calibration_mask])
            mean_aupr = _mean_power_aupr(
                early_parts["labels"][calibration_mask], probability, early_parts["power"][calibration_mask]
            )
            calibration_protocol.append({"variant": name, "threshold": threshold, "mean_power_aupr": mean_aupr})
        selected = max(calibration_protocol, key=lambda row: row["mean_power_aupr"])
        calibration_fallback = False
        if args.recall_constrained:
            three_state_row = next(row for row in calibration_protocol if row["variant"] == "three_state")
            positive = calibration_mask & early_parts["labels"].astype(bool)
            recall_drops = []
            for power in POWER_LEVELS:
                mask = positive & (early_parts["power"] == power)
                base_recall = float((early_parts["e5"][mask] >= frozen_e5_mft_threshold).mean())
                repaired_recall = float((three_state_calibration[mask[calibration_mask]] >= frozen_e5_mft_threshold).mean())
                recall_drops.append(base_recall - repaired_recall)
            base_calibration_aupr = _mean_power_aupr(
                early_parts["labels"][calibration_mask], early_parts["e5"][calibration_mask],
                early_parts["power"][calibration_mask])
            calibration_fallback = bool(
                alpha == 0.0 or max(recall_drops) > 0.01
                or three_state_row["mean_power_aupr"] < base_calibration_aupr
            )
            selected = {
                "variant": "E5_plus_ISRJ" if calibration_fallback else "three_state",
                "threshold": frozen_e5_mft_threshold,
                "mean_power_aupr": base_calibration_aupr if calibration_fallback else three_state_row["mean_power_aupr"],
            }
        joblib.dump(router, seed_root / "mft_three_state_router.joblib")
        joblib.dump(delay_model, seed_root / "mft_delay_auxiliary.joblib")
        np.savez_compressed(seed_root / "reference_stats.npz", median=reference_median, scale=reference_scale)
        router_protocol = {
            "seed": seed, "router_fit_identities": int(len(router_identity)),
            "calibration_identities": int(len(calibration_identity)), "alpha": alpha,
            "confidence": confidence, "combined_threshold": combined_threshold,
            "delay_threshold": delay_threshold, "ood_threshold": ood_threshold,
            "calibration_candidates": calibration_protocol, "selected_variant": selected["variant"],
            "selected_threshold": selected["threshold"], "cv_candidates": cv_rows,
            "recall_constrained": bool(args.recall_constrained),
            "calibration_fallback": calibration_fallback,
            "explicit_JNR_used": False, "locked_test_used": False,
        }
        (seed_root / "protocol.json").write_text(json.dumps(router_protocol, indent=2), encoding="utf-8")
        for power in POWER_LEVELS:
            started = time.perf_counter()
            matrices = extract_physics_features(files[power], development_identity, args.batch_size)
            feature_seconds = time.perf_counter() - started
            combined_probability = combined_model.predict_proba(matrices["combined"])[:, 1]
            delay_probability = delay_model.predict_proba(matrices["mft_delay"])[:, 1]
            saved = np.load(b2_seed_root / f"power_{power}dB.npz")
            e5 = saved["probabilities_E5"].astype(np.float64)
            hard = saved["probabilities_B2-Hard"].astype(np.float64)
            features = reliability_features(
                e5[:, MFT_INDEX], combined_probability, delay_probability,
                matrices["combined"], reference_median, reference_scale,
            )
            raw = router.predict_proba(features)
            actions = np.zeros((len(features), 3), dtype=np.float64)
            actions[:, router.classes_.astype(int)] = raw
            variants = {
                "E5": e5[:, MFT_INDEX],
                "E5_plus_ISRJ": e5[:, MFT_INDEX],
                "B2-Hard": combined_probability,
                "agreement": np.where(
                    (combined_probability >= combined_threshold) == (delay_probability >= delay_threshold),
                    combined_probability, e5[:, MFT_INDEX],
                ),
                "ood_keep_e5": np.where(
                    ((features[:, -2] <= ood_threshold)
                     & ((combined_probability >= combined_threshold) == (delay_probability >= delay_threshold))),
                    combined_probability, e5[:, MFT_INDEX],
                ),
                "three_state": bounded_three_state(
                    e5[:, MFT_INDEX], combined_probability, actions, alpha, confidence
                ),
            }
            for name, mft_probability in variants.items():
                probability = e5.copy()
                thresholds = saved["thresholds_E5"].astype(np.float64)
                if name != "E5":
                    probability[:, 5] = hard[:, 5]
                    thresholds[5] = saved["thresholds_B2-Hard"][5]
                if name == "B2-Hard":
                    thresholds[MFT_INDEX] = saved["thresholds_B2-Hard"][MFT_INDEX]
                elif name not in ("E5", "E5_plus_ISRJ"):
                    protocol_row = next(row for row in calibration_protocol if row["variant"] == name)
                    thresholds[MFT_INDEX] = frozen_e5_mft_threshold if args.recall_constrained else protocol_row["threshold"]
                probability[:, MFT_INDEX] = mft_probability
                metrics = metrics_from_probabilities(saved["labels"], probability, thresholds, saved["weak_mask"])
                predicted = probability >= thresholds[None, :]
                positive = saved["labels"][:, MFT_INDEX].astype(bool)
                all_power_rows.append({
                    "seed": seed, "power_dB": power, "variant": name,
                    "macro_aupr": float(metrics["macro_aupr"]), "macro_f1": float(metrics["macro_f1"]),
                    "hamming": float(metrics["hamming_loss"]),
                    "subset_accuracy": float(np.all(predicted == saved["labels"].astype(bool), axis=1).mean()),
                    "weak_recall": float(metrics["weak_component_recall"]),
                    "MFT_aupr": float(metrics["per_label_aupr"][MFT_INDEX]),
                    "MFT_fpr": float(metrics["per_label_fpr"][MFT_INDEX]),
                    "MFT_recall": float(predicted[positive, MFT_INDEX].mean()),
                    "ISRJ_fpr": float(metrics["per_label_fpr"][5]),
                    "feature_seconds": feature_seconds,
                })
            selected_name = selected["variant"]
            selected_probability = variants[selected_name]
            state_rows.append({
                "seed": seed, "power_dB": power, "selected_variant": selected_name,
                "positive_rescue_rate": float((actions[:, 1] >= confidence).mean()),
                "negative_protection_rate": float((actions[:, 2] >= confidence).mean()),
                "changed_from_E5_rate": float((np.abs(selected_probability - e5[:, MFT_INDEX]) > 1e-8).mean()),
                "OOD_rate": float((features[:, -2] > ood_threshold).mean()),
            })
        router_rows.append(router_protocol)
        print(json.dumps({"seed": seed, "selected": selected["variant"],
                          "alpha": alpha, "confidence": confidence}), flush=True)
    variant_results = {}
    for variant in ("E5_plus_ISRJ", "B2-Hard", "agreement", "ood_keep_e5", "three_state"):
        rows = [row for row in all_power_rows if row["variant"] in ("E5", variant)]
        renamed = []
        for row in rows:
            copy = dict(row)
            copy["variant"] = "E5" if row["variant"] == "E5" else "B2-Hard"
            renamed.append(copy)
        variant_results[variant] = summarize_gates(renamed)
    selected_rows = []
    selected_by_seed = {row["seed"]: row["selected_variant"] for row in router_rows}
    for row in all_power_rows:
        if row["variant"] == "E5" or row["variant"] == selected_by_seed[row["seed"]]:
            copy = dict(row)
            copy["variant"] = "E5" if row["variant"] == "E5" else "B2-Hard"
            selected_rows.append(copy)
    selected_result = summarize_gates(selected_rows)
    max_recall_drop = max(
        np.mean([row["MFT_recall"] for row in selected_rows if row["variant"] == "E5" and row["power_dB"] == power])
        - np.mean([row["MFT_recall"] for row in selected_rows if row["variant"] == "B2-Hard" and row["power_dB"] == power])
        for power in POWER_LEVELS
    )
    selected_result["gate_values"]["max_MFT_recall_drop"] = float(max_recall_drop)
    selected_result["gates"]["MFT_recall_drop_within_0p01"] = bool(max_recall_drop <= 0.01)
    selected_result["B2_passed"] = bool(all(selected_result["gates"].values()))
    result = {
        "protocol": {"router_fit": "early_stop first half", "calibration": "early_stop second half",
                     "development_read_once": True, "explicit_JNR_used": False,
                     "legacy_screen_only": True, "locked_test_used": False},
        "router_protocols": router_rows, "variant_results": variant_results,
        "selected_by_seed_result": selected_result,
        "B2R_passed": selected_result["B2_passed"], "locked_test_used": False,
    }
    _write_csv(args.output_root / "power_runs.csv", all_power_rows)
    _write_csv(args.output_root / "state_rates.csv", state_rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected_by_seed_result": selected_result,
                      "B2R_passed": result["B2R_passed"]}, indent=2))


if __name__ == "__main__":
    main()
