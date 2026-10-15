"""B4 matched hard-negative screen on the frozen B2-RC/B3-R pipeline."""

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
from sklearn.metrics import average_precision_score

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features
from cj0724.b3r_unified_pipeline import read_and_normalize
from cj0724.dual_axis import CLASS_NAMES, POWER_LEVELS
from cj0724.models import build_benchmark_model
from cj0724.route_b_mechanism_router import SEEDS, baseline_root
from cj0724.shared_physics_frontend import shared_feature_batch


TARGETS = {"MFT": 3, "ISRJ": 5}
FEATURES = {"MFT": "combined", "ISRJ": "isrj_periodicity"}
VARIANTS = ("random", "neural_high_confidence", "physics_similar",
            "power_cardinality_composition_matched")


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def companion_signatures(labels: np.ndarray, target_index: int) -> np.ndarray:
    matrix = np.delete(np.asarray(labels, dtype=np.uint8), target_index, axis=1)
    powers = (1 << np.arange(matrix.shape[1], dtype=np.int64))[None, :]
    return (matrix.astype(np.int64) * powers).sum(axis=1)


def hard_negative_weights(
    labels: np.ndarray, e5: np.ndarray, physics: np.ndarray, powers: np.ndarray,
    target_index: int, variant: str, seed: int,
) -> np.ndarray:
    """Return identity-balanced weights; paired views do not add total identity mass."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown B4 variant: {variant}")
    matrix = np.asarray(labels, dtype=np.uint8)
    target = matrix[:, target_index].astype(bool)
    negative_indices = np.flatnonzero(~target)
    raw_negative = np.ones(len(negative_indices), dtype=np.float64)
    if variant == "random":
        rng = np.random.default_rng(int(seed) + 1009 * int(target_index))
        hard = rng.random(len(negative_indices))
    elif variant == "neural_high_confidence":
        hard = _percentile(np.asarray(e5)[negative_indices])
    elif variant == "physics_similar":
        hard = _percentile(np.asarray(physics)[negative_indices])
    else:
        signatures = companion_signatures(matrix, target_index)
        hard = np.zeros(len(negative_indices), dtype=np.float64)
        for power in np.unique(powers):
            for signature in np.unique(signatures[negative_indices]):
                local = np.flatnonzero(
                    (~target) & (powers == power) & (signatures == signature)
                )
                if not len(local):
                    continue
                positions = np.searchsorted(negative_indices, local)
                hard[positions] = 0.5 * (
                    _percentile(np.asarray(e5)[local])
                    + _percentile(np.asarray(physics)[local])
                )
                positive_support = int(np.sum(target & (powers == power) & (signatures == signature)))
                if positive_support:
                    raw_negative[positions] *= positive_support / len(local)
    raw_negative *= np.where(hard >= 0.5, 5.0, 0.2)
    weights = np.zeros(len(matrix), dtype=np.float64)
    # Every identity has nine views, so the common 1/9 factor cancels after
    # normalization. Positive and negative total mass are held equal.
    weights[target] = 0.5 / max(int(target.sum()), 1)
    weights[negative_indices] = 0.5 * raw_negative / max(raw_negative.sum(), 1e-12)
    return (weights * len(matrix)).astype(np.float64)


def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    target = np.asarray(labels).astype(bool)
    predicted = np.asarray(probability) >= float(threshold)
    return {
        "aupr": float(average_precision_score(target, probability)),
        "fpr": float(predicted[~target].mean()),
        "recall": float(predicted[target].mean()),
    }


@torch.no_grad()
def infer_neural_and_physics(
    path: Path, indices: np.ndarray, model: torch.nn.Module,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    started = time.perf_counter()
    iq, _, _ = read_and_normalize(path, indices)
    tensor = torch.from_numpy(iq).to(device)
    neural_parts = []
    feature_parts = {"mft_delay": [], "isrj_periodicity": [], "combined": []}
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        neural_parts.append(torch.sigmoid(model(batch)).cpu().numpy())
        features = shared_feature_batch(batch)
        for key in feature_parts:
            feature_parts[key].append(features[key].cpu().numpy())
    return (
        np.concatenate(neural_parts),
        {key: np.concatenate(parts) for key, parts in feature_parts.items()},
        time.perf_counter() - started,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _combo_name(label: np.ndarray) -> str:
    return "+".join(name for name, value in zip(CLASS_NAMES, label) if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b4_matched_hard_negatives_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    run_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir()
        baseline = baseline_root(seed)
        route_root = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        repair_root = Path(f"outputs/b2rc_recall_constrained_20260725/seed_{seed}")
        b2_root = Path(f"outputs/b2_power_boundary_20260725/seed_{seed}")
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        files = {power: Path(audit["levels"][str(power)]["path"]) for power in POWER_LEVELS}
        split = np.load(baseline / "split_indices.npz")
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        roles = {
            "train": np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64)),
            "early": np.sort(split["early_stop"].astype(np.int64)),
            "development": np.sort(split["development"].astype(np.int64)),
        }
        all_indices = np.sort(np.concatenate(list(roles.values())))
        lookup = {int(index): position for position, index in enumerate(all_indices)}
        role_positions = {
            role: np.asarray([lookup[int(index)] for index in indices], dtype=np.int64)
            for role, indices in roles.items()
        }
        with h5py.File(files[10], "r") as handle:
            all_labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
        model.load_state_dict(torch.load(baseline / "best.pt", map_location=device, weights_only=False)["model_state"])
        model.eval()
        old_models = {
            "MFT": joblib.load(route_root / "mft_physics_classifier.joblib"),
            "ISRJ": joblib.load(route_root / "isrj_physics_classifier.joblib"),
        }
        delay_model = joblib.load(repair_root / "mft_delay_auxiliary.joblib")
        router = joblib.load(repair_root / "mft_three_state_router.joblib")
        reference = np.load(repair_root / "reference_stats.npz")
        router_protocol = json.loads((repair_root / "protocol.json").read_text(encoding="utf-8"))
        feature_names = set(FEATURES.values()) | {"mft_delay"}
        data = {role: {"labels": [], "powers": [], "e5": [], "features": {key: [] for key in feature_names}}
                for role in roles}
        inference_seconds = 0.0
        for power in POWER_LEVELS:
            neural, features, elapsed = infer_neural_and_physics(files[power], all_indices, model, device, args.batch_size)
            inference_seconds += elapsed
            for role, positions in role_positions.items():
                data[role]["labels"].append(all_labels[roles[role]])
                data[role]["powers"].append(np.full(len(positions), power, dtype=np.int16))
                data[role]["e5"].append(neural[positions])
                for key in data[role]["features"]:
                    data[role]["features"][key].append(features[key][positions])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for role in data:
            data[role]["labels"] = np.concatenate(data[role]["labels"])
            data[role]["powers"] = np.concatenate(data[role]["powers"])
            data[role]["e5"] = np.concatenate(data[role]["e5"])
            data[role]["features"] = {key: np.concatenate(parts) for key, parts in data[role]["features"].items()}
        trained: dict[str, dict[str, object]] = {target: {} for target in TARGETS}
        fit_seconds = 0.0
        for target, target_index in TARGETS.items():
            feature_name = FEATURES[target]
            train_old_probability = old_models[target].predict_proba(data["train"]["features"][feature_name])[:, 1]
            for variant in VARIANTS:
                weights = hard_negative_weights(
                    data["train"]["labels"], data["train"]["e5"][:, target_index],
                    train_old_probability, data["train"]["powers"], target_index, variant, seed,
                )
                classifier = HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                    l2_regularization=1.0, random_state=seed,
                )
                tick = time.perf_counter()
                classifier.fit(data["train"]["features"][feature_name],
                               data["train"]["labels"][:, target_index], sample_weight=weights)
                fit_seconds += time.perf_counter() - tick
                trained[target][variant] = classifier
                joblib.dump(classifier, seed_root / f"{target.lower()}_{variant}.joblib")
        early_identity_count = len(roles["early"])
        calibration_identity = roles["early"][early_identity_count // 2:]
        repeated_identity = np.tile(roles["early"], len(POWER_LEVELS))
        calibration_mask = np.isin(repeated_identity, calibration_identity)
        selected: dict[str, str] = {}
        old_early_probability = {
            target: old_models[target].predict_proba(data["early"]["features"][FEATURES[target]])[:, 1]
            for target in TARGETS
        }
        delay_early = delay_model.predict_proba(data["early"]["features"]["mft_delay"])[:, 1]
        router_features = reliability_features(
            data["early"]["e5"][:, 3], old_early_probability["MFT"], delay_early,
            data["early"]["features"]["combined"], reference["median"], reference["scale"],
        )
        raw_actions = router.predict_proba(router_features)
        actions = np.zeros((len(raw_actions), 3), dtype=np.float64)
        actions[:, router.classes_.astype(int)] = raw_actions
        old_early_final = {
            "MFT": bounded_three_state(
                data["early"]["e5"][:, 3], old_early_probability["MFT"], actions,
                router_protocol["alpha"], router_protocol["confidence"],
            ),
            "ISRJ": old_early_probability["ISRJ"],
        }
        for target, target_index in TARGETS.items():
            threshold = float(np.load(b2_root / "power_10dB.npz")[
                "thresholds_E5" if target == "MFT" else "thresholds_B2-Hard"
            ][target_index])
            label = data["early"]["labels"][:, target_index]
            reference_metric = binary_metrics(label[calibration_mask], old_early_final[target][calibration_mask], threshold)
            candidates = []
            for variant in VARIANTS:
                probability = trained[target][variant].predict_proba(
                    data["early"]["features"][FEATURES[target]])[:, 1]
                if target == "MFT":
                    probability = bounded_three_state(
                        data["early"]["e5"][:, 3], probability, actions,
                        router_protocol["alpha"], router_protocol["confidence"],
                    )
                metric = binary_metrics(label[calibration_mask], probability[calibration_mask], threshold)
                row = {
                    "seed": seed, "target": target, "variant": variant,
                    **{f"reference_{key}": value for key, value in reference_metric.items()},
                    **metric,
                    "fpr_reduction": reference_metric["fpr"] - metric["fpr"],
                    "aupr_delta": metric["aupr"] - reference_metric["aupr"],
                    "recall_drop": reference_metric["recall"] - metric["recall"],
                }
                row["eligible"] = bool(row["aupr_delta"] >= 0.0 and row["recall_drop"] <= 0.01)
                candidates.append(row)
                selection_rows.append(row)
            eligible = [row for row in candidates if row["eligible"] and row["fpr_reduction"] > 0.0]
            selected[target] = max(eligible, key=lambda row: (row["fpr_reduction"], row["aupr_delta"]))["variant"] if eligible else "B2-RC"
        (seed_root / "selection.json").write_text(json.dumps({
            "seed": seed, "selected": selected, "selection_role": "early-stop second half",
            "development_used": False, "locked_test_used": False,
        }, indent=2), encoding="utf-8")
        for power_index, power in enumerate(POWER_LEVELS):
            start = power_index * len(roles["development"])
            stop = start + len(roles["development"])
            labels = data["development"]["labels"][start:stop]
            features = {key: value[start:stop] for key, value in data["development"]["features"].items()}
            e5 = data["development"]["e5"][start:stop]
            saved = np.load(b2_root / f"power_{power}dB.npz")
            thresholds = saved["thresholds_E5"].astype(np.float64)
            thresholds[5] = saved["thresholds_B2-Hard"][5]
            old_mft = old_models["MFT"].predict_proba(features["combined"])[:, 1]
            old_isrj = old_models["ISRJ"].predict_proba(features["isrj_periodicity"])[:, 1]
            delay_probability = delay_model.predict_proba(features["mft_delay"])[:, 1]
            router_feature = reliability_features(
                e5[:, 3], old_mft, delay_probability, features["combined"],
                reference["median"], reference["scale"],
            )
            raw = router.predict_proba(router_feature)
            dev_actions = np.zeros((len(raw), 3), dtype=np.float64)
            dev_actions[:, router.classes_.astype(int)] = raw
            reference_probability = e5.copy()
            reference_probability[:, 3] = bounded_three_state(
                e5[:, 3], old_mft, dev_actions, router_protocol["alpha"], router_protocol["confidence"])
            reference_probability[:, 5] = old_isrj
            candidate_probability = reference_probability.copy()
            if selected["MFT"] != "B2-RC":
                new_mft = trained["MFT"][selected["MFT"]].predict_proba(features["combined"])[:, 1]
                candidate_probability[:, 3] = bounded_three_state(
                    e5[:, 3], new_mft, dev_actions, router_protocol["alpha"], router_protocol["confidence"])
            if selected["ISRJ"] != "B2-RC":
                candidate_probability[:, 5] = trained["ISRJ"][selected["ISRJ"]].predict_proba(features["isrj_periodicity"])[:, 1]
            reference_metrics = metrics_from_probabilities(labels, reference_probability, thresholds, saved["weak_mask"])
            candidate_metrics = metrics_from_probabilities(labels, candidate_probability, thresholds, saved["weak_mask"])
            reference_predicted = reference_probability >= thresholds[None, :]
            candidate_predicted = candidate_probability >= thresholds[None, :]
            row: dict[str, object] = {
                "seed": seed, "power_dB": power,
                "selected_MFT": selected["MFT"], "selected_ISRJ": selected["ISRJ"],
                "reference_macro_aupr": reference_metrics["macro_aupr"],
                "candidate_macro_aupr": candidate_metrics["macro_aupr"],
                "delta_macro_aupr": candidate_metrics["macro_aupr"] - reference_metrics["macro_aupr"],
                "reference_hamming": reference_metrics["hamming_loss"],
                "candidate_hamming": candidate_metrics["hamming_loss"],
                "delta_hamming": candidate_metrics["hamming_loss"] - reference_metrics["hamming_loss"],
                "reference_weak_recall": reference_metrics["weak_component_recall"],
                "candidate_weak_recall": candidate_metrics["weak_component_recall"],
                "delta_weak_recall": candidate_metrics["weak_component_recall"] - reference_metrics["weak_component_recall"],
            }
            for target, target_index in TARGETS.items():
                positive = labels[:, target_index].astype(bool)
                row[f"reference_{target}_aupr"] = reference_metrics["per_label_aupr"][target_index]
                row[f"candidate_{target}_aupr"] = candidate_metrics["per_label_aupr"][target_index]
                row[f"delta_{target}_aupr"] = candidate_metrics["per_label_aupr"][target_index] - reference_metrics["per_label_aupr"][target_index]
                row[f"reference_{target}_fpr"] = reference_metrics["per_label_fpr"][target_index]
                row[f"candidate_{target}_fpr"] = candidate_metrics["per_label_fpr"][target_index]
                row[f"{target}_fpr_reduction"] = reference_metrics["per_label_fpr"][target_index] - candidate_metrics["per_label_fpr"][target_index]
                row[f"reference_{target}_recall"] = float(reference_predicted[positive, target_index].mean())
                row[f"candidate_{target}_recall"] = float(candidate_predicted[positive, target_index].mean())
                row[f"{target}_recall_drop"] = row[f"reference_{target}_recall"] - row[f"candidate_{target}_recall"]
                for bits in np.unique(labels[~positive], axis=0):
                    mask = (~positive) & np.all(labels == bits, axis=1)
                    combination_rows.append({
                        "seed": seed, "power_dB": power, "target": target,
                        "combination": _combo_name(bits), "support": int(mask.sum()),
                        "reference_fpr": float(reference_predicted[mask, target_index].mean()),
                        "candidate_fpr": float(candidate_predicted[mask, target_index].mean()),
                    })
            run_rows.append(row)
            print(json.dumps({"seed": seed, "power": power, "selected": selected,
                              "MFT_fpr_reduction": row["MFT_fpr_reduction"],
                              "ISRJ_fpr_reduction": row["ISRJ_fpr_reduction"]}), flush=True)
        cost_rows.append({"seed": seed, "inference_seconds": inference_seconds,
                          "fit_seconds": fit_seconds,
                          "model_bytes": int(sum(path.stat().st_size for path in seed_root.glob("*.joblib")))})
    target_summary = {}
    target_gates = {}
    for target in TARGETS:
        fpr = np.asarray([row[f"{target}_fpr_reduction"] for row in run_rows], dtype=np.float64)
        aupr = np.asarray([row[f"delta_{target}_aupr"] for row in run_rows], dtype=np.float64)
        power_recall_drop = [
            float(np.mean([row[f"{target}_recall_drop"] for row in run_rows if row["power_dB"] == power]))
            for power in POWER_LEVELS
        ]
        same_direction = 0
        seed_rows = []
        for seed in SEEDS:
            chosen = [row for row in run_rows if row["seed"] == seed]
            seed_fpr = float(np.mean([row[f"{target}_fpr_reduction"] for row in chosen]))
            seed_aupr = float(np.mean([row[f"delta_{target}_aupr"] for row in chosen]))
            same_direction += int(seed_fpr > 0.0 and seed_aupr >= 0.0)
            seed_rows.append({"seed": seed, "mean_fpr_reduction": seed_fpr, "mean_aupr_delta": seed_aupr})
        target_summary[target] = {
            "mean_fpr_reduction": float(fpr.mean()), "mean_aupr_delta": float(aupr.mean()),
            "max_power_mean_recall_drop": float(max(power_recall_drop)),
            "same_direction_seeds": same_direction, "seed_rows": seed_rows,
        }
        target_gates[target] = {
            "mean_fpr_reduction_at_least_0p05": target_summary[target]["mean_fpr_reduction"] >= 0.05,
            "mean_aupr_not_lower": target_summary[target]["mean_aupr_delta"] >= 0.0,
            "max_power_recall_drop_at_most_0p01": target_summary[target]["max_power_mean_recall_drop"] <= 0.01,
            "same_direction_at_least_4_seeds": same_direction >= 4,
        }
    mean_hamming_delta = float(np.mean([row["delta_hamming"] for row in run_rows]))
    mean_weak_delta = float(np.nanmean([row["delta_weak_recall"] for row in run_rows]))
    overall_gates = {
        "hamming_not_increased": mean_hamming_delta <= 0.0,
        "weak_recall_drop_at_most_0p005": mean_weak_delta >= -0.005,
    }
    deployment_bytes = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        selection = json.loads((seed_root / "selection.json").read_text(encoding="utf-8"))["selected"]
        deployment_bytes.append(sum(
            (seed_root / f"{target.lower()}_{variant}.joblib").stat().st_size
            for target, variant in selection.items() if variant != "B2-RC"
        ))
    result = {
        "protocol": {"reference": "B2-RC", "seeds": list(SEEDS), "powers_dB": list(POWER_LEVELS),
                     "development_used_for_selection": False, "explicit_JNR_used": False,
                     "locked_test_used": False},
        "target_summary": target_summary, "target_gates": target_gates,
        "overall": {"mean_macro_aupr_delta": float(np.mean([row["delta_macro_aupr"] for row in run_rows])),
                    "mean_hamming_delta": mean_hamming_delta,
                    "mean_weak_recall_delta": mean_weak_delta},
        "overall_gates": overall_gates,
        "B4_passed": bool(all(all(gates.values()) for gates in target_gates.values()) and all(overall_gates.values())),
        "cost": {"mean_inference_seconds_per_seed": float(np.mean([row["inference_seconds"] for row in cost_rows])),
                 "mean_fit_seconds_per_seed": float(np.mean([row["fit_seconds"] for row in cost_rows])),
                 "all_screened_model_bytes_per_seed": float(np.mean([row["model_bytes"] for row in cost_rows])),
                 "mean_deployment_incremental_bytes_per_seed": float(np.mean(deployment_bytes)),
                 "max_deployment_incremental_bytes": int(max(deployment_bytes))},
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "power_runs.csv", run_rows)
    _write_csv(args.output_root / "selection.csv", selection_rows)
    _write_csv(args.output_root / "per_combination.csv", combination_rows)
    _write_csv(args.output_root / "cost.csv", cost_rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
