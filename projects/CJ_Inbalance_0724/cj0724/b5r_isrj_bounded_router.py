"""B5-R: bounded ISRJ routing over frozen B4 hard-negative experts."""

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

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features, robust_reference
from cj0724.b3r_unified_pipeline import read_and_normalize
from cj0724.b4_matched_hard_negatives import VARIANTS, binary_metrics
from cj0724.dual_axis import CLASS_NAMES, POWER_LEVELS
from cj0724.models import build_benchmark_model
from cj0724.route_b_mechanism_router import SEEDS, baseline_root
from cj0724.shared_physics_frontend import shared_feature_batch


ISRJ_INDEX = 5


def isrj_reliability_features(
    e5: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
    physics: np.ndarray, median: np.ndarray, scale: np.ndarray,
    include_e5: bool = True, include_physics: bool = True,
) -> np.ndarray:
    matrix = np.asarray(physics, dtype=np.float32)
    columns = [reference, candidate, np.abs(candidate - reference),
               reference * (1.0 - reference), candidate * (1.0 - candidate)]
    if include_e5:
        columns.extend([e5, np.abs(reference - e5), e5 * (1.0 - e5)])
    if include_physics:
        standardized = np.abs((matrix - median) / scale)
        envelope = np.clip(matrix[:, 120:240], 1e-8, None)
        spectrum = np.clip(matrix[:, 240:300], 1e-8, None)
        envelope_distribution = envelope / np.maximum(envelope.sum(axis=1, keepdims=True), 1e-8)
        spectrum_distribution = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-8)
        columns.extend([
            matrix[:, 300], matrix[:, 301], matrix[:, 302:].mean(axis=1), matrix[:, 302:].std(axis=1),
            envelope.max(axis=1), spectrum.max(axis=1),
            -(envelope_distribution * np.log(envelope_distribution)).sum(axis=1),
            -(spectrum_distribution * np.log(spectrum_distribution)).sum(axis=1),
            np.sqrt(np.mean(np.minimum(standardized, 20.0) ** 2, axis=1)),
            np.max(np.minimum(standardized, 20.0), axis=1),
        ])
    return np.column_stack(columns).astype(np.float32)


def action_targets(labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    positive = np.asarray(labels).astype(bool)
    target = np.zeros(len(positive), dtype=np.uint8)
    target[positive & (candidate > reference)] = 1
    target[(~positive) & (candidate < reference)] = 2
    return target


def two_state_route(
    reference: np.ndarray, candidate: np.ndarray, action_probability: np.ndarray, alpha: float,
) -> np.ndarray:
    positive = action_probability[:, 1]
    negative = action_probability[:, 2]
    total = np.maximum(positive + negative, 1e-8)
    positive, negative = positive / total, negative / total
    output = (reference + alpha * positive * np.maximum(candidate - reference, 0.0)
              - alpha * negative * np.maximum(reference - candidate, 0.0))
    return np.clip(output, np.minimum(reference, candidate), np.maximum(reference, candidate))


def unbounded_residual_route(
    reference: np.ndarray, candidate: np.ndarray, action_probability: np.ndarray,
) -> np.ndarray:
    eps = 1e-6
    logit = np.log(np.clip(reference, eps, 1 - eps) / np.clip(1 - reference, eps, 1 - eps))
    candidate_logit = np.log(np.clip(candidate, eps, 1 - eps) / np.clip(1 - candidate, eps, 1 - eps))
    direction = action_probability[:, 1] - action_probability[:, 2]
    output_logit = logit + 1.5 * direction * np.abs(candidate_logit - logit)
    return 1.0 / (1.0 + np.exp(-np.clip(output_logit, -30.0, 30.0)))


def route_variant(
    name: str, reference: np.ndarray, candidate: np.ndarray,
    action_probability: np.ndarray | None, alpha: float = 1.0, confidence: float = 0.4,
) -> np.ndarray:
    if name == "hard_replace":
        return candidate.copy()
    if name == "fixed_blend":
        return 0.5 * (reference + candidate)
    if action_probability is None:
        raise ValueError(f"{name} requires action probabilities")
    if name == "unbounded_sigmoid_gate":
        return unbounded_residual_route(reference, candidate, action_probability)
    if name == "bounded_two_state":
        return two_state_route(reference, candidate, action_probability, alpha)
    return bounded_three_state(reference, candidate, action_probability, alpha, confidence)


@torch.no_grad()
def infer_inputs(
    path: Path, indices: np.ndarray, model: torch.nn.Module,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    iq, _, _ = read_and_normalize(path, indices)
    tensor = torch.from_numpy(iq).to(device)
    neural, physics = [], []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        neural.append(torch.sigmoid(model(batch))[:, ISRJ_INDEX].cpu().numpy())
        physics.append(shared_feature_batch(batch)["isrj_periodicity"].cpu().numpy())
    return np.concatenate(neural), np.concatenate(physics), time.perf_counter() - started


def fit_router(features: np.ndarray, targets: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=2.0, random_state=seed,
    )
    model.fit(features, targets)
    return model


def action_probabilities(model: HistGradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    output = np.zeros((len(features), 3), dtype=np.float64)
    output[:, model.classes_.astype(int)] = raw
    return output


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
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b5r_isrj_bounded_router_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    run_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir()
        baseline = baseline_root(seed)
        route_root = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        repair_root = Path(f"outputs/b2rc_recall_constrained_20260725/seed_{seed}")
        b2_root = Path(f"outputs/b2_power_boundary_20260725/seed_{seed}")
        b4_root = Path(f"outputs/b4_matched_hard_negatives_20260725/seed_{seed}")
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
        positions = {role: np.asarray([lookup[int(index)] for index in value], dtype=np.int64)
                     for role, value in roles.items()}
        with h5py.File(files[10], "r") as handle:
            labels_all = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
        model.load_state_dict(torch.load(baseline / "best.pt", map_location=device, weights_only=False)["model_state"])
        model.eval()
        data = {role: {"labels": [], "powers": [], "e5": [], "physics": []} for role in roles}
        inference_seconds = 0.0
        for power in POWER_LEVELS:
            neural, physics, elapsed = infer_inputs(files[power], all_indices, model, device, args.batch_size)
            inference_seconds += elapsed
            for role in roles:
                data[role]["labels"].append(labels_all[roles[role]])
                data[role]["powers"].append(np.full(len(roles[role]), power, dtype=np.int16))
                data[role]["e5"].append(neural[positions[role]])
                data[role]["physics"].append(physics[positions[role]])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for role in data:
            data[role] = {key: np.concatenate(value) for key, value in data[role].items()}
        reference_model = joblib.load(route_root / "isrj_physics_classifier.joblib")
        candidate_models = {variant: joblib.load(b4_root / f"isrj_{variant}.joblib") for variant in VARIANTS}
        median, scale = robust_reference(data["train"]["physics"])
        early_count = len(roles["early"])
        repeated_identity = np.tile(roles["early"], len(POWER_LEVELS))
        fit_identity = roles["early"][:early_count // 2]
        calibration_identity = roles["early"][early_count // 2:]
        fit_mask = np.isin(repeated_identity, fit_identity)
        calibration_mask = np.isin(repeated_identity, calibration_identity)
        early_reference = reference_model.predict_proba(data["early"]["physics"])[:, 1]
        threshold = float(np.load(b2_root / "power_10dB.npz")["thresholds_B2-Hard"][ISRJ_INDEX])
        reference_metric = binary_metrics(
            data["early"]["labels"][calibration_mask, ISRJ_INDEX],
            early_reference[calibration_mask], threshold,
        )
        candidates = []
        router_artifacts = {}
        fit_seconds = 0.0
        for expert_name, expert in candidate_models.items():
            early_candidate = expert.predict_proba(data["early"]["physics"])[:, 1]
            feature_sets = {
                "full": isrj_reliability_features(data["early"]["e5"], early_reference, early_candidate,
                                                   data["early"]["physics"], median, scale),
                "without_physics": isrj_reliability_features(
                    data["early"]["e5"], early_reference, early_candidate,
                    data["early"]["physics"], median, scale, include_physics=False),
                "without_e5": isrj_reliability_features(
                    data["early"]["e5"], early_reference, early_candidate,
                    data["early"]["physics"], median, scale, include_e5=False),
            }
            targets = action_targets(data["early"]["labels"][:, ISRJ_INDEX], early_reference, early_candidate)
            routers = {}
            for feature_name, feature in feature_sets.items():
                tick = time.perf_counter()
                routers[feature_name] = fit_router(feature[fit_mask], targets[fit_mask], seed)
                fit_seconds += time.perf_counter() - tick
                joblib.dump(routers[feature_name], seed_root / f"{expert_name}_{feature_name}_router.joblib")
            actions = {name: action_probabilities(router, feature_sets[name]) for name, router in routers.items()}
            specifications = [
                ("hard_replace", "full", 1.0, 0.0),
                ("fixed_blend", "full", 0.5, 0.0),
                ("unbounded_sigmoid_gate", "full", 1.0, 0.0),
                ("bounded_two_state", "full", 1.0, 0.0),
            ]
            for alpha in (0.5, 1.0):
                for confidence in (0.4, 0.6, 0.8):
                    specifications.extend([
                        ("bounded_three_state", "full", alpha, confidence),
                        ("without_physics_reliability", "without_physics", alpha, confidence),
                        ("without_E5_uncertainty", "without_e5", alpha, confidence),
                    ])
            for ablation, feature_name, alpha, confidence in specifications:
                route_name = "bounded_three_state" if ablation.startswith("without_") else ablation
                probability = route_variant(
                    route_name, early_reference, early_candidate,
                    actions.get(feature_name), alpha=alpha, confidence=confidence,
                )
                metric = binary_metrics(
                    data["early"]["labels"][calibration_mask, ISRJ_INDEX],
                    probability[calibration_mask], threshold,
                )
                row = {
                    "seed": seed, "expert": expert_name, "ablation": ablation,
                    "feature_set": feature_name, "alpha": alpha, "confidence": confidence,
                    **{f"reference_{key}": value for key, value in reference_metric.items()}, **metric,
                    "fpr_reduction": reference_metric["fpr"] - metric["fpr"],
                    "aupr_delta": metric["aupr"] - reference_metric["aupr"],
                    "recall_drop": reference_metric["recall"] - metric["recall"],
                }
                row["eligible"] = bool(row["aupr_delta"] >= 0.0 and row["recall_drop"] <= 0.005)
                candidates.append(row)
                selection_rows.append(row)
            router_artifacts[expert_name] = {"routers": routers}
        eligible = [row for row in candidates if row["eligible"] and row["fpr_reduction"] > 0.0]
        selected = max(eligible, key=lambda row: (row["fpr_reduction"], row["aupr_delta"])) if eligible else {
            "expert": "B2-RC", "ablation": "keep_B2RC", "feature_set": "none",
            "alpha": 0.0, "confidence": 1.0,
        }
        (seed_root / "protocol.json").write_text(json.dumps({
            "seed": seed, "selected": selected, "fit_role": "early-stop first half",
            "calibration_role": "early-stop second half", "development_used_for_selection": False,
            "threshold": threshold, "explicit_JNR_used": False, "locked_test_used": False,
        }, indent=2), encoding="utf-8")
        reference_mft_model = joblib.load(route_root / "mft_physics_classifier.joblib")
        delay_model = joblib.load(repair_root / "mft_delay_auxiliary.joblib")
        mft_router = joblib.load(repair_root / "mft_three_state_router.joblib")
        mft_stats = np.load(repair_root / "reference_stats.npz")
        mft_protocol = json.loads((repair_root / "protocol.json").read_text(encoding="utf-8"))
        for power_index, power in enumerate(POWER_LEVELS):
            start = power_index * len(roles["development"])
            stop = start + len(roles["development"])
            labels = data["development"]["labels"][start:stop]
            isrj_physics = data["development"]["physics"][start:stop]
            e5_isrj = data["development"]["e5"][start:stop]
            saved = np.load(b2_root / f"power_{power}dB.npz")
            e5_full = saved["probabilities_E5"].astype(np.float64)
            thresholds = saved["thresholds_E5"].astype(np.float64)
            thresholds[ISRJ_INDEX] = saved["thresholds_B2-Hard"][ISRJ_INDEX]
            # Reuse the frozen B2-RC MFT path exactly.
            combined = __import__("cj0724.b1_physics_sufficiency", fromlist=["extract_physics_features"]).extract_physics_features(
                files[power], roles["development"], args.batch_size)
            old_mft = reference_mft_model.predict_proba(combined["combined"])[:, 1]
            delay = delay_model.predict_proba(combined["mft_delay"])[:, 1]
            mft_features = reliability_features(e5_full[:, 3], old_mft, delay, combined["combined"],
                                                mft_stats["median"], mft_stats["scale"])
            mft_actions = action_probabilities(mft_router, mft_features)
            reference_probability = e5_full.copy()
            reference_probability[:, 3] = bounded_three_state(
                e5_full[:, 3], old_mft, mft_actions, mft_protocol["alpha"], mft_protocol["confidence"])
            reference_isrj = reference_model.predict_proba(isrj_physics)[:, 1]
            reference_probability[:, ISRJ_INDEX] = reference_isrj
            candidate_probability = reference_probability.copy()
            state_action = np.zeros((len(labels), 3), dtype=np.float64)
            if selected["expert"] != "B2-RC":
                expert = candidate_models[selected["expert"]]
                candidate_isrj = expert.predict_proba(isrj_physics)[:, 1]
                feature = isrj_reliability_features(
                    e5_isrj, reference_isrj, candidate_isrj, isrj_physics, median, scale,
                    include_e5=selected["feature_set"] != "without_e5",
                    include_physics=selected["feature_set"] != "without_physics",
                )
                router = router_artifacts[selected["expert"]]["routers"][selected["feature_set"]]
                state_action = action_probabilities(router, feature)
                route_name = "bounded_three_state" if selected["ablation"].startswith("without_") else selected["ablation"]
                candidate_probability[:, ISRJ_INDEX] = route_variant(
                    route_name, reference_isrj, candidate_isrj, state_action,
                    selected["alpha"], selected["confidence"],
                )
            reference_metrics = metrics_from_probabilities(labels, reference_probability, thresholds, saved["weak_mask"])
            candidate_metrics = metrics_from_probabilities(labels, candidate_probability, thresholds, saved["weak_mask"])
            reference_predicted = reference_probability >= thresholds[None, :]
            candidate_predicted = candidate_probability >= thresholds[None, :]
            positive = labels[:, ISRJ_INDEX].astype(bool)
            row = {
                "seed": seed, "power_dB": power, "expert": selected["expert"],
                "ablation": selected["ablation"], "alpha": selected["alpha"], "confidence": selected["confidence"],
                "reference_macro_aupr": reference_metrics["macro_aupr"],
                "candidate_macro_aupr": candidate_metrics["macro_aupr"],
                "delta_macro_aupr": candidate_metrics["macro_aupr"] - reference_metrics["macro_aupr"],
                "reference_hamming": reference_metrics["hamming_loss"],
                "candidate_hamming": candidate_metrics["hamming_loss"],
                "delta_hamming": candidate_metrics["hamming_loss"] - reference_metrics["hamming_loss"],
                "reference_weak_recall": reference_metrics["weak_component_recall"],
                "candidate_weak_recall": candidate_metrics["weak_component_recall"],
                "delta_weak_recall": candidate_metrics["weak_component_recall"] - reference_metrics["weak_component_recall"],
                "reference_ISRJ_aupr": reference_metrics["per_label_aupr"][ISRJ_INDEX],
                "candidate_ISRJ_aupr": candidate_metrics["per_label_aupr"][ISRJ_INDEX],
                "delta_ISRJ_aupr": candidate_metrics["per_label_aupr"][ISRJ_INDEX] - reference_metrics["per_label_aupr"][ISRJ_INDEX],
                "reference_ISRJ_fpr": reference_metrics["per_label_fpr"][ISRJ_INDEX],
                "candidate_ISRJ_fpr": candidate_metrics["per_label_fpr"][ISRJ_INDEX],
                "ISRJ_fpr_reduction": reference_metrics["per_label_fpr"][ISRJ_INDEX] - candidate_metrics["per_label_fpr"][ISRJ_INDEX],
                "reference_ISRJ_recall": float(reference_predicted[positive, ISRJ_INDEX].mean()),
                "candidate_ISRJ_recall": float(candidate_predicted[positive, ISRJ_INDEX].mean()),
            }
            row["ISRJ_recall_drop"] = row["reference_ISRJ_recall"] - row["candidate_ISRJ_recall"]
            run_rows.append(row)
            state_rows.append({
                "seed": seed, "power_dB": power, "expert": selected["expert"], "ablation": selected["ablation"],
                "positive_rescue_rate": float((state_action[:, 1] >= selected["confidence"]).mean()) if selected["expert"] != "B2-RC" else 0.0,
                "negative_protection_rate": float((state_action[:, 2] >= selected["confidence"]).mean()) if selected["expert"] != "B2-RC" else 0.0,
                "unknown_rate": float((state_action[:, 0] >= np.maximum(state_action[:, 1], state_action[:, 2])).mean()) if selected["expert"] != "B2-RC" else 1.0,
            })
            for bits in np.unique(labels[~positive], axis=0):
                mask = (~positive) & np.all(labels == bits, axis=1)
                combination_rows.append({
                    "seed": seed, "power_dB": power, "combination": _combo_name(bits), "support": int(mask.sum()),
                    "reference_fpr": float(reference_predicted[mask, ISRJ_INDEX].mean()),
                    "candidate_fpr": float(candidate_predicted[mask, ISRJ_INDEX].mean()),
                })
            print(json.dumps({"seed": seed, "power": power, "selected": {k: selected[k] for k in ("expert", "ablation")},
                              "macro_delta": row["delta_macro_aupr"], "fpr_reduction": row["ISRJ_fpr_reduction"],
                              "recall_drop": row["ISRJ_recall_drop"]}), flush=True)
        selected_bytes = 0
        if selected["expert"] != "B2-RC":
            selected_bytes += (b4_root / f"isrj_{selected['expert']}.joblib").stat().st_size
            selected_bytes += (seed_root / f"{selected['expert']}_{selected['feature_set']}_router.joblib").stat().st_size
        cost_rows.append({"seed": seed, "inference_seconds": inference_seconds, "router_fit_seconds": fit_seconds,
                          "deployment_incremental_bytes": selected_bytes})
    macro_delta = float(np.mean([row["delta_macro_aupr"] for row in run_rows]))
    hamming_delta = float(np.mean([row["delta_hamming"] for row in run_rows]))
    weak_delta = float(np.nanmean([row["delta_weak_recall"] for row in run_rows]))
    power_recall_drop = [float(np.mean([row["ISRJ_recall_drop"] for row in run_rows if row["power_dB"] == power]))
                         for power in POWER_LEVELS]
    seed_rows = []
    for seed in SEEDS:
        selected_rows = [row for row in run_rows if row["seed"] == seed]
        seed_rows.append({
            "seed": seed, "mean_macro_aupr_delta": float(np.mean([row["delta_macro_aupr"] for row in selected_rows])),
            "mean_hamming_delta": float(np.mean([row["delta_hamming"] for row in selected_rows])),
            "mean_ISRJ_fpr_reduction": float(np.mean([row["ISRJ_fpr_reduction"] for row in selected_rows])),
            "mean_ISRJ_aupr_delta": float(np.mean([row["delta_ISRJ_aupr"] for row in selected_rows])),
        })
    same_direction = sum(row["mean_macro_aupr_delta"] > 0 and row["mean_hamming_delta"] <= 0 for row in seed_rows)
    summary = {
        "mean_macro_aupr_delta": macro_delta, "mean_hamming_delta": hamming_delta,
        "mean_weak_recall_delta": weak_delta, "max_power_mean_ISRJ_recall_drop": max(power_recall_drop),
        "same_direction_seeds": int(same_direction),
        "mean_ISRJ_fpr_reduction": float(np.mean([row["ISRJ_fpr_reduction"] for row in run_rows])),
        "mean_ISRJ_aupr_delta": float(np.mean([row["delta_ISRJ_aupr"] for row in run_rows])),
        "seed_rows": seed_rows,
    }
    gates = {
        "macro_aupr_gain_at_least_0p005": macro_delta >= 0.005,
        "hamming_not_increased": hamming_delta <= 0.0,
        "weak_recall_drop_at_most_0p005": weak_delta >= -0.005,
        "max_power_ISRJ_recall_drop_at_most_0p005": max(power_recall_drop) <= 0.005,
        "same_direction_at_least_4_seeds": same_direction >= 4,
    }
    result = {
        "protocol": {"reference": "B2-RC", "seeds": list(SEEDS), "powers_dB": list(POWER_LEVELS),
                     "development_used_for_selection": False, "explicit_JNR_used": False,
                     "locked_test_used": False},
        "summary": summary, "gates": gates, "B5R_passed": bool(all(gates.values())),
        "cost": {"mean_inference_seconds_per_seed": float(np.mean([row["inference_seconds"] for row in cost_rows])),
                 "mean_router_fit_seconds_per_seed": float(np.mean([row["router_fit_seconds"] for row in cost_rows])),
                 "mean_deployment_incremental_bytes": float(np.mean([row["deployment_incremental_bytes"] for row in cost_rows])),
                 "max_deployment_incremental_bytes": int(max(row["deployment_incremental_bytes"] for row in cost_rows))},
        "stop_hard_negative_branch": not bool(all(gates.values())), "locked_test_used": False,
    }
    _write_csv(args.output_root / "power_runs.csv", run_rows)
    _write_csv(args.output_root / "selection.csv", selection_rows)
    _write_csv(args.output_root / "state_rates.csv", state_rows)
    _write_csv(args.output_root / "per_combination.csv", combination_rows)
    _write_csv(args.output_root / "cost.csv", cost_rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
