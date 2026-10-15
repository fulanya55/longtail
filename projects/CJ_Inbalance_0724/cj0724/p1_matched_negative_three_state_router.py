"""P1 repair: matched difficult-negative three-state routing for MFT/VFT/ISRJ."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cj0724 import CLASS_NAMES
from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment, discover_power_train_files, read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _level_metrics, _weak_mask
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
    _identity_split,
    _load_main,
    _load_unet,
    align_payloads,
    bounded_blend,
    classification_metrics,
    collect_main,
    expert_probabilities,
    extract_unet_role,
    fit_frozen_experts,
    merge_records,
    router_features,
)
from cj0724.oracle_swept_isolation_experiment import set_seed, sha256_indices
from cj0724.paper_core_evidence import load_normalizer


ACCEPT_EXPERT = 0
KEEP_MAIN = 1
ABSTAIN = 2


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


def split_three_ways(indices: np.ndarray, identities: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.unique(identities[indices])
    shuffled = np.random.default_rng(seed).permutation(values)
    first = len(shuffled) // 3
    second = 2 * len(shuffled) // 3
    groups = [set(shuffled[:first]), set(shuffled[first:second]), set(shuffled[second:])]
    output = []
    for group in groups:
        output.append(indices[np.asarray([value in group for value in identities[indices]], dtype=bool)].astype(np.int64))
    return output[0], output[1], output[2]


def matched_group_mask(labels: np.ndarray, powers: np.ndarray, target_index: int) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.uint8)
    selected_columns = [index for index in range(truth.shape[1]) if index != target_index]
    other = truth[:, selected_columns]
    weights = (1 << np.arange(other.shape[1], dtype=np.int64))[None, :]
    code = (other.astype(np.int64) * weights).sum(axis=1)
    mask = np.zeros(len(truth), dtype=bool)
    for power in np.unique(powers):
        for value in np.unique(code[powers == power]):
            group = (powers == power) & (code == value)
            target_values = np.unique(truth[group, target_index])
            if len(target_values) == 2:
                mask |= group
    return mask


def three_state_targets(
    labels: np.ndarray,
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    target_index: int,
    local_index: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    truth = labels[:, target_index].astype(np.float64)
    base = np.clip(main_probability[:, target_index].astype(np.float64), 1e-6, 1.0 - 1e-6)
    expert = np.clip(expert_probability[:, local_index].astype(np.float64), 1e-6, 1.0 - 1e-6)
    base_loss = -(truth * np.log(base) + (1.0 - truth) * np.log(1.0 - base))
    expert_loss = -(truth * np.log(expert) + (1.0 - truth) * np.log(1.0 - expert))
    gain = base_loss - expert_loss
    target = np.full(len(truth), ABSTAIN, dtype=np.uint8)
    target[gain >= margin] = ACCEPT_EXPERT
    target[gain <= -margin] = KEEP_MAIN
    return target, gain.astype(np.float32)


class ConstantThreeStateRouter:
    def __init__(self, state: int):
        self.state = int(state)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        output = np.zeros((len(features), 3), dtype=np.float64)
        output[:, self.state] = 1.0
        return output


def fit_three_state_routers(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    labels: np.ndarray,
    powers: np.ndarray,
    seed: int,
    margin: float,
    negative_weight: float,
) -> tuple[list[object], list[dict[str, object]]]:
    routers, audits = [], []
    for local_index, target_index in enumerate(TARGET_INDICES):
        matched = matched_group_mask(labels, powers, int(target_index))
        targets, gain = three_state_targets(
            labels, main_probability, expert_probability, int(target_index), local_index, margin
        )
        features = router_features(main_probability, expert_probability, local_index)
        sample_weight = np.ones(int(matched.sum()), dtype=np.float64)
        negative = labels[matched, target_index] == 0
        sample_weight[negative] = float(negative_weight)
        unique = np.unique(targets[matched])
        if len(unique) == 1:
            router = ConstantThreeStateRouter(int(unique[0]))
        else:
            router = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2500, random_state=seed),
            )
            router.fit(features[matched], targets[matched], logisticregression__sample_weight=sample_weight)
        routers.append(router)
        counts = np.bincount(targets[matched], minlength=3)
        audits.append({
            "target": TARGET_NAMES[local_index],
            "matched_samples": int(matched.sum()),
            "matched_positive_samples": int((matched & (labels[:, target_index] == 1)).sum()),
            "matched_negative_samples": int((matched & (labels[:, target_index] == 0)).sum()),
            "accept_expert": int(counts[ACCEPT_EXPERT]),
            "keep_main": int(counts[KEEP_MAIN]),
            "abstain": int(counts[ABSTAIN]),
            "mean_accept_gain": float(gain[matched & (targets == ACCEPT_EXPERT)].mean()) if np.any(matched & (targets == ACCEPT_EXPERT)) else 0.0,
            "mean_keep_main_gain": float(gain[matched & (targets == KEEP_MAIN)].mean()) if np.any(matched & (targets == KEEP_MAIN)) else 0.0,
        })
    return routers, audits


def state_probabilities(routers: list[object], main_probability: np.ndarray, expert_probability: np.ndarray) -> np.ndarray:
    output = np.zeros((len(main_probability), len(TARGET_INDICES), 3), dtype=np.float32)
    for local_index, router in enumerate(routers):
        raw = router.predict_proba(router_features(main_probability, expert_probability, local_index))
        if raw.shape[1] == 3:
            output[:, local_index] = raw
        else:
            classes = router[-1].classes_.astype(np.int64)
            output[:, local_index, classes] = raw
    return output


def routed_probability(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    states: np.ndarray,
    alpha: float,
    accept_threshold: float,
) -> np.ndarray:
    route = states[:, :, ACCEPT_EXPERT]
    return bounded_blend(main_probability, expert_probability, route, alpha, accept_threshold)


def threshold_at_fpr(labels: np.ndarray, probability: np.ndarray, cap: float) -> float:
    truth = labels.astype(bool)
    best = (-1.0, -1.0, 1.0)
    for threshold in np.linspace(0.001, 0.999, 999):
        predicted = probability >= threshold
        positive, negative = truth, ~truth
        recall = float(predicted[positive].mean()) if positive.any() else 0.0
        fpr = float(predicted[negative].mean()) if negative.any() else 0.0
        if fpr <= cap + 1e-12:
            candidate = (recall, -fpr, float(threshold))
            if candidate > best:
                best = candidate
    return float(best[2])


def recalibrate_target_thresholds(
    records: list[dict[str, np.ndarray]],
    original_thresholds: np.ndarray,
    alpha: float,
    accept_threshold: float,
    cap: float,
) -> np.ndarray:
    labels = np.concatenate([record["labels"] for record in records])
    probabilities = np.concatenate([
        routed_probability(record["probabilities"], record["expert"], record["states"], alpha, accept_threshold)
        for record in records
    ])
    thresholds = np.asarray(original_thresholds, dtype=np.float64).copy()
    for target_index in TARGET_INDICES:
        thresholds[target_index] = threshold_at_fpr(labels[:, target_index], probabilities[:, target_index], cap)
    return thresholds.astype(np.float32)


def score_records(
    records: list[dict[str, np.ndarray]],
    thresholds: np.ndarray,
    alpha: float,
    accept_threshold: float,
) -> dict[str, object]:
    power_metrics = []
    for record in records:
        final = routed_probability(record["probabilities"], record["expert"], record["states"], alpha, accept_threshold)
        power_metrics.append(classification_metrics(record["labels"], final, thresholds))
    extreme_recall = float(np.mean([
        metric["per_class"][target_index]["recall"]
        for metric, record in zip(power_metrics, records)
        if int(record["power"][0]) in (10, 50)
        for target_index in TARGET_INDICES
    ]))
    return {
        "extreme_recall": extreme_recall,
        "mean_macro_aupr": float(np.mean([metric["macro_aupr"] for metric in power_metrics])),
        "target_fpr": {
            CLASS_NAMES[target_index]: float(np.mean([metric["per_class"][target_index]["fpr"] for metric in power_metrics]))
            for target_index in TARGET_INDICES
        },
    }


def select_candidate(
    threshold_records: list[dict[str, np.ndarray]],
    safety_records: list[dict[str, np.ndarray]],
    original_thresholds: np.ndarray,
    recalibration_cap: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    fixed_baseline = score_records(safety_records, original_thresholds, 0.0, 1.0)
    recalibrated_base_thresholds = recalibrate_target_thresholds(
        threshold_records, original_thresholds, 0.0, 1.0, recalibration_cap
    )
    recalibrated_baseline = score_records(safety_records, recalibrated_base_thresholds, 0.0, 1.0)
    rows = []
    for alpha in (0.25, 0.5, 0.75):
        for accept_threshold in (0.5, 0.6, 0.7, 0.8):
            fixed = score_records(safety_records, original_thresholds, alpha, accept_threshold)
            recalibrated_thresholds = recalibrate_target_thresholds(
                threshold_records, original_thresholds, alpha, accept_threshold, recalibration_cap
            )
            recalibrated = score_records(safety_records, recalibrated_thresholds, alpha, accept_threshold)
            fixed_fpr_noninferior = all(
                fixed["target_fpr"][name] <= fixed_baseline["target_fpr"][name] + 0.005
                for name in TARGET_NAMES
            )
            absolute_fpr_safe = max(recalibrated["target_fpr"].values()) <= 0.10 + 1e-12
            row = {
                "alpha": alpha,
                "accept_threshold": accept_threshold,
                "fixed": fixed,
                "recalibrated": recalibrated,
                "recalibrated_thresholds": recalibrated_thresholds.tolist(),
                "fixed_macro_aupr_delta": fixed["mean_macro_aupr"] - fixed_baseline["mean_macro_aupr"],
                "recalibrated_extreme_recall_gain": recalibrated["extreme_recall"] - recalibrated_baseline["extreme_recall"],
                "fixed_fpr_noninferior": fixed_fpr_noninferior,
                "absolute_fpr_safe": absolute_fpr_safe,
            }
            row["eligible"] = bool(
                row["fixed_macro_aupr_delta"] >= -0.005
                and fixed_fpr_noninferior
                and absolute_fpr_safe
            )
            rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["recalibrated"]["extreme_recall"],
                row["fixed"]["mean_macro_aupr"],
                -max(row["recalibrated"]["target_fpr"].values()),
            ),
        )
    else:
        selected = {
            "alpha": 0.0,
            "accept_threshold": 1.0,
            "fixed": fixed_baseline,
            "recalibrated": recalibrated_baseline,
            "recalibrated_thresholds": recalibrated_base_thresholds.tolist(),
            "fixed_macro_aupr_delta": 0.0,
            "recalibrated_extreme_recall_gain": 0.0,
            "fixed_fpr_noninferior": True,
            "absolute_fpr_safe": max(recalibrated_baseline["target_fpr"].values()) <= 0.10,
            "eligible": False,
            "fallback_reason": "no matched-negative candidate passed both relative and absolute FPR safety",
        }
    baseline = {
        "fixed": fixed_baseline,
        "recalibrated": recalibrated_baseline,
        "recalibrated_thresholds": recalibrated_base_thresholds.tolist(),
    }
    return selected, rows, baseline


def evaluate_tune(
    records: list[dict[str, np.ndarray]],
    original_thresholds: np.ndarray,
    selected: dict[str, object],
    scale: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    recalibrated_thresholds = np.asarray(selected["recalibrated_thresholds"], dtype=np.float32)
    rows, saved = [], {"main": [], "final": [], "labels": [], "sample_indices": [], "powers": [], "prediction": [], "target": [], "mask": []}
    for record in records:
        final = routed_probability(record["probabilities"], record["expert"], record["states"], selected["alpha"], selected["accept_threshold"])
        fixed_base = classification_metrics(record["labels"], record["probabilities"], original_thresholds)
        fixed_final = classification_metrics(record["labels"], final, original_thresholds)
        recal_base = classification_metrics(record["labels"], record["probabilities"], recalibrated_thresholds)
        recal_final = classification_metrics(record["labels"], final, recalibrated_thresholds)
        common = {"sample_indices": record["sample_indices"], "labels": record["labels"], "prediction": record["prediction"], "target": record["target"], "mask": record["mask"], "thresholds": recalibrated_thresholds, "scale": scale}
        weak = _weak_mask(record["labels"].astype(bool), int(record["power"][0]))
        base_gate = _level_metrics({**common, "probabilities": record["probabilities"]}, weak)
        final_gate = _level_metrics({**common, "probabilities": final}, weak)
        rows.append({
            "power_level_db": int(record["power"][0]),
            "fixed_base": fixed_base,
            "fixed_final": fixed_final,
            "recalibrated_base": recal_base,
            "recalibrated_final": recal_final,
            "base_parameter_gate": base_gate,
            "final_parameter_gate": final_gate,
            "accept_coverage": {name: float((record["states"][:, local, ACCEPT_EXPERT] >= selected["accept_threshold"]).mean()) if selected["alpha"] > 0 else 0.0 for local, name in enumerate(TARGET_NAMES)},
        })
        for key, value in (("main", record["probabilities"]), ("final", final), ("labels", record["labels"]), ("sample_indices", record["sample_indices"]), ("powers", record["power"]), ("prediction", record["prediction"]), ("target", record["target"]), ("mask", record["mask"])):
            saved[key].append(value)
    saved = {key: np.concatenate(value) for key, value in saved.items()}

    def extreme(field: str) -> float:
        return float(np.mean([
            row[field]["per_class"][target_index]["recall"]
            for row in rows if row["power_level_db"] in (10, 50) for target_index in TARGET_INDICES
        ]))

    fixed_base_fpr = {CLASS_NAMES[index]: float(np.mean([row["fixed_base"]["per_class"][index]["fpr"] for row in rows])) for index in TARGET_INDICES}
    fixed_final_fpr = {CLASS_NAMES[index]: float(np.mean([row["fixed_final"]["per_class"][index]["fpr"] for row in rows])) for index in TARGET_INDICES}
    recal_final_fpr = {CLASS_NAMES[index]: float(np.mean([row["recalibrated_final"]["per_class"][index]["fpr"] for row in rows])) for index in TARGET_INDICES}
    aggregate = {
        "fixed_macro_aupr_delta": float(np.mean([row["fixed_final"]["macro_aupr"] - row["fixed_base"]["macro_aupr"] for row in rows])),
        "fixed_base_target_fpr": fixed_base_fpr,
        "fixed_final_target_fpr": fixed_final_fpr,
        "recalibrated_final_target_fpr": recal_final_fpr,
        "recalibrated_extreme_recall_base": extreme("recalibrated_base"),
        "recalibrated_extreme_recall_final": extreme("recalibrated_final"),
        "recalibrated_extreme_recall_gain": extreme("recalibrated_final") - extreme("recalibrated_base"),
        "mean_parameter_coverage_delta": float(np.mean([row["final_parameter_gate"]["predicted_gate_parameter_coverage"] - row["base_parameter_gate"]["predicted_gate_parameter_coverage"] for row in rows])),
        "mean_miss_penalty_nmae_delta": float(np.mean([row["final_parameter_gate"]["miss_penalty_parameter_nmae"] - row["base_parameter_gate"]["miss_penalty_parameter_nmae"] for row in rows])),
    }
    return rows, aggregate, saved


def paired_identity_bootstrap(seed_payloads: list[dict[str, np.ndarray]], replicates: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    recall_deltas, aupr_deltas = [], []
    for _ in range(replicates):
        seed_recall, seed_aupr = [], []
        for payload in seed_payloads:
            identities = np.unique(payload["sample_indices"])
            sampled = identities[rng.integers(0, len(identities), len(identities))]
            positions = np.concatenate([np.flatnonzero(payload["sample_indices"] == identity) for identity in sampled])
            labels = payload["labels"][positions]
            powers = payload["powers"][positions]
            main = payload["main"][positions]
            final = payload["final"][positions]
            thresholds = payload["thresholds"]
            main_predicted = main >= thresholds[None, :]
            final_predicted = final >= thresholds[None, :]
            base_values, final_values = [], []
            for power in (10, 50):
                for target_index in TARGET_INDICES:
                    selected = (powers == power) & labels[:, target_index].astype(bool)
                    base_values.append(float(main_predicted[selected, target_index].mean()))
                    final_values.append(float(final_predicted[selected, target_index].mean()))
            seed_recall.append(float(np.mean(final_values) - np.mean(base_values)))
            seed_aupr.append(float(sk_metrics.average_precision_score(labels, final, average="macro") - sk_metrics.average_precision_score(labels, main, average="macro")))
        recall_deltas.append(float(np.mean(seed_recall)))
        aupr_deltas.append(float(np.mean(seed_aupr)))

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values)
        return {"mean": float(array.mean()), "ci95_low": float(np.quantile(array, 0.025)), "ci95_high": float(np.quantile(array, 0.975)), "probability_positive": float((array > 0).mean()), "replicates": int(replicates)}

    return {"extreme_recall_delta": summarize(recall_deltas), "macro_aupr_delta": summarize(aupr_deltas)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/p1_matched_negative_router_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument("--recalibration-fpr-cap", type=float, default=0.08)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    run_root = project / args.run_root
    separator_root = project / args.separator_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles_file = np.load(run_root / "role_indices.npz")
    roles = {name: roles_file[name].astype(np.int64) for name in ("train", "early", "calibration", "tune")}
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    root_identities, root_labels = read_identity_metadata(protocol["data_file"])
    power_identities, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_identities, power_identities) or not np.array_equal(root_labels, power_labels):
        raise RuntimeError("root and power identities differ")
    selected_rows = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    normalizer = load_normalizer(protocol["normalizer"])
    receipt = {
        "purpose": "matched difficult-negative three-state repair for correctly mapped MFT/VFT/ISRJ experts",
        "label_mapping": {"slice": "3:6", "targets": list(TARGET_NAMES), "global_indices": TARGET_INDICES.tolist()},
        "seeds": args.seeds,
        "powers_dB": list(POWER_LEVELS),
        "router_states": ["accept_expert", "keep_main", "abstain"],
        "matched_rule": "same power and identical five non-target labels; require both target-positive and target-negative",
        "margin_nats": args.margin,
        "negative_sample_weight": args.negative_weight,
        "calibration_roles": "identity-disjoint router-fit / threshold-calibration / safety-selection thirds",
        "relative_safety": "fixed original thresholds; each target FPR <= baseline + 0.005",
        "absolute_safety": f"target thresholds recalibrated at FPR cap {args.recalibration_fpr_cap}; safety target FPR <= 0.10",
        "confirm_used": False,
        "locked_test_used": False,
    }
    (output_root / "protocol.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    results, selection_rows, power_csv_rows, bootstrap_payloads = [], [], [], []
    started = time.perf_counter()
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        main_model, checkpoint_path = _load_main(selected_rows[seed], protocol, project, device)
        unet_path = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_path, device)
        threshold_value = selected_rows[seed]["thresholds"]
        original_thresholds = np.asarray(json.loads(threshold_value) if isinstance(threshold_value, str) else threshold_value, dtype=np.float32)
        train_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["train"], args.batch_size, seed, device)
        early_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["early"], args.batch_size, seed, device)
        experts = fit_frozen_experts(train_unet, early_unet, seed)
        fit_indices, threshold_indices, safety_indices = split_three_ways(roles["calibration"], root_identities, seed + 9000)
        role_records = {"fit": [], "threshold": [], "safety": [], "tune": []}
        for role_name, indices in (("fit", fit_indices), ("threshold", threshold_indices), ("safety", safety_indices), ("tune", roles["tune"])):
            for power in POWER_LEVELS:
                main = collect_main(main_model, files[power], indices, args.batch_size, device, normalizer)
                unet_payload = extract_unet_role(unet, files[power], files[50], root_labels, indices, args.batch_size, seed + power, device)
                align_payloads(main, unet_payload)
                role_records[role_name].append({
                    **{name: main[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
                    "expert": expert_probabilities(unet_payload, experts),
                    "power": np.full(len(indices), power, dtype=np.int16),
                })
                print(json.dumps({"seed": seed, "role": role_name, "power_dB": power, "samples": len(indices)}), flush=True)
        fit_merged = merge_records([{**record, "route": np.zeros((len(record["labels"]), 3), dtype=np.float32)} for record in role_records["fit"]])
        routers, router_audit = fit_three_state_routers(
            fit_merged["probabilities"], fit_merged["expert"], fit_merged["labels"], fit_merged["power"], seed, args.margin, args.negative_weight
        )
        for role_name in role_records:
            for record in role_records[role_name]:
                record["states"] = state_probabilities(routers, record["probabilities"], record["expert"])
        selected, candidates, baseline = select_candidate(
            role_records["threshold"], role_records["safety"], original_thresholds, args.recalibration_fpr_cap
        )
        for row in candidates:
            selection_rows.append({"seed": seed, "alpha": row["alpha"], "accept_threshold": row["accept_threshold"], "fixed_macro_aupr_delta": row["fixed_macro_aupr_delta"], "recalibrated_extreme_recall_gain": row["recalibrated_extreme_recall_gain"], "fixed_fpr_noninferior": row["fixed_fpr_noninferior"], "absolute_fpr_safe": row["absolute_fpr_safe"], "eligible": row["eligible"], **{f"fixed_{name}_fpr": row["fixed"]["target_fpr"][name] for name in TARGET_NAMES}, **{f"recal_{name}_fpr": row["recalibrated"]["target_fpr"][name] for name in TARGET_NAMES}})
        tune_rows, tune_aggregate, saved = evaluate_tune(role_records["tune"], original_thresholds, selected, normalizer.scale.numpy())
        for row in tune_rows:
            power_csv_rows.append({"seed": seed, "power_dB": row["power_level_db"], "fixed_macro_aupr_delta": row["fixed_final"]["macro_aupr"] - row["fixed_base"]["macro_aupr"], **{f"fixed_base_{name}_fpr": row["fixed_base"]["per_class"][index]["fpr"] for index, name in zip(TARGET_INDICES, TARGET_NAMES)}, **{f"fixed_final_{name}_fpr": row["fixed_final"]["per_class"][index]["fpr"] for index, name in zip(TARGET_INDICES, TARGET_NAMES)}, **{f"recal_base_{name}_recall": row["recalibrated_base"]["per_class"][index]["recall"] for index, name in zip(TARGET_INDICES, TARGET_NAMES)}, **{f"recal_final_{name}_recall": row["recalibrated_final"]["per_class"][index]["recall"] for index, name in zip(TARGET_INDICES, TARGET_NAMES)}, **{f"recal_final_{name}_fpr": row["recalibrated_final"]["per_class"][index]["fpr"] for index, name in zip(TARGET_INDICES, TARGET_NAMES)}})
        recal_thresholds = np.asarray(selected["recalibrated_thresholds"], dtype=np.float32)
        np.savez_compressed(seed_dir / "tune_per_sample.npz", sample_index=saved["sample_indices"], receiver_noise_seed=root_identities[saved["sample_indices"]], power_db=saved["powers"], labels=saved["labels"].astype(np.uint8), main_probability=saved["main"], final_probability=saved["final"], recalibrated_thresholds=recal_thresholds, parameter_prediction=saved["prediction"], parameter_target=saved["target"], parameter_mask=saved["mask"].astype(np.uint8))
        bootstrap_payloads.append({**saved, "thresholds": recal_thresholds})
        result = {
            "seed": seed,
            "main_checkpoint": str(checkpoint_path),
            "unet_checkpoint": str(unet_path),
            "identity_audit": {
                "router_fit": {"count": len(fit_indices), "sha256": sha256_indices(fit_indices)},
                "threshold_calibration": {"count": len(threshold_indices), "sha256": sha256_indices(threshold_indices)},
                "safety_selection": {"count": len(safety_indices), "sha256": sha256_indices(safety_indices)},
                "pairwise_overlap": 0,
                "tune_overlap": 0,
            },
            "router_audit": router_audit,
            "baseline": baseline,
            "selected": selected,
            "tune_by_power": tune_rows,
            "tune_aggregate": tune_aggregate,
            "confirm_used": False,
            "locked_test_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts, routers, role_records
        if device.type == "cuda": torch.cuda.empty_cache()

    bootstrap = paired_identity_bootstrap(bootstrap_payloads, args.bootstrap_replicates, 20260727)
    gains = [row["tune_aggregate"]["recalibrated_extreme_recall_gain"] for row in results]
    macro_deltas = [row["tune_aggregate"]["fixed_macro_aupr_delta"] for row in results]
    relative_safe = all(
        all(row["tune_aggregate"]["fixed_final_target_fpr"][name] <= row["tune_aggregate"]["fixed_base_target_fpr"][name] + 0.005 for name in TARGET_NAMES)
        for row in results
    )
    absolute_safe = all(max(row["tune_aggregate"]["recalibrated_final_target_fpr"].values()) <= 0.10 + 1e-12 for row in results)
    gates = {
        "mean_extreme_recall_gain_at_least_0p10": float(np.mean(gains)) >= 0.10,
        "three_of_three_positive_direction": all(value > 0 for value in gains),
        "mean_macro_aupr_loss_at_most_0p005": float(np.mean(macro_deltas)) >= -0.005,
        "fixed_threshold_FPR_noninferior": relative_safe,
        "recalibrated_absolute_FPR_at_most_0p10": absolute_safe,
        "bootstrap_recall_CI_positive": bootstrap["extreme_recall_delta"]["ci95_low"] > 0.0,
    }
    summary_payload = {
        "experiment": "P1 matched difficult-negative three-state router",
        "label_mapping": "MFT/VFT/ISRJ = global indices 3/4/5",
        "aggregate": {
            "mean_extreme_recall_gain": float(np.mean(gains)),
            "per_seed_extreme_recall_gain": gains,
            "mean_macro_aupr_delta": float(np.mean(macro_deltas)),
            "relative_FPR_safe": relative_safe,
            "absolute_FPR_safe": absolute_safe,
            "mean_parameter_coverage_delta": float(np.mean([row["tune_aggregate"]["mean_parameter_coverage_delta"] for row in results])),
            "mean_miss_penalty_nmae_delta": float(np.mean([row["tune_aggregate"]["mean_miss_penalty_nmae_delta"] for row in results])),
        },
        "bootstrap": bootstrap,
        "gates": gates,
        "promotion_passed": bool(all(gates.values())),
        "identity_alignment": alignment,
        "elapsed_seconds": float(time.perf_counter() - started),
        "confirm_used": False,
        "locked_test_used": False,
    }
    _write_csv(output_root / "selection.csv", selection_rows)
    _write_csv(output_root / "per_power.csv", power_csv_rows)
    (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
