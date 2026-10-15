"""P1: safely route frozen MFT/VFT/ISRJ experts into the frozen six-class MLCNN.

The legacy swept-isolation code selects labels[:, 3:6].  Consequently its
three outputs are MFT, VFT and ISRJ; this module deliberately corrects the
older SawFM name attached to output zero and never routes it to SawFM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES
from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.data import V73IQDataset
from cj0724.dual_axis import (
    POWER_LEVELS,
    audit_power_alignment,
    discover_power_train_files,
    read_identity_metadata,
)
from cj0724.identity_safe_power_parameter_eval import _collect, _level_metrics, _weak_mask
from cj0724.isrj_separator_physics_head import clipped_logit, feature_view, fit_physics_models
from cj0724.multitask import PARAMETER_COLUMNS, build_multitask_model
from cj0724.oracle_swept_isolation_experiment import SweptIsolationDataset, make_loader, set_seed, sha256_indices
from cj0724.paper_core_evidence import load_normalizer
from cj0724.shared_physics_frontend import shared_feature_batch
from cj0724.vft_separator_frequency_head import instantaneous_frequency_features


TARGET_INDICES = np.asarray([3, 4, 5], dtype=np.int64)
TARGET_NAMES = ("MFT", "VFT", "ISRJ")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _load_unet(checkpoint: Path, device: torch.device) -> ConditionalComplexUNetSeparator:
    model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_unet_role(
    model: ConditionalComplexUNetSeparator,
    power_file: Path,
    file_50: Path,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = SweptIsolationDataset(power_file, file_50, labels, indices, "raw_10db")
    loader = make_loader(dataset, batch_size, False, seed)
    output = {
        "sample_indices": [], "labels": [], "probabilities": [],
        "mft": [], "trajectory": [], "isrj": [],
    }
    start = time.perf_counter()
    for batch in loader:
        components = model.forward_components(batch["iq"].to(device, non_blocking=True), None)
        separated = components["normalized_estimate"].float()
        output["sample_indices"].append(batch["sample_index"].numpy())
        output["labels"].append(batch["label"].numpy())
        output["probabilities"].append(torch.sigmoid(components["logits"].float()).cpu().numpy())
        physics = shared_feature_batch(separated)
        output["mft"].append(physics["mft_delay"].cpu().numpy())
        output["trajectory"].append(
            instantaneous_frequency_features(separated)["if_trajectory"].cpu().numpy()
        )
        output["isrj"].append(physics["isrj_periodicity"].cpu().numpy())
    dataset.close()
    result = {
        name: np.concatenate(parts).astype(
            np.int64 if name == "sample_indices" else np.uint8 if name == "labels" else np.float32
        )
        for name, parts in output.items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - start)
    return result


def fit_frozen_experts(train: dict[str, np.ndarray], early: dict[str, np.ndarray], seed: int):
    """Reconstruct the already-selected VFT and ISRJ expert configurations."""
    vft_physics = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    )
    vft_physics.fit(train["trajectory"], train["labels"][:, 1])
    vft_early = vft_physics.predict_proba(early["trajectory"])[:, 1]
    vft_stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    vft_stacker.fit(
        np.column_stack((clipped_logit(early["probabilities"][:, 1]), clipped_logit(vft_early))),
        early["labels"][:, 1],
    )

    isrj_train = {"features": train["isrj"], "labels": train["labels"]}
    isrj_models = fit_physics_models(isrj_train, seed)
    isrj_names = sorted(isrj_models)
    isrj_early = {
        name: model.predict_proba(feature_view(early["isrj"], view))[:, 1]
        for name, (model, view) in isrj_models.items()
    }
    isrj_stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    isrj_stacker.fit(
        np.column_stack(
            [clipped_logit(early["probabilities"][:, 2])]
            + [clipped_logit(isrj_early[name]) for name in isrj_names]
        ),
        early["labels"][:, 2],
    )
    return {
        "vft_physics": vft_physics,
        "vft_stacker": vft_stacker,
        "isrj_models": isrj_models,
        "isrj_names": isrj_names,
        "isrj_stacker": isrj_stacker,
    }


def expert_probabilities(payload: dict[str, np.ndarray], experts: dict[str, object]) -> np.ndarray:
    vft_physics = experts["vft_physics"].predict_proba(payload["trajectory"])[:, 1]
    vft_matrix = np.column_stack(
        (clipped_logit(payload["probabilities"][:, 1]), clipped_logit(vft_physics))
    )
    vft = experts["vft_stacker"].predict_proba(vft_matrix)[:, 1]
    isrj_physics = {
        name: model.predict_proba(feature_view(payload["isrj"], view))[:, 1]
        for name, (model, view) in experts["isrj_models"].items()
    }
    isrj_matrix = np.column_stack(
        [clipped_logit(payload["probabilities"][:, 2])]
        + [clipped_logit(isrj_physics[name]) for name in experts["isrj_names"]]
    )
    isrj = experts["isrj_stacker"].predict_proba(isrj_matrix)[:, 1]
    return np.column_stack((payload["probabilities"][:, 0], vft, isrj)).astype(np.float32)


def _load_main(row: dict[str, object], protocol: dict[str, object], project: Path, device: torch.device):
    model = build_multitask_model(
        row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
        protocol["parameter_hidden_dim"], protocol["legacy_attention"],
    ).to(device)
    checkpoint_path = Path(str(row["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = project / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_path


def collect_main(
    model, power_file: Path, indices: np.ndarray, batch_size: int, device: torch.device, normalizer
) -> dict[str, np.ndarray]:
    dataset = V73IQDataset(power_file, indices, PARAMETER_COLUMNS)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
    )
    start = time.perf_counter()
    output = _collect(model, loader, device, normalizer)
    dataset.close()
    output["probabilities"] = torch.sigmoid(torch.from_numpy(output["logits"])).numpy().astype(np.float32)
    output["elapsed_seconds"] = float(time.perf_counter() - start)
    return output


def align_payloads(main: dict[str, np.ndarray], unet: dict[str, np.ndarray]) -> None:
    if not np.array_equal(main["sample_indices"], unet["sample_indices"]):
        raise RuntimeError("main and expert sample order differ")
    if not np.array_equal(main["labels"][:, TARGET_INDICES].astype(np.uint8), unet["labels"]):
        raise RuntimeError("swept label mapping is not global labels[:, 3:6]")


def router_features(main_probability: np.ndarray, expert_probability: np.ndarray, target: int) -> np.ndarray:
    base = main_probability[:, TARGET_INDICES]
    return np.column_stack(
        (
            main_probability,
            expert_probability,
            base[:, target],
            expert_probability[:, target],
            np.abs(expert_probability[:, target] - base[:, target]),
            np.minimum(base[:, target], 1.0 - base[:, target]),
            np.minimum(expert_probability[:, target], 1.0 - expert_probability[:, target]),
        )
    ).astype(np.float32)


class ConstantRouter:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        value = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - value, value))


def fit_routers(main_probability: np.ndarray, expert_probability: np.ndarray, labels: np.ndarray, seed: int):
    routers, prevalence = [], []
    epsilon, margin = 1e-6, 0.05
    for local, global_index in enumerate(TARGET_INDICES):
        truth = labels[:, global_index].astype(np.float64)
        base = np.clip(main_probability[:, global_index], epsilon, 1.0 - epsilon)
        expert = np.clip(expert_probability[:, local], epsilon, 1.0 - epsilon)
        base_loss = -(truth * np.log(base) + (1.0 - truth) * np.log(1.0 - base))
        expert_loss = -(truth * np.log(expert) + (1.0 - truth) * np.log(1.0 - expert))
        advantage = (expert_loss + margin < base_loss).astype(np.uint8)
        prevalence.append(float(advantage.mean()))
        if np.unique(advantage).size < 2:
            routers.append(ConstantRouter(float(advantage[0])))
        else:
            router = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
            )
            router.fit(router_features(main_probability, expert_probability, local), advantage)
            routers.append(router)
    return routers, prevalence


def route_probabilities(routers, main_probability: np.ndarray, expert_probability: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            router.predict_proba(router_features(main_probability, expert_probability, local))[:, 1]
            for local, router in enumerate(routers)
        ]
    ).astype(np.float32)


def bounded_blend(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    route_probability: np.ndarray,
    alpha: float,
    route_threshold: float,
) -> np.ndarray:
    final = np.asarray(main_probability, dtype=np.float32).copy()
    if alpha <= 0.0:
        return final
    active = route_probability >= float(route_threshold)
    base = final[:, TARGET_INDICES]
    blended = base + float(alpha) * active * (expert_probability - base)
    final[:, TARGET_INDICES] = np.clip(blended, 0.0, 1.0)
    return final


def classification_metrics(labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray):
    truth = labels.astype(bool)
    predicted = probability >= thresholds[None, :]
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        positive, negative = truth[:, index], ~truth[:, index]
        per_class.append({
            "class": name,
            "aupr": float(sk_metrics.average_precision_score(positive, probability[:, index])),
            "recall": float(predicted[positive, index].mean()) if positive.any() else float("nan"),
            "fpr": float(predicted[negative, index].mean()) if negative.any() else float("nan"),
        })
    return {
        "macro_aupr": float(np.mean([row["aupr"] for row in per_class])),
        "macro_f1": float(sk_metrics.f1_score(truth.ravel(), predicted.ravel())),
        "hamming_loss": float(np.not_equal(truth, predicted).mean()),
        "subset_accuracy": float(np.all(truth == predicted, axis=1).mean()),
        "per_class": per_class,
    }


def merge_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = ("sample_indices", "labels", "probabilities", "prediction", "target", "mask", "expert", "route", "power")
    return {name: np.concatenate([record[name] for record in records]) for name in names}


def candidate_score(records, thresholds, alpha, route_threshold):
    by_power = []
    routed_fpr = []
    for record in records:
        final = bounded_blend(record["probabilities"], record["expert"], record["route"], alpha, route_threshold)
        metric = classification_metrics(record["labels"], final, thresholds)
        by_power.append((int(record["power"][0]), metric))
    extreme = [metric for power, metric in by_power if power in (10, 50)]
    recall = float(np.mean([
        metric["per_class"][index]["recall"] for metric in extreme for index in TARGET_INDICES
    ]))
    macro = float(np.mean([metric["macro_aupr"] for _, metric in by_power]))
    for index in TARGET_INDICES:
        routed_fpr.append(float(np.mean([metric["per_class"][index]["fpr"] for _, metric in by_power])))
    return {"extreme_recall": recall, "macro_aupr": macro, "routed_fpr": routed_fpr}


def select_blend(records, thresholds):
    baseline = candidate_score(records, thresholds, 0.0, 1.0)
    rows = []
    for alpha in (0.25, 0.5, 0.75):
        for route_threshold in (0.5, 0.6, 0.7, 0.8):
            score = candidate_score(records, thresholds, alpha, route_threshold)
            score.update({"alpha": alpha, "route_threshold": route_threshold})
            score["macro_aupr_delta"] = score["macro_aupr"] - baseline["macro_aupr"]
            score["extreme_recall_gain"] = score["extreme_recall"] - baseline["extreme_recall"]
            score["eligible"] = bool(
                score["macro_aupr_delta"] >= -0.005
                and max(score["routed_fpr"]) <= 0.10 + 1e-12
            )
            rows.append(score)
    eligible = [row for row in rows if row["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda row: (row["extreme_recall"], row["macro_aupr"], -max(row["routed_fpr"])))
    else:
        selected = {
            **baseline, "alpha": 0.0, "route_threshold": 1.0,
            "macro_aupr_delta": 0.0, "extreme_recall_gain": 0.0,
            "eligible": False, "fallback_reason": "no preregistered candidate met safety constraints",
        }
    return selected, rows, baseline


def evaluate_tune(records, thresholds, alpha, route_threshold, scale):
    rows = []
    for record in records:
        final = bounded_blend(record["probabilities"], record["expert"], record["route"], alpha, route_threshold)
        base_class = classification_metrics(record["labels"], record["probabilities"], thresholds)
        final_class = classification_metrics(record["labels"], final, thresholds)
        common = {
            "sample_indices": record["sample_indices"], "labels": record["labels"],
            "prediction": record["prediction"], "target": record["target"], "mask": record["mask"],
            "thresholds": thresholds, "scale": scale,
        }
        weak = _weak_mask(record["labels"].astype(bool), int(record["power"][0]))
        base_gate = _level_metrics({**common, "probabilities": record["probabilities"]}, weak)
        final_gate = _level_metrics({**common, "probabilities": final}, weak)
        rows.append({
            "power_level_db": int(record["power"][0]),
            "base_classification": base_class,
            "final_classification": final_class,
            "base_parameter_gate": base_gate,
            "final_parameter_gate": final_gate,
            "route_coverage": {
                name: float((record["route"][:, local] >= route_threshold).mean()) if alpha > 0 else 0.0
                for local, name in enumerate(TARGET_NAMES)
            },
        })
    extreme_base = float(np.mean([
        row["base_classification"]["per_class"][index]["recall"]
        for row in rows if row["power_level_db"] in (10, 50) for index in TARGET_INDICES
    ]))
    extreme_final = float(np.mean([
        row["final_classification"]["per_class"][index]["recall"]
        for row in rows if row["power_level_db"] in (10, 50) for index in TARGET_INDICES
    ]))
    macro_delta = float(np.mean([
        row["final_classification"]["macro_aupr"] - row["base_classification"]["macro_aupr"] for row in rows
    ]))
    routed_fpr = {
        CLASS_NAMES[index]: float(np.mean([row["final_classification"]["per_class"][index]["fpr"] for row in rows]))
        for index in TARGET_INDICES
    }
    return rows, {
        "extreme_three_class_recall_base": extreme_base,
        "extreme_three_class_recall_final": extreme_final,
        "extreme_three_class_recall_gain": extreme_final - extreme_base,
        "mean_nine_power_macro_aupr_delta": macro_delta,
        "mean_routed_class_fpr": routed_fpr,
        "mean_parameter_coverage_delta": float(np.mean([
            row["final_parameter_gate"]["predicted_gate_parameter_coverage"]
            - row["base_parameter_gate"]["predicted_gate_parameter_coverage"] for row in rows
        ])),
        "mean_miss_penalty_parameter_nmae_delta": float(np.mean([
            row["final_parameter_gate"]["miss_penalty_parameter_nmae"]
            - row["base_parameter_gate"]["miss_penalty_parameter_nmae"] for row in rows
        ])),
    }


def _identity_split(indices: np.ndarray, identities: np.ndarray, seed: int):
    values = np.unique(identities[indices])
    permutation = np.random.default_rng(seed).permutation(values)
    fit_ids = set(permutation[: len(permutation) // 2].tolist())
    fit = indices[np.asarray([value in fit_ids for value in identities[indices]], dtype=bool)]
    safety = indices[np.asarray([value not in fit_ids for value in identities[indices]], dtype=bool)]
    return fit.astype(np.int64), safety.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/p1_swept_expert_router_20260726")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
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
        raise RuntimeError("root and power-sweep identities are not aligned")
    normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {
        int(row["seed"]): row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    }
    results = []
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        main_model, main_checkpoint = _load_main(selected_rows[seed], protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)
        thresholds_value = selected_rows[seed]["thresholds"]
        thresholds = np.asarray(
            json.loads(thresholds_value) if isinstance(thresholds_value, str) else thresholds_value,
            dtype=np.float32,
        )

        train_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["train"], args.batch_size, seed, device)
        early_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["early"], args.batch_size, seed, device)
        experts = fit_frozen_experts(train_unet, early_unet, seed)
        fit_indices, safety_indices = _identity_split(roles["calibration"], root_identities, seed + 7000)
        if np.intersect1d(root_identities[fit_indices], root_identities[safety_indices]).size:
            raise RuntimeError("router fit/safety identity overlap")

        role_records = {"fit": [], "safety": [], "tune": []}
        runtime = {"unet_expert_fit_seconds": train_unet["elapsed_seconds"] + early_unet["elapsed_seconds"]}
        for role_name, indices in (("fit", fit_indices), ("safety", safety_indices), ("tune", roles["tune"])):
            for level in POWER_LEVELS:
                main = collect_main(main_model, files[level], indices, args.batch_size, device, normalizer)
                unet_payload = extract_unet_role(
                    unet, files[level], files[50], root_labels, indices, args.batch_size, seed + level, device
                )
                align_payloads(main, unet_payload)
                record = {
                    **{name: main[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
                    "expert": expert_probabilities(unet_payload, experts),
                    "power": np.full(len(indices), level, dtype=np.int16),
                }
                runtime[f"{role_name}_{level}db_seconds"] = main["elapsed_seconds"] + unet_payload["elapsed_seconds"]
                role_records[role_name].append(record)

        fit_merged = merge_records([{**record, "route": np.zeros((len(record["labels"]), 3), np.float32)} for record in role_records["fit"]])
        routers, advantage_prevalence = fit_routers(
            fit_merged["probabilities"], fit_merged["expert"], fit_merged["labels"], seed
        )
        for role_name in ("fit", "safety", "tune"):
            for record in role_records[role_name]:
                record["route"] = route_probabilities(routers, record["probabilities"], record["expert"])
        selected, candidate_rows, safety_baseline = select_blend(role_records["safety"], thresholds)
        power_rows, aggregate = evaluate_tune(
            role_records["tune"], thresholds, selected["alpha"], selected["route_threshold"], normalizer.scale.numpy()
        )
        final_parts = [
            bounded_blend(record["probabilities"], record["expert"], record["route"], selected["alpha"], selected["route_threshold"])
            for record in role_records["tune"]
        ]
        tune = merge_records(role_records["tune"])
        np.savez_compressed(
            seed_dir / "tune_per_sample.npz",
            sample_index=tune["sample_indices"], receiver_noise_seed=root_identities[tune["sample_indices"]],
            power_db=tune["power"], labels=tune["labels"].astype(np.uint8),
            main_probability=tune["probabilities"], expert_probability=tune["expert"],
            router_probability=tune["route"], final_probability=np.concatenate(final_parts),
            thresholds=thresholds, parameter_prediction=tune["prediction"],
            parameter_target=tune["target"], parameter_mask=tune["mask"].astype(np.uint8),
        )
        result = {
            "seed": seed,
            "label_mapping": {"slice": "3:6", "targets": TARGET_NAMES, "global_indices": TARGET_INDICES},
            "frozen_main_checkpoint": str(main_checkpoint),
            "frozen_unet_checkpoint": str(unet_checkpoint),
            "expert_reconstruction": {"MFT": "unet_output_0", "VFT": "stack_trajectory_leaf15", "ISRJ": "stack_all_periodicity"},
            "router_advantage_prevalence": dict(zip(TARGET_NAMES, advantage_prevalence)),
            "identity_audit": {
                "fit_samples": len(fit_indices), "safety_samples": len(safety_indices),
                "fit_hash": _hash(fit_indices), "safety_hash": _hash(safety_indices),
                "fit_safety_identity_overlap": 0, "tune_fit_overlap": int(np.intersect1d(roles["tune"], fit_indices).size),
                "tune_safety_overlap": int(np.intersect1d(roles["tune"], safety_indices).size),
            },
            "selected_blend": selected,
            "safety_baseline": safety_baseline,
            "safety_candidates": candidate_rows,
            "tune_by_power": power_rows,
            "tune_aggregate": aggregate,
            "runtime_seconds": runtime,
            "locked_test_used": False,
            "confirm_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gains = [row["tune_aggregate"]["extreme_three_class_recall_gain"] for row in results]
    macro_deltas = [row["tune_aggregate"]["mean_nine_power_macro_aupr_delta"] for row in results]
    fprs = [max(row["tune_aggregate"]["mean_routed_class_fpr"].values()) for row in results]
    promotion = bool(
        np.mean(gains) >= 0.10 and min(gains) > 0.0
        and np.mean(macro_deltas) >= -0.005 and max(fprs) <= 0.10 + 1e-12
    )
    final_summary = {
        "experiment": "P1 six-class bounded swept-expert routing",
        "label_mapping_correction": "legacy output 0 is MFT, not SawFM",
        "seeds": args.seeds,
        "mean_extreme_three_class_recall_gain": float(np.mean(gains)),
        "per_seed_extreme_three_class_recall_gain": gains,
        "same_direction_seeds": int(sum(value > 0 for value in gains)),
        "mean_nine_power_macro_aupr_delta": float(np.mean(macro_deltas)),
        "maximum_seed_mean_routed_fpr": float(max(fprs)),
        "promotion_passed": promotion,
        "next_gate": "enter P2 parameter-gate evaluation" if promotion else "stop P1 integration; do not start end-to-end joint training",
        "identity_alignment_audit": alignment,
        "locked_test_used": False,
        "confirm_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(final_summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(final_summary), indent=2))


if __name__ == "__main__":
    main()
