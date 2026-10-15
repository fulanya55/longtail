"""Route frozen C2 local parameter experts with identity-safe visibility evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import get_class_names  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.models.supervised_joint import temporal_local_distribution  # noqa: E402
from experiments.run_route_b_c2_frozen_local_replacement import (  # noqa: E402
    PROMOTED_TARGETS,
    _normalizers,
)
from experiments.run_route_b_c2_local_parameter_probe import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    ScalarParameterProbe,
    _load_model,
    _sha256,
    _target_info,
    _write_json,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


GATE_VARIANTS = ("G0_ModelEvidence", "G1_RawRMSVisibility")
CONTEXT_GATE_VARIANTS = (
    "G2_MultiLabelContext",
    "G3_MultiLabelContextRMS",
)
ALL_GATE_VARIANTS = GATE_VARIANTS + CONTEXT_GATE_VARIANTS


def _identity_split(indices, identities, seed):
    values = np.asarray(indices, dtype=np.int64)
    identity_values = np.asarray(identities, dtype=np.int64)[values]
    order = np.arange(len(values), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    split = len(order) // 2
    fit = values[order[:split]]
    selection = values[order[split:]]
    if set(identity_values[order[:split]].tolist()).intersection(
        identity_values[order[split:]].tolist()
    ):
        raise ValueError("router fit and selection identities overlap")
    return np.sort(fit), np.sort(selection)


def _load_expert(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    input_dim = int(state["regressor.1.weight"].shape[1])
    hidden_dim = int(state["regressor.1.weight"].shape[0])
    model = ScalarParameterProbe(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _mapped_indices(base_path, power_path, base_indices):
    base_identity, base_labels = load_level_identity(base_path)
    power_identity, power_labels = load_level_identity(power_path)
    lookup = {int(identity): int(index) for index, identity in enumerate(power_identity)}
    mapped = np.asarray(
        [lookup[int(base_identity[int(index)])] for index in base_indices], dtype=np.int64
    )
    if not np.array_equal(base_labels[np.asarray(base_indices, dtype=np.int64)], power_labels[mapped]):
        raise ValueError("identity mapping changed labels")
    return mapped, base_identity[np.asarray(base_indices, dtype=np.int64)]


def _extract_level(
    model,
    experts,
    expert_checkpoints,
    power_path,
    mapped_indices,
    base_indices,
    identities,
    normalization,
    normalizer_mean,
    normalizer_scale,
    device,
    batch_size,
):
    dataset = V73SignalDataset(
        power_path,
        mapped_indices,
        class_mode="six",
        normalization=normalization,
        include_parameters=False,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    class_names = get_class_names("six")
    output = {
        "base_indices": np.asarray(base_indices, dtype=np.int64),
        "power_indices": np.asarray(mapped_indices, dtype=np.int64),
        "identities": np.asarray(identities, dtype=np.int64),
        "raw_rms": [],
        "class_probabilities": [],
        "c2_predictions": [],
        "local_predictions": [],
        "local_z_rms": [],
        "local_z_max": [],
    }
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                temporal = model.backbone.forward_temporal_features(signal)
                pooled = model.backbone.global_pool(temporal).squeeze(-1)
                logits = model.backbone.forward_classifier(pooled)
                embeddings = model.label_feature_bank(pooled)
                normalized_parameters = model.parameter_heads(embeddings)
                raw_parameters = (
                    normalized_parameters
                    * torch.as_tensor(normalizer_scale, device=device, dtype=normalized_parameters.dtype)[None]
                    + torch.as_tensor(normalizer_mean, device=device, dtype=normalized_parameters.dtype)[None]
                )
                local_features = temporal_local_distribution(temporal).float()
                local_predictions = []
                local_z_rms = []
                local_z_max = []
                for class_name, parameter_name in PROMOTED_TARGETS:
                    key = (class_name, parameter_name)
                    expert = experts[key]
                    checkpoint = expert_checkpoints[key]
                    mean = checkpoint["feature_mean"].to(device=device, dtype=local_features.dtype)
                    std = checkpoint["feature_std"].to(device=device, dtype=local_features.dtype)
                    z = (local_features - mean) / std
                    prediction = expert(z) * float(checkpoint["target_scale"]) + float(
                        checkpoint["target_mean"]
                    )
                    local_predictions.append(prediction)
                    local_z_rms.append(z.square().mean(dim=1).sqrt())
                    local_z_max.append(z.abs().amax(dim=1))
                c2_predictions = []
                for class_name, parameter_name in PROMOTED_TARGETS:
                    class_index, parameter_index, _ = _target_info(
                        class_names, class_name, parameter_name
                    )
                    c2_predictions.append(raw_parameters[:, class_index, parameter_index])
                output["raw_rms"].append(np.asarray(batch["raw_rms"], dtype=np.float32))
                output["class_probabilities"].append(logits.sigmoid().cpu().numpy())
                output["c2_predictions"].append(torch.stack(c2_predictions, dim=1).cpu().numpy())
                output["local_predictions"].append(torch.stack(local_predictions, dim=1).cpu().numpy())
                output["local_z_rms"].append(torch.stack(local_z_rms, dim=1).cpu().numpy())
                output["local_z_max"].append(torch.stack(local_z_max, dim=1).cpu().numpy())
    finally:
        dataset.close()
    for key in (
        "raw_rms",
        "class_probabilities",
        "c2_predictions",
        "local_predictions",
        "local_z_rms",
        "local_z_max",
    ):
        output[key] = np.concatenate(output[key], axis=0).astype(np.float32)
    if not np.array_equal(output["power_indices"], np.asarray(mapped_indices, dtype=np.int64)):
        raise ValueError("power extraction order drifted")
    return output


def _gate_features(payload, target_offset, class_index, target_mean, target_scale, variant):
    c2 = payload["c2_predictions"][:, target_offset]
    local = payload["local_predictions"][:, target_offset]
    scale = max(float(target_scale), 1.0e-12)
    features = [
        np.abs(local - c2) / scale,
        np.abs(c2 - float(target_mean)) / scale,
        np.abs(local - float(target_mean)) / scale,
        payload["local_z_rms"][:, target_offset],
        payload["local_z_max"][:, target_offset],
        payload["class_probabilities"][:, class_index],
    ]
    if variant in CONTEXT_GATE_VARIANTS:
        probabilities = np.clip(
            payload["class_probabilities"].astype(np.float64), 1.0e-6, 1.0 - 1.0e-6
        )
        other = np.delete(probabilities, int(class_index), axis=1)
        features.extend(
            [
                *[probabilities[:, index] for index in range(probabilities.shape[1])],
                probabilities.sum(axis=1),
                (probabilities >= 0.5).sum(axis=1),
                -(
                    probabilities * np.log(probabilities)
                    + (1.0 - probabilities) * np.log(1.0 - probabilities)
                ).mean(axis=1),
                other.max(axis=1),
                probabilities[:, class_index] - other.max(axis=1),
            ]
        )
    if variant in ("G1_RawRMSVisibility", "G3_MultiLabelContextRMS"):
        features.append(np.log(np.maximum(payload["raw_rms"], 1.0e-12)))
    return np.stack(features, axis=1).astype(np.float32)


def _target_payload(payload, target_offset, base_labels, base_values, base_mask, class_index, column_index):
    indices = payload["base_indices"]
    valid = (
        (base_labels[indices, class_index] == 1)
        & base_mask[indices, column_index]
        & np.isfinite(base_values[indices, column_index])
    )
    return valid, base_values[indices, column_index].astype(np.float32)


def _fit_gate(features, c2, local, target, scale, seed):
    advantage = (np.abs(local - target) + 0.0025 * float(scale)) < np.abs(c2 - target)
    if len(np.unique(advantage)) < 2:
        return None, {"advantage_rate": float(advantage.mean()), "constant": True}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=int(seed),
        ),
    )
    model.fit(features, advantage.astype(np.uint8))
    return model, {"advantage_rate": float(advantage.mean()), "constant": False}


def _predict_gate(model, features, constant_probability):
    if model is None:
        return np.full(len(features), float(constant_probability), dtype=np.float64)
    return model.predict_proba(features)[:, 1].astype(np.float64)


def _apply_identity_consensus(level_payloads, probabilities, beta):
    """Shrink view-specific gate probabilities toward same-identity median evidence."""

    strength = float(beta)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("identity consensus beta must be in [0, 1]")
    identity_values = sorted(
        {
            int(identity)
            for payload in level_payloads.values()
            for identity in payload["identities"]
        }
    )
    per_identity = {identity: [] for identity in identity_values}
    for level, payload in level_payloads.items():
        values = np.asarray(probabilities[level], dtype=np.float64)
        if len(values) != len(payload["identities"]):
            raise ValueError("gate probability and identity lengths differ")
        for identity, value in zip(payload["identities"], values):
            per_identity[int(identity)].append(float(value))
    median = {
        identity: float(np.median(values)) for identity, values in per_identity.items()
    }
    return {
        level: (
            (1.0 - strength) * np.asarray(probabilities[level], dtype=np.float64)
            + strength
            * np.asarray(
                [median[int(identity)] for identity in payload["identities"]],
                dtype=np.float64,
            )
        )
        for level, payload in level_payloads.items()
    }


def _predicted_context_keys(payload, class_index):
    decisions = np.asarray(payload["class_probabilities"], dtype=np.float64) >= 0.5
    decisions[:, int(class_index)] = True
    return np.asarray(
        [
            "".join(str(int(value)) for value in np.delete(row, int(class_index)))
            for row in decisions
        ]
    )


def _fit_predicted_context_allowlist(
    level_payloads,
    probabilities,
    target_offset,
    target_meta,
    threshold,
    local_weight,
    minimum_count,
):
    groups = {}
    for level, payload in level_payloads.items():
        valid, target = _target_payload(
            payload,
            target_offset,
            target_meta["labels"],
            target_meta["values"],
            target_meta["mask"],
            target_meta["class_index"],
            target_meta["column_index"],
        )
        gate = np.asarray(probabilities[level])[valid] >= float(threshold)
        c2 = payload["c2_predictions"][valid, target_offset]
        local = payload["local_predictions"][valid, target_offset]
        expected = target[valid]
        blended = c2 + float(local_weight) * (local - c2)
        gain = np.abs(c2 - expected) - np.abs(blended - expected)
        keys = _predicted_context_keys(payload, target_meta["class_index"])[valid]
        for key in np.unique(keys[gate]):
            selected = gate & (keys == key)
            row = groups.setdefault(str(key), {"count": 0, "gain_sum": 0.0})
            row["count"] += int(selected.sum())
            row["gain_sum"] += float(gain[selected].sum())
    allowlist = sorted(
        key
        for key, row in groups.items()
        if row["count"] >= int(minimum_count) and row["gain_sum"] > 0.0
    )
    return allowlist, groups


def _serialize_gate(model, variant, constant_probability):
    feature_names = [
        "normalized_local_c2_disagreement",
        "normalized_c2_distance_from_training_mean",
        "normalized_local_distance_from_training_mean",
        "local_feature_z_rms",
        "local_feature_z_max",
        "class_probability",
    ]
    if variant in CONTEXT_GATE_VARIANTS:
        feature_names.extend(
            [
                "probability_FMB",
                "probability_SawFM",
                "probability_SinFM",
                "probability_MFT",
                "probability_VFT",
                "probability_ISRJ",
                "soft_label_cardinality",
                "hard_label_cardinality_at_0p5",
                "mean_bernoulli_entropy",
                "maximum_other_label_probability",
                "target_probability_margin_over_other_labels",
            ]
        )
    if variant in ("G1_RawRMSVisibility", "G3_MultiLabelContextRMS"):
        feature_names.append("log_raw_rms")
    if model is None:
        return {
            "type": "constant",
            "constant_probability": float(constant_probability),
            "feature_names": feature_names,
        }
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {
        "type": "standardized_logistic_regression",
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "coefficient": classifier.coef_[0].astype(np.float64).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def _routing_metrics(
    level_payloads,
    probabilities,
    target_offset,
    target_meta,
    threshold,
    local_weight=1.0,
    predicted_context_allowlist=None,
):
    class_index = target_meta["class_index"]
    column_index = target_meta["column_index"]
    scale = target_meta["scale"]
    by_level = {}
    all_c2_error = []
    all_local_error = []
    all_routed_error = []
    all_gate = []
    all_base_indices = []
    for level, payload in level_payloads.items():
        valid, target = _target_payload(
            payload,
            target_offset,
            target_meta["labels"],
            target_meta["values"],
            target_meta["mask"],
            class_index,
            column_index,
        )
        c2 = payload["c2_predictions"][valid, target_offset]
        local = payload["local_predictions"][valid, target_offset]
        expected = target[valid]
        gate = probabilities[level][valid] >= float(threshold)
        if predicted_context_allowlist is not None:
            context_keys = _predicted_context_keys(
                payload, target_meta["class_index"]
            )[valid]
            gate &= np.isin(context_keys, list(predicted_context_allowlist))
        blended = c2 + float(local_weight) * (local - c2)
        routed = np.where(gate, blended, c2)
        c2_error = np.abs(c2 - expected) / scale
        local_error = np.abs(local - expected) / scale
        routed_error = np.abs(routed - expected) / scale
        c2_mean = float(c2_error.mean())
        local_mean = float(local_error.mean())
        routed_mean = float(routed_error.mean())
        by_level[str(level)] = {
            "count": int(len(expected)),
            "c2_nmae": c2_mean,
            "always_local_nmae": local_mean,
            "routed_nmae": routed_mean,
            "always_local_relative_reduction": float(
                (c2_mean - local_mean) / max(c2_mean, 1.0e-12)
            ),
            "routed_relative_reduction": float(
                (c2_mean - routed_mean) / max(c2_mean, 1.0e-12)
            ),
            "route_coverage": float(gate.mean()),
            "oracle_local_better_rate": float((local_error < c2_error).mean()),
        }
        all_c2_error.append(c2_error)
        all_local_error.append(local_error)
        all_routed_error.append(routed_error)
        all_gate.append(gate)
        all_base_indices.append(payload["base_indices"][valid])
    c2_error = np.concatenate(all_c2_error)
    local_error = np.concatenate(all_local_error)
    routed_error = np.concatenate(all_routed_error)
    gate = np.concatenate(all_gate)
    base_indices = np.concatenate(all_base_indices).astype(np.int64)
    combination_rows = {}
    combination_keys = np.asarray(
        [
            "".join(str(int(value)) for value in target_meta["labels"][index].tolist())
            for index in base_indices
        ]
    )
    for combination in sorted(np.unique(combination_keys)):
        selected = combination_keys == combination
        combo_c2 = float(c2_error[selected].mean())
        combo_routed = float(routed_error[selected].mean())
        combination_rows[str(combination)] = {
            "count": int(selected.sum()),
            "c2_nmae": combo_c2,
            "routed_nmae": combo_routed,
            "routed_relative_reduction": float(
                (combo_c2 - combo_routed) / max(combo_c2, 1.0e-12)
            ),
            "route_coverage": float(gate[selected].mean()),
        }
    identity_level_gate = {}
    for level, payload in level_payloads.items():
        valid, _ = _target_payload(
            payload,
            target_offset,
            target_meta["labels"],
            target_meta["values"],
            target_meta["mask"],
            target_meta["class_index"],
            target_meta["column_index"],
        )
        level_gate = probabilities[level][valid] >= float(threshold)
        for identity, decision in zip(payload["identities"][valid], level_gate):
            identity_level_gate.setdefault(int(identity), []).append(
                (int(level), int(decision))
            )
    flip_counts = []
    monotonic = []
    for values in identity_level_gate.values():
        decisions = np.asarray(
            [decision for _, decision in sorted(values)], dtype=np.int16
        )
        differences = np.diff(decisions)
        flip_counts.append(int(np.abs(differences).sum()))
        monotonic.append(bool((differences >= 0).all() or (differences <= 0).all()))
    c2_mean = float(c2_error.mean())
    local_mean = float(local_error.mean())
    routed_mean = float(routed_error.mean())
    return {
        "c2_nmae": c2_mean,
        "always_local_nmae": local_mean,
        "routed_nmae": routed_mean,
        "always_local_relative_reduction": float(
            (c2_mean - local_mean) / max(c2_mean, 1.0e-12)
        ),
        "routed_relative_reduction": float(
            (c2_mean - routed_mean) / max(c2_mean, 1.0e-12)
        ),
        "route_coverage": float(gate.mean()),
        "local_weight": float(local_weight),
        "minimum_level_routed_reduction": float(
            min(row["routed_relative_reduction"] for row in by_level.values())
        ),
        "minimum_combination_routed_reduction": float(
            min(row["routed_relative_reduction"] for row in combination_rows.values())
        ),
        "negative_combination_count": int(
            sum(
                row["routed_relative_reduction"] < 0.0
                for row in combination_rows.values()
            )
        ),
        "identity_route_flip_mean": float(np.mean(flip_counts)),
        "identity_three_or_more_flip_rate": float(
            np.mean(np.asarray(flip_counts) >= 3)
        ),
        "identity_monotonic_route_rate": float(np.mean(monotonic)),
        "by_level": by_level,
        "by_combination": combination_rows,
    }


def _select_threshold(
    level_payloads,
    probabilities,
    target_offset,
    target_meta,
    minimum_combination_reduction=-1.0,
    consensus_betas=(0.0,),
    context_safety_modes=(False,),
    context_minimum_count=30,
):
    candidates = []
    for consensus_beta in consensus_betas:
        consensus_probabilities = _apply_identity_consensus(
            level_payloads, probabilities, consensus_beta
        )
        for local_weight in (0.25, 0.50, 0.75):
            for threshold in np.concatenate(
                [np.arange(0.40, 0.91, 0.025), np.asarray([0.95, 0.975, 1.01])]
            ):
                for context_safety in context_safety_modes:
                    allowlist = None
                    context_groups = {}
                    if bool(context_safety):
                        allowlist, context_groups = _fit_predicted_context_allowlist(
                            level_payloads,
                            consensus_probabilities,
                            target_offset,
                            target_meta,
                            threshold,
                            local_weight,
                            context_minimum_count,
                        )
                    metrics = _routing_metrics(
                        level_payloads,
                        consensus_probabilities,
                        target_offset,
                        target_meta,
                        threshold,
                        local_weight=local_weight,
                        predicted_context_allowlist=allowlist,
                    )
                    feasible = (
                        metrics["routed_relative_reduction"] > 0.0
                        and metrics["minimum_level_routed_reduction"] >= 0.0
                        and metrics["minimum_combination_routed_reduction"]
                        >= float(minimum_combination_reduction)
                    )
                    candidates.append(
                        {
                            "threshold": float(threshold),
                            "identity_consensus_beta": float(consensus_beta),
                            "predicted_context_safety": bool(context_safety),
                            "predicted_context_allowlist": allowlist,
                            "predicted_context_groups": context_groups,
                            "feasible": bool(feasible),
                            **metrics,
                        }
                    )
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        selected = candidates[-1]
    else:
        selected = max(
            feasible,
            key=lambda row: (
                row["routed_relative_reduction"],
                row["minimum_combination_routed_reduction"],
                row["minimum_level_routed_reduction"],
                -row["identity_route_flip_mean"],
                -row["route_coverage"],
            ),
        )
    return selected, candidates


def _fit_and_select_seed(args, seed, config, base_path, power_files, labels, values, mask, device):
    plan = build_run_plan(config, "bce", int(seed))
    fit_indices, selection_indices = _identity_split(
        plan["threshold_indices"], load_level_identity(base_path)[0], 310000 + int(seed)
    )
    evaluation_indices = np.asarray(plan["calibration_indices"], dtype=np.int64)
    fit_identity = set(load_level_identity(base_path)[0][fit_indices].tolist())
    selection_identity = set(load_level_identity(base_path)[0][selection_indices].tolist())
    evaluation_identity = set(
        load_level_identity(base_path)[0][evaluation_indices].tolist()
    )
    if (
        fit_identity & selection_identity
        or fit_identity & evaluation_identity
        or selection_identity & evaluation_identity
    ):
        raise ValueError("router roles overlap by receiver identity")

    run_dir = (
        DEFAULT_CHECKPOINTS[int(seed)]
        if args.checkpoint_root is None
        else Path(args.checkpoint_root) / "seed_{}".format(seed)
    )
    model, checkpoint_path = _load_model(run_dir, device)
    _, normalizer_mean, normalizer_scale = _normalizers(run_dir)
    experts = {}
    expert_checkpoints = {}
    expert_hashes = {}
    for class_name, parameter_name in PROMOTED_TARGETS:
        path = (
            args.expert_root
            / "seed_{}".format(seed)
            / "{}_{}_local_expert.pth".format(class_name, parameter_name)
        )
        experts[(class_name, parameter_name)], expert_checkpoints[(class_name, parameter_name)] = _load_expert(path, device)
        expert_hashes["{}.{}".format(class_name, parameter_name)] = _sha256(path)

    roles = {}
    for role, base_indices in (
        ("fit", fit_indices),
        ("selection", selection_indices),
        ("confirmatory", evaluation_indices),
    ):
        roles[role] = {}
        for level in args.levels:
            mapped, identities = _mapped_indices(base_path, power_files[int(level)], base_indices)
            roles[role][int(level)] = _extract_level(
                model,
                experts,
                expert_checkpoints,
                power_files[int(level)],
                mapped,
                base_indices,
                identities,
                config["normalization"],
                normalizer_mean,
                normalizer_scale,
                device,
                args.batch_size,
            )

    class_names = get_class_names("six")
    targets = []
    prediction_payload = {}
    for target_offset, (class_name, parameter_name) in enumerate(PROMOTED_TARGETS):
        class_index, parameter_index, column_index = _target_info(
            class_names, class_name, parameter_name
        )
        metadata = {
            "class_index": class_index,
            "parameter_index": parameter_index,
            "column_index": column_index,
            "mean": float(normalizer_mean[class_index, parameter_index]),
            "scale": float(normalizer_scale[class_index, parameter_index]),
            "labels": labels,
            "values": values,
            "mask": mask,
        }
        variants = []
        variant_runtime = []
        for variant_offset, variant in enumerate(args.gate_variants):
            fit_x, fit_c2, fit_local, fit_target = [], [], [], []
            for level, payload in roles["fit"].items():
                valid, target = _target_payload(
                    payload, target_offset, labels, values, mask, class_index, column_index
                )
                fit_x.append(
                    _gate_features(
                        payload,
                        target_offset,
                        class_index,
                        metadata["mean"],
                        metadata["scale"],
                        variant,
                    )[valid]
                )
                fit_c2.append(payload["c2_predictions"][valid, target_offset])
                fit_local.append(payload["local_predictions"][valid, target_offset])
                fit_target.append(target[valid])
            gate, fit_info = _fit_gate(
                np.concatenate(fit_x),
                np.concatenate(fit_c2),
                np.concatenate(fit_local),
                np.concatenate(fit_target),
                metadata["scale"],
                320000 + int(seed) + target_offset * 100 + variant_offset,
            )
            constant_probability = fit_info["advantage_rate"]
            selection_probabilities = {
                level: _predict_gate(
                    gate,
                    _gate_features(
                        payload,
                        target_offset,
                        class_index,
                        metadata["mean"],
                        metadata["scale"],
                        variant,
                    ),
                    constant_probability,
                )
                for level, payload in roles["selection"].items()
            }
            selected, candidates = _select_threshold(
                roles["selection"],
                selection_probabilities,
                target_offset,
                metadata,
                minimum_combination_reduction=args.minimum_combination_reduction,
                consensus_betas=args.consensus_betas,
                context_safety_modes=args.context_safety_modes,
                context_minimum_count=args.context_minimum_count,
            )
            confirmatory_probabilities = {
                level: _predict_gate(
                    gate,
                    _gate_features(
                        payload,
                        target_offset,
                        class_index,
                        metadata["mean"],
                        metadata["scale"],
                        variant,
                    ),
                    constant_probability,
                )
                for level, payload in roles["confirmatory"].items()
            }
            confirmatory_probabilities = _apply_identity_consensus(
                roles["confirmatory"],
                confirmatory_probabilities,
                selected["identity_consensus_beta"],
            )
            confirmatory_metrics = _routing_metrics(
                roles["confirmatory"],
                confirmatory_probabilities,
                target_offset,
                metadata,
                selected["threshold"],
                local_weight=selected["local_weight"],
                predicted_context_allowlist=selected[
                    "predicted_context_allowlist"
                ],
            )
            variants.append(
                {
                    "variant": variant,
                    "fit": fit_info,
                    "router_state": _serialize_gate(
                        gate, variant, constant_probability
                    ),
                    "selected_threshold": float(selected["threshold"]),
                    "selected_local_weight": float(selected["local_weight"]),
                    "selected_identity_consensus_beta": float(
                        selected["identity_consensus_beta"]
                    ),
                    "selected_predicted_context_safety": bool(
                        selected["predicted_context_safety"]
                    ),
                    "selected_predicted_context_allowlist": selected[
                        "predicted_context_allowlist"
                    ],
                    "selection": selected,
                    "confirmatory": confirmatory_metrics,
                    "threshold_candidates": candidates,
                }
            )
            variant_runtime.append(
                {
                    "gate": gate,
                    "constant_probability": constant_probability,
                    "confirmatory_probabilities": confirmatory_probabilities,
                }
            )
        chosen_index = max(
            range(len(variants)),
            key=lambda index: (
                variants[index]["selection"]["feasible"],
                variants[index]["selection"]["routed_relative_reduction"],
            ),
        )
        chosen = variants[chosen_index]
        chosen_runtime = variant_runtime[chosen_index]
        prediction_parts = {
            key: []
            for key in (
                "base_indices",
                "power_indices",
                "identities",
                "levels",
                "targets",
                "c2_predictions",
                "local_predictions",
                "routing_probabilities",
                "routing_decisions",
                "routed_predictions",
                "raw_rms",
                "class_probabilities",
            )
        }
        for level, payload in roles["confirmatory"].items():
            valid, target = _target_payload(
                payload,
                target_offset,
                labels,
                values,
                mask,
                class_index,
                column_index,
            )
            probability = chosen_runtime["confirmatory_probabilities"][level][valid]
            c2_prediction = payload["c2_predictions"][valid, target_offset]
            local_prediction = payload["local_predictions"][valid, target_offset]
            decision = probability >= float(chosen["selected_threshold"])
            if chosen["selected_predicted_context_allowlist"] is not None:
                decision &= np.isin(
                    _predicted_context_keys(payload, class_index)[valid],
                    chosen["selected_predicted_context_allowlist"],
                )
            blended_prediction = c2_prediction + float(
                chosen["selected_local_weight"]
            ) * (local_prediction - c2_prediction)
            count = int(valid.sum())
            prediction_parts["base_indices"].append(payload["base_indices"][valid])
            prediction_parts["power_indices"].append(payload["power_indices"][valid])
            prediction_parts["identities"].append(payload["identities"][valid])
            prediction_parts["levels"].append(np.full(count, int(level), dtype=np.int16))
            prediction_parts["targets"].append(target[valid])
            prediction_parts["c2_predictions"].append(c2_prediction)
            prediction_parts["local_predictions"].append(local_prediction)
            prediction_parts["routing_probabilities"].append(probability.astype(np.float32))
            prediction_parts["routing_decisions"].append(decision.astype(np.uint8))
            prediction_parts["routed_predictions"].append(
                np.where(decision, blended_prediction, c2_prediction).astype(np.float32)
            )
            prediction_parts["raw_rms"].append(payload["raw_rms"][valid])
            prediction_parts["class_probabilities"].append(
                payload["class_probabilities"][valid, class_index]
            )
        prefix = "{}_{}".format(class_name, parameter_name)
        for key, parts in prediction_parts.items():
            prediction_payload["{}_{}".format(prefix, key)] = np.concatenate(parts)
        targets.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "variants": variants,
                "chosen_variant": chosen["variant"],
                "chosen_threshold": chosen["selected_threshold"],
                "chosen_local_weight": chosen["selected_local_weight"],
                "chosen_identity_consensus_beta": chosen[
                    "selected_identity_consensus_beta"
                ],
                "chosen_predicted_context_safety": chosen[
                    "selected_predicted_context_safety"
                ],
                "chosen_predicted_context_allowlist": chosen[
                    "selected_predicted_context_allowlist"
                ],
                "chosen_router_state": chosen["router_state"],
                "chosen_selection": chosen["selection"],
                "chosen_confirmatory": chosen["confirmatory"],
            }
        )
    receipt = {
        "status": "completed",
        "seed": int(seed),
        "levels": [int(level) for level in args.levels],
        "split_sha256": plan["split_sha256"],
        "c2_checkpoint_sha256": _sha256(checkpoint_path),
        "expert_sha256": expert_hashes,
        "router_fit_identity_count": len(fit_identity),
        "router_selection_identity_count": len(selection_identity),
        "confirmatory_identity_count": len(evaluation_identity),
        "identity_overlap": 0,
        "classification_outputs_modified": False,
        "locked_test_used": False,
        "targets": targets,
    }
    seed_root = args.output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(seed_root / "routing_predictions.npz"), **prediction_payload)
    receipt["routing_predictions_sha256"] = _sha256(
        seed_root / "routing_predictions.npz"
    )
    _write_json(seed_root / "run_receipt.json", receipt)
    return receipt


def _aggregate(receipts, levels):
    targets = []
    all_seed_c2 = []
    all_seed_routed = []
    for class_name, parameter_name in PROMOTED_TARGETS:
        selected = [
            next(
                target
                for target in receipt["targets"]
                if target["class"] == class_name and target["parameter"] == parameter_name
            )
            for receipt in receipts
        ]
        c2 = np.asarray([row["chosen_confirmatory"]["c2_nmae"] for row in selected])
        local = np.asarray([row["chosen_confirmatory"]["always_local_nmae"] for row in selected])
        routed = np.asarray([row["chosen_confirmatory"]["routed_nmae"] for row in selected])
        reductions = 1.0 - routed / c2
        level_rows = {}
        for level in levels:
            level_c2 = np.asarray(
                [row["chosen_confirmatory"]["by_level"][str(level)]["c2_nmae"] for row in selected]
            )
            level_routed = np.asarray(
                [row["chosen_confirmatory"]["by_level"][str(level)]["routed_nmae"] for row in selected]
            )
            level_rows[str(level)] = {
                "c2_nmae_mean": float(level_c2.mean()),
                "routed_nmae_mean": float(level_routed.mean()),
                "relative_reduction": float(1.0 - level_routed.mean() / level_c2.mean()),
            }
        all_seed_c2.append(c2)
        all_seed_routed.append(routed)
        targets.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "c2_nmae_mean": float(c2.mean()),
                "always_local_nmae_mean": float(local.mean()),
                "routed_nmae_mean": float(routed.mean()),
                "routed_relative_reduction": float(1.0 - routed.mean() / c2.mean()),
                "per_seed_reductions": reductions.tolist(),
                "positive_seed_count": int((reductions > 0.0).sum()),
                "chosen_variants": [row["chosen_variant"] for row in selected],
                "by_level": level_rows,
            }
        )
    seed_c2 = np.stack(all_seed_c2, axis=1).mean(axis=1)
    seed_routed = np.stack(all_seed_routed, axis=1).mean(axis=1)
    combined_reductions = 1.0 - seed_routed / seed_c2
    minimum_level_reduction = min(
        row["by_level"][str(level)]["relative_reduction"]
        for row in targets
        for level in levels
    )
    checks = {
        "combined_relative_reduction_at_least_2pct": float(
            1.0 - seed_routed.mean() / seed_c2.mean()
        )
        >= 0.02,
        "combined_3_of_3_seeds_positive": bool(
            len(receipts) == 3 and (combined_reductions > 0.0).all()
        ),
        "each_target_mean_positive": all(row["routed_relative_reduction"] > 0.0 for row in targets),
        "worst_target_level_not_below_minus_1pct": minimum_level_reduction >= -0.01,
        "identity_and_locked_test_audit": all(
            receipt["identity_overlap"] == 0
            and not receipt["locked_test_used"]
            and not receipt["classification_outputs_modified"]
            for receipt in receipts
        ),
    }
    return {
        "targets": targets,
        "combined_c2_nmae_mean": float(seed_c2.mean()),
        "combined_routed_nmae_mean": float(seed_routed.mean()),
        "combined_relative_reduction": float(1.0 - seed_routed.mean() / seed_c2.mean()),
        "combined_per_seed_reductions": combined_reductions.tolist(),
        "minimum_target_level_reduction": float(minimum_level_reduction),
        "promotion_gate": {"checks": checks, "passed": bool(all(checks.values()))},
    }


def run(args):
    config = load_config(args.config)
    base_path = _data_path(config)
    labels = read_labels(base_path)
    values, mask = read_parameters(base_path)
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    missing = [int(level) for level in args.levels if int(level) not in power_files]
    if missing:
        raise ValueError("power levels unavailable: {}".format(missing))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    receipts = []
    for seed in args.seeds:
        print("visibility-safe local routing seed={}".format(seed), flush=True)
        receipts.append(
            _fit_and_select_seed(
                args,
                int(seed),
                config,
                base_path,
                power_files,
                labels,
                values,
                mask,
                device,
            )
        )
    payload = {
        "status": "completed",
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "levels": [int(level) for level in args.levels],
            "router_fit_role": "first identity-disjoint half of threshold role",
            "router_selection_role": "second identity-disjoint half of threshold role",
            "method_development_role": (
                "early_stop is not read by this runner; method development, if any, "
                "occurred in prior runs"
            ),
            "final_evaluation_role": "calibration role, untouched by routing development",
            "gate_variants": list(args.gate_variants),
            "minimum_selection_combination_reduction": float(
                args.minimum_combination_reduction
            ),
            "identity_consensus_betas": [float(value) for value in args.consensus_betas],
            "predicted_context_safety_modes": [
                bool(value) for value in args.context_safety_modes
            ],
            "predicted_context_minimum_count": int(args.context_minimum_count),
            "targets": [list(target) for target in PROMOTED_TARGETS],
        },
        "aggregate": _aggregate(receipts, args.levels),
        "runs": receipts,
        "classification_outputs_modified": False,
        "locked_test_used": False,
    }
    _write_json(args.output_root / "route_b_c2_visibility_safe_routing_summary.json", payload)
    print(json.dumps({"aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/student_b/supervised_joint_six_class.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[10, 15, 20, 25, 30, 35, 40, 45, 50]
    )
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_frozen_local_replacement_20260725",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_visibility_safe_routing_confirmatory_20260725",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--gate-variants",
        nargs="+",
        choices=ALL_GATE_VARIANTS,
        default=list(GATE_VARIANTS),
    )
    parser.add_argument("--minimum-combination-reduction", type=float, default=-1.0)
    parser.add_argument(
        "--consensus-betas",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument(
        "--context-safety-modes",
        nargs="+",
        choices=("off", "on"),
        default=["off"],
    )
    parser.add_argument("--context-minimum-count", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.context_safety_modes = [value == "on" for value in args.context_safety_modes]
    return args


if __name__ == "__main__":
    run(parse_args())
