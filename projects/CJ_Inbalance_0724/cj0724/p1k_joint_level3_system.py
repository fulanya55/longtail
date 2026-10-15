"""P1k: frozen joint ISRJ-risk, SawFM-period and VFT-Doppler system.

The experiment uses fresh train identities for route fitting and per-power
finite-sample calibration.  Test files are discovered only after a complete
method-freeze receipt is written.  The locked test is never accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _level_metrics, _saw_prediction, _weak_mask
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    _jsonable,
    _load_main,
    _load_unet,
    align_payloads,
    classification_metrics,
    collect_main,
    expert_probabilities,
    extract_unet_role,
    fit_frozen_experts,
)
from cj0724.multitask import PARAMETER_COLUMNS
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.p1f_isrj_confidence_bound_router import (
    ISRJ_GLOBAL,
    POWERS,
    apply_route,
    buffered_isrj_threshold,
    ensemble_scores,
    fit_score_ensemble,
    select_coverage_route,
    stratified_train_roles,
    transition_files,
)
from cj0724.p1i_vft_reliability_attachment import collect_expert
from cj0724.p1j_vft_safe_attachment import buffered_vft_threshold
from cj0724.paper_core_evidence import gated_metrics, load_normalizer, macro_nmae
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.run_physics_parameter_specialists import _positive, _stratified_limit


MODEL_SEEDS = (2041, 2042, 2043)
VFT_CLASS_INDEX = 4
SAW_PARAMETER_INDEX = (1, 1)
VFT_PARAMETER_INDEX = (4, 1)
SAW_PARAMETER_COLUMN = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def audit_paired_power_metadata(files: dict[int, Path]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Verify that every power contains the same identities and labels in the same order."""
    reference_identities: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    per_power = {}
    for power, path in sorted(files.items()):
        identities, labels = read_identity_metadata(path)
        identities = np.asarray(identities, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.uint8)
        if reference_identities is None:
            reference_identities, reference_labels = identities, labels
        elif not np.array_equal(reference_identities, identities) or not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"paired identity/label alignment failed at {power} dB")
        per_power[str(power)] = {
            "count": int(len(identities)),
            "identity_hash": _hash_int64(identities),
            "label_hash": hashlib.sha256(labels.tobytes()).hexdigest(),
        }
    if reference_identities is None or reference_labels is None:
        raise ValueError("no paired-power files were supplied")
    return reference_identities, reference_labels, {
        "powers_db": sorted(int(power) for power in files),
        "identities_aligned": True,
        "labels_aligned": True,
        "per_power": per_power,
    }


@torch.inference_mode()
def collect_saw(
    model: torch.nn.Module,
    path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    mean: float,
    scale: float,
) -> dict[str, np.ndarray]:
    dataset = V73IQDataset(path, indices, [SAW_PARAMETER_COLUMN])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prediction = _saw_prediction(model, loader, device, mean, scale)
    sample_index = np.asarray(indices, dtype=np.int64).copy()
    dataset.close()
    return {"sample_index": sample_index, "prediction": prediction.astype(np.float32)}


def collect_joint_records(
    main_model,
    unet,
    experts,
    saw_model,
    saw_mean: float,
    saw_scale: float,
    files: dict[int, Path],
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    device: torch.device,
    normalizer,
    powers: tuple[int, ...] = POWERS,
    lazy_thresholds: np.ndarray | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
    records = []
    timing = {"main_seconds": 0.0, "unet_physics_seconds": 0.0,
              "vft_expert_seconds": 0.0, "saw_expert_seconds": 0.0}
    for power in powers:
        main = collect_main(main_model, files[power], indices, batch_size, device, normalizer)
        timing["main_seconds"] += float(main["elapsed_seconds"])
        if lazy_thresholds is None:
            isrj_keep = saw_keep = vft_keep = np.ones(len(indices), dtype=bool)
        else:
            isrj_keep, saw_keep, vft_keep = lazy_candidate_masks(main["probabilities"], lazy_thresholds)
        position = {int(sample_index): offset for offset, sample_index in enumerate(main["sample_indices"])}
        started = time.perf_counter()
        unet_indices = np.asarray(indices, dtype=np.int64)[isrj_keep]
        unet_record = extract_unet_role(
            unet, files[power], files[power], labels, unet_indices,
            batch_size, seed + power, device,
        )
        if lazy_thresholds is None:
            align_payloads(main, unet_record)
            expert_probability = expert_probabilities(unet_record, experts)
            physical = {name: unet_record[name] for name in ("mft", "trajectory", "isrj")}
        else:
            subset_expert = expert_probabilities(unet_record, experts)
            expert_probability = main["probabilities"][:, TARGET_INDICES].copy()
            physical = {
                name: np.zeros((len(indices), *unet_record[name].shape[1:]), dtype=np.float32)
                for name in ("mft", "trajectory", "isrj")
            }
            for subset_offset, sample_index in enumerate(unet_record["sample_indices"]):
                full_offset = position[int(sample_index)]
                expert_probability[full_offset] = subset_expert[subset_offset]
                for name in physical:
                    physical[name][full_offset] = unet_record[name][subset_offset]
        timing["unet_physics_seconds"] += float(time.perf_counter() - started)
        started = time.perf_counter()
        vft_indices = np.asarray(indices, dtype=np.int64)[vft_keep]
        vft_subset = collect_expert(files[power], vft_indices, batch_size, device)
        vft_prediction = main["prediction"][:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]].copy()
        vft_score = np.zeros(len(indices), dtype=np.float32)
        for subset_offset, sample_index in enumerate(vft_subset["sample_index"]):
            full_offset = position[int(sample_index)]
            vft_prediction[full_offset] = vft_subset["prediction"][subset_offset]
            vft_score[full_offset] = vft_subset["score"][subset_offset]
        timing["vft_expert_seconds"] += float(time.perf_counter() - started)
        started = time.perf_counter()
        saw_indices = np.asarray(indices, dtype=np.int64)[saw_keep]
        saw_subset = collect_saw(
            saw_model, files[power], saw_indices, batch_size, device, saw_mean, saw_scale,
        )
        saw_prediction = main["prediction"][:, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]].copy()
        for subset_offset, sample_index in enumerate(saw_subset["sample_index"]):
            saw_prediction[position[int(sample_index)]] = saw_subset["prediction"][subset_offset]
        timing["saw_expert_seconds"] += float(time.perf_counter() - started)
        records.append({
            "sample_indices": main["sample_indices"].astype(np.int64),
            "power": np.full(len(indices), power, dtype=np.int16),
            "labels": main["labels"].astype(np.uint8),
            "main": main["probabilities"].astype(np.float32),
            "expert": expert_probability.astype(np.float32),
            "mft": physical["mft"],
            "trajectory": physical["trajectory"],
            "isrj": physical["isrj"],
            "prediction": main["prediction"].astype(np.float32),
            "target": main["target"].astype(np.float32),
            "mask": main["mask"].astype(np.uint8),
            "saw_prediction": saw_prediction.astype(np.float32),
            "vft_prediction": vft_prediction.astype(np.float32),
            "vft_score": vft_score,
        })
    timing["joint_seconds"] = float(sum(timing.values()))
    timing["sample_views"] = int(len(indices) * len(powers))
    timing["lazy_experts"] = bool(lazy_thresholds is not None)
    if lazy_thresholds is not None:
        timing["isrj_candidate_coverage"] = float(np.mean([
            np.mean(lazy_candidate_masks(record["main"], lazy_thresholds)[0]) for record in records
        ]))
        timing["saw_candidate_coverage"] = float(np.mean([
            np.mean(lazy_candidate_masks(record["main"], lazy_thresholds)[1]) for record in records
        ]))
        timing["vft_candidate_coverage"] = float(np.mean([
            np.mean(lazy_candidate_masks(record["main"], lazy_thresholds)[2]) for record in records
        ]))
    return records, timing


def lazy_candidate_masks(
    probabilities: np.ndarray, thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen inference-only prefilters that cannot change negative-gate parameters."""
    probability = np.asarray(probabilities)
    threshold = np.asarray(thresholds)
    return (
        probability[:, ISRJ_GLOBAL] < threshold[ISRJ_GLOBAL],
        probability[:, SAW_PARAMETER_INDEX[0]] >= threshold[SAW_PARAMETER_INDEX[0]],
        probability[:, VFT_CLASS_INDEX] >= threshold[VFT_CLASS_INDEX],
    )


def role_payload(records: list[dict[str, np.ndarray]], indices: np.ndarray) -> dict[str, np.ndarray]:
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    output: dict[str, list[np.ndarray]] = {name: [] for name in records[0]}
    for record in records:
        keep = np.asarray([int(value) in selected for value in record["sample_indices"]], dtype=bool)
        for name, value in record.items():
            output[name].append(value[keep])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def vft_threshold_payload(payload: dict[str, np.ndarray], probabilities: np.ndarray) -> dict[str, np.ndarray]:
    return {"power": payload["power"], "labels": payload["labels"], "probability": probabilities}


def fused_prediction(payload: dict[str, np.ndarray]) -> np.ndarray:
    prediction = payload["prediction"].copy()
    prediction[:, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]] = payload["saw_prediction"]
    prediction[:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]] = payload["vft_prediction"]
    return prediction.astype(np.float32)


def worst_combo_predicted_gate_nmae(record: dict[str, np.ndarray]) -> float:
    labels = record["labels"].astype(bool)
    predicted = record["probabilities"] >= record["thresholds"][None, :]
    valid = record["mask"].astype(bool) & labels[:, :, None] & predicted[:, :, None]
    combinations = np.packbits(record["labels"], axis=1, bitorder="little")[:, 0]
    values = []
    for combination in np.unique(combinations):
        selected_rows = combinations == combination
        selected_valid = valid & selected_rows[:, None, None]
        if selected_valid.sum() >= 2:
            values.append(macro_nmae(
                record["prediction"], record["target"], selected_valid, record["scale"],
            ))
    return float(max(values)) if values else float("nan")


def target_parameter_metrics(record: dict[str, np.ndarray], parameter_index: tuple[int, int]) -> dict[str, float]:
    class_index, value_index = parameter_index
    truth = record["labels"][:, class_index].astype(bool)
    predicted = record["probabilities"][:, class_index] >= record["thresholds"][class_index]
    valid = truth & record["mask"][:, class_index, value_index].astype(bool)
    finite = np.isfinite(record["target"][:, class_index, value_index])
    valid &= finite
    error = np.abs(
        record["prediction"][:, class_index, value_index]
        - record["target"][:, class_index, value_index]
    ) / float(record["scale"][class_index, value_index])
    gated = valid & predicted
    return {
        "oracle_nmae": float(error[valid].mean()),
        "predicted_gate_nmae": float(error[gated].mean()),
        "miss_penalty_nmae": float(np.where(predicted, error, 1.0)[valid].mean()),
        "coverage": float(gated.sum() / max(valid.sum(), 1)),
        "valid_count": int(valid.sum()), "gated_count": int(gated.sum()),
    }


def variant_record(
    payload: dict[str, np.ndarray], probabilities: np.ndarray,
    thresholds: np.ndarray, prediction: np.ndarray, scale: np.ndarray,
    rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if rows is None:
        rows = np.arange(len(payload["labels"]), dtype=np.int64)
    return {
        "sample_indices": payload["sample_indices"][rows],
        "labels": payload["labels"][rows],
        "probabilities": probabilities[rows],
        "prediction": prediction[rows],
        "target": payload["target"][rows],
        "mask": payload["mask"][rows],
        "thresholds": thresholds,
        "scale": scale,
    }


def evaluate_variant(
    payload: dict[str, np.ndarray], probabilities: np.ndarray,
    thresholds: np.ndarray, prediction: np.ndarray, scale: np.ndarray,
    powers: tuple[int, ...] = POWERS,
) -> dict[str, object]:
    complete = variant_record(payload, probabilities, thresholds, prediction, scale)
    aggregate = gated_metrics(complete)
    aggregate["worst_combo_predicted_gate_nmae"] = worst_combo_predicted_gate_nmae(complete)
    aggregate["sawfm_period"] = target_parameter_metrics(complete, SAW_PARAMETER_INDEX)
    aggregate["vft_doppler"] = target_parameter_metrics(complete, VFT_PARAMETER_INDEX)
    aggregate["classification"] = classification_metrics(
        complete["labels"], complete["probabilities"], thresholds,
    )
    by_power = {}
    for power in powers:
        rows = np.flatnonzero(payload["power"] == power)
        record = variant_record(payload, probabilities, thresholds, prediction, scale, rows)
        weak = _weak_mask(record["labels"].astype(bool), power)
        metrics = _level_metrics(record, weak)
        metrics["worst_combo_predicted_gate_nmae"] = worst_combo_predicted_gate_nmae(record)
        metrics["sawfm_period"] = target_parameter_metrics(record, SAW_PARAMETER_INDEX)
        metrics["vft_doppler"] = target_parameter_metrics(record, VFT_PARAMETER_INDEX)
        metrics["classification"] = classification_metrics(record["labels"], record["probabilities"], thresholds)
        by_power[str(power)] = metrics
    return {"aggregate": aggregate, "by_power": by_power}


def metric_value(record: dict[str, np.ndarray], metric: str) -> float:
    if metric in gated_metrics(record):
        return float(gated_metrics(record)[metric])
    classification = classification_metrics(record["labels"], record["probabilities"], record["thresholds"])
    if metric == "isrj_recall":
        return float(classification["per_class"][ISRJ_GLOBAL]["recall"])
    if metric == "vft_recall":
        return float(classification["per_class"][VFT_CLASS_INDEX]["recall"])
    if metric == "worst_combo_predicted_gate_nmae":
        return worst_combo_predicted_gate_nmae(record)
    raise KeyError(metric)


def paired_identity_bootstrap(
    rows: list[dict[str, object]], candidate: str, baseline: str,
    metric: str, replicates: int, seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = [
        metric_value(row["records"][candidate], metric) - metric_value(row["records"][baseline], metric)
        for row in rows
    ]
    samples = []
    for _ in range(replicates):
        deltas = []
        for row in rows:
            identities = np.asarray(row["identity"])
            unique = np.unique(identities)
            sampled = rng.choice(unique, size=len(unique), replace=True)
            sampled_rows = np.concatenate([np.flatnonzero(identities == value) for value in sampled])
            variant_metrics = {}
            for name in (candidate, baseline):
                source = row["records"][name]
                variant_metrics[name] = {
                    key: value[sampled_rows]
                    if isinstance(value, np.ndarray) and value.shape[:1] == (len(identities),)
                    else value
                    for key, value in source.items()
                }
            deltas.append(metric_value(variant_metrics[candidate], metric) - metric_value(variant_metrics[baseline], metric))
        samples.append(float(np.mean(deltas)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_delta_positive": float((values > 0).mean()),
        "replicates": int(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--specialist-promotion-summary")
    parser.add_argument("--original-power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--data-root", default="generated_data/p1k_joint_level3_seed800000000_rechunked")
    parser.add_argument("--output-root", default="outputs/p1k_joint_level3_system_20260801")
    parser.add_argument("--experiment-name", default="P1k joint Level-3 system confirmation")
    parser.add_argument(
        "--isrj-threshold-policy", choices=("routed_calibrated", "base_calibrated"),
        default="routed_calibrated",
    )
    parser.add_argument("--isrj-route-score", choices=("lower", "mean"), default="lower")
    parser.add_argument("--route-alpha", type=float, default=0.75)
    parser.add_argument("--lazy-experts", action="store_true")
    parser.add_argument("--powers", nargs="+", type=int, default=list(POWERS))
    parser.add_argument(
        "--excluded-identity-files", nargs="*", default=[],
        help="Metadata-only MAT files whose identities must not overlap fresh train or test.",
    )
    parser.add_argument("--defer-statistics", action="store_true")
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    powers = tuple(int(power) for power in args.powers)
    canonical_powers = tuple(range(10, 51, 5))
    if not powers or len(set(powers)) != len(powers) or any(power not in canonical_powers for power in powers):
        raise ValueError(f"powers must be unique values from {canonical_powers}")

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, separator_root = project / args.run_root, project / args.separator_root
    specialist_root, data_root = project / args.specialist_root, project / args.data_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    specialist_promotion = None
    if args.specialist_promotion_summary:
        promotion_path = project / args.specialist_promotion_summary
        specialist_promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        if not specialist_promotion.get("promotion_passed", False):
            raise RuntimeError("specialist promotion summary did not pass its frozen development gates")
        specialist_promotion = {
            "path": str(promotion_path.resolve()),
            "sha256": _sha256(promotion_path),
            "experiment": specialist_promotion.get("experiment"),
            "promotion_gates": specialist_promotion.get("promotion_gates"),
            "p2a_test_opened": specialist_promotion.get("p2a_test_opened"),
            "locked_test_used": specialist_promotion.get("locked_test_used"),
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    original_train = role_file["train"].astype(np.int64)
    original_early = role_file["early"].astype(np.int64)
    original_identities, original_labels = read_identity_metadata(protocol["data_file"])
    with h5py.File(protocol["data_file"], "r") as handle:
        root_parameter_values = np.asarray(handle["parameter_values"], dtype=np.float64)
        root_parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    original_power_root = Path(args.original_power_root)
    original_files = {
        power: original_power_root / f"level_{list(range(10, 51, 5)).index(power)+1:02d}__jnr_{power}dB" / "closedset_6class_train.mat"
        for power in (10, 50)
    }
    normalizer = load_normalizer(protocol["normalizer"])
    scale = normalizer.scale.numpy()
    selected_rows = {
        int(row["seed"]): row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed"
        and row.get("checkpoint_objective") == "parameter"
    }
    train_files = transition_files(data_root, "train", powers)
    train_identities, train_labels, train_pair_audit = audit_paired_power_metadata(train_files)
    if np.intersect1d(train_identities, original_identities).size:
        raise RuntimeError("P1k train identities overlap original identities")
    excluded_identities = []
    for raw_path in args.excluded_identity_files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project / path
        identities, _ = read_identity_metadata(path)
        excluded_identities.append(np.asarray(identities, dtype=np.int64))
        if np.intersect1d(train_identities, identities).size:
            raise RuntimeError(f"fresh train identities overlap excluded identity file: {path}")
    roles = stratified_train_roles(train_labels, 20260805)
    all_train_indices = np.arange(len(train_labels), dtype=np.int64)

    frozen_models: dict[int, dict[str, object]] = {}
    seed_freezes = []
    for seed in MODEL_SEEDS:
        set_seed(seed)
        row = selected_rows[seed]
        raw = row["thresholds"]
        original_thresholds = np.asarray(json.loads(raw) if isinstance(raw, str) else raw, dtype=np.float32)
        main_model, main_checkpoint = _load_main(row, protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)
        expert_train = extract_unet_role(
            unet, original_files[10], original_files[50], original_labels,
            original_train, args.batch_size, seed, device,
        )
        expert_early = extract_unet_role(
            unet, original_files[10], original_files[50], original_labels,
            original_early, args.batch_size, seed + 1, device,
        )
        experts = fit_frozen_experts(expert_train, expert_early, seed)
        saw_model = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        saw_model.load_state_dict(torch.load(saw_checkpoint, map_location=device, weights_only=True))
        saw_model.eval()
        saw_positive = _positive(original_train, original_labels, root_parameter_mask, 1, SAW_PARAMETER_COLUMN)
        saw_indices = _stratified_limit(
            saw_positive, root_parameter_values[SAW_PARAMETER_COLUMN], 2048, seed + 101, "scalar",
        )
        saw_mean = float(np.mean(root_parameter_values[SAW_PARAMETER_COLUMN, saw_indices]))
        saw_scale = float(np.std(root_parameter_values[SAW_PARAMETER_COLUMN, saw_indices]))
        train_records, train_timing = collect_joint_records(
            main_model, unet, experts, saw_model, saw_mean, saw_scale,
            train_files, train_labels, all_train_indices, args.batch_size,
            seed, device, normalizer, powers,
        )
        payloads = {name: role_payload(train_records, indices) for name, indices in roles.items()}
        ensemble, target_audit = fit_score_ensemble(
            payloads["fit"], float(original_thresholds[ISRJ_GLOBAL]), seed + 9000,
        )
        selection_scores = ensemble_scores(
            ensemble, payloads["selection"], float(original_thresholds[ISRJ_GLOBAL]),
        )
        selection = select_coverage_route(
            payloads["selection"], selection_scores[args.isrj_route_score],
            float(original_thresholds[ISRJ_GLOBAL]), args.route_alpha,
        )
        calibration_scores = ensemble_scores(
            ensemble, payloads["calibration"], float(original_thresholds[ISRJ_GLOBAL]),
        )
        routed_calibration, _ = apply_route(
            payloads["calibration"], calibration_scores[args.isrj_route_score], selection, args.route_alpha,
        )
        base_isrj = buffered_isrj_threshold(
            payloads["calibration"], payloads["calibration"]["main"], powers,
        )
        routed_isrj = buffered_isrj_threshold(payloads["calibration"], routed_calibration, powers)
        vft_calibration = buffered_vft_threshold(vft_threshold_payload(
            payloads["calibration"], payloads["calibration"]["main"],
        ), powers)
        safe_thresholds = original_thresholds.copy()
        safe_thresholds[ISRJ_GLOBAL] = float(base_isrj["threshold"])
        safe_thresholds[VFT_CLASS_INDEX] = float(vft_calibration["threshold"])
        routed_thresholds = safe_thresholds.copy()
        routed_thresholds[ISRJ_GLOBAL] = float(
            base_isrj["threshold"]
            if args.isrj_threshold_policy == "base_calibrated"
            else routed_isrj["threshold"]
        )
        seed_freeze = {
            "seed": seed, "main_checkpoint": str(main_checkpoint),
            "unet_checkpoint": str(unet_checkpoint), "saw_checkpoint": str(saw_checkpoint),
            "original_thresholds": original_thresholds,
            "safe_thresholds": safe_thresholds, "routed_thresholds": routed_thresholds,
            "isrj_selection": selection, "base_isrj_calibration": base_isrj,
            "routed_isrj_calibration": routed_isrj, "vft_calibration": vft_calibration,
            "fit_target_audit": target_audit, "saw_mean": saw_mean, "saw_scale": saw_scale,
            "role_hashes": {
                name: {"index": _hash_int64(values), "identity": _hash_int64(train_identities[values])}
                for name, values in roles.items()
            },
            "train_timing": train_timing,
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "seed_freeze.json").write_text(json.dumps(_jsonable(seed_freeze), indent=2), encoding="utf-8")
        joblib.dump({"experts": experts, "ensemble": ensemble}, seed_dir / "frozen_models.joblib")
        frozen_models[seed] = {
            "main": main_model, "unet": unet, "saw": saw_model,
            "experts": experts, "ensemble": ensemble,
        }
        seed_freezes.append(seed_freeze)

    method_freeze = {
        "experiment": args.experiment_name,
        "single_hypothesis": (
            "a risk-calibrated base ISRJ threshold remains valid after a selection-stage route that is constrained to reduce FPR"
            if args.isrj_threshold_policy == "base_calibrated"
            else "individually promoted physics modules retain joint end-to-end benefit under predicted-label gates"
        ),
        "fresh_data_root": str(data_root), "powers_db": list(powers),
        "variants": ["original_iq_grouped", "safe_calibrated_iq_grouped", "safe_isrj_route", "joint_physics_system"],
        "joint_rule": {
            "classification": f"P1g {args.isrj_route_score}-score ISRJ rescue plus per-power CP-buffered ISRJ/VFT thresholds",
            "isrj_threshold_policy": args.isrj_threshold_policy,
            "isrj_route_score": args.isrj_route_score,
            "route_alpha": args.route_alpha,
            "parameters": "unconditional SawFM-period and aligned VFT-Doppler replacement; all other outputs unchanged",
            "mft": "not attached because promotion gate failed",
        },
        "specialist_promotion_receipt": specialist_promotion,
        "lazy_expert_rule": ({
            "isrj": "run UNet physics only for base-negative ISRJ candidates",
            "sawfm": "run period expert only for main-threshold-positive SawFM candidates",
            "vft": "run Doppler expert only for safe-threshold-positive VFT candidates",
            "latency_multiplier_gate": 1.5,
        } if args.lazy_experts else None),
        "promotion_gates": {
            "isrj_mean_recall_gain_vs_safe_main_min": 0.05,
            "isrj_three_of_three_positive": True,
            "maximum_isrj_and_vft_power_fpr": 0.10,
            "macro_aupr_delta_vs_original_min": -0.005,
            "full_predicted_gate_nmae_improvement_vs_safe_route_min": 0.05,
            "full_three_of_three_parameter_improvement": True,
            "miss_penalty_nonworse_than_original": True,
            "worst_combo_parameter_nmae_nonworse_than_safe_route": True,
            "paired_bootstrap_parameter_and_miss_ci_high_below_zero": True,
            "no_power_recall_decline_below_minus_0p02": len(powers) == 9,
        },
        "seed_freezes": seed_freezes,
        "test_used_at_freeze": False, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    freeze_path = output_root / "method_freeze.json"
    freeze_path.write_text(json.dumps(_jsonable(method_freeze), indent=2), encoding="utf-8")
    freeze_hash = _sha256(freeze_path)
    (output_root / "test_access_started.json").write_text(json.dumps({
        "method_freeze_sha256": freeze_hash, "started_unix": time.time(),
        "purpose": "one-time fresh P1k test after complete joint-method freeze",
    }, indent=2), encoding="utf-8")

    test_files = transition_files(data_root, "test", powers)
    test_identities, test_labels, test_pair_audit = audit_paired_power_metadata(test_files)
    excluded_test_overlap = [
        int(np.intersect1d(test_identities, identities).size) for identities in excluded_identities
    ]
    identity_audit = {
        "train_count": int(len(train_identities)), "test_count": int(len(test_identities)),
        "train_hash": _hash_int64(train_identities), "test_hash": _hash_int64(test_identities),
        "train_test_overlap": int(np.intersect1d(train_identities, test_identities).size),
        "test_original_overlap": int(np.intersect1d(test_identities, original_identities).size),
        "train_power_pairing": train_pair_audit,
        "test_power_pairing": test_pair_audit,
        "excluded_identity_file_count": int(len(excluded_identities)),
        "test_excluded_overlaps": excluded_test_overlap,
    }
    if (
        identity_audit["train_test_overlap"]
        or identity_audit["test_original_overlap"]
        or any(excluded_test_overlap)
    ):
        raise RuntimeError("P1k test identity isolation failed")
    test_indices = np.arange(len(test_labels), dtype=np.int64)
    results = []
    bootstrap_rows = []
    for seed_freeze in seed_freezes:
        seed = int(seed_freeze["seed"])
        models = frozen_models[seed]
        payload_records, timing = collect_joint_records(
            models["main"], models["unet"], models["experts"], models["saw"],
            float(seed_freeze["saw_mean"]), float(seed_freeze["saw_scale"]),
            test_files, test_labels, test_indices, args.batch_size,
            seed + 100000, device, normalizer, powers,
            np.asarray(seed_freeze["routed_thresholds"], dtype=np.float32) if args.lazy_experts else None,
        )
        payload = role_payload(payload_records, test_indices)
        original_thresholds = np.asarray(seed_freeze["original_thresholds"], dtype=np.float32)
        safe_thresholds = np.asarray(seed_freeze["safe_thresholds"], dtype=np.float32)
        routed_thresholds = np.asarray(seed_freeze["routed_thresholds"], dtype=np.float32)
        scores = ensemble_scores(models["ensemble"], payload, float(original_thresholds[ISRJ_GLOBAL]))
        routed_probability, route_accept = apply_route(
            payload, scores[args.isrj_route_score], seed_freeze["isrj_selection"], args.route_alpha,
        )
        joint_prediction = fused_prediction(payload)
        variant_arrays = {
            "original_iq_grouped": (payload["main"], original_thresholds, payload["prediction"]),
            "safe_calibrated_iq_grouped": (payload["main"], safe_thresholds, payload["prediction"]),
            "safe_isrj_route": (routed_probability, routed_thresholds, payload["prediction"]),
            "joint_physics_system": (routed_probability, routed_thresholds, joint_prediction),
        }
        variants = {
            name: evaluate_variant(payload, probability, thresholds, prediction, scale, powers)
            for name, (probability, thresholds, prediction) in variant_arrays.items()
        }
        records = {
            name: variant_record(payload, probability, thresholds, prediction, scale)
            for name, (probability, thresholds, prediction) in variant_arrays.items()
        }
        repeated_identity = test_identities[payload["sample_indices"]]
        bootstrap_rows.append({"identity": repeated_identity, "records": records})
        result = {
            "seed": seed, "variants": variants,
            "route_accept_coverage": float(route_accept.mean()), "timing": timing,
            "method_freeze_sha256": freeze_hash, "identity_audit": identity_audit,
            "test_used_once": True, "test_fit_or_selection": False,
            "confirm_used": False, "locked_test_used": False,
        }
        saved = {
            "sample_index": payload["sample_indices"], "receiver_noise_seed": repeated_identity,
            "power_db": payload["power"], "labels": payload["labels"],
            "target": payload["target"], "mask": payload["mask"],
            "original_probability": payload["main"], "routed_probability": routed_probability,
            "base_prediction": payload["prediction"], "joint_prediction": joint_prediction,
            "saw_prediction": payload["saw_prediction"], "vft_prediction": payload["vft_prediction"],
            "isrj_route_score": scores[args.isrj_route_score], "isrj_route_accept": route_accept.astype(np.uint8),
            "original_thresholds": original_thresholds, "safe_thresholds": safe_thresholds,
            "routed_thresholds": routed_thresholds,
        }
        seed_dir = output_root / f"seed_{seed}"
        np.savez_compressed(seed_dir / "test_per_sample.npz", **saved)
        (seed_dir / "test_result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)

    if args.defer_statistics:
        preliminary = {
            "experiment": method_freeze["experiment"], "seed_results": results,
            "method_freeze_sha256": freeze_hash, "identity_audit": identity_audit,
            "statistics_deferred": True,
            "test_used_once": True, "test_fit_or_selection": False,
            "confirm_used": False, "locked_test_used": False,
        }
        (output_root / "results_preliminary.json").write_text(
            json.dumps(_jsonable(preliminary), indent=2), encoding="utf-8",
        )
        print(json.dumps({
            "experiment": method_freeze["experiment"], "statistics_deferred": True,
            "method_freeze_sha256": freeze_hash, "identity_audit": identity_audit,
        }, indent=2))
        return

    comparisons = {}
    comparison_specs = {
        "joint_vs_original": ("joint_physics_system", "original_iq_grouped"),
        "joint_vs_safe_calibrated": ("joint_physics_system", "safe_calibrated_iq_grouped"),
        "joint_vs_safe_route": ("joint_physics_system", "safe_isrj_route"),
        "safe_route_vs_safe_calibrated": ("safe_isrj_route", "safe_calibrated_iq_grouped"),
    }
    metrics = (
        "classification_macro_aupr", "isrj_recall", "predicted_gate_parameter_nmae",
        "miss_penalty_parameter_nmae", "worst_combo_predicted_gate_nmae",
    )
    bootstrap_seed = 20260810
    for comparison_index, (name, (candidate, baseline)) in enumerate(comparison_specs.items()):
        comparisons[name] = {
            metric: paired_identity_bootstrap(
                bootstrap_rows, candidate, baseline, metric, args.bootstrap_replicates,
                bootstrap_seed + comparison_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    def aggregate_metric(variant: str, key: str) -> np.ndarray:
        return np.asarray([row["variants"][variant]["aggregate"][key] for row in results], dtype=np.float64)

    original_nmae = aggregate_metric("original_iq_grouped", "predicted_gate_parameter_nmae")
    route_nmae = aggregate_metric("safe_isrj_route", "predicted_gate_parameter_nmae")
    joint_nmae = aggregate_metric("joint_physics_system", "predicted_gate_parameter_nmae")
    original_miss = aggregate_metric("original_iq_grouped", "miss_penalty_parameter_nmae")
    joint_miss = aggregate_metric("joint_physics_system", "miss_penalty_parameter_nmae")
    route_worst = aggregate_metric("safe_isrj_route", "worst_combo_predicted_gate_nmae")
    joint_worst = aggregate_metric("joint_physics_system", "worst_combo_predicted_gate_nmae")
    isrj_gain = []
    macro_delta = []
    maximum_isrj_fpr = 0.0
    maximum_vft_fpr = 0.0
    for row in results:
        safe = row["variants"]["safe_calibrated_iq_grouped"]
        joint = row["variants"]["joint_physics_system"]
        seed_power_gain = []
        for power in powers:
            base_class = safe["by_power"][str(power)]["classification"]
            joint_class = joint["by_power"][str(power)]["classification"]
            seed_power_gain.append(
                joint_class["per_class"][ISRJ_GLOBAL]["recall"]
                - base_class["per_class"][ISRJ_GLOBAL]["recall"]
            )
            maximum_isrj_fpr = max(maximum_isrj_fpr, joint_class["per_class"][ISRJ_GLOBAL]["fpr"])
            maximum_vft_fpr = max(maximum_vft_fpr, joint_class["per_class"][VFT_CLASS_INDEX]["fpr"])
        isrj_gain.append(float(np.mean(seed_power_gain)))
        macro_delta.append(
            joint["aggregate"]["classification_macro_aupr"]
            - row["variants"]["original_iq_grouped"]["aggregate"]["classification_macro_aupr"]
        )
    nmae_improvement = 1.0 - joint_nmae / route_nmae
    ci_parameter = comparisons["joint_vs_safe_route"]["predicted_gate_parameter_nmae"]
    ci_miss = comparisons["joint_vs_original"]["miss_penalty_parameter_nmae"]
    gates = {
        "isrj_mean_recall_gain_at_least_0p05": bool(np.mean(isrj_gain) >= 0.05),
        "isrj_three_of_three_positive": bool(np.all(np.asarray(isrj_gain) > 0)),
        "maximum_isrj_and_vft_power_fpr_at_most_0p10": bool(max(maximum_isrj_fpr, maximum_vft_fpr) <= 0.10 + 1e-12),
        "macro_aupr_noninferior_vs_original": bool(np.mean(macro_delta) >= -0.005),
        "full_parameter_nmae_improvement_at_least_0p05": bool(np.mean(nmae_improvement) >= 0.05),
        "full_three_of_three_parameter_improvement": bool(np.all(nmae_improvement > 0)),
        "miss_penalty_nonworse_than_original": bool(np.mean(joint_miss - original_miss) <= 0),
        "worst_combo_nonworse_than_safe_route": bool(np.mean(joint_worst - route_worst) <= 0),
        "paired_bootstrap_parameter_and_miss_ci_high_below_zero": bool(
            ci_parameter["ci95_high"] < 0 and ci_miss["ci95_high"] < 0
        ),
    }
    timing_rows = [row["timing"] for row in results]
    main_parameters = sum(parameter.numel() for parameter in frozen_models[MODEL_SEEDS[0]]["main"].parameters())
    unet_parameters = sum(parameter.numel() for parameter in frozen_models[MODEL_SEEDS[0]]["unet"].parameters())
    saw_parameters = sum(parameter.numel() for parameter in frozen_models[MODEL_SEEDS[0]]["saw"].parameters())
    efficiency = {
        "parameters": {
            "main": int(main_parameters), "unet": int(unet_parameters), "saw": int(saw_parameters),
            "joint_neural_total": int(main_parameters + unet_parameters + saw_parameters),
            "vft_aligned_expert_trainable": 0,
        },
        "measured_test_collection": {
            key: float(np.mean([timing[key] for timing in timing_rows]))
            for key in ("main_seconds", "unet_physics_seconds", "vft_expert_seconds", "saw_expert_seconds", "joint_seconds")
        },
    }
    efficiency["measured_test_collection"]["joint_to_main_latency_multiplier"] = float(
        efficiency["measured_test_collection"]["joint_seconds"]
        / efficiency["measured_test_collection"]["main_seconds"]
    )
    output = {
        "experiment": method_freeze["experiment"],
        "recognition": {
            "mean_isrj_recall_gain_vs_safe_calibrated": float(np.mean(isrj_gain)),
            "per_seed_isrj_recall_gain": isrj_gain,
            "maximum_isrj_power_fpr": float(maximum_isrj_fpr),
            "maximum_vft_power_fpr": float(maximum_vft_fpr),
            "mean_macro_aupr_delta_vs_original": float(np.mean(macro_delta)),
        },
        "parameters": {
            "mean_full_predicted_gate_nmae_improvement_vs_safe_route": float(np.mean(nmae_improvement)),
            "per_seed_improvement": nmae_improvement,
            "mean_joint_nmae": float(np.mean(joint_nmae)),
            "mean_safe_route_nmae": float(np.mean(route_nmae)),
            "mean_original_nmae": float(np.mean(original_nmae)),
            "mean_miss_penalty_delta_vs_original": float(np.mean(joint_miss - original_miss)),
            "mean_worst_combo_delta_vs_safe_route": float(np.mean(joint_worst - route_worst)),
        },
        "paired_bootstrap": comparisons, "efficiency": efficiency,
        "gates": gates, "level3_validated": bool(all(gates.values())),
        "method_freeze_sha256": freeze_hash, "identity_audit": identity_audit,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_used_once": True, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(output), indent=2))


if __name__ == "__main__":
    main()
