"""P2j: attach the frozen calibrated SawFM separator to the frozen P2a system."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.data import V73IQDataset
from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main, _load_unet
from cj0724.p1f_isrj_confidence_bound_router import (
    ISRJ_GLOBAL,
    apply_route,
    ensemble_scores,
    transition_files,
)
from cj0724.p1k_joint_level3_system import (
    MODEL_SEEDS,
    SAW_PARAMETER_COLUMN,
    SAW_PARAMETER_INDEX,
    audit_paired_power_metadata,
    collect_joint_records,
    evaluate_variant,
    fused_prediction,
    paired_identity_bootstrap,
    role_payload,
    variant_record,
)
from cj0724.p2a_failure_diagnostics import per_parameter_nmae
from cj0724.paper_core_evidence import load_normalizer
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = tuple(range(10, 51, 5))
COEFFICIENTS = {
    2041: (1.0538332944014202, 7.92056473920431e-06),
    2042: (1.0485594935630393, 5.973457154872555e-06),
    2043: (1.1260814638565335, 2.9172537901531955e-06),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def collect_separator_period(
    separator: torch.nn.Module,
    period_expert: torch.nn.Module,
    path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    mean: float,
    scale: float,
    coefficient_a: float,
    coefficient_b: float,
) -> dict[str, np.ndarray | float]:
    """Infer uncalibrated and frozen-affine period estimates without reading labels."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        empty = np.empty(0, dtype=np.float32)
        return {"sample_index": indices, "uncalibrated": empty, "calibrated": empty, "seconds": 0.0}
    dataset = V73IQDataset(path, indices, [SAW_PARAMETER_COLUMN])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    separator.eval()
    period_expert.eval()
    sample_indices, uncalibrated = [], []
    started = time.perf_counter()
    try:
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            estimate = separator.forward_components(iq, None)["normalized_estimate"].float()
            prediction = period_expert(estimate)["prediction"].float() * float(scale) + float(mean)
            sample_indices.append(batch["sample_index"].numpy().astype(np.int64))
            uncalibrated.append(prediction.cpu().numpy().astype(np.float32))
    finally:
        dataset.close()
    raw = np.concatenate(uncalibrated)
    calibrated = float(coefficient_a) * raw.astype(np.float64) + float(coefficient_b)
    return {
        "sample_index": np.concatenate(sample_indices),
        "uncalibrated": raw,
        "calibrated": calibrated.astype(np.float32),
        "seconds": float(time.perf_counter() - started),
    }


def attach_saw_predictions(
    old_joint: np.ndarray,
    candidate_positions: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only SawFM sweep period at frozen predicted-positive positions."""
    positions = np.asarray(candidate_positions, dtype=np.int64)
    if len(positions) != len(uncalibrated) or len(positions) != len(calibrated):
        raise ValueError("candidate positions and predictions must have equal length")
    uncalibrated_joint = np.asarray(old_joint).copy()
    calibrated_joint = np.asarray(old_joint).copy()
    uncalibrated_joint[positions, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]] = uncalibrated
    calibrated_joint[positions, SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1]] = calibrated
    return uncalibrated_joint, calibrated_joint


def relative_improvement(candidate: float, baseline: float) -> float:
    return float(1.0 - float(candidate) / float(baseline))


def build_promotion_gates(
    seed_rows: list[dict[str, object]],
    bootstrap_vs_old: dict[str, float],
    classification_unchanged: bool,
    other_parameters_unchanged: bool,
) -> dict[str, bool]:
    full_vs_safe = np.asarray([row["full_improvement_vs_safe"] for row in seed_rows], dtype=np.float64)
    full_vs_old = np.asarray([row["full_improvement_vs_old"] for row in seed_rows], dtype=np.float64)
    miss_delta = np.asarray([row["miss_delta_vs_old"] for row in seed_rows], dtype=np.float64)
    worst_delta = np.asarray([row["worst_combo_delta_vs_old"] for row in seed_rows], dtype=np.float64)
    high_reduction = np.asarray([row["high_power_saw_reduction_vs_old"] for row in seed_rows], dtype=np.float64)
    low_ratios = [float(value) for row in seed_rows for value in row["low_power_saw_ratios_vs_old"]]
    return {
        "mean_full_nmae_improvement_vs_safe_at_least_0p05": bool(full_vs_safe.mean() >= 0.05),
        "all_three_full_nmae_improvement_vs_safe_at_least_0p05": bool(np.all(full_vs_safe >= 0.05)),
        "all_three_full_nmae_improvement_vs_old_positive": bool(np.all(full_vs_old > 0.0)),
        "bootstrap_full_nmae_delta_vs_old_ci95_high_at_most_zero": bool(bootstrap_vs_old["ci95_high"] <= 0.0),
        "mean_miss_penalty_delta_vs_old_at_most_zero": bool(miss_delta.mean() <= 0.0),
        "mean_worst_combo_delta_vs_old_at_most_zero": bool(worst_delta.mean() <= 0.0),
        "mean_high_power_saw_reduction_vs_old_at_least_0p30": bool(high_reduction.mean() >= 0.30),
        "all_model_seeds_high_power_saw_reduction_positive": bool(np.all(high_reduction > 0.0)),
        "each_low_power_saw_ratio_vs_old_at_most_1p05": bool(all(value <= 1.05 for value in low_ratios)),
        "classification_arrays_unchanged": bool(classification_unchanged),
        "other_11_parameter_outputs_unchanged": bool(other_parameters_unchanged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--reference-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--p2i-summary", default="outputs/p2i_sawfm_calibrated_separator_confirmation_20260801/summary.json")
    parser.add_argument("--data-root", default="generated_data/p2j_full_parameter_attachment_seed2300000000_rechunked")
    parser.add_argument("--preregister", default="configs/p2j_full_parameter_predicted_gate_attachment_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2j_full_parameter_predicted_gate_attachment_20260801")
    parser.add_argument("--expected-identities", type=int, default=1260)
    parser.add_argument("--required-promotion-summary")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--powers", nargs="+", type=int, default=list(POWERS))
    args = parser.parse_args()
    powers = tuple(int(value) for value in args.powers)
    if not powers or len(set(powers)) != len(powers) or any(value not in POWERS for value in powers):
        raise ValueError(f"powers must be unique values from {POWERS}")

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root = project / args.run_root
    reference_root = project / args.reference_root
    separator_root = project / args.separator_root
    p2i_summary_path = project / args.p2i_summary
    data_root = project / args.data_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    required_promotion = None
    if args.required_promotion_summary:
        required_path = project / args.required_promotion_summary
        required_payload = json.loads(required_path.read_text(encoding="utf-8"))
        if not required_payload.get("promotion_passed", False):
            raise RuntimeError("required promotion summary did not pass")
        required_promotion = {
            "path": str(required_path.resolve()),
            "sha256": sha256_file(required_path),
            "experiment": required_payload.get("experiment"),
        }

    p2i = json.loads(p2i_summary_path.read_text(encoding="utf-8"))
    if not p2i.get("promotion_passed", False):
        raise RuntimeError("P2i confirmation did not pass")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    training_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    selected_rows = {
        int(row["seed"]): row for row in training_summary
        if row["model"] == "mlcnn_iq_grouped_task_routed"
        and row.get("checkpoint_objective") == "parameter"
    }
    normalizer = load_normalizer(protocol["normalizer"])
    scale_array = normalizer.scale.numpy()

    files = transition_files(data_root, "train", powers)
    identities, labels, alignment = audit_paired_power_metadata(files)
    if len(identities) != args.expected_identities:
        raise RuntimeError(f"expected {args.expected_identities} identities, got {len(identities)}")
    original_identities, _ = read_identity_metadata(protocol["data_file"])
    original_overlap = int(np.intersect1d(identities, original_identities).size)
    exclusions = []
    for raw_path in args.excluded_identity_files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project / path
        prior, _ = read_identity_metadata(path)
        overlap = int(np.intersect1d(identities, prior).size)
        exclusions.append({"path": str(path.resolve()), "overlap_count": overlap})
    if original_overlap or any(row["overlap_count"] for row in exclusions):
        raise RuntimeError("P2j identity isolation failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = np.arange(len(labels), dtype=np.int64)
    seed_rows, bootstrap_rows = [], []
    classification_unchanged = True
    other_parameters_unchanged = True
    for seed in MODEL_SEEDS:
        seed_started = time.perf_counter()
        freeze_path = reference_root / f"seed_{seed}" / "seed_freeze.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        frozen = joblib.load(reference_root / f"seed_{seed}" / "frozen_models.joblib")
        main_model, _ = _load_main(selected_rows[seed], protocol, project, device)
        unet = _load_unet(Path(freeze["unet_checkpoint"]), device)
        old_saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        old_saw.load_state_dict(torch.load(freeze["saw_checkpoint"], map_location=device, weights_only=True))
        old_saw.eval()
        separator_checkpoint = separator_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
        separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
        separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
        separator.eval()

        records, base_timing = collect_joint_records(
            main_model, unet, frozen["experts"], old_saw,
            float(freeze["saw_mean"]), float(freeze["saw_scale"]),
            files, labels, indices, args.batch_size, seed, device, normalizer, powers,
        )
        payload = role_payload(records, indices)
        original_thresholds = np.asarray(freeze["original_thresholds"], dtype=np.float32)
        routed_thresholds = np.asarray(freeze["routed_thresholds"], dtype=np.float32)
        scores = ensemble_scores(frozen["ensemble"], payload, float(original_thresholds[ISRJ_GLOBAL]))
        routed_probability, route_accept = apply_route(
            payload, scores["lower"], freeze["isrj_selection"], 1.0,
        )
        old_joint = fused_prediction(payload)
        candidate_gate = routed_probability[:, SAW_PARAMETER_INDEX[0]] >= routed_thresholds[SAW_PARAMETER_INDEX[0]]
        candidate_positions, uncalibrated_values, calibrated_values = [], [], []
        separator_seconds = 0.0
        coefficient_a, coefficient_b = COEFFICIENTS[seed]
        for power in powers:
            positions = np.flatnonzero((payload["power"] == power) & candidate_gate)
            evaluation = collect_separator_period(
                separator, old_saw, files[power], payload["sample_indices"][positions],
                args.batch_size, device, float(freeze["saw_mean"]), float(freeze["saw_scale"]),
                coefficient_a, coefficient_b,
            )
            if not np.array_equal(evaluation["sample_index"], payload["sample_indices"][positions]):
                raise RuntimeError(f"P2j SawFM candidate order mismatch at {power} dB")
            candidate_positions.append(positions)
            uncalibrated_values.append(evaluation["uncalibrated"])
            calibrated_values.append(evaluation["calibrated"])
            separator_seconds += float(evaluation["seconds"])
        candidate_positions_array = np.concatenate(candidate_positions)
        uncalibrated_joint, calibrated_joint = attach_saw_predictions(
            old_joint, candidate_positions_array,
            np.concatenate(uncalibrated_values), np.concatenate(calibrated_values),
        )

        variant_arrays = {
            "R0_safe_isrj_route": payload["prediction"],
            "R1_old_joint": old_joint,
            "R2_uncalibrated_separator": uncalibrated_joint,
            "R3_calibrated_separator": calibrated_joint,
        }
        variants = {
            name: evaluate_variant(payload, routed_probability, routed_thresholds, prediction, scale_array, powers)
            for name, prediction in variant_arrays.items()
        }
        records_for_bootstrap = {
            name: variant_record(payload, routed_probability, routed_thresholds, prediction, scale_array)
            for name, prediction in variant_arrays.items()
        }
        bootstrap_rows.append({"identity": identities[payload["sample_indices"]], "records": records_for_bootstrap})

        safe_nmae = variants["R0_safe_isrj_route"]["aggregate"]["predicted_gate_parameter_nmae"]
        old_nmae = variants["R1_old_joint"]["aggregate"]["predicted_gate_parameter_nmae"]
        candidate_nmae = variants["R3_calibrated_separator"]["aggregate"]["predicted_gate_parameter_nmae"]
        old_miss = variants["R1_old_joint"]["aggregate"]["miss_penalty_parameter_nmae"]
        candidate_miss = variants["R3_calibrated_separator"]["aggregate"]["miss_penalty_parameter_nmae"]
        old_worst = variants["R1_old_joint"]["aggregate"]["worst_combo_predicted_gate_nmae"]
        candidate_worst = variants["R3_calibrated_separator"]["aggregate"]["worst_combo_predicted_gate_nmae"]
        high_old, high_candidate, low_ratios = [], [], []
        for power in powers:
            old_saw_metric = variants["R1_old_joint"]["by_power"][str(power)]["sawfm_period"]["predicted_gate_nmae"]
            candidate_saw_metric = variants["R3_calibrated_separator"]["by_power"][str(power)]["sawfm_period"]["predicted_gate_nmae"]
            if power >= 35:
                high_old.append(old_saw_metric)
                high_candidate.append(candidate_saw_metric)
            else:
                low_ratios.append(float(candidate_saw_metric / old_saw_metric))

        parameter_old = per_parameter_nmae(
            payload["labels"], payload["mask"], routed_probability, routed_thresholds,
            old_joint, payload["target"], scale_array,
        )
        parameter_candidate = per_parameter_nmae(
            payload["labels"], payload["mask"], routed_probability, routed_thresholds,
            calibrated_joint, payload["target"], scale_array,
        )
        unchanged_mask = np.ones((6, 2), dtype=bool)
        unchanged_mask[SAW_PARAMETER_INDEX] = False
        other_parameters_unchanged &= bool(np.array_equal(old_joint[:, unchanged_mask], calibrated_joint[:, unchanged_mask]))
        classification_unchanged &= bool(
            np.array_equal(records_for_bootstrap["R1_old_joint"]["probabilities"], records_for_bootstrap["R3_calibrated_separator"]["probabilities"])
            and np.array_equal(records_for_bootstrap["R1_old_joint"]["thresholds"], records_for_bootstrap["R3_calibrated_separator"]["thresholds"])
        )
        seed_row = {
            "seed": seed,
            "variants": variants,
            "full_improvement_vs_safe": relative_improvement(candidate_nmae, safe_nmae),
            "full_improvement_vs_old": relative_improvement(candidate_nmae, old_nmae),
            "miss_delta_vs_old": float(candidate_miss - old_miss),
            "worst_combo_delta_vs_old": float(candidate_worst - old_worst),
            "high_power_saw_reduction_vs_old": relative_improvement(float(np.mean(high_candidate)), float(np.mean(high_old))),
            "low_power_saw_ratios_vs_old": low_ratios,
            "per_parameter_old_nmae": parameter_old,
            "per_parameter_candidate_nmae": parameter_candidate,
            "candidate_coverage": float(candidate_gate.mean()),
            "route_accept_coverage": float(route_accept.mean()),
            "timing": {
                **base_timing,
                "candidate_separator_seconds": separator_seconds,
                "candidate_total_seconds": float(base_timing["joint_seconds"] + separator_seconds),
                "elapsed_seed_seconds": float(time.perf_counter() - seed_started),
            },
            "receipts": {
                "reference_seed_freeze_sha256": sha256_file(freeze_path),
                "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
                "coefficients": {"a": coefficient_a, "b": coefficient_b},
            },
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        np.savez_compressed(
            seed_dir / "development_per_sample.npz",
            receiver_noise_seed=identities[payload["sample_indices"]],
            sample_index=payload["sample_indices"], power_db=payload["power"], labels=payload["labels"],
            target=payload["target"], mask=payload["mask"], routed_probability=routed_probability,
            routed_thresholds=routed_thresholds, parameter_scale=scale_array,
            safe_prediction=payload["prediction"], old_joint_prediction=old_joint,
            uncalibrated_joint_prediction=uncalibrated_joint, calibrated_joint_prediction=calibrated_joint,
        )
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(seed_row), indent=2), encoding="utf-8")
        seed_rows.append(seed_row)
        print(json.dumps({
            "seed": seed,
            "full_improvement_vs_safe": seed_row["full_improvement_vs_safe"],
            "full_improvement_vs_old": seed_row["full_improvement_vs_old"],
            "high_power_saw_reduction_vs_old": seed_row["high_power_saw_reduction_vs_old"],
            "max_low_power_saw_ratio_vs_old": max(low_ratios),
        }), flush=True)
        del main_model, unet, old_saw, separator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bootstrap_vs_old = paired_identity_bootstrap(
        bootstrap_rows, "R3_calibrated_separator", "R1_old_joint",
        "predicted_gate_parameter_nmae", args.bootstrap_replicates, 20260821,
    )
    bootstrap_miss_vs_old = paired_identity_bootstrap(
        bootstrap_rows, "R3_calibrated_separator", "R1_old_joint",
        "miss_penalty_parameter_nmae", args.bootstrap_replicates, 20260822,
    )
    gates = build_promotion_gates(
        seed_rows, bootstrap_vs_old, classification_unchanged, other_parameters_unchanged,
    )
    passed = bool(all(gates.values()))
    if passed:
        interpretation = "P2j passed; freeze the calibrated predicted-gate SawFM attachment for two new identity-seed confirmation runs."
    else:
        high_pass = gates["mean_high_power_saw_reduction_vs_old_at_least_0p30"]
        low_pass = gates["each_low_power_saw_ratio_vs_old_at_most_1p05"]
        if high_pass and not low_pass:
            interpretation = "P2j isolated a low-power negative-transfer boundary; develop an inference-only bounded reliability selector on a new seed."
        elif high_pass:
            interpretation = "The SawFM mechanism transfers, but a system-level or worst-combination gate failed; diagnose saved development predictions before any new model change."
        else:
            interpretation = "The confirmed high-power component estimator did not transfer to the full system; stop attachment and inspect predicted-gate composition coverage."
    summary = {
        "experiment": "P2j full 12-parameter predicted-gate SawFM attachment development",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "reference_method_freeze_sha256": sha256_file(reference_root / "method_freeze.json"),
        "p2i_summary_sha256": sha256_file(p2i_summary_path),
        "required_promotion": required_promotion,
        "identity_audit": {
            "identity_count": int(len(identities)),
            "identity_sha256": hashlib.sha256(np.asarray(identities, dtype="<i8").tobytes()).hexdigest(),
            "power_alignment": alignment,
            "overlap_with_original_training": original_overlap,
            "excluded_prior_splits": exclusions,
        },
        "promotion_gates": gates,
        "promotion_passed": passed,
        "paired_identity_bootstrap": {
            "calibrated_vs_old_predicted_gate_parameter_nmae": bootstrap_vs_old,
            "calibrated_vs_old_miss_penalty_parameter_nmae": bootstrap_miss_vs_old,
        },
        "aggregate": {
            "mean_full_improvement_vs_safe": float(np.mean([row["full_improvement_vs_safe"] for row in seed_rows])),
            "mean_full_improvement_vs_old": float(np.mean([row["full_improvement_vs_old"] for row in seed_rows])),
            "mean_high_power_saw_reduction_vs_old": float(np.mean([row["high_power_saw_reduction_vs_old"] for row in seed_rows])),
            "maximum_low_power_saw_ratio_vs_old": float(max(value for row in seed_rows for value in row["low_power_saw_ratios_vs_old"])),
            "mean_candidate_coverage": float(np.mean([row["candidate_coverage"] for row in seed_rows])),
            "mean_candidate_latency_seconds": float(np.mean([row["timing"]["candidate_separator_seconds"] for row in seed_rows])),
        },
        "interpretation": interpretation,
        "runs": seed_rows,
        "development_only": True,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "promotion_gates": gates, "promotion_passed": passed,
        "aggregate": summary["aggregate"], "interpretation": interpretation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
