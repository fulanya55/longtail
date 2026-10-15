"""P1i: attach the frozen P1h VFT Doppler expert through reliability gating."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main, collect_main
from cj0724.multitask import PARAMETER_COLUMNS
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles
from cj0724.paper_core_evidence import load_normalizer
from cj0724.vft_doppler_aligned_expert import (
    POWERS,
    VFT_CLASS_INDEX,
    VFT_DOPPLER_COLUMN,
    estimate_aligned_doppler,
    read_vft_metadata,
    transition_file,
)


VFT_PARAMETER_INDEX = (4, 1)
MODEL_SEEDS = (2041, 2042, 2043)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


@torch.inference_mode()
def collect_expert(path: Path, indices: np.ndarray, batch_size: int, device: torch.device):
    dataset = V73IQDataset(path, indices, [VFT_DOPPLER_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    output = {"sample_index": [], "prediction": [], "score": []}
    for batch in loader:
        estimate = estimate_aligned_doppler(batch["iq"].to(device, non_blocking=True))
        output["sample_index"].append(batch["sample_index"].numpy())
        output["prediction"].append(estimate["prediction_hz"].cpu().numpy())
        output["score"].append(estimate["peak_score"].cpu().numpy())
    dataset.close()
    return {name: np.concatenate(parts) for name, parts in output.items()}


def collect_payload(model, files, identities, indices, batch_size, device, normalizer):
    parts = {name: [] for name in (
        "sample_index", "identity", "power", "labels", "probability", "base_prediction",
        "target", "valid", "expert_prediction", "expert_score",
    )}
    for power in POWERS:
        metadata = read_vft_metadata(files[power])
        if not np.array_equal(metadata["identity"], identities):
            raise RuntimeError("power identity order differs")
        main = collect_main(model, files[power], indices, batch_size, device, normalizer)
        expert = collect_expert(files[power], indices, batch_size, device)
        if not np.array_equal(main["sample_indices"], expert["sample_index"]):
            raise RuntimeError("main and expert sample order differs")
        parts["sample_index"].append(indices)
        parts["identity"].append(identities[indices])
        parts["power"].append(np.full(len(indices), power, dtype=np.int16))
        parts["labels"].append(main["labels"].astype(np.uint8))
        parts["probability"].append(main["probabilities"].astype(np.float32))
        parts["base_prediction"].append(main["prediction"][:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]])
        parts["target"].append(main["target"][:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]])
        parts["valid"].append(
            main["mask"][:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]].astype(bool)
            & (main["labels"][:, VFT_CLASS_INDEX] > 0)
            & np.isfinite(main["target"][:, VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1]])
        )
        parts["expert_prediction"].append(expert["prediction"].astype(np.float32))
        parts["expert_score"].append(expert["score"].astype(np.float32))
    return {name: np.concatenate(values) for name, values in parts.items()}


def role_payload(payload, role_indices):
    keep = np.isin(payload["sample_index"], np.asarray(role_indices, dtype=np.int64))
    return {name: value[keep] for name, value in payload.items()}


def parameter_metrics(payload, prediction, class_threshold, scale):
    valid = payload["valid"]
    predicted = payload["probability"][:, VFT_CLASS_INDEX] >= float(class_threshold)
    gated = valid & predicted
    normalized_error = np.abs(prediction - payload["target"]) / float(scale)
    combo = np.packbits(payload["labels"], axis=1, bitorder="little")[:, 0]
    combo_values = []
    for value in np.unique(combo[gated]):
        selected = gated & (combo == value)
        if selected.sum() >= 2:
            combo_values.append(float(normalized_error[selected].mean()))
    return {
        "oracle_nmae": float(normalized_error[valid].mean()),
        "predicted_gate_nmae": float(normalized_error[gated].mean()),
        "miss_penalty_nmae": float(np.where(predicted, normalized_error, 1.0)[valid].mean()),
        "prediction_gate_coverage": float(gated.sum() / max(valid.sum(), 1)),
        "worst_combo_predicted_gate_nmae": float(max(combo_values)) if combo_values else float("nan"),
        "valid_count": int(valid.sum()),
        "gated_count": int(gated.sum()),
    }


def candidate_prediction(payload, cutoff):
    accept = payload["expert_score"] >= float(cutoff)
    prediction = np.where(accept, payload["expert_prediction"], payload["base_prediction"])
    return prediction.astype(np.float32), accept


def select_reliability_cutoff(payload, class_threshold, scale):
    predicted = payload["probability"][:, VFT_CLASS_INDEX] >= float(class_threshold)
    selected_scores = payload["expert_score"][payload["valid"] & predicted]
    candidates = np.unique(np.concatenate((
        np.asarray([0.0]), np.quantile(selected_scores, [0.25, 0.50, 0.75, 0.90])
    )))
    baseline = parameter_metrics(payload, payload["base_prediction"], class_threshold, scale)
    unconditional, _ = candidate_prediction(payload, 0.0)
    unconditional_metrics = parameter_metrics(payload, unconditional, class_threshold, scale)
    rows = []
    for cutoff in candidates:
        prediction, accept = candidate_prediction(payload, cutoff)
        metrics = parameter_metrics(payload, prediction, class_threshold, scale)
        gated = payload["valid"] & predicted
        accept_coverage = float(accept[gated].mean()) if gated.any() else 0.0
        rows.append({"cutoff": float(cutoff), "accept_coverage": accept_coverage, **metrics})
    eligible = [row for row in rows if (
        row["accept_coverage"] >= 0.25
        and row["predicted_gate_nmae"] <= baseline["predicted_gate_nmae"]
        and row["worst_combo_predicted_gate_nmae"] <= unconditional_metrics["worst_combo_predicted_gate_nmae"]
    )]
    chosen = min(
        eligible or rows,
        key=lambda row: (row["worst_combo_predicted_gate_nmae"], row["predicted_gate_nmae"], -row["accept_coverage"]),
    )
    return {"chosen": chosen, "candidates": rows, "baseline": baseline,
            "unconditional": unconditional_metrics}


def classification_metrics(payload, thresholds):
    labels = payload["labels"].astype(bool)
    probabilities = payload["probability"]
    predicted = probabilities >= thresholds[None, :]
    truth = labels[:, VFT_CLASS_INDEX]
    return {
        "macro_aupr": float(average_precision_score(labels, probabilities, average="macro")),
        "vft_recall": float(predicted[truth, VFT_CLASS_INDEX].mean()),
        "vft_fpr": float(predicted[~truth, VFT_CLASS_INDEX].mean()),
    }


def grouped_bootstrap(records, metric, replicates=1000, seed=20260801):
    rng = np.random.default_rng(seed)
    observed = []
    for record in records:
        observed.append(record["reliability"][metric] - record["baseline"][metric])
    values = []
    for _ in range(replicates):
        seed_deltas = []
        for record in records:
            payload = record["payload"]
            identities = np.unique(payload["identity"])
            sampled = rng.choice(identities, len(identities), replace=True)
            rows = np.concatenate([np.flatnonzero(payload["identity"] == value) for value in sampled])
            subset = {name: value[rows] for name, value in payload.items()}
            base = parameter_metrics(subset, subset["base_prediction"], record["class_threshold"], record["scale"])
            candidate, _ = candidate_prediction(subset, record["cutoff"])
            final = parameter_metrics(subset, candidate, record["class_threshold"], record["scale"])
            seed_deltas.append(final[metric] - base[metric])
        values.append(float(np.mean(seed_deltas)))
    values = np.asarray(values)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "replicates": int(replicates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--data-root", default="generated_data/p1i_vft_attachment_seed600000000_rechunked")
    parser.add_argument("--output-root", default="outputs/p1i_vft_reliability_gated_attachment_20260801")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, data_root, output_root = project / args.run_root, project / args.data_root, project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    selected_rows = {int(row["seed"]): row for row in summary if (
        row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    )}
    normalizer = load_normalizer(protocol["normalizer"])
    scale = float(normalizer.scale[VFT_PARAMETER_INDEX])
    train_files = {power: transition_file(data_root, "train", power) for power in POWERS}
    train_identities, train_labels = read_vft_metadata(train_files[20])["identity"], read_vft_metadata(train_files[20])["labels"]
    roles = stratified_train_roles(train_labels, 20260801)
    indices = np.arange(len(train_identities), dtype=np.int64)

    frozen_models, seed_freezes = {}, []
    for seed in MODEL_SEEDS:
        row = selected_rows[seed]
        raw_thresholds = row["thresholds"]
        thresholds = np.asarray(
            json.loads(raw_thresholds) if isinstance(raw_thresholds, str) else raw_thresholds,
            dtype=np.float32,
        )
        model, checkpoint = _load_main(row, protocol, project, device)
        payload = collect_payload(model, train_files, train_identities, indices, args.batch_size, device, normalizer)
        selection = role_payload(payload, roles["selection"])
        calibration = role_payload(payload, roles["calibration"])
        selected = select_reliability_cutoff(selection, thresholds[VFT_CLASS_INDEX], scale)
        cutoff = float(selected["chosen"]["cutoff"])
        calibration_prediction, calibration_accept = candidate_prediction(calibration, cutoff)
        calibration_metrics = parameter_metrics(calibration, calibration_prediction, thresholds[VFT_CLASS_INDEX], scale)
        calibration_base = parameter_metrics(calibration, calibration["base_prediction"], thresholds[VFT_CLASS_INDEX], scale)
        calibration_unconditional = parameter_metrics(calibration, calibration["expert_prediction"], thresholds[VFT_CLASS_INDEX], scale)
        fallback = None
        if (calibration_metrics["predicted_gate_nmae"] > calibration_base["predicted_gate_nmae"]
                or calibration_metrics["worst_combo_predicted_gate_nmae"] > calibration_unconditional["worst_combo_predicted_gate_nmae"]):
            cutoff, fallback = 0.0, "calibration_fallback_to_unconditional"
            calibration_prediction, calibration_accept = candidate_prediction(calibration, cutoff)
            calibration_metrics = parameter_metrics(calibration, calibration_prediction, thresholds[VFT_CLASS_INDEX], scale)
        seed_freezes.append({
            "seed": seed, "checkpoint": str(checkpoint), "thresholds": thresholds,
            "selected_cutoff": cutoff, "selection": selected,
            "calibration_metrics": calibration_metrics,
            "calibration_accept_coverage": float(calibration_accept.mean()), "fallback": fallback,
            "role_hashes": {name: _hash_int64(values) for name, values in roles.items()},
        })
        frozen_models[seed] = model

    freeze = {
        "experiment": "P1i predicted-label VFT Doppler reliability attachment",
        "recent_evidence": {
            "P1g": "per-power risk buffer validated Level-1 classification safety",
            "P1h": "aligned VFT Doppler expert passed 7/7 parameter gates",
            "P1h_boundary": "some FMB/MFT/SawFM compound combinations still regress",
        },
        "roles": "40 identities per combination split 20 fit / 10 selection / 10 calibration",
        "variants": ["IQ-Grouped baseline", "unconditional aligned expert", "reliability-gated aligned expert"],
        "selection_rule": "minimize worst-combination then mean predicted-gate NMAE; expert coverage >=0.25; no worse than baseline mean or unconditional worst-combination",
        "calibration_fallback": "fallback to unconditional if selected gate is worse than baseline mean or unconditional worst-combination",
        "promotion_gates": {
            "mean_predicted_gate_nmae_improvement_vs_baseline_min": 0.10,
            "three_of_three_improvement": True,
            "miss_penalty_nonworse": True,
            "worst_combo_improvement_vs_unconditional_min": 0.05,
            "paired_bootstrap_nmae_ci_high_below_zero": True,
            "maximum_vft_power_fpr": 0.10,
            "classification_outputs_unchanged": True,
        },
        "seed_freezes": seed_freezes, "test_used_at_freeze": False,
        "test_fit_or_selection": False, "confirm_used": False, "locked_test_used": False,
    }
    freeze_path = output_root / "method_freeze.json"
    freeze_path.write_text(json.dumps(_jsonable(freeze), indent=2), encoding="utf-8")
    freeze_hash = _sha256(freeze_path)
    (output_root / "test_access_started.json").write_text(json.dumps({
        "method_freeze_sha256": freeze_hash, "started_unix": time.time(),
        "purpose": "one-time P1i fresh test after complete attachment freeze",
    }, indent=2), encoding="utf-8")

    test_files = {power: transition_file(data_root, "test", power) for power in POWERS}
    test_meta = read_vft_metadata(test_files[20])
    if np.intersect1d(train_identities, test_meta["identity"]).size:
        raise RuntimeError("P1i train/test identity overlap")
    test_indices = np.arange(len(test_meta["identity"]), dtype=np.int64)
    results, bootstrap_records = [], []
    for seed_freeze in seed_freezes:
        seed = int(seed_freeze["seed"])
        thresholds = np.asarray(seed_freeze["thresholds"], dtype=np.float32)
        payload = collect_payload(frozen_models[seed], test_files, test_meta["identity"], test_indices,
                                  args.batch_size, device, normalizer)
        base_prediction = payload["base_prediction"]
        unconditional_prediction = payload["expert_prediction"]
        reliability_prediction, accept = candidate_prediction(payload, seed_freeze["selected_cutoff"])
        variants = {
            "baseline": parameter_metrics(payload, base_prediction, thresholds[VFT_CLASS_INDEX], scale),
            "unconditional": parameter_metrics(payload, unconditional_prediction, thresholds[VFT_CLASS_INDEX], scale),
            "reliability": parameter_metrics(payload, reliability_prediction, thresholds[VFT_CLASS_INDEX], scale),
        }
        by_power = {}
        for power in POWERS:
            keep = payload["power"] == power
            sub = {name: value[keep] for name, value in payload.items()}
            by_power[str(power)] = {
                "classification": classification_metrics(sub, thresholds),
                "baseline": parameter_metrics(sub, sub["base_prediction"], thresholds[VFT_CLASS_INDEX], scale),
                "unconditional": parameter_metrics(sub, sub["expert_prediction"], thresholds[VFT_CLASS_INDEX], scale),
                "reliability": parameter_metrics(
                    sub, candidate_prediction(sub, seed_freeze["selected_cutoff"])[0],
                    thresholds[VFT_CLASS_INDEX], scale,
                ),
            }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "test_per_sample.npz", **payload,
                            reliability_prediction=reliability_prediction,
                            reliability_accept=accept.astype(np.uint8), thresholds=thresholds)
        result = {
            "seed": seed, "cutoff": seed_freeze["selected_cutoff"], "variants": variants,
            "by_power": by_power, "method_freeze_sha256": freeze_hash,
            "test_used_once": True, "test_fit_or_selection": False,
            "confirm_used": False, "locked_test_used": False,
        }
        (seed_dir / "test_result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        bootstrap_records.append({
            "payload": payload, "baseline": variants["baseline"], "reliability": variants["reliability"],
            "class_threshold": float(thresholds[VFT_CLASS_INDEX]), "scale": scale,
            "cutoff": float(seed_freeze["selected_cutoff"]),
        })

    ci_nmae = grouped_bootstrap(bootstrap_records, "predicted_gate_nmae", args.bootstrap_replicates)
    ci_miss = grouped_bootstrap(bootstrap_records, "miss_penalty_nmae", args.bootstrap_replicates, 20260802)
    base_nmae = np.asarray([row["variants"]["baseline"]["predicted_gate_nmae"] for row in results])
    rel_nmae = np.asarray([row["variants"]["reliability"]["predicted_gate_nmae"] for row in results])
    uncond_worst = np.asarray([row["variants"]["unconditional"]["worst_combo_predicted_gate_nmae"] for row in results])
    rel_worst = np.asarray([row["variants"]["reliability"]["worst_combo_predicted_gate_nmae"] for row in results])
    improvement = 1.0 - rel_nmae / base_nmae
    worst_improvement = 1.0 - rel_worst.mean() / uncond_worst.mean()
    max_fpr = max(row["by_power"][str(power)]["classification"]["vft_fpr"]
                  for row in results for power in POWERS)
    gates = {
        "mean_predicted_gate_nmae_improvement_at_least_0p10": bool(improvement.mean() >= 0.10),
        "three_of_three_improvement": bool(np.all(improvement > 0)),
        "miss_penalty_nonworse": bool(ci_miss["observed_mean_delta"] <= 0),
        "worst_combo_improvement_vs_unconditional_at_least_0p05": bool(worst_improvement >= 0.05),
        "paired_bootstrap_nmae_ci_high_below_zero": bool(ci_nmae["ci95_high"] < 0),
        "maximum_vft_power_fpr_at_most_0p10": bool(max_fpr <= 0.10 + 1e-12),
        "classification_outputs_unchanged": True,
    }
    summary_payload = {
        "experiment": freeze["experiment"],
        "mean_predicted_gate_nmae_improvement_vs_baseline": float(improvement.mean()),
        "per_seed_improvement": improvement,
        "mean_worst_combo_improvement_vs_unconditional": float(worst_improvement),
        "maximum_vft_power_fpr": float(max_fpr),
        "paired_bootstrap": {"predicted_gate_nmae": ci_nmae, "miss_penalty_nmae": ci_miss},
        "gates": gates, "innovation_validated": bool(all(gates.values())),
        "method_freeze_sha256": freeze_hash, "elapsed_seconds": float(time.perf_counter() - started),
        "test_used_once": True, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary_payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary_payload), indent=2))


if __name__ == "__main__":
    main()
