"""P1j: unconditional VFT Doppler expert plus per-power safe classification calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cj0724.independent_worst_power_calibration import clopper_pearson_upper
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles
from cj0724.p1i_vft_reliability_attachment import (
    MODEL_SEEDS,
    VFT_CLASS_INDEX,
    VFT_PARAMETER_INDEX,
    classification_metrics,
    collect_payload,
    parameter_metrics,
    role_payload,
)
from cj0724.paper_core_evidence import load_normalizer
from cj0724.vft_doppler_aligned_expert import POWERS, read_vft_metadata, transition_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def buffered_vft_threshold(payload, powers=POWERS):
    rows = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        by_power, feasible = [], True
        for power in powers:
            selected = payload["power"] == power
            truth = payload["labels"][selected, VFT_CLASS_INDEX].astype(bool)
            predicted = payload["probability"][selected, VFT_CLASS_INDEX] >= threshold
            fp, negatives = int(predicted[~truth].sum()), int((~truth).sum())
            fpr = fp / max(negatives, 1)
            upper = clopper_pearson_upper(fp, negatives)
            feasible &= fpr <= 0.06 + 1e-12 and upper <= 0.08 + 1e-12
            by_power.append({
                "power_db": power, "false_positives": fp, "negatives": negatives,
                "observed_fpr": fpr, "cp95_upper": upper,
                "recall": float(predicted[truth].mean()),
            })
        if feasible:
            rows.append({
                "threshold": float(threshold),
                "mean_recall": float(np.mean([row["recall"] for row in by_power])),
                "by_power": by_power,
            })
    if not rows:
        return {"threshold": 1.0, "mean_recall": 0.0, "by_power": [], "feasible": False}
    selected = max(rows, key=lambda row: (row["mean_recall"], -row["threshold"]))
    selected.update({"feasible": True, "candidate_count": len(rows)})
    return selected


def paired_bootstrap(records, candidate_key, baseline_key, metric, replicates, seed):
    rng = np.random.default_rng(seed)
    observed = [row[candidate_key][metric] - row[baseline_key][metric] for row in records]
    samples = []
    for _ in range(replicates):
        deltas = []
        for row in records:
            payload = row["payload"]
            identities = np.unique(payload["identity"])
            sampled = rng.choice(identities, len(identities), replace=True)
            indices = np.concatenate([np.flatnonzero(payload["identity"] == value) for value in sampled])
            subset = {name: value[indices] for name, value in payload.items()}
            candidate_prediction = subset["expert_prediction"] if candidate_key == "safe_expert" else subset["base_prediction"]
            baseline_prediction = subset["base_prediction"]
            candidate_threshold = row["safe_threshold"] if candidate_key.startswith("safe") else row["original_threshold"]
            baseline_threshold = row["safe_threshold"] if baseline_key.startswith("safe") else row["original_threshold"]
            cm = parameter_metrics(subset, candidate_prediction, candidate_threshold, row["scale"])[metric]
            bm = parameter_metrics(subset, baseline_prediction, baseline_threshold, row["scale"])[metric]
            deltas.append(cm - bm)
        samples.append(float(np.mean(deltas)))
    values = np.asarray(samples)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "replicates": int(replicates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--data-root", default="generated_data/p1j_vft_safe_attachment_seed700000000_rechunked")
    parser.add_argument("--output-root", default="outputs/p1j_vft_safe_attachment_20260801")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    start = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, data_root, output_root = project / args.run_root, project / args.data_root, project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    rows = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed"
            and row.get("checkpoint_objective") == "parameter"}
    normalizer = load_normalizer(protocol["normalizer"])
    scale = float(normalizer.scale[VFT_PARAMETER_INDEX])
    train_files = {power: transition_file(data_root, "train", power) for power in POWERS}
    train_meta = read_vft_metadata(train_files[20])
    roles = stratified_train_roles(train_meta["labels"], 20260803)
    indices = np.arange(len(train_meta["identity"]), dtype=np.int64)
    models, freezes = {}, []
    for seed in MODEL_SEEDS:
        row = rows[seed]
        raw = row["thresholds"]
        original = np.asarray(json.loads(raw) if isinstance(raw, str) else raw, dtype=np.float32)
        model, checkpoint = _load_main(row, protocol, project, device)
        payload = collect_payload(model, train_files, train_meta["identity"], indices,
                                  args.batch_size, device, normalizer)
        calibration = role_payload(payload, roles["calibration"])
        calibrated = buffered_vft_threshold(calibration)
        safe = original.copy(); safe[VFT_CLASS_INDEX] = calibrated["threshold"]
        freezes.append({
            "seed": seed, "checkpoint": str(checkpoint), "original_thresholds": original,
            "safe_thresholds": safe, "vft_calibration": calibrated,
        })
        models[seed] = model

    freeze = {
        "experiment": "P1j safe predicted-gate VFT Doppler attachment",
        "hypothesis": "unconditional aligned parameter replacement plus per-power CP-buffered VFT classification threshold",
        "variants": ["original IQ-Grouped", "safe IQ-Grouped", "safe unconditional aligned expert"],
        "threshold_buffer": {"each_power_observed_fpr_max": 0.06, "each_power_cp95_max": 0.08},
        "promotion_gates": {
            "safe_expert_predicted_nmae_improvement_vs_safe_baseline_min": 0.30,
            "three_of_three_improvement": True,
            "maximum_vft_power_fpr": 0.10,
            "vft_recall_retention_vs_original_min": 0.75,
            "safe_expert_miss_penalty_nonworse_than_original": True,
            "worst_combo_improvement_vs_safe_baseline_min": 0.15,
            "paired_bootstrap_nmae_and_miss_ci_high_below_zero": True,
        },
        "seed_freezes": freezes, "test_used_at_freeze": False,
        "test_fit_or_selection": False, "confirm_used": False, "locked_test_used": False,
    }
    freeze_path = output_root / "method_freeze.json"
    freeze_path.write_text(json.dumps(_jsonable(freeze), indent=2), encoding="utf-8")
    freeze_hash = _sha256(freeze_path)
    (output_root / "test_access_started.json").write_text(json.dumps({
        "method_freeze_sha256": freeze_hash, "started_unix": time.time(),
        "purpose": "one-time P1j test after safe attachment freeze",
    }, indent=2), encoding="utf-8")

    test_files = {power: transition_file(data_root, "test", power) for power in POWERS}
    test_meta = read_vft_metadata(test_files[20])
    if np.intersect1d(train_meta["identity"], test_meta["identity"]).size:
        raise RuntimeError("P1j train/test identity overlap")
    test_indices = np.arange(len(test_meta["identity"]), dtype=np.int64)
    results, boot = [], []
    for seed_freeze in freezes:
        seed = seed_freeze["seed"]
        original = np.asarray(seed_freeze["original_thresholds"], dtype=np.float32)
        safe = np.asarray(seed_freeze["safe_thresholds"], dtype=np.float32)
        payload = collect_payload(models[seed], test_files, test_meta["identity"], test_indices,
                                  args.batch_size, device, normalizer)
        variants = {
            "original_baseline": parameter_metrics(payload, payload["base_prediction"], original[VFT_CLASS_INDEX], scale),
            "safe_baseline": parameter_metrics(payload, payload["base_prediction"], safe[VFT_CLASS_INDEX], scale),
            "safe_expert": parameter_metrics(payload, payload["expert_prediction"], safe[VFT_CLASS_INDEX], scale),
        }
        by_power = {}
        for power in POWERS:
            keep = payload["power"] == power
            sub = {name: value[keep] for name, value in payload.items()}
            by_power[str(power)] = {
                "original_classification": classification_metrics(sub, original),
                "safe_classification": classification_metrics(sub, safe),
                "safe_expert": parameter_metrics(sub, sub["expert_prediction"], safe[VFT_CLASS_INDEX], scale),
            }
        result = {"seed": seed, "variants": variants, "by_power": by_power,
                  "safe_vft_threshold": float(safe[VFT_CLASS_INDEX]),
                  "method_freeze_sha256": freeze_hash, "test_used_once": True,
                  "test_fit_or_selection": False, "confirm_used": False, "locked_test_used": False}
        seed_dir = output_root / f"seed_{seed}"; seed_dir.mkdir()
        np.savez_compressed(seed_dir / "test_per_sample.npz", **payload,
                            original_thresholds=original, safe_thresholds=safe)
        (seed_dir / "test_result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        boot.append({"payload": payload, **variants, "original_threshold": float(original[VFT_CLASS_INDEX]),
                     "safe_threshold": float(safe[VFT_CLASS_INDEX]), "scale": scale})

    ci_nmae = paired_bootstrap(boot, "safe_expert", "safe_baseline", "predicted_gate_nmae",
                               args.bootstrap_replicates, 20260803)
    ci_miss = paired_bootstrap(boot, "safe_expert", "original_baseline", "miss_penalty_nmae",
                               args.bootstrap_replicates, 20260804)
    safe_base = np.asarray([r["variants"]["safe_baseline"]["predicted_gate_nmae"] for r in results])
    safe_exp = np.asarray([r["variants"]["safe_expert"]["predicted_gate_nmae"] for r in results])
    improvement = 1.0 - safe_exp / safe_base
    worst = 1.0 - np.mean([r["variants"]["safe_expert"]["worst_combo_predicted_gate_nmae"] for r in results]) / np.mean([
        r["variants"]["safe_baseline"]["worst_combo_predicted_gate_nmae"] for r in results])
    max_fpr = max(r["by_power"][str(p)]["safe_classification"]["vft_fpr"] for r in results for p in POWERS)
    original_recall = np.mean([r["by_power"][str(p)]["original_classification"]["vft_recall"] for r in results for p in POWERS])
    safe_recall = np.mean([r["by_power"][str(p)]["safe_classification"]["vft_recall"] for r in results for p in POWERS])
    retention = float(safe_recall / original_recall)
    gates = {
        "mean_predicted_gate_nmae_improvement_at_least_0p30": bool(improvement.mean() >= 0.30),
        "three_of_three_improvement": bool(np.all(improvement > 0)),
        "maximum_vft_power_fpr_at_most_0p10": bool(max_fpr <= 0.10 + 1e-12),
        "vft_recall_retention_at_least_0p75": bool(retention >= 0.75),
        "safe_expert_miss_penalty_nonworse_than_original": bool(ci_miss["observed_mean_delta"] <= 0),
        "worst_combo_improvement_at_least_0p15": bool(worst >= 0.15),
        "paired_bootstrap_nmae_and_miss_ci_high_below_zero": bool(ci_nmae["ci95_high"] < 0 and ci_miss["ci95_high"] < 0),
    }
    output = {
        "experiment": freeze["experiment"],
        "mean_predicted_gate_nmae_improvement_vs_safe_baseline": float(improvement.mean()),
        "per_seed_improvement": improvement, "worst_combo_improvement_vs_safe_baseline": float(worst),
        "maximum_vft_power_fpr": float(max_fpr), "vft_recall_retention_vs_original": retention,
        "paired_bootstrap": {"predicted_gate_nmae": ci_nmae, "miss_penalty_vs_original": ci_miss},
        "gates": gates, "innovation_validated": bool(all(gates.values())),
        "method_freeze_sha256": freeze_hash, "elapsed_seconds": float(time.perf_counter() - start),
        "test_used_once": True, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(output), indent=2))


if __name__ == "__main__":
    main()
