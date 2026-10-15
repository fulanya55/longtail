"""Frozen receiver-identity paired bootstrap audit for Route-A R2."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "route_a_dual_axis_next" / "r2_formal_primary"
OUTPUT_ROOT = ROOT / "outputs" / "route_a_dual_axis_next" / "r2_receiver_bootstrap_audit"
PROTOCOL_PATH = ROOT / "docs" / "plans" / "route_a_dual_axis_r2_receiver_bootstrap_audit_protocol_20260807.md"
SCRIPT_PATH = Path(__file__).resolve()
SEEDS = (260811, 260812, 260813, 260814, 260815)
CELLS = ("FB-VH", "FB-VL", "LT-VH", "LT-VL")
VARIANTS = (
    "M0_B0_Focal",
    "MF_B0_DB",
    "MV_B0_RawSaw",
    "MFV_B0_DB_RawSaw",
)
CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SAW_INDEX = 1
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 26082401
BATCH_SIZE = 250


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def weighted_mean(values: np.ndarray, counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    weights = counts[:, selected].astype(np.float64, copy=False)
    denominator = weights.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("bootstrap replicate has an empty required cohort")
    return (weights @ np.asarray(values, dtype=np.float64)[selected]) / denominator


def weighted_average_precision(target: np.ndarray, score: np.ndarray, counts: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    sorted_counts = counts[:, order].astype(np.float64, copy=False)
    group_total = np.add.reduceat(sorted_counts, starts, axis=1)
    group_positive = np.add.reduceat(sorted_counts * target[order], starts, axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    cumulative_positive = np.cumsum(group_positive, axis=1)
    total_positive = cumulative_positive[:, -1]
    if np.any(total_positive <= 0):
        raise ValueError("bootstrap replicate has no positives for AUPR")
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    return np.sum(group_positive * precision, axis=1) / total_positive


def weighted_auc(target: np.ndarray, score: np.ndarray, counts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    order = np.argsort(score[valid], kind="mergesort")
    selected_score = score[valid][order]
    selected_target = target[valid][order]
    selected_counts = counts[:, valid][:, order].astype(np.float64, copy=False)
    starts = np.flatnonzero(np.r_[True, selected_score[1:] != selected_score[:-1]])
    group_total = np.add.reduceat(selected_counts, starts, axis=1)
    group_positive = np.add.reduceat(selected_counts * selected_target, starts, axis=1)
    group_negative = group_total - group_positive
    negative_before = np.cumsum(group_negative, axis=1) - group_negative
    numerator = np.sum(group_positive * (negative_before + 0.5 * group_negative), axis=1)
    total_positive = group_positive.sum(axis=1)
    total_negative = group_negative.sum(axis=1)
    if np.any(total_positive <= 0) or np.any(total_negative <= 0):
        raise ValueError("bootstrap replicate has an empty AUROC class")
    return numerator / (total_positive * total_negative)


def macro_aupr(payload: dict, variant: str, view: str, counts: np.ndarray) -> np.ndarray:
    target = payload["targets"]
    score = payload[f"{variant}_{view}_logits"]
    return np.mean(
        [weighted_average_precision(target[:, index], score[:, index], counts) for index in range(len(CLASS_NAMES))],
        axis=0,
    )


def label_aupr(payload: dict, variant: str, view: str, label_index: int, counts: np.ndarray) -> np.ndarray:
    return weighted_average_precision(
        payload["targets"][:, label_index],
        payload[f"{variant}_{view}_logits"][:, label_index],
        counts,
    )


def saw_cohorts(targets: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = targets[:, SAW_INDEX] == 1
    weak = positive & (targets.sum(axis=1) > 1) if view == "VL" else np.zeros(len(targets), dtype=bool)
    strong = positive & ~weak
    negative = ~positive
    return weak, strong, negative


def saw_recall(payload: dict, summary: dict, variant: str, view: str, counts: np.ndarray, cohort: str) -> np.ndarray:
    weak, strong, _ = saw_cohorts(payload["targets"], view)
    mask = weak if cohort == "weak" else strong
    threshold = float(summary["variants"][variant]["thresholds"][SAW_INDEX])
    prediction = payload[f"{variant}_{view}_probabilities"][:, SAW_INDEX] >= threshold
    return weighted_mean(prediction.astype(np.float64), counts, mask)


def saw_fpr(payload: dict, summary: dict, variant: str, view: str, counts: np.ndarray) -> np.ndarray:
    _, _, negative = saw_cohorts(payload["targets"], view)
    threshold = float(summary["variants"][variant]["thresholds"][SAW_INDEX])
    prediction = payload[f"{variant}_{view}_probabilities"][:, SAW_INDEX] >= threshold
    return weighted_mean(prediction.astype(np.float64), counts, negative)


def saw_weak_auc(payload: dict, variant: str, view: str, counts: np.ndarray) -> np.ndarray:
    weak, _, negative = saw_cohorts(payload["targets"], view)
    valid = weak | negative
    target = weak.astype(np.uint8)
    score = payload[f"{variant}_{view}_logits"][:, SAW_INDEX]
    return weighted_auc(target, score, counts, valid)


def load_and_audit_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray, list[dict]]:
    terminal_path = FORMAL_ROOT / "terminal_receipt.json"
    formal_summary_path = FORMAL_ROOT / "formal_primary_summary.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    formal_summary = json.loads(formal_summary_path.read_text(encoding="utf-8"))
    if terminal["status"] != "completed" or formal_summary["status"] != "completed":
        raise ValueError("formal run is not terminal-completed")
    if any(bool(terminal.get(key)) for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("formal terminal receipt reports forbidden test use")
    if sha256(formal_summary_path) != terminal["summary_sha256"]:
        raise ValueError("formal summary hash mismatch")

    payloads: dict[tuple[int, str], dict] = {}
    summaries: dict[tuple[int, str], dict] = {}
    receipt_rows: list[dict] = []
    reference_ids = None
    reference_targets = None
    for seed in SEEDS:
        for cell in CELLS:
            cell_root = FORMAL_ROOT / f"seed_{seed}" / cell
            receipt_path = cell_root / "run_receipt.json"
            summary_path = cell_root / "cell_summary.json"
            prediction_path = cell_root / "evaluation_predictions.npz"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if receipt["status"] != "completed" or summary["status"] != "completed":
                raise ValueError(f"incomplete cell: {seed}/{cell}")
            if any(bool(receipt.get(key)) for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
                raise ValueError(f"forbidden test use in cell: {seed}/{cell}")
            if sha256(prediction_path) != summary["predictions_sha256"]:
                raise ValueError(f"prediction hash mismatch: {seed}/{cell}")
            with np.load(prediction_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            ids = np.asarray(payload["sample_ids"], dtype=np.int64)
            targets = np.asarray(payload["targets"], dtype=np.uint8)
            if len(ids) != 2520 or len(np.unique(ids)) != 2520:
                raise ValueError(f"invalid identity count: {seed}/{cell}")
            if reference_ids is None:
                reference_ids = ids.copy()
                reference_targets = targets.copy()
            elif not np.array_equal(ids, reference_ids) or not np.array_equal(targets, reference_targets):
                raise ValueError(f"identity/target alignment failure: {seed}/{cell}")
            payloads[(seed, cell)] = payload
            summaries[(seed, cell)] = summary
            receipt_rows.append({
                "seed": seed,
                "cell": cell,
                "status": receipt["status"],
                "locked_test_used": bool(receipt.get("locked_test_used")),
                "test_v2_used": bool(receipt.get("test_v2_used")),
                "test_v3_used": bool(receipt.get("test_v3_used")),
                "prediction_sha256": summary["predictions_sha256"],
            })
    return payloads, summaries, reference_ids, reference_targets, receipt_rows


def calculate_batch(payloads: dict, summaries: dict, counts: np.ndarray) -> dict[str, np.ndarray]:
    frequency_lt = []
    frequency_fb = []
    visibility_auc = []
    visibility_recall = []
    visibility_fpr = []
    visibility_strong = []
    combined_macro = []
    combined_recall = []
    safety = {name: [] for name in CLASS_NAMES if name != "SawFM"}

    for seed in SEEDS:
        lt_cells = []
        fb_cells = []
        for cell in ("LT-VH", "LT-VL"):
            view = cell.split("-")[1]
            payload = payloads[(seed, cell)]
            lt_cells.append(macro_aupr(payload, "MF_B0_DB", view, counts) - macro_aupr(payload, "M0_B0_Focal", view, counts))
        for cell in ("FB-VH", "FB-VL"):
            view = cell.split("-")[1]
            payload = payloads[(seed, cell)]
            fb_cells.append(macro_aupr(payload, "MF_B0_DB", view, counts) - macro_aupr(payload, "M0_B0_Focal", view, counts))
        frequency_lt.append(np.mean(lt_cells, axis=0))
        frequency_fb.append(np.mean(fb_cells, axis=0))

        seed_auc = []
        seed_recall = []
        seed_fpr = []
        seed_strong = []
        for cell in ("FB-VL", "LT-VL"):
            payload = payloads[(seed, cell)]
            summary = summaries[(seed, cell)]
            seed_auc.append(saw_weak_auc(payload, "MV_B0_RawSaw", "VL", counts) - saw_weak_auc(payload, "M0_B0_Focal", "VL", counts))
            seed_recall.append(saw_recall(payload, summary, "MV_B0_RawSaw", "VL", counts, "weak") - saw_recall(payload, summary, "M0_B0_Focal", "VL", counts, "weak"))
            seed_fpr.append(saw_fpr(payload, summary, "MV_B0_RawSaw", "VL", counts))
            seed_strong.append(saw_recall(payload, summary, "MV_B0_RawSaw", "VL", counts, "strong") - saw_recall(payload, summary, "M0_B0_Focal", "VL", counts, "strong"))
        visibility_auc.append(np.mean(seed_auc, axis=0))
        visibility_recall.append(np.mean(seed_recall, axis=0))
        visibility_fpr.append(np.mean(seed_fpr, axis=0))
        visibility_strong.append(np.mean(seed_strong, axis=0))

        payload = payloads[(seed, "LT-VL")]
        summary = summaries[(seed, "LT-VL")]
        mfv_macro = macro_aupr(payload, "MFV_B0_DB_RawSaw", "VL", counts)
        mf_macro = macro_aupr(payload, "MF_B0_DB", "VL", counts)
        mv_macro = macro_aupr(payload, "MV_B0_RawSaw", "VL", counts)
        combined_macro.append(mfv_macro - np.maximum(mf_macro, mv_macro))
        mfv_recall = saw_recall(payload, summary, "MFV_B0_DB_RawSaw", "VL", counts, "weak")
        mf_recall = saw_recall(payload, summary, "MF_B0_DB", "VL", counts, "weak")
        mv_recall = saw_recall(payload, summary, "MV_B0_RawSaw", "VL", counts, "weak")
        combined_recall.append(mfv_recall - np.maximum(mf_recall, mv_recall))
        for index, name in enumerate(CLASS_NAMES):
            if name == "SawFM":
                continue
            safety[name].append(
                label_aupr(payload, "MFV_B0_DB_RawSaw", "VL", index, counts)
                - label_aupr(payload, "M0_B0_Focal", "VL", index, counts)
            )

    result = {
        "frequency_lt_macro_aupr_delta": np.mean(frequency_lt, axis=0),
        "frequency_fb_macro_aupr_delta": np.mean(frequency_fb, axis=0),
        "visibility_weak_auroc_delta": np.mean(visibility_auc, axis=0),
        "visibility_weak_recall_delta": np.mean(visibility_recall, axis=0),
        "visibility_candidate_fpr": np.mean(visibility_fpr, axis=0),
        "visibility_strong_recall_delta": np.mean(visibility_strong, axis=0),
        "combined_macro_aupr_delta_vs_best": np.mean(combined_macro, axis=0),
        "combined_weak_recall_delta_vs_best": np.mean(combined_recall, axis=0),
    }
    for name, values in safety.items():
        result[f"non_saw_{name}_aupr_delta"] = np.mean(values, axis=0)
    return result


def summarize(samples: np.ndarray, point: float) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std(ddof=1)),
        "bootstrap_95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "repetitions": int(len(values)),
    }


def main() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payloads, summaries, sample_ids, targets, receipt_rows = load_and_audit_inputs()
    identity_count = len(sample_ids)

    unit_counts = np.ones((1, identity_count), dtype=np.uint16)
    point_arrays = calculate_batch(payloads, summaries, unit_counts)
    point = {name: float(values[0]) for name, values in point_arrays.items()}

    formal = json.loads((FORMAL_ROOT / "formal_primary_summary.json").read_text(encoding="utf-8"))["aggregate"]
    expected = {
        "frequency_lt_macro_aupr_delta": formal["frequency"]["lt_macro_aupr_delta"]["mean"],
        "frequency_fb_macro_aupr_delta": formal["frequency"]["fb_macro_aupr_delta"]["mean"],
        "visibility_weak_auroc_delta": formal["visibility"]["weak_auroc_delta"]["mean"],
        "visibility_weak_recall_delta": formal["visibility"]["weak_recall_delta"]["mean"],
        "visibility_candidate_fpr": formal["visibility"]["candidate_fpr"]["mean"],
        "visibility_strong_recall_delta": formal["visibility"]["strong_recall_delta"]["mean"],
        "combined_macro_aupr_delta_vs_best": formal["combined"]["macro_aupr_delta_vs_best"]["mean"],
        "combined_weak_recall_delta_vs_best": formal["combined"]["weak_recall_delta_vs_best"]["mean"],
    }
    for name in CLASS_NAMES:
        if name != "SawFM":
            expected[f"non_saw_{name}_aupr_delta"] = formal["non_saw_safety_by_label"][name]["mean"]
    reproduction_error = {name: abs(point[name] - float(value)) for name, value in expected.items()}
    if max(reproduction_error.values()) > 1e-10:
        raise ValueError(f"formal point-estimate reproduction failed: {reproduction_error}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    collected = {name: [] for name in point}
    probabilities = np.full(identity_count, 1.0 / identity_count, dtype=np.float64)
    remaining = BOOTSTRAP_REPETITIONS
    while remaining:
        current = min(BATCH_SIZE, remaining)
        counts = rng.multinomial(identity_count, probabilities, size=current).astype(np.uint16)
        batch = calculate_batch(payloads, summaries, counts)
        for name, values in batch.items():
            collected[name].append(np.asarray(values, dtype=np.float64))
        remaining -= current

    samples = {name: np.concatenate(parts) for name, parts in collected.items()}
    np.savez_compressed(
        OUTPUT_ROOT / "paired_bootstrap_samples.npz",
        sample_ids=sample_ids,
        targets=targets,
        **samples,
    )
    metrics = {name: summarize(values, point[name]) for name, values in samples.items()}
    macro_gate = (
        point["combined_macro_aupr_delta_vs_best"] >= 0.005
        and metrics["combined_macro_aupr_delta_vs_best"]["bootstrap_95_ci"][0] > 0.0
    )
    recall_gate = (
        point["combined_weak_recall_delta_vs_best"] >= 0.020
        and metrics["combined_weak_recall_delta_vs_best"]["bootstrap_95_ci"][0] > 0.0
    )
    gates = {
        "MF_frequency_main_effect": bool(formal["gates"]["MF_frequency_main_effect"]),
        "MV_visibility_main_effect": bool(formal["gates"]["MV_visibility_main_effect"]),
        "MFV_macro_with_receiver_ci": bool(macro_gate),
        "MFV_weak_recall_with_receiver_ci": bool(recall_gate),
        "MFV_independent_increment": bool(macro_gate or recall_gate),
        "non_saw_safety": bool(formal["gates"]["non_saw_safety"]),
        "controls_admitted": bool(formal["gates"]["MF_frequency_main_effect"] and formal["gates"]["MV_visibility_main_effect"]),
    }
    summary = {
        "status": "completed",
        "evidence_level": "development_receiver_identity_paired_bootstrap",
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "formal_summary_path": str(FORMAL_ROOT / "formal_primary_summary.json"),
        "formal_summary_sha256": sha256(FORMAL_ROOT / "formal_primary_summary.json"),
        "identity_count": identity_count,
        "identity_alignment": True,
        "target_alignment": True,
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "bootstrap": {
            "unit": "receiver_sample_identity",
            "paired_across": "all_seed_cell_variant_predictions",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "rng_seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
            "best_comparator": "replicatewise_max_MF_MV_per_seed",
        },
        "point_reproduction_absolute_error": reproduction_error,
        "metrics": metrics,
        "gates": gates,
        "decision": "negative_closure_no_positive_controls" if not gates["controls_admitted"] else "positive_controls_admitted",
        "receipt_audit": receipt_rows,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    summary_path = OUTPUT_ROOT / "receiver_bootstrap_summary.json"
    write_json(summary_path, summary)
    receipt = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "samples_path": str(OUTPUT_ROOT / "paired_bootstrap_samples.npz"),
        "samples_sha256": sha256(OUTPUT_ROOT / "paired_bootstrap_samples.npz"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_ROOT / "run_receipt.json", receipt)
    print(json.dumps({"status": "completed", "gates": gates, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
