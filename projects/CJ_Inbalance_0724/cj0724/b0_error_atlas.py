"""Build a read-only MFT/ISRJ error atlas from frozen D11 E0 predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGETS = {"MFT": (3, 28), "ISRJ": (5, 53)}
POWER_LEVELS = (10, 15, 20, 25, 30, 35, 40, 45, 50)
DEFAULT_ROOT = Path("outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce")


def _read_referenced_matrix(handle: h5py.File, key: str, indices: np.ndarray, width: int = 6) -> np.ndarray:
    """Resolve a MATLAB cell/reference vector into a sample-major numeric matrix."""
    references = handle[key]
    rows = []
    for index in indices:
        reference = references[0, int(index)]
        value = np.asarray(handle[reference], dtype=np.float64).reshape(-1)
        if value.size < width:
            value = np.pad(value, (0, width - value.size), constant_values=np.nan)
        rows.append(value[:width])
    return np.stack(rows, axis=0)


def _align(values: np.ndarray, source_indices: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    lookup = {int(index): position for position, index in enumerate(source_indices)}
    return values[[lookup[int(index)] for index in target_indices]]


def _combo_name(label: np.ndarray) -> str:
    return "+".join(name for name, active in zip(CLASS_NAMES, label) if active) or "none"


def _binary_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    target = target.astype(bool)
    predicted = probability >= threshold
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    return {
        "aupr": float(average_precision_score(target, probability)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "positive_support": int(target.sum()),
        "negative_support": int((~target).sum()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b0_mft_isrj_error_atlas_20260725"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260724, 20260725, 20260726))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    overall_rows, combo_rows, cardinality_rows, parameter_rows = [], [], [], []
    for seed in args.seeds:
        run = args.baseline_root / f"seed_{seed}"
        audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
        thresholds = np.asarray(json.loads((run / "thresholds.json").read_text(encoding="utf-8"))["thresholds"])
        predictions = np.load(run / "development_predictions.npz")
        development_indices = np.sort(np.load(run / "split_indices.npz")["development"].astype(np.int64))
        for power in POWER_LEVELS:
            path = Path(audit["levels"][str(power)]["path"])
            source_indices = predictions[f"sample_indices_{power}"].astype(np.int64)
            labels = _align(predictions[f"labels_{power}"], source_indices, development_indices).astype(np.uint8)
            probabilities = _align(predictions[f"probabilities_{power}"], source_indices, development_indices)
            weak = _align(predictions[f"weak_mask_{power}"], source_indices, development_indices).astype(bool)
            with h5py.File(path, "r") as handle:
                parameter_values = np.asarray(handle["parameter_values"][:, development_indices], dtype=np.float64)
                relative_power = _read_referenced_matrix(
                    handle, "relative_power_dB", development_indices
                )
            for target_name, (target_index, parameter_column) in TARGETS.items():
                target = labels[:, target_index]
                probability = probabilities[:, target_index]
                overall_rows.append({
                    "seed": seed, "power_dB": power, "target": target_name,
                    "threshold": float(thresholds[target_index]),
                    "weak_positive_recall": _binary_metrics(
                        target[weak[:, target_index]], probability[weak[:, target_index]], thresholds[target_index]
                    )["recall"] if np.any(weak[:, target_index]) else float("nan"),
                    **_binary_metrics(target, probability, thresholds[target_index]),
                })
                combo_ids = labels.dot(1 << np.arange(6, dtype=np.int64))
                for combo_id in sorted(np.unique(combo_ids)):
                    mask = combo_ids == combo_id
                    combo_rows.append({
                        "seed": seed, "power_dB": power, "target": target_name,
                        "combination_id": int(combo_id), "combination": _combo_name(labels[mask][0]),
                        "cardinality": int(labels[mask][0].sum()), "support": int(mask.sum()),
                        **_binary_metrics(target[mask], probability[mask], thresholds[target_index]),
                    })
                for cardinality in range(1, 7):
                    mask = labels.sum(axis=1) == cardinality
                    cardinality_rows.append({
                        "seed": seed, "power_dB": power, "target": target_name,
                        "cardinality": cardinality, "support": int(mask.sum()),
                        **_binary_metrics(target[mask], probability[mask], thresholds[target_index]),
                    })
                positive = target.astype(bool) & np.isfinite(parameter_values[parameter_column])
                if np.any(positive):
                    values = parameter_values[parameter_column, positive]
                    edges = np.unique(np.quantile(values, np.linspace(0, 1, 5)))
                    if len(edges) > 1:
                        bins = np.clip(np.digitize(values, edges[1:-1], right=True), 0, len(edges) - 2)
                        positive_indices = np.flatnonzero(positive)
                        for bin_index in range(len(edges) - 1):
                            selected = positive_indices[bins == bin_index]
                            parameter_rows.append({
                                "seed": seed, "power_dB": power, "target": target_name,
                                "slice": "target_parameter", "bin": bin_index,
                                "lower": float(edges[bin_index]), "upper": float(edges[bin_index + 1]),
                                "support": int(len(selected)),
                                "recall": float((probability[selected] >= thresholds[target_index]).mean()),
                                "probability_mean": float(probability[selected].mean()),
                            })
                active = target.astype(bool)
                if np.any(active):
                    rp = relative_power[active, target_index]
                    active_indices = np.flatnonzero(active)
                    for lower, upper in ((-100, -10), (-10, -5), (-5, -1e-9), (0, 1e-9)):
                        local = (rp >= lower) & (rp < upper if upper < 0 else rp <= upper)
                        if np.any(local):
                            selected = active_indices[local]
                            parameter_rows.append({
                                "seed": seed, "power_dB": power, "target": target_name,
                                "slice": "relative_power_dB", "bin": f"[{lower},{upper}]",
                                "lower": lower, "upper": upper, "support": int(len(selected)),
                                "recall": float((probability[selected] >= thresholds[target_index]).mean()),
                                "probability_mean": float(probability[selected].mean()),
                            })
    _write_csv(args.output_root / "overall.csv", overall_rows)
    _write_csv(args.output_root / "per_combination.csv", combo_rows)
    _write_csv(args.output_root / "per_cardinality.csv", cardinality_rows)
    _write_csv(args.output_root / "parameter_visibility_slices.csv", parameter_rows)
    aggregate = []
    for target in TARGETS:
        for power in POWER_LEVELS:
            selected = [row for row in overall_rows if row["target"] == target and row["power_dB"] == power]
            aggregate.append({
                "target": target, "power_dB": power,
                **{f"{metric}_mean": float(np.nanmean([row[metric] for row in selected]))
                   for metric in ("aupr", "f1", "recall", "fpr", "weak_positive_recall")},
            })
    hard_negative = sorted(
        [row for row in combo_rows if row["negative_support"] > 0],
        key=lambda row: (-row["fpr"], row["target"], row["power_dB"]),
    )[:100]
    result = {
        "protocol": {"seeds": args.seeds, "power_levels": POWER_LEVELS, "targets": TARGETS,
                     "prediction_source": str(args.baseline_root), "locked_test_used": False},
        "power_aggregates": aggregate,
        "top_hard_negative_rows": hard_negative,
        "locked_test_used": False,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
