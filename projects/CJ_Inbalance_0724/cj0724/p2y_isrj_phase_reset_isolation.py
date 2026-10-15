"""Test whether a frozen separator preserves the P2x ISRJ width mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    ARRIVAL_COLUMN,
    ISRJ_CLASS_INDEX,
    POWERS,
    REGIMES,
    REPLICAS_COLUMN,
    WIDTH_COLUMN,
    metrics,
    reconstruct_isrj_component,
    sha256_file,
)
from cj0724.p2x_isrj_phase_reset_ood import estimate_width_phase_resets


MODEL_SEEDS = (2041, 2042, 2043)


@torch.no_grad()
def load_and_separate(
    model: ConditionalComplexUNetSeparator,
    path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = V73IQDataset(path, indices, [WIDTH_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    raw_parts, separated_parts = [], []
    try:
        model.eval()
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            separated = model.forward_components(iq, None)["normalized_estimate"]
            raw_parts.append(iq.cpu().numpy())
            separated_parts.append(separated.cpu().numpy())
    finally:
        dataset.close()
    return np.concatenate(raw_parts), np.concatenate(separated_parts)


def waveform_cosine(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    estimate_complex = estimate[:, 0].astype(np.float64) + 1j * estimate[:, 1].astype(np.float64)
    target_complex = target[:, 0].astype(np.float64) + 1j * target[:, 1].astype(np.float64)
    numerator = np.abs(np.sum(estimate_complex.conj() * target_complex, axis=1))
    denominator = np.linalg.norm(estimate_complex, axis=1) * np.linalg.norm(target_complex, axis=1)
    return numerator / np.maximum(denominator, 1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2y_isrj_phase_reset_isolation_seed4200000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725/unet_wave")
    parser.add_argument("--preregister", default="configs/p2y_isrj_phase_reset_isolation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2y_isrj_phase_reset_isolation_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files)

    data_records, identity_reference = {}, None
    for regime in REGIMES:
        files = transition_files(project / args.data_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
            width_mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, ISRJ_CLASS_INDEX] > 0) & width_mask & np.isfinite(values[WIDTH_COLUMN]))
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2y count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2y regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2y identity overlap")
        target = values[WIDTH_COLUMN, selected]
        exact_iq = reconstruct_isrj_component(target, values[REPLICAS_COLUMN, selected], values[ARRIVAL_COLUMN, selected])
        exact_prediction = estimate_width_phase_resets(exact_iq)["prediction"]
        data_records[regime] = {
            "files": files, "identities": identities, "labels": labels, "selected": selected,
            "target": target, "exact_iq": exact_iq, "exact_prediction": exact_prediction,
            "alignment": alignment, "exclusions": exclusions,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, checkpoint_receipts = [], []
    for model_seed in MODEL_SEEDS:
        checkpoint_path = project / args.separator_root / f"seed_{model_seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_receipts.append({
            "model_seed": model_seed, "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path), "epoch": int(checkpoint["epoch"]),
        })
        for regime in REGIMES:
            record = data_records[regime]
            arrays = {
                "identity": [], "power": [], "combination_size": [], "target": [],
                "exact_prediction": [], "raw_prediction": [], "separated_prediction": [],
                "waveform_cosine": [],
            }
            for power in POWERS:
                raw_iq, separated_iq = load_and_separate(model, record["files"][power], record["selected"], args.batch_size, device)
                raw_prediction = estimate_width_phase_resets(raw_iq)["prediction"]
                separated_prediction = estimate_width_phase_resets(separated_iq)["prediction"]
                arrays["identity"].append(record["identities"][record["selected"]])
                arrays["power"].append(np.full(len(record["selected"]), power, dtype=np.int16))
                arrays["combination_size"].append(record["labels"][record["selected"]].sum(axis=1).astype(np.int8))
                arrays["target"].append(record["target"])
                arrays["exact_prediction"].append(record["exact_prediction"])
                arrays["raw_prediction"].append(raw_prediction)
                arrays["separated_prediction"].append(separated_prediction)
                arrays["waveform_cosine"].append(waveform_cosine(separated_iq, record["exact_iq"]))
            combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
            compound = combined["combination_size"] >= 2
            variants = {}
            for name, key in (
                ("C1_exact_phase_reset", "exact_prediction"),
                ("C2_raw_phase_reset", "raw_prediction"),
                ("C3_frozen_separator_phase_reset", "separated_prediction"),
            ):
                variants[name] = {
                    "overall": metrics(combined[key], combined["target"]),
                    "compound": metrics(combined[key][compound], combined["target"][compound]),
                    "by_power": {
                        str(power): metrics(combined[key][combined["power"] == power], combined["target"][combined["power"] == power])
                        for power in POWERS
                    },
                }
            row = {
                "model_seed": model_seed, "regime": regime, "variants": variants,
                "waveform_cosine_mean": float(combined["waveform_cosine"].mean()),
                "waveform_cosine_by_power": {
                    str(power): float(combined["waveform_cosine"][combined["power"] == power].mean()) for power in POWERS
                },
            }
            run_dir = output_root / f"seed_{model_seed}" / regime
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **combined)
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "model_seed": model_seed, "regime": regime,
                "raw": variants["C2_raw_phase_reset"]["overall"],
                "separated": variants["C3_frozen_separator_phase_reset"]["overall"],
                "cosine": row["waveform_cosine_mean"],
            }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    exact_rows = [row["variants"]["C1_exact_phase_reset"]["overall"] for row in rows if row["model_seed"] == MODEL_SEEDS[0]]
    separated = lambda row, subset="overall": row["variants"]["C3_frozen_separator_phase_reset"][subset]
    raw = lambda row: row["variants"]["C2_raw_phase_reset"]["overall"]
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for record in data_records.values() for ex in record["exclusions"])),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(item["mae_reduction_vs_constant"] >= 0.10 for item in exact_rows)),
        "exact_each_regime_pearson_at_least_0p30": bool(all(item["pearson"] >= 0.30 for item in exact_rows)),
        "separated_all_15_reduction_vs_constant_at_least_0p10": bool(all(separated(row)["mae_reduction_vs_constant"] >= 0.10 for row in rows)),
        "separated_all_15_pearson_at_least_0p30": bool(all(separated(row)["pearson"] >= 0.30 for row in rows)),
        "separated_compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(separated(row, "compound")["mae_reduction_vs_constant"] >= 0.10 for row in rows)),
        "separated_compound_all_15_pearson_at_least_0p30": bool(all(separated(row, "compound")["pearson"] >= 0.30 for row in rows)),
        "separated_all_15_mae_better_than_raw": bool(all(separated(row)["mae_s"] < raw(row)["mae_s"] for row in rows)),
        "separated_all_15_waveform_cosine_at_least_0p30": bool(all(row["waveform_cosine_mean"] >= 0.30 for row in rows)),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2y frozen separator preservation of ISRJ phase-reset width",
        "preregister_sha256": sha256_file(preregister),
        "device": str(device), "checkpoints": checkpoint_receipts,
        "identity_audit": {
            "identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference),
            "paired_across_regimes": True,
            "per_regime": {regime: {"alignment": record["alignment"], "exclusions": record["exclusions"]} for regime, record in data_records.items()},
        },
        "promotion_gates": gates, "development_passed": passed, "runs": rows,
        "interpretation": (
            "The frozen separator preserves the phase-reset width mechanism across 3/3 seeds; next test predicted-label gating on fresh identities."
            if passed else
            "The frozen generic separator does not reliably preserve ISRJ width; do not attach the P2x expert or tune on seed420."
        ),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
