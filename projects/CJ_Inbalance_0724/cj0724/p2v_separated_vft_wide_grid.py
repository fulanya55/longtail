"""Screen frozen swept-component separation before signed wide-grid VFT estimation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_unet
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import MODEL_SEEDS, _hash_int64
from cj0724.p2u_vft_wide_grid_ood import CONSTANT_HZ, POWERS, REGIMES, regression_metrics
from cj0724.vft_doppler_aligned_expert import CHIRP_RATE_HZ_PER_S, VFT_CLASS_INDEX, VFT_DOPPLER_COLUMN, estimate_aligned_doppler


VARIANTS = {
    "R0_raw_frozen": {"input": "raw", "doppler_min_hz": 1000.0, "doppler_max_hz": 7000.0, "doppler_step_hz": 100.0},
    "R1_raw_wide": {"input": "raw", "doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0},
    "R2_separated_wide": {"input": "separated", "doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0},
    "R3_separated_wrong_reference": {"input": "separated", "doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0, "chirp_rate_hz_per_s": 0.8 * CHIRP_RATE_HZ_PER_S},
    "R4_separated_no_delay": {"input": "separated", "doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0, "search_delay": False},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimator_kwargs(specification: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in specification.items() if key != "input"}


@torch.inference_mode()
def predict_file(
    separator: torch.nn.Module,
    path: Path,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = V73IQDataset(path, indices, [VFT_DOPPLER_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    output = {name: [] for name in VARIANTS}
    sample_indices = []
    separator.eval()
    try:
        for batch in loader:
            raw = batch["iq"].to(device, non_blocking=True)
            separated = separator.forward_components(raw, None)["normalized_estimate"].float()
            sample_indices.append(batch["sample_index"].numpy())
            for name, specification in VARIANTS.items():
                source = raw if specification["input"] == "raw" else separated
                prediction = estimate_aligned_doppler(source, **estimator_kwargs(specification))["prediction_hz"]
                output[name].append(prediction.cpu().numpy())
    finally:
        dataset.close()
    return {"sample_index": np.concatenate(sample_indices), **{name: np.concatenate(parts) for name, parts in output.items()}}


def summarize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    result = {}
    compound = arrays["combination_size"] >= 2
    for name in VARIANTS:
        result[name] = {
            "overall": regression_metrics(arrays[name], arrays["target"]),
            "compound": regression_metrics(arrays[name][compound], arrays["target"][compound]),
            "by_power": {
                str(power): regression_metrics(arrays[name][arrays["power"] == power], arrays["target"][arrays["power"] == power])
                for power in POWERS
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2v_separated_vft_ood_seed3600000000_{regime}_rechunked")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725/unet_wave")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p2v_separated_vft_wide_grid_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2v_separated_vft_wide_grid_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    preregister = project / args.preregister
    separator_root = project / args.separator_root
    output_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_records, identity_reference = [], None
    for regime in REGIMES:
        files = transition_files(project / args.data_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        selected = np.flatnonzero(labels[:, VFT_CLASS_INDEX] > 0)
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2v count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2v regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2v identity overlap")
        data_records.append({"regime": regime, "files": files, "identities": identities, "labels": labels, "selected": selected, "alignment": alignment, "exclusions": exclusions})

    runs = []
    for seed in MODEL_SEEDS:
        checkpoint = separator_root / f"seed_{seed}" / "best.pt"
        separator = _load_unet(checkpoint, device)
        for record in data_records:
            arrays = {"identity": [], "power": [], "combination_size": [], "target": []}
            for name in VARIANTS:
                arrays[name] = []
            for power in POWERS:
                path = record["files"][power]
                with h5py.File(path, "r") as handle:
                    target_all = np.asarray(handle["parameter_values"][VFT_DOPPLER_COLUMN, :], dtype=np.float64)
                prediction = predict_file(separator, path, record["selected"], device, args.batch_size)
                if not np.array_equal(prediction["sample_index"], record["selected"]):
                    raise RuntimeError("P2v sample order mismatch")
                arrays["identity"].append(record["identities"][record["selected"]])
                arrays["power"].append(np.full(len(record["selected"]), power, dtype=np.int16))
                arrays["combination_size"].append(record["labels"][record["selected"]].sum(axis=1).astype(np.int8))
                arrays["target"].append(target_all[record["selected"]])
                for name in VARIANTS:
                    arrays[name].append(prediction[name])
            combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
            variants = summarize_arrays(combined)
            run_dir = output_root / f"seed_{seed}" / record["regime"]
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **combined)
            row = {
                "seed": seed,
                "regime": record["regime"],
                "variants": variants,
                "separator_checkpoint_sha256": sha256_file(checkpoint),
            }
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            runs.append(row)
            print(json.dumps({
                "seed": seed,
                "regime": record["regime"],
                "raw_wide": variants["R1_raw_wide"]["overall"],
                "separated_wide": variants["R2_separated_wide"]["overall"],
            }), flush=True)
        del separator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_regime = {}
    for regime in REGIMES:
        subset = [row for row in runs if row["regime"] == regime]
        per_regime[regime] = {}
        for name in VARIANTS:
            per_regime[regime][name] = {
                "mae_hz_mean": float(np.mean([row["variants"][name]["overall"]["mae_hz"] for row in subset])),
                "pearson_mean": float(np.mean([row["variants"][name]["overall"]["pearson"] for row in subset])),
                "reduction_vs_constant_mean": float(np.mean([row["variants"][name]["overall"]["mae_reduction_vs_constant"] for row in subset])),
                "compound_mae_hz_mean": float(np.mean([row["variants"][name]["compound"]["mae_hz"] for row in subset])),
                "compound_pearson_mean": float(np.mean([row["variants"][name]["compound"]["pearson"] for row in subset])),
                "compound_reduction_vs_constant_mean": float(np.mean([row["variants"][name]["compound"]["mae_reduction_vs_constant"] for row in subset])),
            }
    candidate = {regime: per_regime[regime]["R2_separated_wide"] for regime in REGIMES}
    raw_frozen = {regime: per_regime[regime]["R0_raw_frozen"] for regime in REGIMES}
    raw_wide = {regime: per_regime[regime]["R1_raw_wide"] for regime in REGIMES}
    ood_regimes = tuple(regime for regime in REGIMES if regime != "ID_control")
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in data_records for ex in row["exclusions"])),
        "candidate_ID_mae_ratio_vs_raw_frozen_at_most_1p05": bool(candidate["ID_control"]["mae_hz_mean"] <= 1.05 * raw_frozen["ID_control"]["mae_hz_mean"]),
        "candidate_ID_pearson_delta_vs_raw_frozen_at_least_minus_0p02": bool(candidate["ID_control"]["pearson_mean"] >= raw_frozen["ID_control"]["pearson_mean"] - 0.02),
        "candidate_each_OOD_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["reduction_vs_constant_mean"] >= 0.10 for r in ood_regimes)),
        "candidate_each_OOD_pearson_at_least_0p30": bool(all(candidate[r]["pearson_mean"] >= 0.30 for r in ood_regimes)),
        "candidate_each_OOD_mae_reduction_vs_raw_wide_at_least_0p10": bool(all(1.0 - candidate[r]["mae_hz_mean"] / raw_wide[r]["mae_hz_mean"] >= 0.10 for r in ood_regimes)),
        "candidate_each_OOD_three_seed_mae_direction_vs_raw_wide": bool(all(
            row["variants"]["R2_separated_wide"]["overall"]["mae_hz"] < row["variants"]["R1_raw_wide"]["overall"]["mae_hz"]
            for row in runs if row["regime"] in ood_regimes
        )),
        "candidate_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["compound_reduction_vs_constant_mean"] >= 0.10 for r in REGIMES)),
        "candidate_compound_each_regime_pearson_at_least_0p30": bool(all(candidate[r]["compound_pearson_mean"] >= 0.30 for r in REGIMES)),
        "candidate_mean_mae_beats_controls": bool(all(
            np.mean([candidate[r]["mae_hz_mean"] for r in REGIMES])
            < np.mean([per_regime[r][control]["mae_hz_mean"] for r in REGIMES])
            for control in ("R3_separated_wrong_reference", "R4_separated_no_delay")
        )),
    }
    passed = bool(all(gates.values()))
    summary_path = output_root / "summary.json"
    summary = {
        "experiment": "P2v separated wide-grid VFT Doppler development",
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": {
            "identity_count": int(len(identity_reference)),
            "identity_sha256": _hash_int64(identity_reference),
            "paired_across_regimes": True,
            "records": [{"regime": row["regime"], "alignment": row["alignment"], "excluded_prior_splits": row["exclusions"]} for row in data_records],
        },
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": per_regime,
        "runs": runs,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if passed:
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "frozen swept-component separator plus wide signed VFT delay-Doppler grid",
            "separator_checkpoint_sha256": {
                str(seed): sha256_file(separator_root / f"seed_{seed}" / "best.pt") for seed in MODEL_SEEDS
            },
            "grid_hz": [-30000.0, 10000.0, 250.0],
            "summary_sha256": sha256_file(summary_path),
            "preregister_sha256": sha256_file(preregister),
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "per_regime": per_regime}, indent=2))


if __name__ == "__main__":
    main()
