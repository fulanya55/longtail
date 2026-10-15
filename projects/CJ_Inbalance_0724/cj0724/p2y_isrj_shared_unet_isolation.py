"""P2y: test whether the frozen shared U-Net preserves P2x ISRJ width evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_unet
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    ARRIVAL_COLUMN,
    ISRJ_CLASS_INDEX,
    POWERS,
    REGIMES,
    REPLICAS_COLUMN,
    WIDTH_COLUMN,
    deterministic_shuffle,
    metrics,
    reconstruct_isrj_component,
)
from cj0724.p2x_isrj_phase_reset_ood import estimate_width_phase_resets


MODEL_SEEDS = (2041, 2042, 2043)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def frozen_unet_estimate(
    model,
    path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = V73IQDataset(path, indices, [WIDTH_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts: list[np.ndarray] = []
    try:
        for batch in loader:
            output = model.forward_components(batch["iq"].to(device), None)["normalized_estimate"]
            parts.append(output.float().cpu().numpy())
    finally:
        dataset.close()
    return np.concatenate(parts).astype(np.float32, copy=False)


def waveform_cosine(estimate: np.ndarray, exact: np.ndarray) -> np.ndarray:
    lhs = torch.from_numpy(np.asarray(estimate, dtype=np.float32)).flatten(1)
    rhs = torch.from_numpy(np.asarray(exact, dtype=np.float32)).flatten(1)
    return F.cosine_similarity(lhs, rhs, dim=1).numpy().astype(np.float32)


def evaluate_prediction(prediction: np.ndarray, target: np.ndarray, combination_size: np.ndarray) -> dict[str, object]:
    compound = combination_size >= 2
    return {
        "overall": metrics(prediction, target),
        "compound": metrics(prediction[compound], target[compound]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root-template",
        default="generated_data/p2y_isrj_shared_unet_seed4150000000_{regime}_rechunked",
    )
    parser.add_argument(
        "--separator-root", default="outputs/conditional_unet_separator_full_20260725"
    )
    parser.add_argument(
        "--training-protocol",
        default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json",
    )
    parser.add_argument(
        "--preregister", default="configs/p2y_isrj_shared_unet_isolation_preregister_20260801.json"
    )
    parser.add_argument(
        "--output-root", default="outputs/p2y_isrj_shared_unet_isolation_development_20260801"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(
        Path(value) if Path(value).is_absolute() else project / value
        for value in args.excluded_identity_files
    )

    models = {}
    checkpoint_hashes = {}
    for seed in MODEL_SEEDS:
        checkpoint = project / args.separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        models[seed] = _load_unet(checkpoint, device)
        checkpoint_hashes[str(seed)] = sha256_file(checkpoint)

    identity_reference = None
    regime_records = []
    for regime in REGIMES:
        files = transition_files(project / args.data_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
            mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero(
            (labels[:, ISRJ_CLASS_INDEX] > 0) & mask & np.isfinite(values[WIDTH_COLUMN])
        )
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2y count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2y regimes are not identity paired")

        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            overlap = int(np.intersect1d(identities, prior).size)
            exclusions.append({"path": str(path.resolve()), "overlap_count": overlap})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2y identity overlap")

        target = values[WIDTH_COLUMN, selected]
        combination_size = labels[selected].sum(axis=1).astype(np.int8)
        exact = reconstruct_isrj_component(
            target, values[REPLICAS_COLUMN, selected], values[ARRIVAL_COLUMN, selected]
        )
        exact_prediction = estimate_width_phase_resets(exact)["prediction"]
        seed_records = {}
        for model_seed, model in models.items():
            arrays = {
                "identity": [], "power": [], "target": [], "combination_size": [],
                "raw_prediction": [], "shared_prediction": [], "exact_prediction": [],
                "shuffled_prediction": [], "waveform_cosine": [],
            }
            for power in POWERS:
                raw_dataset = V73IQDataset(files[power], selected, [WIDTH_COLUMN])
                try:
                    raw = np.stack([raw_dataset[position]["iq"].numpy() for position in range(len(raw_dataset))])
                finally:
                    raw_dataset.close()
                separated = frozen_unet_estimate(model, files[power], selected, args.batch_size, device)
                arrays["identity"].append(identities[selected])
                arrays["power"].append(np.full(len(selected), power, dtype=np.int16))
                arrays["target"].append(target)
                arrays["combination_size"].append(combination_size)
                arrays["raw_prediction"].append(estimate_width_phase_resets(raw)["prediction"])
                arrays["shared_prediction"].append(estimate_width_phase_resets(separated)["prediction"])
                arrays["exact_prediction"].append(exact_prediction)
                arrays["shuffled_prediction"].append(
                    estimate_width_phase_resets(deterministic_shuffle(separated))["prediction"]
                )
                arrays["waveform_cosine"].append(waveform_cosine(separated, exact))
            combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
            variants = {
                "C1_raw_phase_reset": evaluate_prediction(
                    combined["raw_prediction"], combined["target"], combined["combination_size"]
                ),
                "C2_shared_unet_phase_reset": evaluate_prediction(
                    combined["shared_prediction"], combined["target"], combined["combination_size"]
                ),
                "C3_exact_phase_reset": evaluate_prediction(
                    combined["exact_prediction"], combined["target"], combined["combination_size"]
                ),
                "C4_time_shuffled_shared_unet": evaluate_prediction(
                    combined["shuffled_prediction"], combined["target"], combined["combination_size"]
                ),
            }
            variants["C2_shared_unet_phase_reset"]["by_power"] = {
                str(power): metrics(
                    combined["shared_prediction"][combined["power"] == power],
                    combined["target"][combined["power"] == power],
                )
                for power in POWERS
            }
            seed_records[str(model_seed)] = {
                "variants": variants,
                "mean_waveform_cosine": float(combined["waveform_cosine"].mean()),
            }
            run_dir = output_root / regime / f"seed_{model_seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **combined)
            (run_dir / "result.json").write_text(
                json.dumps(_jsonable(seed_records[str(model_seed)]), indent=2), encoding="utf-8"
            )
            print(json.dumps({
                "regime": regime, "model_seed": model_seed,
                "raw": variants["C1_raw_phase_reset"]["overall"],
                "shared": variants["C2_shared_unet_phase_reset"]["overall"],
                "cosine": seed_records[str(model_seed)]["mean_waveform_cosine"],
            }), flush=True)
        regime_records.append({
            "regime": regime, "alignment": alignment, "exclusions": exclusions,
            "model_seeds": seed_records,
        })

    def candidate(row, seed, subset="overall"):
        return row["model_seeds"][str(seed)]["variants"]["C2_shared_unet_phase_reset"][subset]

    def baseline(row, seed):
        return row["model_seeds"][str(seed)]["variants"]["C1_raw_phase_reset"]["overall"]

    per_seed = {}
    for seed in MODEL_SEEDS:
        reductions = []
        for row in regime_records:
            raw_mae = baseline(row, seed)["mae_s"]
            shared_mae = candidate(row, seed)["mae_s"]
            reductions.append(1.0 - shared_mae / max(raw_mae, 1.0e-12))
        id_row = next(row for row in regime_records if row["regime"] == "ID_control")
        per_seed[str(seed)] = {
            "mean_mae_reduction_vs_raw": float(np.mean(reductions)),
            "ID_control_mae_reduction_vs_raw": float(reductions[regime_records.index(id_row)]),
            "mean_waveform_cosine": float(np.mean([
                row["model_seeds"][str(seed)]["mean_waveform_cosine"] for row in regime_records
            ])),
        }

    gates = {
        "identity_overlap_zero": bool(all(
            exclusion["overlap_count"] == 0
            for row in regime_records for exclusion in row["exclusions"]
        )),
        "each_seed_each_regime_candidate_reduction_vs_constant_at_least_0p10": bool(all(
            candidate(row, seed)["mae_reduction_vs_constant"] >= 0.10
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_each_regime_candidate_pearson_at_least_0p30": bool(all(
            candidate(row, seed)["pearson"] >= 0.30
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_each_regime_candidate_catastrophic_at_most_0p10": bool(all(
            candidate(row, seed)["catastrophic_fraction"] <= 0.10
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_each_regime_compound_reduction_vs_constant_at_least_0p10": bool(all(
            candidate(row, seed, "compound")["mae_reduction_vs_constant"] >= 0.10
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_each_regime_compound_pearson_at_least_0p30": bool(all(
            candidate(row, seed, "compound")["pearson"] >= 0.30
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_mean_mae_reduction_vs_raw_at_least_0p20": bool(all(
            per_seed[str(seed)]["mean_mae_reduction_vs_raw"] >= 0.20 for seed in MODEL_SEEDS
        )),
        "each_seed_ID_control_mae_reduction_vs_raw_at_least_0p20": bool(all(
            per_seed[str(seed)]["ID_control_mae_reduction_vs_raw"] >= 0.20 for seed in MODEL_SEEDS
        )),
        "each_seed_each_regime_candidate_better_than_shuffled": bool(all(
            candidate(row, seed)["mae_s"]
            < row["model_seeds"][str(seed)]["variants"]["C4_time_shuffled_shared_unet"]["overall"]["mae_s"]
            for seed in MODEL_SEEDS for row in regime_records
        )),
        "each_seed_mean_waveform_cosine_at_least_0p30": bool(all(
            per_seed[str(seed)]["mean_waveform_cosine"] >= 0.30 for seed in MODEL_SEEDS
        )),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2y frozen shared U-Net ISRJ phase-reset isolation transfer",
        "preregister_sha256": sha256_file(preregister),
        "device": str(device),
        "checkpoint_sha256": checkpoint_hashes,
        "identity_audit": {
            "identity_count": int(len(identity_reference)),
            "identity_sha256": _hash_int64(identity_reference),
            "paired_across_regimes": True,
        },
        "per_seed": per_seed,
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": {row["regime"]: row["model_seeds"] for row in regime_records},
        "interpretation": (
            "The frozen shared U-Net preserves P2x width evidence; freeze it for a fresh full-system attachment study."
            if passed else
            "The shared U-Net does not preserve sufficient ISRJ phase-reset evidence; do not attach or route it. The P2x exact oracle permits a separately preregistered dedicated lightweight separator."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "development_passed": passed,
        "promotion_gates": gates,
        "per_seed": per_seed,
        "interpretation": summary["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
