"""P2g two fresh identity-seed confirmation for frozen P2f S1 separators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2e_sawfm_component_oracle import SAW_PERIOD_COLUMN
from cj0724.p2f_dedicated_sawfm_separator import POWERS, SawSeparatorDataset, evaluate_separator, metric
from cj0724.physics_parameter_specialists import build_physics_specialist


MODEL_SEEDS = (2041, 2042, 2043)
DATA_SEEDS = (1800000000, 1900000000)
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_power_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    power = np.tile(np.asarray(POWERS, dtype=np.int16), len(target) // len(POWERS))
    return {str(level): metric(prediction[power == level], target[power == level]) for level in POWERS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs=2, default=[
        "generated_data/p2g_sawfm_separator_confirmation_seed1800000000_rechunked",
        "generated_data/p2g_sawfm_separator_confirmation_seed1900000000_rechunked",
    ])
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--preregister", default="configs/p2g_sawfm_separator_confirmation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2g_sawfm_separator_two_seed_confirmation_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    roots = [project / path for path in args.data_roots]
    reference_root = project / args.reference_freeze_root
    specialist_root = project / args.specialist_root
    separator_root = project / args.separator_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    data_records = []
    all_identity_sets = []
    excluded_paths = [Path(path) for path in args.excluded_identity_files]
    for data_seed, root in zip(DATA_SEEDS, roots):
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, SAW_CLASS_INDEX] > 0) & mask[SAW_PERIOD_COLUMN])
        if len(selected) != 320:
            raise RuntimeError(f"P2g seed {data_seed} expected 320 SawFM positives, got {len(selected)}")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] != 0 for row in exclusions):
            raise RuntimeError(f"P2g seed {data_seed} overlaps a prior split")
        data_records.append({
            "data_seed": data_seed, "root": root, "files": files,
            "identities": identities, "selected": selected,
            "alignment": alignment, "exclusions": exclusions,
        })
        all_identity_sets.append(identities)
    cross_overlap = int(np.intersect1d(all_identity_sets[0], all_identity_sets[1]).size)
    if cross_overlap != 0:
        raise RuntimeError("P2g confirmation identity seeds overlap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for record in data_records:
        dataset = SawSeparatorDataset(record["files"], record["selected"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        for model_seed in MODEL_SEEDS:
            freeze = json.loads((reference_root / f"seed_{model_seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
            mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
            expert_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{model_seed}" / "best.pt"
            separator_checkpoint = separator_root / f"seed_{model_seed}" / "S1_waveform_only" / "best.pt"
            period_expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
            period_expert.load_state_dict(torch.load(expert_checkpoint, map_location=device, weights_only=True))
            separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
            evaluation = evaluate_separator(separator, period_expert, loader, device, mean, scale)
            raw_metrics = metric(evaluation["raw_prediction"], evaluation["target"])
            separated_metrics = metric(evaluation["separated_prediction"], evaluation["target"])
            row = {
                "data_seed": record["data_seed"], "model_seed": model_seed,
                "raw": raw_metrics, "separated": separated_metrics,
                "period_reduction_vs_raw": float(1.0 - separated_metrics["mae"] / raw_metrics["mae"]),
                "waveform_cosine": evaluation["waveform_cosine"],
                "per_power_raw": per_power_metrics(evaluation["raw_prediction"], evaluation["target"]),
                "per_power_separated": per_power_metrics(evaluation["separated_prediction"], evaluation["target"]),
                "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
                "expert_checkpoint_sha256": sha256_file(expert_checkpoint),
            }
            run_dir = output_root / f"data_seed_{record['data_seed']}" / f"model_seed_{model_seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(
                run_dir / "predictions.npz", sample_index=evaluation["sample_index"], target=evaluation["target"],
                raw_prediction=evaluation["raw_prediction"], separated_prediction=evaluation["separated_prediction"],
            )
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            del separator, period_expert
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"data_seed": record["data_seed"], "model_seed": model_seed, "reduction": row["period_reduction_vs_raw"], "pearson": separated_metrics["pearson"], "cosine": row["waveform_cosine"]}), flush=True)
        dataset.close()

    gates = {
        "all_six_reduction_vs_raw_at_least_0p20": bool(all(row["period_reduction_vs_raw"] >= 0.20 for row in rows)),
        "all_six_pearson_at_least_0p60": bool(all(row["separated"]["pearson"] >= 0.60 for row in rows)),
        "all_six_waveform_cosine_at_least_0p50": bool(all(row["waveform_cosine"] >= 0.50 for row in rows)),
        "mean_reduction_vs_raw_at_least_0p25": bool(np.mean([row["period_reduction_vs_raw"] for row in rows]) >= 0.25),
        "both_data_seed_means_positive": bool(all(np.mean([row["period_reduction_vs_raw"] for row in rows if row["data_seed"] == seed]) > 0.0 for seed in DATA_SEEDS)),
    }
    passed = bool(all(gates.values()))
    interpretation = (
        "P2f S1 is confirmed on two fresh identity seeds; freeze it for a separately preregistered end-to-end 12-parameter attachment."
        if passed else
        "P2f S1 did not confirm across all frozen checkpoints and data seeds; do not attach or select checkpoints."
    )
    summary = {
        "experiment": "P2g frozen S1 two-seed confirmation",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister),
        "identity_audit": {
            "cross_seed_overlap": cross_overlap,
            "data_seeds": [{
                "data_seed": record["data_seed"],
                "identity_count": int(len(record["identities"])),
                "identity_sha256": _hash_int64(record["identities"]),
                "SawFM_positive_identity_sha256": _hash_int64(record["identities"][record["selected"]]),
                "alignment": record["alignment"], "excluded_prior_splits": record["exclusions"],
            } for record in data_records],
        },
        "promotion_gates": gates, "promotion_passed": passed,
        "mean_reduction_vs_raw": float(np.mean([row["period_reduction_vs_raw"] for row in rows])),
        "per_data_seed_mean_reduction": {str(seed): float(np.mean([row["period_reduction_vs_raw"] for row in rows if row["data_seed"] == seed])) for seed in DATA_SEEDS},
        "interpretation": interpretation, "runs": rows,
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
