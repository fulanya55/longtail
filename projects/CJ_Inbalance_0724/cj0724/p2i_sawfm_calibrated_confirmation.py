"""P2i confirm frozen calibrated SawFM separators on two fresh identity seeds."""

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
DATA_SEEDS = (2100000000, 2200000000)
COEFFICIENTS = {
    2041: (1.0538332944014202, 7.92056473920431e-06),
    2042: (1.0485594935630393, 5.973457154872555e-06),
    2043: (1.1260814638565335, 2.9172537901531955e-06),
}
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduction(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    return float(1.0 - candidate["mae"] / baseline["mae"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs=2, default=[
        "generated_data/p2i_sawfm_calibrated_confirmation_seed2100000000_rechunked",
        "generated_data/p2i_sawfm_calibrated_confirmation_seed2200000000_rechunked",
    ])
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--p2h-summary", default="outputs/p2h_sawfm_affine_calibration_development_20260801/summary.json")
    parser.add_argument("--preregister", default="configs/p2i_sawfm_calibrated_separator_confirmation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2i_sawfm_calibrated_separator_confirmation_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    roots = [project / path for path in args.data_roots]
    reference_root = project / args.reference_freeze_root
    specialist_root = project / args.specialist_root
    separator_root = project / args.separator_root
    p2h_summary_path = project / args.p2h_summary
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    p2h = json.loads(p2h_summary_path.read_text(encoding="utf-8"))
    p2h_coefficients = {
        int(run["seed"]): (float(run["coefficients"]["affine_a"]), float(run["coefficients"]["affine_b"]))
        for run in p2h["runs"]
    }
    for seed in MODEL_SEEDS:
        if not np.allclose(p2h_coefficients[seed], COEFFICIENTS[seed], rtol=0.0, atol=1.0e-15):
            raise RuntimeError(f"P2i frozen coefficients differ from P2h for seed {seed}")

    data_records, identity_sets = [], []
    excluded_paths = [Path(path) for path in args.excluded_identity_files]
    for data_seed, root in zip(DATA_SEEDS, roots):
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, SAW_CLASS_INDEX] > 0) & mask[SAW_PERIOD_COLUMN])
        if len(selected) != 320:
            raise RuntimeError(f"P2i seed {data_seed} expected 320 SawFM positives, got {len(selected)}")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] != 0 for row in exclusions):
            raise RuntimeError(f"P2i seed {data_seed} overlaps a prior split")
        data_records.append({
            "data_seed": data_seed, "files": files, "identities": identities,
            "selected": selected, "alignment": alignment, "exclusions": exclusions,
        })
        identity_sets.append(identities)
    cross_overlap = int(np.intersect1d(identity_sets[0], identity_sets[1]).size)
    if cross_overlap != 0:
        raise RuntimeError("P2i confirmation identity seeds overlap")

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
            expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
            expert.load_state_dict(torch.load(expert_checkpoint, map_location=device, weights_only=True))
            separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
            evaluation = evaluate_separator(separator, expert, loader, device, mean, scale)
            a, b = COEFFICIENTS[model_seed]
            calibrated_prediction = a * np.asarray(evaluation["separated_prediction"], dtype=np.float64) + b
            raw_metrics = metric(evaluation["raw_prediction"], evaluation["target"])
            uncalibrated_metrics = metric(evaluation["separated_prediction"], evaluation["target"])
            calibrated_metrics = metric(calibrated_prediction, evaluation["target"])
            row = {
                "data_seed": record["data_seed"], "model_seed": model_seed,
                "coefficients": {"a": a, "b": b},
                "raw": raw_metrics, "uncalibrated": uncalibrated_metrics, "calibrated": calibrated_metrics,
                "uncalibrated_reduction_vs_raw": reduction(uncalibrated_metrics, raw_metrics),
                "calibrated_reduction_vs_raw": reduction(calibrated_metrics, raw_metrics),
                "calibrated_improvement_over_uncalibrated": float(reduction(calibrated_metrics, raw_metrics) - reduction(uncalibrated_metrics, raw_metrics)),
                "waveform_cosine": evaluation["waveform_cosine"],
                "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
                "expert_checkpoint_sha256": sha256_file(expert_checkpoint),
            }
            run_dir = output_root / f"data_seed_{record['data_seed']}" / f"model_seed_{model_seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(
                run_dir / "predictions.npz", sample_index=evaluation["sample_index"], target=evaluation["target"],
                raw_prediction=evaluation["raw_prediction"], uncalibrated_prediction=evaluation["separated_prediction"],
                calibrated_prediction=calibrated_prediction,
            )
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            del separator, expert
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({
                "data_seed": record["data_seed"], "model_seed": model_seed,
                "calibrated_reduction": row["calibrated_reduction_vs_raw"],
                "improvement_over_uncalibrated": row["calibrated_improvement_over_uncalibrated"],
                "pearson": calibrated_metrics["pearson"],
            }), flush=True)
        dataset.close()

    gates = {
        "all_six_calibrated_reduction_vs_raw_at_least_0p35": bool(all(row["calibrated_reduction_vs_raw"] >= 0.35 for row in rows)),
        "all_six_improvement_over_uncalibrated_at_least_0p10": bool(all(row["calibrated_improvement_over_uncalibrated"] >= 0.10 for row in rows)),
        "all_six_calibrated_pearson_at_least_0p60": bool(all(row["calibrated"]["pearson"] >= 0.60 for row in rows)),
        "mean_calibrated_reduction_vs_raw_at_least_0p40": bool(np.mean([row["calibrated_reduction_vs_raw"] for row in rows]) >= 0.40),
        "both_data_seed_means_positive": bool(all(np.mean([row["calibrated_reduction_vs_raw"] for row in rows if row["data_seed"] == seed]) > 0.0 for seed in DATA_SEEDS)),
    }
    passed = bool(all(gates.values()))
    interpretation = (
        "P2i confirmed the frozen calibrated SawFM separator on two fresh identity seeds; freeze it for a separately preregistered full 12-parameter predicted-gate attachment."
        if passed else
        "P2i did not confirm; do not refit coefficients or select checkpoints on confirmation data."
    )
    summary = {
        "experiment": "P2i frozen calibrated SawFM separator confirmation",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister),
        "p2h_summary_sha256": sha256_file(p2h_summary_path),
        "identity_audit": {
            "cross_seed_overlap": cross_overlap,
            "data_seeds": [{
                "data_seed": record["data_seed"], "identity_count": int(len(record["identities"])),
                "identity_sha256": _hash_int64(record["identities"]),
                "SawFM_positive_identity_sha256": _hash_int64(record["identities"][record["selected"]]),
                "alignment": record["alignment"], "excluded_prior_splits": record["exclusions"],
            } for record in data_records],
        },
        "promotion_gates": gates, "promotion_passed": passed,
        "mean_calibrated_reduction_vs_raw": float(np.mean([row["calibrated_reduction_vs_raw"] for row in rows])),
        "per_data_seed_mean_reduction": {str(seed): float(np.mean([row["calibrated_reduction_vs_raw"] for row in rows if row["data_seed"] == seed])) for seed in DATA_SEEDS},
        "interpretation": interpretation, "runs": rows,
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
