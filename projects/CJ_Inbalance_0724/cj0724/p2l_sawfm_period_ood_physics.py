"""P2l physical-control evaluation for ID/near/far SawFM period regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2e_sawfm_component_oracle import SAW_PERIOD_COLUMN
from cj0724.p2f_dedicated_sawfm_separator import SawSeparatorDataset, metric
from cj0724.p2j_full_parameter_attachment import COEFFICIENTS, MODEL_SEEDS
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = (25, 35, 50)
REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduction(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    return float(1.0 - candidate["mae"] / baseline["mae"])


@torch.inference_mode()
def evaluate_physics(
    separator: torch.nn.Module,
    period_expert: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
    coefficient_a: float,
    coefficient_b: float,
) -> dict[str, np.ndarray]:
    separator.eval()
    period_expert.eval()
    outputs = {name: [] for name in (
        "sample_index", "power", "target", "constant", "raw", "exact_component",
        "separated_uncalibrated", "separated_calibrated",
    )}
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
        exact = batch["target_iq"].to(device)[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
        separated = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
        raw_prediction = period_expert(flat_iq)["prediction"].float() * float(scale) + float(mean)
        exact_prediction = period_expert(exact)["prediction"].float() * float(scale) + float(mean)
        separated_prediction = period_expert(separated)["prediction"].float() * float(scale) + float(mean)
        calibrated = float(coefficient_a) * separated_prediction + float(coefficient_b)
        outputs["sample_index"].append(np.repeat(batch["sample_index"].numpy(), power_count))
        outputs["power"].append(np.tile(np.asarray(POWERS, dtype=np.int16), batch_size))
        outputs["target"].append(np.repeat(batch["period"].numpy(), power_count))
        outputs["constant"].append(np.full(batch_size * power_count, mean, dtype=np.float32))
        outputs["raw"].append(raw_prediction.cpu().numpy())
        outputs["exact_component"].append(exact_prediction.cpu().numpy())
        outputs["separated_uncalibrated"].append(separated_prediction.cpu().numpy())
        outputs["separated_calibrated"].append(calibrated.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def summarize_predictions(predictions: dict[str, np.ndarray]) -> dict[str, object]:
    target = predictions["target"]
    variants = {
        name: metric(predictions[name], target)
        for name in ("constant", "raw", "exact_component", "separated_uncalibrated", "separated_calibrated")
    }
    by_power = {}
    for power in POWERS:
        selected = predictions["power"] == power
        by_power[str(power)] = {
            name: metric(predictions[name][selected], target[selected]) for name in variants
        }
    return {
        "variants": variants,
        "exact_reduction_vs_constant": reduction(variants["exact_component"], variants["constant"]),
        "calibrated_reduction_vs_raw": reduction(variants["separated_calibrated"], variants["raw"]),
        "uncalibrated_reduction_vs_raw": reduction(variants["separated_uncalibrated"], variants["raw"]),
        "by_power": by_power,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2l_sawfm_period_ood_seed2600000000_{regime}_rechunked")
    parser.add_argument("--reference-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--candidate-freeze", default="outputs/p2k_full_parameter_confirmation_20260801/method_freeze.json")
    parser.add_argument("--preregister", default="configs/p2l_sawfm_period_ood_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2l_sawfm_period_ood_physics_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    reference_root = project / args.reference_root
    separator_root = project / args.separator_root
    specialist_root = project / args.specialist_root
    candidate_freeze = project / args.candidate_freeze
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(candidate_freeze.read_text(encoding="utf-8"))
    if frozen.get("candidate") != "P2k-confirmed calibrated separated-SawFM full-parameter attachment":
        raise RuntimeError("unexpected P2k candidate freeze")

    excluded_paths = [Path(path) if Path(path).is_absolute() else project / path for path in args.excluded_identity_files]
    regime_records = []
    reference_identities = None
    for regime in REGIMES:
        root = project / args.data_root_template.format(regime=regime)
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        selected = np.flatnonzero(labels[:, SAW_CLASS_INDEX] > 0)
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2l {regime} expected 630 identities and 320 SawFM positives")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError(f"P2l {regime} overlaps a prior split")
        if reference_identities is None:
            reference_identities = identities
        elif not np.array_equal(reference_identities, identities):
            raise RuntimeError("P2l regimes are not identity paired")
        regime_records.append({
            "regime": regime, "root": root, "files": files, "identities": identities,
            "selected": selected, "alignment": alignment, "exclusions": exclusions,
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for record in regime_records:
        dataset = SawSeparatorDataset(record["files"], record["selected"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        for seed in MODEL_SEEDS:
            freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
            mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
            period_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
            separator_checkpoint = separator_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
            period_expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
            period_expert.load_state_dict(torch.load(period_checkpoint, map_location=device, weights_only=True))
            separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
            a, b = COEFFICIENTS[seed]
            predictions = evaluate_physics(separator, period_expert, loader, device, mean, scale, a, b)
            summary = summarize_predictions(predictions)
            row = {
                "regime": record["regime"], "seed": seed, **summary,
                "period_checkpoint_sha256": sha256_file(period_checkpoint),
                "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
                "coefficients": {"a": a, "b": b},
            }
            run_dir = output_root / record["regime"] / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **predictions)
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "regime": record["regime"], "seed": seed,
                "exact_reduction_vs_constant": row["exact_reduction_vs_constant"],
                "calibrated_reduction_vs_raw": row["calibrated_reduction_vs_raw"],
                "exact_pearson": row["variants"]["exact_component"]["pearson"],
                "calibrated_pearson": row["variants"]["separated_calibrated"]["pearson"],
            }), flush=True)
            del separator, period_expert
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()

    near = [row for row in rows if row["regime"] in ("near_low", "near_high")]
    far = [row for row in rows if row["regime"] in ("far_low", "far_high")]
    physical_gates = {
        "near_exact_reduction_vs_constant_mean_at_least_0p20": bool(np.mean([row["exact_reduction_vs_constant"] for row in near]) >= 0.20),
        "near_calibrated_reduction_vs_raw_mean_at_least_0p10": bool(np.mean([row["calibrated_reduction_vs_raw"] for row in near]) >= 0.10),
        "far_exact_component_pearson_all_at_least_0p30": bool(all(row["variants"]["exact_component"]["pearson"] >= 0.30 for row in far)),
    }
    summary = {
        "experiment": "P2l SawFM period OOD physical controls",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "candidate_freeze_sha256": sha256_file(candidate_freeze),
        "identity_audit": {
            "paired_across_regimes": True,
            "identity_count": int(len(reference_identities)),
            "identity_sha256": _hash_int64(reference_identities),
            "regimes": [{
                "regime": record["regime"], "alignment": record["alignment"],
                "excluded_prior_splits": record["exclusions"],
            } for record in regime_records],
        },
        "physical_gates": physical_gates,
        "physical_gate_passed": bool(all(physical_gates.values())),
        "runs": rows,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"physical_gates": physical_gates, "physical_gate_passed": summary["physical_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
