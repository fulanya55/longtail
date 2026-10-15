"""P2m fixed FFT-ACF SawFM period estimator physical-sufficiency screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2f_dedicated_sawfm_separator import SawSeparatorDataset, metric
from cj0724.p2j_full_parameter_attachment import COEFFICIENTS, MODEL_SEEDS
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = (25, 35, 50)
REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
SAMPLE_RATE_HZ = 10.0e6
EXACT_MAE_LIMIT_US = {"ID_control": 2.0, "near_low": 1.0, "near_high": 2.0, "far_low": 0.3, "far_high": 1.0}
SEPARATED_PEARSON_LIMIT = {"ID_control": 0.55, "near_low": 0.50, "near_high": 0.15, "far_low": 0.85, "far_high": 0.25}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analytic_period_estimate(iq: np.ndarray, sample_rate_hz: float = SAMPLE_RATE_HZ) -> dict[str, np.ndarray]:
    """Apply the preregistered non-trainable FFT autocorrelation peak rule."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [samples, 2, time]")
    periods, selected_scores, strongest_lags, peak_counts = [], [], [], []
    for row in iq:
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        length = len(signal)
        spectrum = np.fft.fft(signal, n=2 * length)
        autocorrelation = np.fft.ifft(spectrum * np.conj(spectrum)).real[:length] / float(length)
        autocorrelation /= max(float(autocorrelation[0]), 1.0e-12)
        low, high = 40, min(2800, length)
        peaks, _ = find_peaks(autocorrelation[low:high], prominence=0.03, distance=20)
        lags = peaks + low
        if len(lags) == 0:
            selected_lag, selected_score, strongest_lag = low, float(autocorrelation[low]), low
        else:
            scores = autocorrelation[lags]
            strongest_position = int(np.argmax(scores))
            strongest_lag = int(lags[strongest_position])
            fraction = 0.20 if strongest_lag < 800 else 0.60
            eligible = lags[scores >= max(0.10, fraction * float(scores[strongest_position]))]
            selected_lag = int(np.min(eligible)) if len(eligible) else strongest_lag
            selected_score = float(autocorrelation[selected_lag])
        periods.append(selected_lag / float(sample_rate_hz))
        selected_scores.append(selected_score)
        strongest_lags.append(strongest_lag)
        peak_counts.append(len(lags))
    return {
        "prediction": np.asarray(periods, dtype=np.float64),
        "selected_peak_score": np.asarray(selected_scores, dtype=np.float64),
        "strongest_peak_lag": np.asarray(strongest_lags, dtype=np.int64),
        "peak_count": np.asarray(peak_counts, dtype=np.int64),
    }


@torch.inference_mode()
def evaluate_models(
    separator: torch.nn.Module,
    period_expert: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
    coefficient_a: float,
    coefficient_b: float,
) -> dict[str, np.ndarray]:
    output = {name: [] for name in (
        "sample_index", "power", "target", "learned_exact", "learned_separated",
        "analytic_exact", "analytic_separated", "exact_score", "separated_score",
    )}
    separator.eval()
    period_expert.eval()
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
        exact = batch["target_iq"].to(device)[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
        separated = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
        learned_exact = period_expert(exact)["prediction"].float() * float(scale) + float(mean)
        learned_separated = period_expert(separated)["prediction"].float() * float(scale) + float(mean)
        learned_separated = float(coefficient_a) * learned_separated + float(coefficient_b)
        exact_analytic = analytic_period_estimate(exact.cpu().numpy())
        separated_analytic = analytic_period_estimate(separated.cpu().numpy())
        output["sample_index"].append(np.repeat(batch["sample_index"].numpy(), power_count))
        output["power"].append(np.tile(np.asarray(POWERS, dtype=np.int16), batch_size))
        output["target"].append(np.repeat(batch["period"].numpy(), power_count))
        output["learned_exact"].append(learned_exact.cpu().numpy())
        output["learned_separated"].append(learned_separated.cpu().numpy())
        output["analytic_exact"].append(exact_analytic["prediction"])
        output["analytic_separated"].append(separated_analytic["prediction"])
        output["exact_score"].append(exact_analytic["selected_peak_score"])
        output["separated_score"].append(separated_analytic["selected_peak_score"])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def summarize(predictions: dict[str, np.ndarray]) -> dict[str, object]:
    target = predictions["target"]
    variants = {name: metric(predictions[name], target) for name in (
        "learned_exact", "learned_separated", "analytic_exact", "analytic_separated",
    )}
    return {
        "variants": variants,
        "analytic_exact_reduction_vs_learned": float(1.0 - variants["analytic_exact"]["mae"] / variants["learned_exact"]["mae"]),
        "analytic_separated_reduction_vs_learned": float(1.0 - variants["analytic_separated"]["mae"] / variants["learned_separated"]["mae"]),
        "mean_exact_peak_score": float(np.mean(predictions["exact_score"])),
        "mean_separated_peak_score": float(np.mean(predictions["separated_score"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2m_analytic_saw_period_seed2700000000_{regime}_rechunked")
    parser.add_argument("--reference-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--candidate-freeze", default="outputs/p2k_full_parameter_confirmation_20260801/method_freeze.json")
    parser.add_argument("--preregister", default="configs/p2m_analytic_saw_period_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2m_analytic_saw_period_physics_20260801")
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
    excluded_paths = [Path(path) if Path(path).is_absolute() else project / path for path in args.excluded_identity_files]

    records, paired_identities = [], None
    for regime in REGIMES:
        root = project / args.data_root_template.format(regime=regime)
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        selected = np.flatnonzero(labels[:, 1] > 0)
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2m {regime} data count mismatch")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError(f"P2m {regime} overlaps prior data")
        if paired_identities is None:
            paired_identities = identities
        elif not np.array_equal(paired_identities, identities):
            raise RuntimeError("P2m regimes are not identity paired")
        records.append({
            "regime": regime, "files": files, "selected": selected,
            "alignment": alignment, "exclusions": exclusions,
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for record in records:
        dataset = SawSeparatorDataset(record["files"], record["selected"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        for seed in MODEL_SEEDS:
            freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
            period_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
            separator_checkpoint = separator_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
            expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
            expert.load_state_dict(torch.load(period_checkpoint, map_location=device, weights_only=True))
            separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
            a, b = COEFFICIENTS[seed]
            predictions = evaluate_models(
                separator, expert, loader, device, float(freeze["saw_mean"]), float(freeze["saw_scale"]), a, b,
            )
            row = {
                "regime": record["regime"], "seed": seed, **summarize(predictions),
                "period_checkpoint_sha256": sha256_file(period_checkpoint),
                "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
            }
            run_dir = output_root / record["regime"] / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **predictions)
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "regime": record["regime"], "seed": seed,
                "exact_mae_us": row["variants"]["analytic_exact"]["mae_us"],
                "exact_pearson": row["variants"]["analytic_exact"]["pearson"],
                "separated_mae_us": row["variants"]["analytic_separated"]["mae_us"],
                "separated_pearson": row["variants"]["analytic_separated"]["pearson"],
                "separated_reduction_vs_learned": row["analytic_separated_reduction_vs_learned"],
            }), flush=True)
            del separator, expert
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()

    per_regime = {}
    for regime in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        per_regime[regime] = {
            "analytic_exact_mae_us_mean": float(np.mean([row["variants"]["analytic_exact"]["mae_us"] for row in subset])),
            "analytic_exact_pearson_mean": float(np.mean([row["variants"]["analytic_exact"]["pearson"] for row in subset])),
            "analytic_separated_mae_us_mean": float(np.mean([row["variants"]["analytic_separated"]["mae_us"] for row in subset])),
            "analytic_separated_pearson_mean": float(np.mean([row["variants"]["analytic_separated"]["pearson"] for row in subset])),
            "analytic_separated_reduction_vs_learned_mean": float(np.mean([row["analytic_separated_reduction_vs_learned"] for row in subset])),
        }
    gates = {
        "analytic_exact_mean_pearson_each_regime_at_least_0p60": bool(all(value["analytic_exact_pearson_mean"] >= 0.60 for value in per_regime.values())),
        "analytic_exact_mean_mae_within_regime_limits": bool(all(per_regime[name]["analytic_exact_mae_us_mean"] <= EXACT_MAE_LIMIT_US[name] for name in REGIMES)),
        "analytic_separated_mean_reduction_vs_learned_each_regime_at_least_0p15": bool(all(value["analytic_separated_reduction_vs_learned_mean"] >= 0.15 for value in per_regime.values())),
        "analytic_separated_all_15_runs_improve": bool(all(row["analytic_separated_reduction_vs_learned"] > 0.0 for row in rows)),
        "analytic_separated_mean_pearson_within_regime_limits": bool(all(per_regime[name]["analytic_separated_pearson_mean"] >= SEPARATED_PEARSON_LIMIT[name] for name in REGIMES)),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2m fixed FFT-ACF SawFM period OOD physical sufficiency",
        "protocol": vars(args),
        "algorithm": {
            "lag_samples": [40, 2799], "prominence": 0.03, "distance": 20,
            "strongest_peak_boundary": 800, "fractions": [0.20, 0.60], "absolute_floor": 0.10,
        },
        "preregister_sha256": sha256_file(preregister),
        "candidate_freeze_sha256": sha256_file(candidate_freeze),
        "identity_audit": {
            "identity_count": int(len(paired_identities)), "identity_sha256": _hash_int64(paired_identities),
            "paired_across_regimes": True,
            "regimes": [{"regime": row["regime"], "alignment": row["alignment"], "excluded_prior_splits": row["exclusions"]} for row in records],
        },
        "promotion_gates": gates, "promotion_passed": passed,
        "per_regime": per_regime, "runs": rows,
        "interpretation": (
            "P2m passed; freeze the analytic estimator for a separately preregistered full-system attachment."
            if passed else
            "P2m failed a frozen physical gate; do not attach or retune on this seed."
        ),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "per_regime": per_regime, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
