"""P2p preregistered conservative harmonic-consensus SawFM period screen."""

from __future__ import annotations

import argparse
import json
import math
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
from cj0724.p2m_analytic_saw_period import (
    EXACT_MAE_LIMIT_US,
    POWERS,
    REGIMES,
    SAMPLE_RATE_HZ,
    SEPARATED_PEARSON_LIMIT,
    analytic_period_estimate,
    sha256_file,
)
from cj0724.physics_parameter_specialists import build_physics_specialist


DIVISORS = (3, 4, 5, 6, 7, 8)


def select_harmonic_consensus_lag(
    autocorrelation: np.ndarray,
    baseline_lag: int,
) -> dict[str, float | int | bool]:
    """Conservatively backtrack a baseline peak to a supported integer divisor."""
    autocorrelation = np.asarray(autocorrelation, dtype=np.float64)
    low, high = 40, min(2800, len(autocorrelation))
    secondary, _ = find_peaks(autocorrelation[low:high], prominence=0.01, distance=10)
    peaks = secondary + low
    valid = []
    for divisor in DIVISORS:
        candidate = int(round(baseline_lag / divisor))
        if candidate < low:
            continue
        tolerance = max(3, int(round(0.02 * candidate)))
        matched_scores = []
        fundamental_matched = False
        for harmonic in range(1, divisor + 1):
            expected = harmonic * candidate
            if expected >= high:
                break
            nearby = peaks[np.abs(peaks - expected) <= tolerance]
            if len(nearby) == 0:
                continue
            best = int(nearby[np.argmax(autocorrelation[nearby])])
            matched_scores.append(float(autocorrelation[best]))
            fundamental_matched |= harmonic == 1
        expected_harmonics = min(divisor, (high - 1) // candidate)
        support_count = len(matched_scores)
        support_fraction = support_count / max(expected_harmonics, 1)
        fundamental_score = float(np.max(
            autocorrelation[max(low, candidate - tolerance):min(high, candidate + tolerance + 1)]
        ))
        if (
            fundamental_matched
            and fundamental_score >= 0.02
            and support_count >= 3
            and support_fraction >= 0.75
        ):
            valid.append({
                "lag": candidate,
                "divisor": divisor,
                "support_count": support_count,
                "support_fraction": support_fraction,
                "mean_matched_acf": float(np.mean(matched_scores)),
                "fundamental_score": fundamental_score,
            })
    if not valid:
        return {
            "lag": int(baseline_lag), "backtracked": False, "divisor": 1,
            "support_count": 1, "support_fraction": 1.0,
            "mean_matched_acf": float(autocorrelation[baseline_lag]),
            "fundamental_score": float(autocorrelation[baseline_lag]),
        }
    best = max(valid, key=lambda row: (row["support_fraction"], row["mean_matched_acf"], row["lag"]))
    return {**best, "backtracked": True}


def harmonic_consensus_period_estimate(iq: np.ndarray) -> dict[str, np.ndarray]:
    """Run frozen P2m, then apply only the preregistered divisor-consensus rule."""
    iq = np.asarray(iq, dtype=np.float32)
    baseline = analytic_period_estimate(iq)
    prediction, flags, divisors, supports = [], [], [], []
    for row, baseline_prediction in zip(iq, baseline["prediction"]):
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        length = len(signal)
        spectrum = np.fft.fft(signal, n=2 * length)
        acf = np.fft.ifft(spectrum * np.conj(spectrum)).real[:length] / float(length)
        acf /= max(float(acf[0]), 1.0e-12)
        baseline_lag = int(round(float(baseline_prediction) * SAMPLE_RATE_HZ))
        decision = select_harmonic_consensus_lag(acf, baseline_lag)
        prediction.append(float(decision["lag"]) / SAMPLE_RATE_HZ)
        flags.append(bool(decision["backtracked"]))
        divisors.append(int(decision["divisor"]))
        supports.append(float(decision["support_fraction"]))
    return {
        "prediction": np.asarray(prediction, dtype=np.float64),
        "backtracked": np.asarray(flags, dtype=bool),
        "divisor": np.asarray(divisors, dtype=np.int16),
        "support_fraction": np.asarray(supports, dtype=np.float64),
        "baseline_prediction": baseline["prediction"],
    }


@torch.inference_mode()
def evaluate_models(separator, period_expert, loader, device, mean, scale, coefficient_a, coefficient_b):
    names = (
        "sample_index", "power", "target", "learned_exact", "learned_separated",
        "baseline_exact", "baseline_separated", "consensus_exact", "consensus_separated",
        "exact_backtracked", "separated_backtracked", "exact_divisor", "separated_divisor",
        "exact_support", "separated_support",
    )
    output = {name: [] for name in names}
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
        exact_consensus = harmonic_consensus_period_estimate(exact.cpu().numpy())
        separated_consensus = harmonic_consensus_period_estimate(separated.cpu().numpy())
        output["sample_index"].append(np.repeat(batch["sample_index"].numpy(), power_count))
        output["power"].append(np.tile(np.asarray(POWERS, dtype=np.int16), batch_size))
        output["target"].append(np.repeat(batch["period"].numpy(), power_count))
        output["learned_exact"].append(learned_exact.cpu().numpy())
        output["learned_separated"].append(learned_separated.cpu().numpy())
        output["baseline_exact"].append(exact_consensus["baseline_prediction"])
        output["baseline_separated"].append(separated_consensus["baseline_prediction"])
        output["consensus_exact"].append(exact_consensus["prediction"])
        output["consensus_separated"].append(separated_consensus["prediction"])
        output["exact_backtracked"].append(exact_consensus["backtracked"])
        output["separated_backtracked"].append(separated_consensus["backtracked"])
        output["exact_divisor"].append(exact_consensus["divisor"])
        output["separated_divisor"].append(separated_consensus["divisor"])
        output["exact_support"].append(exact_consensus["support_fraction"])
        output["separated_support"].append(separated_consensus["support_fraction"])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def summarize(predictions: dict[str, np.ndarray]) -> dict[str, object]:
    target = predictions["target"]
    variants = {name: metric(predictions[name], target) for name in (
        "learned_exact", "learned_separated", "baseline_exact", "baseline_separated",
        "consensus_exact", "consensus_separated",
    )}
    baseline_error = np.abs(predictions["baseline_exact"] - target) * 1.0e6
    consensus_error = np.abs(predictions["consensus_exact"] - target) * 1.0e6
    return {
        "variants": variants,
        "consensus_separated_reduction_vs_learned": float(
            1.0 - variants["consensus_separated"]["mae"] / variants["learned_separated"]["mae"]
        ),
        "exact_backtrack_rate": float(np.mean(predictions["exact_backtracked"])),
        "separated_backtrack_rate": float(np.mean(predictions["separated_backtracked"])),
        "baseline_exact_catastrophic_count": int(np.sum(baseline_error >= 1.0)),
        "consensus_exact_catastrophic_count": int(np.sum(consensus_error >= 1.0)),
        "introduced_exact_catastrophic_count": int(np.sum((baseline_error < 0.1) & (consensus_error >= 1.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2p_harmonic_consensus_dev_seed3000000000_{regime}_rechunked")
    parser.add_argument("--reference-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--candidate-freeze", default="outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json")
    parser.add_argument("--preregister", default="configs/p2p_harmonic_consensus_dev_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2p_harmonic_consensus_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
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
            raise RuntimeError(f"P2p {regime} data count mismatch")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError(f"P2p {regime} overlaps prior data")
        if paired_identities is None:
            paired_identities = identities
        elif not np.array_equal(paired_identities, identities):
            raise RuntimeError("P2p regimes are not identity paired")
        records.append({"regime": regime, "files": files, "selected": selected, "alignment": alignment, "exclusions": exclusions})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_root = project / args.reference_root
    separator_root = project / args.separator_root
    specialist_root = project / args.specialist_root
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
            row = {"regime": record["regime"], "seed": seed, **summarize(predictions)}
            run_dir = output_root / record["regime"] / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", **predictions)
            (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "regime": record["regime"], "seed": seed,
                "baseline_exact_mae_us": row["variants"]["baseline_exact"]["mae_us"],
                "consensus_exact_mae_us": row["variants"]["consensus_exact"]["mae_us"],
                "baseline_separated_pearson": row["variants"]["baseline_separated"]["pearson"],
                "consensus_separated_pearson": row["variants"]["consensus_separated"]["pearson"],
                "exact_backtrack_rate": row["exact_backtrack_rate"],
                "separated_backtrack_rate": row["separated_backtrack_rate"],
            }), flush=True)
            del separator, expert
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()

    per_regime = {}
    for regime in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        mean_variant = {
            name: {
                "mae_us": float(np.mean([row["variants"][name]["mae_us"] for row in subset])),
                "pearson": float(np.mean([row["variants"][name]["pearson"] for row in subset])),
            }
            for name in ("baseline_exact", "baseline_separated", "consensus_exact", "consensus_separated")
        }
        per_regime[regime] = mean_variant

    baseline_catastrophic = sum(row["baseline_exact_catastrophic_count"] for row in rows[::3])
    consensus_catastrophic = sum(row["consensus_exact_catastrophic_count"] for row in rows[::3])
    introduced_catastrophic = sum(row["introduced_exact_catastrophic_count"] for row in rows[::3])
    gates = {
        "identity_overlap_with_P2i_through_P2o_is_zero": bool(all(
            exclusion["overlap_count"] == 0 for record in records for exclusion in record["exclusions"]
        )),
        "consensus_exact_mean_pearson_each_regime_at_least_0p60": bool(all(
            per_regime[name]["consensus_exact"]["pearson"] >= 0.60 for name in REGIMES
        )),
        "consensus_exact_mean_mae_within_limits": bool(all(
            per_regime[name]["consensus_exact"]["mae_us"] <= EXACT_MAE_LIMIT_US[name] for name in REGIMES
        )),
        "consensus_separated_mean_pearson_within_limits": bool(all(
            per_regime[name]["consensus_separated"]["pearson"] >= SEPARATED_PEARSON_LIMIT[name] for name in REGIMES
        )),
        "consensus_separated_all_15_runs_improve_vs_learned": bool(all(
            row["consensus_separated_reduction_vs_learned"] > 0.0 for row in rows
        )),
        "consensus_exact_catastrophic_count_at_most_half_baseline": bool(
            consensus_catastrophic <= 0.5 * baseline_catastrophic
        ),
        "consensus_exact_introduced_no_catastrophic_errors": bool(introduced_catastrophic == 0),
        "consensus_separated_mean_mae_nonworse_each_regime": bool(all(
            per_regime[name]["consensus_separated"]["mae_us"] <= per_regime[name]["baseline_separated"]["mae_us"]
            for name in REGIMES
        )),
        "consensus_separated_mean_pearson_nonworse_each_regime": bool(all(
            per_regime[name]["consensus_separated"]["pearson"] >= per_regime[name]["baseline_separated"]["pearson"]
            for name in REGIMES
        )),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2p preregistered harmonic-consensus SawFM development",
        "protocol": vars(args),
        "algorithm": {
            "divisors": list(DIVISORS), "secondary_peak_prominence": 0.01,
            "secondary_peak_distance": 10, "minimum_fundamental_acf": 0.02,
            "minimum_supported_harmonics": 3, "minimum_support_fraction": 0.75,
        },
        "preregister_sha256": sha256_file(project / args.preregister),
        "candidate_freeze_sha256": sha256_file(project / args.candidate_freeze),
        "identity_audit": {
            "identity_count": int(len(paired_identities)), "identity_sha256": _hash_int64(paired_identities),
            "paired_across_regimes": True,
            "regimes": [{"regime": row["regime"], "alignment": row["alignment"], "excluded_prior_splits": row["exclusions"]} for row in records],
        },
        "catastrophic_error_audit": {
            "baseline_exact_count": int(baseline_catastrophic),
            "consensus_exact_count": int(consensus_catastrophic),
            "introduced_exact_count": int(introduced_catastrophic),
        },
        "development_gates": gates,
        "development_passed": passed,
        "per_regime": per_regime,
        "runs": rows,
        "interpretation": (
            "P2p passed development; freeze constants before a new-seed confirmation."
            if passed else
            "P2p failed a preregistered development gate; do not confirm or attach to the full system."
        ),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "development_gates": gates, "development_passed": passed,
        "catastrophic_error_audit": summary["catastrophic_error_audit"],
        "per_regime": per_regime, "interpretation": summary["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
