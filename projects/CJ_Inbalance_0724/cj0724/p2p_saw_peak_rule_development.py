"""Develop a composition-safe SawFM autocorrelation peak selector on P2m development identities only."""

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
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2f_dedicated_sawfm_separator import SawSeparatorDataset, metric
from cj0724.p2m_analytic_saw_period import POWERS, REGIMES, SAMPLE_RATE_HZ
from cj0724.p2j_full_parameter_attachment import MODEL_SEEDS


PEARSON_TARGET = {
    "ID_control": 0.60,
    "near_low": 0.55,
    "near_high": 0.30,
    "far_low": 0.90,
    "far_high": 0.40,
}
EXACT_MAE_LIMIT_US = {
    "ID_control": 2.0,
    "near_low": 1.0,
    "near_high": 2.0,
    "far_low": 0.3,
    "far_high": 1.0,
}
CANDIDATES = ("legacy", "relative_0p75", "relative_0p90", "strongest", "harmonic_consensus")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_peak_lag(lags: np.ndarray, scores: np.ndarray, rule: str) -> int:
    """Select one lag from detected peaks using a deterministic preregistered rule."""
    lags = np.asarray(lags, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(lags) == 0:
        return 40
    strongest_position = int(np.argmax(scores))
    strongest_lag = int(lags[strongest_position])
    strongest_score = float(scores[strongest_position])
    if rule == "strongest":
        return strongest_lag
    if rule in {"legacy", "relative_0p75", "relative_0p90"}:
        if rule == "legacy":
            fraction = 0.20 if strongest_lag < 800 else 0.60
        else:
            fraction = 0.75 if rule == "relative_0p75" else 0.90
        eligible = lags[scores >= max(0.10, fraction * strongest_score)]
        return int(np.min(eligible)) if len(eligible) else strongest_lag
    if rule != "harmonic_consensus":
        raise KeyError(rule)

    # A true fundamental is supported by peaks near its integer multiples.  The
    # score combines normalized peak strength and ordinal coverage, and avoids
    # using labels or a learned calibration.
    normalized = np.maximum(scores, 0.0) / max(strongest_score, 1.0e-12)
    values = []
    for candidate_index, candidate_lag in enumerate(lags):
        support = 0.0
        coverage = 0
        maximum_order = min(6, int(2799 // candidate_lag))
        for order in range(1, maximum_order + 1):
            expected = order * int(candidate_lag)
            tolerance = max(8, int(round(0.025 * expected)))
            distances = np.abs(lags - expected)
            nearest = int(np.argmin(distances))
            if int(distances[nearest]) <= tolerance:
                support += float(normalized[nearest]) / float(order)
                coverage += 1
        # Require the candidate itself to be credible and reward repeated
        # ordinal support without introducing a target-dependent scale prior.
        values.append(float(normalized[candidate_index]) + support + 0.20 * coverage)
    values_array = np.asarray(values, dtype=np.float64)
    best = float(np.max(values_array))
    eligible = lags[values_array >= best - 1.0e-12]
    return int(np.min(eligible))


def estimate_candidates(iq: np.ndarray) -> dict[str, np.ndarray]:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [samples, 2, time]")
    output = {name: [] for name in CANDIDATES}
    for row in iq:
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        length = len(signal)
        spectrum = np.fft.fft(signal, n=2 * length)
        autocorrelation = np.fft.ifft(spectrum * np.conj(spectrum)).real[:length] / float(length)
        autocorrelation /= max(float(autocorrelation[0]), 1.0e-12)
        low, high = 40, min(2800, length)
        peaks, _ = find_peaks(autocorrelation[low:high], prominence=0.03, distance=20)
        lags = peaks + low
        scores = autocorrelation[lags] if len(lags) else np.empty(0, dtype=np.float64)
        for name in CANDIDATES:
            output[name].append(select_peak_lag(lags, scores, name) / SAMPLE_RATE_HZ)
    return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}


@torch.inference_mode()
def evaluate_separator(separator: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    exact_parts, separated_parts, target_parts = [], [], []
    separator.eval()
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
        exact = batch["target_iq"].to(device)[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
        separated = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
        exact_parts.append(exact.cpu().numpy())
        separated_parts.append(separated.cpu().numpy())
        target_parts.append(np.repeat(batch["period"].numpy(), power_count))
    return {
        "exact": np.concatenate(exact_parts),
        "separated": np.concatenate(separated_parts),
        "target": np.concatenate(target_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2m_analytic_saw_period_seed2700000000_{regime}_rechunked")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--separator-variant", default="S1_waveform_only")
    parser.add_argument("--baseline-root", default="outputs/p2m_analytic_saw_period_physics_20260801")
    parser.add_argument("--preregister", default="configs/p2p_saw_peak_rule_development_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2p_saw_peak_rule_development_20260801")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    separator_root = project / args.separator_root
    baseline_root = project / args.baseline_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for regime in REGIMES:
        files = transition_files(project / args.data_root_template.format(regime=regime), "train", POWERS)
        # Match the frozen P2m positive-only protocol exactly.
        baseline_prediction = np.load(baseline_root / regime / f"seed_{MODEL_SEEDS[0]}" / "predictions.npz")
        selected_indices = np.unique(baseline_prediction["sample_index"])
        baseline_prediction.close()
        dataset = SawSeparatorDataset(files, selected_indices)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        for seed in MODEL_SEEDS:
            checkpoint = separator_root / f"seed_{seed}" / args.separator_variant / "best.pt"
            separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            separator.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            signals = evaluate_separator(separator, loader, device)
            exact_predictions = estimate_candidates(signals["exact"])
            separated_predictions = estimate_candidates(signals["separated"])
            for candidate in CANDIDATES:
                rows.append({
                    "regime": regime,
                    "seed": seed,
                    "candidate": candidate,
                    "exact": metric(exact_predictions[candidate], signals["target"]),
                    "separated": metric(separated_predictions[candidate], signals["target"]),
                    "separator_checkpoint_sha256": sha256_file(checkpoint),
                })
            run_dir = output_root / regime / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(
                run_dir / "predictions.npz",
                target=signals["target"],
                **{f"exact_{name}": exact_predictions[name] for name in CANDIDATES},
                **{f"separated_{name}": separated_predictions[name] for name in CANDIDATES},
            )
            del separator
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()

    candidate_summary = {}
    legacy = {}
    for regime in REGIMES:
        subset = [row for row in rows if row["regime"] == regime and row["candidate"] == "legacy"]
        legacy[regime] = {
            "separated_mae_us_mean": float(np.mean([row["separated"]["mae_us"] for row in subset])),
            "separated_pearson_by_seed": {str(row["seed"]): float(row["separated"]["pearson"]) for row in subset},
        }

    passing = []
    for candidate in CANDIDATES:
        by_regime = {}
        for regime in REGIMES:
            subset = [row for row in rows if row["regime"] == regime and row["candidate"] == candidate]
            by_regime[regime] = {
                "exact_mae_us_mean": float(np.mean([row["exact"]["mae_us"] for row in subset])),
                "exact_pearson_mean": float(np.mean([row["exact"]["pearson"] for row in subset])),
                "separated_mae_us_mean": float(np.mean([row["separated"]["mae_us"] for row in subset])),
                "separated_pearson_mean": float(np.mean([row["separated"]["pearson"] for row in subset])),
                "separated_pearson_by_seed": {str(row["seed"]): float(row["separated"]["pearson"]) for row in subset},
            }
        gates = {
            "exact_physics": bool(all(
                value["exact_pearson_mean"] >= 0.60 and value["exact_mae_us_mean"] <= EXACT_MAE_LIMIT_US[regime]
                for regime, value in by_regime.items()
            )),
            "separated_pearson": bool(all(
                value["separated_pearson_mean"] >= PEARSON_TARGET[regime]
                for regime, value in by_regime.items()
            )),
            "separated_mae_vs_legacy": bool(all(
                value["separated_mae_us_mean"] <= 1.10 * legacy[regime]["separated_mae_us_mean"]
                for regime, value in by_regime.items()
            )),
            "hard_regimes_three_seed_direction": bool(all(
                by_regime[regime]["separated_pearson_by_seed"][str(seed)]
                > legacy[regime]["separated_pearson_by_seed"][str(seed)]
                for regime in ("near_high", "far_high") for seed in MODEL_SEEDS
            )),
        }
        margins = [by_regime[regime]["separated_pearson_mean"] - PEARSON_TARGET[regime] for regime in REGIMES]
        candidate_summary[candidate] = {
            "per_regime": by_regime,
            "gates": gates,
            "passed": bool(all(gates.values())),
            "selection_score": {
                "minimum_pearson_margin": float(min(margins)),
                "mean_separated_pearson": float(np.mean([by_regime[r]["separated_pearson_mean"] for r in REGIMES])),
                "mean_separated_mae_us": float(np.mean([by_regime[r]["separated_mae_us_mean"] for r in REGIMES])),
            },
        }
        if candidate_summary[candidate]["passed"] and candidate != "legacy":
            passing.append(candidate)

    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                candidate_summary[name]["selection_score"]["minimum_pearson_margin"],
                candidate_summary[name]["selection_score"]["mean_separated_pearson"],
                -candidate_summary[name]["selection_score"]["mean_separated_mae_us"],
            ),
        )
    summary = {
        "experiment": "P2p SawFM autocorrelation peak-rule development",
        "separator_variant": args.separator_variant,
        "preregister_sha256": sha256_file(preregister),
        "candidate_summary": candidate_summary,
        "selected_candidate": selected,
        "development_passed": selected is not None,
        "next_action": (
            "Freeze the selected rule before any new-identity validation."
            if selected else
            "No candidate passed; retain P2n as development-only and formulate a new mechanism on fresh development data."
        ),
        "confirmation_identities_used_for_selection": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
        "runs": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if selected:
        freeze = {
            "method": "SawFM autocorrelation peak rule",
            "selected_candidate": selected,
            "source_summary": str((output_root / "summary.json").resolve()),
            "source_preregister_sha256": sha256_file(preregister),
            "development_identity_seed": 2700000000,
            "forbidden_confirmation_identity_seeds": [2800000000, 2900000000],
            "locked_test_used": False,
        }
        (output_root / "method_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({
        "development_passed": summary["development_passed"],
        "selected_candidate": selected,
        "candidate_gates": {name: value["gates"] for name, value in candidate_summary.items()},
        "selection_scores": {name: value["selection_score"] for name, value in candidate_summary.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
