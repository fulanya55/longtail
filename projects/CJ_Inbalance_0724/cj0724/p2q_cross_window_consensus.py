"""P2q cross-window stable correction built around the frozen P2m estimator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import cj0724.p2p_harmonic_consensus as screen
from cj0724.p2m_analytic_saw_period import SAMPLE_RATE_HZ, analytic_period_estimate


WINDOW_STARTS = (0, 500, 1000)
WINDOW_LENGTH = 2000
ALLOWED_RATIOS = (3, 4, 5, 6, 7, 8)


def select_cross_window_lag(
    autocorrelation: np.ndarray,
    full_lag: int,
    window_lags: np.ndarray,
) -> dict[str, float | int | bool]:
    """Correct only when independent windows agree on a supported subharmonic."""
    window_lags = np.asarray(window_lags, dtype=np.int64)
    clusters = []
    for left in range(len(window_lags)):
        for right in range(left + 1, len(window_lags)):
            smaller = int(min(window_lags[left], window_lags[right]))
            tolerance = max(3, int(round(0.03 * smaller)))
            if abs(int(window_lags[left]) - int(window_lags[right])) > tolerance:
                continue
            center = int(round(np.median(window_lags[[left, right]])))
            members = np.abs(window_lags - center) <= max(3, int(round(0.03 * center)))
            candidate = int(round(np.median(window_lags[members])))
            clusters.append((int(np.sum(members)), int(np.max(window_lags[members]) - np.min(window_lags[members])), candidate))
    if not clusters:
        return {"lag": int(full_lag), "corrected": False, "ratio": 1, "agreement_count": 0}
    _, _, candidate = max(clusters, key=lambda row: (row[0], -row[1], row[2]))
    ratio_value = full_lag / max(candidate, 1)
    ratio = min(ALLOWED_RATIOS, key=lambda value: abs(ratio_value - value))
    if abs(ratio_value - ratio) / ratio > 0.03:
        return {"lag": int(full_lag), "corrected": False, "ratio": 1, "agreement_count": 0}

    low, high = 40, min(2800, len(autocorrelation))
    peaks, _ = find_peaks(autocorrelation[low:high], prominence=0.03, distance=20)
    peaks = peaks + low
    tolerance = max(3, int(round(0.02 * candidate)))
    nearby = peaks[np.abs(peaks - candidate) <= tolerance]
    if len(nearby) == 0:
        return {"lag": int(full_lag), "corrected": False, "ratio": 1, "agreement_count": 0}
    selected = int(nearby[np.argmax(autocorrelation[nearby])])
    agreement_count = int(np.sum(np.abs(window_lags - candidate) <= max(3, int(round(0.03 * candidate)))))
    return {"lag": selected, "corrected": True, "ratio": int(ratio), "agreement_count": agreement_count}


def cross_window_period_estimate(iq: np.ndarray) -> dict[str, np.ndarray]:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.shape[-1] != 3000:
        raise ValueError("P2q preregistered input length is 3000 samples")
    baseline = analytic_period_estimate(iq)
    windows = np.concatenate(
        [iq[:, :, start:start + WINDOW_LENGTH] for start in WINDOW_STARTS], axis=0
    )
    window_predictions = analytic_period_estimate(windows)["prediction"].reshape(len(WINDOW_STARTS), len(iq)).T
    predictions, corrected, ratios, agreements = [], [], [], []
    for row, baseline_prediction, window_row in zip(iq, baseline["prediction"], window_predictions):
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        length = len(signal)
        spectrum = np.fft.fft(signal, n=2 * length)
        acf = np.fft.ifft(spectrum * np.conj(spectrum)).real[:length] / float(length)
        acf /= max(float(acf[0]), 1.0e-12)
        full_lag = int(round(float(baseline_prediction) * SAMPLE_RATE_HZ))
        window_lags = np.rint(window_row * SAMPLE_RATE_HZ).astype(np.int64)
        decision = select_cross_window_lag(acf, full_lag, window_lags)
        predictions.append(float(decision["lag"]) / SAMPLE_RATE_HZ)
        corrected.append(bool(decision["corrected"]))
        ratios.append(int(decision["ratio"]))
        agreements.append(float(decision["agreement_count"]) / len(WINDOW_STARTS))
    return {
        "prediction": np.asarray(predictions, dtype=np.float64),
        "backtracked": np.asarray(corrected, dtype=bool),
        "divisor": np.asarray(ratios, dtype=np.int16),
        "support_fraction": np.asarray(agreements, dtype=np.float64),
        "baseline_prediction": baseline["prediction"],
    }


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--data-root-template", "generated_data/p2q_cross_window_consensus_dev_seed3200000000_{regime}_rechunked",
            "--preregister", "configs/p2q_cross_window_consensus_dev_preregister_20260801.json",
            "--output-root", "outputs/p2q_cross_window_consensus_development_20260801",
        ])
    screen.harmonic_consensus_period_estimate = cross_window_period_estimate
    screen.main()

    project = Path(__file__).resolve().parents[1]
    output_arg = "outputs/p2q_cross_window_consensus_development_20260801"
    if "--output-root" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-root") + 1]
    summary_path = project / output_arg / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exact_rate = max(float(row["exact_backtrack_rate"]) for row in summary["runs"])
    separated_rate = max(float(row["separated_backtrack_rate"]) for row in summary["runs"])
    summary["experiment"] = "P2q preregistered cross-window consensus SawFM development"
    summary["algorithm"] = {
        "full_record_samples": 3000,
        "window_length_samples": WINDOW_LENGTH,
        "window_starts": list(WINDOW_STARTS),
        "minimum_agreeing_windows": 2,
        "window_pair_relative_tolerance": 0.03,
        "allowed_full_to_window_ratios": list(ALLOWED_RATIOS),
        "integer_ratio_relative_tolerance": 0.03,
        "full_record_candidate_peak_prominence": 0.03,
        "candidate_peak_relative_tolerance": 0.02,
    }
    summary["development_gates"]["maximum_exact_correction_rate_at_most_0p05"] = exact_rate <= 0.05
    summary["development_gates"]["maximum_separated_correction_rate_at_most_0p10"] = separated_rate <= 0.10
    summary["correction_rate_audit"] = {
        "maximum_exact_rate": exact_rate,
        "maximum_separated_rate": separated_rate,
    }
    summary["development_passed"] = bool(all(summary["development_gates"].values()))
    summary["interpretation"] = (
        "P2q passed development; freeze constants before a fresh-seed confirmation."
        if summary["development_passed"] else
        "P2q failed a preregistered development gate; do not confirm or attach to the full system."
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "final_development_gates": summary["development_gates"],
        "development_passed": summary["development_passed"],
        "correction_rate_audit": summary["correction_rate_audit"],
        "interpretation": summary["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
