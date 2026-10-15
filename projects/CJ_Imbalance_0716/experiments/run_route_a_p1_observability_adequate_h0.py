"""Run P1's preregistered no-fit physical observability gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generate_route_a_p1_observability_adequate_pairs import FS, K, LABELS, LENGTH, PRI, PULSE, PULSE_COUNT


TARGETS = {3: "MFT", 4: "VFT", 5: "ISRJ"}
MFT_BANDS_HZ = ((1.25e6, 1.75e6), (2.75e6, 3.25e6), (3.75e6, 4.25e6))
VFT_BAND_HZ = (15e3, 65e3)
PULSE_PERMUTATION = np.asarray((0, 3, 1, 5, 2, 4), dtype=np.int64)
SAMPLE_PERMUTATION = np.random.default_rng(20260803).permutation(LENGTH)


def _continuous_dechirp(pri: int, chirp_rate: float) -> np.ndarray:
    local = np.arange(pri, dtype=np.float64) / FS
    return np.exp(-1j * np.pi * chirp_rate * local**2)


def _mft_score(signal: np.ndarray, chirp_rate: float) -> float:
    mix = _continuous_dechirp(PRI, chirp_rate)
    segments = signal.reshape(PULSE_COUNT, PRI) * mix[None, :]
    spectrum = np.fft.fft(segments, axis=1)
    frequency = np.abs(np.fft.fftfreq(PRI, 1 / FS))
    selected = np.zeros(PRI, dtype=bool)
    for low, high in MFT_BANDS_HZ:
        selected |= (frequency >= low) & (frequency <= high)
    numerator = np.max(np.abs(spectrum[:, selected]) ** 2)
    denominator = max(float(np.mean(np.abs(spectrum[:, selected]) ** 2)), np.finfo(np.float64).eps)
    return float(numerator / denominator)


def _vft_scores(signal: np.ndarray) -> tuple[float, float]:
    local = np.arange(PULSE, dtype=np.float64) / FS
    mix = np.exp(-1j * np.pi * K * local**2)
    segments = signal.reshape(PULSE_COUNT, PRI)[:, :PULSE] * mix[None, :]
    nfft = 8192
    coefficient = np.fft.fft(segments, n=nfft, axis=1)
    frequency = np.fft.fftfreq(nfft, 1 / FS)
    selected = (frequency >= VFT_BAND_HZ[0]) & (frequency <= VFT_BAND_HZ[1])
    selected_frequency = frequency[selected]
    coefficient = coefficient[:, selected]
    starts = np.arange(PULSE_COUNT, dtype=np.float64) * PRI / FS
    phase = np.exp(-2j * np.pi * starts[:, None] * selected_frequency[None, :])
    phase_permuted = np.exp(-2j * np.pi * starts[PULSE_PERMUTATION, None] * selected_frequency[None, :])
    denom = np.sqrt(np.sum(np.abs(coefficient) ** 2, axis=0)) * np.sqrt(PULSE_COUNT)
    true = np.abs(np.sum(coefficient * phase, axis=0)) / np.maximum(denom, np.finfo(np.float64).eps)
    permuted = np.abs(np.sum(coefficient * phase_permuted, axis=0)) / np.maximum(denom, np.finfo(np.float64).eps)
    return float(np.max(true)), float(np.max(permuted))


def _isrj_score(signal: np.ndarray) -> float:
    envelope = np.abs(signal).astype(np.float64, copy=False)
    envelope -= envelope.mean()
    spectrum = np.fft.fft(envelope, n=2 * envelope.size)
    autocorrelation = np.fft.ifft(spectrum * np.conj(spectrum)).real[: envelope.size]
    autocorrelation /= max(float(autocorrelation[0]), np.finfo(np.float64).eps)
    return float(np.max(autocorrelation[50:1_001]))


def _score_rows(iq: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    true = {name: np.empty(iq.shape[0], dtype=np.float64) for name in TARGETS.values()}
    control = {name: np.empty(iq.shape[0], dtype=np.float64) for name in TARGETS.values()}
    for row, signal in enumerate(iq):
        true["MFT"][row] = _mft_score(signal, K)
        control["MFT"][row] = _mft_score(signal, 0.83 * K)
        true["VFT"][row], control["VFT"][row] = _vft_scores(signal)
        true["ISRJ"][row] = _isrj_score(signal)
        control["ISRJ"][row] = _isrj_score(signal[SAMPLE_PERMUTATION])
    return true, control


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        iq = np.asarray(handle["iq_real"], dtype=np.float64) + 1j * np.asarray(handle["iq_imag"], dtype=np.float64)
        labels = np.asarray(handle["labels"], dtype=np.uint8)
        pair = np.asarray(handle["pair_seed"], dtype=np.uint64)
        receiver = np.asarray(handle["receiver_noise_seed"], dtype=np.uint64)
        target = np.asarray(handle["target_index"], dtype=np.int64)
        arm = np.asarray(handle["visibility_arm"], dtype=np.int64)
    if iq.shape[1] != LENGTH or labels.shape != (iq.shape[0], 6):
        raise ValueError("P1 I/Q/label contract failed")
    return iq, labels, pair, receiver, target, arm


def _triplets(pair: np.ndarray, receiver: np.ndarray, target: np.ndarray, arm: np.ndarray, labels: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    values: dict[str, list[tuple[int, int, int]]] = {name: [] for name in TARGETS.values()}
    for seed in np.unique(pair):
        rows = np.flatnonzero(pair == seed)
        if rows.size != 3 or set(arm[rows]) != {0, 1, 2} or np.unique(receiver[rows]).size != 1 or np.unique(target[rows]).size != 1:
            raise ValueError(f"P1 identity {seed} triplet contract failed")
        current = int(target[rows[0]])
        if current not in TARGETS:
            raise ValueError("P1 unexpected target")
        ordered = rows[np.argsort(arm[rows])]
        if labels[ordered, current].tolist() != [0, 1, 1]:
            raise ValueError(f"P1 target arm labels failed for {seed}")
        if not np.array_equal(np.delete(labels[ordered[0]], current), np.delete(labels[ordered[1]], current)) or not np.array_equal(np.delete(labels[ordered[0]], current), np.delete(labels[ordered[2]], current)):
            raise ValueError(f"P1 nuisance labels failed for {seed}")
        values[TARGETS[current]].append(tuple(map(int, ordered)))
    return {name: tuple(np.asarray([item[i] for item in items], dtype=np.int64) for i in range(3)) for name, items in values.items()}


def _metrics(absent: np.ndarray, weak: np.ndarray, strong: np.ndarray, absent_control: np.ndarray, weak_control: np.ndarray) -> dict[str, float]:
    y = np.r_[np.zeros(absent.size, dtype=np.uint8), np.ones(weak.size, dtype=np.uint8)]
    true = np.r_[absent, weak]; control = np.r_[absent_control, weak_control]
    return {
        "true_auroc": float(roc_auc_score(y, true)), "control_auroc": float(roc_auc_score(y, control)),
        "true_minus_control_auroc": float(roc_auc_score(y, true) - roc_auc_score(y, control)),
        "true_aupr": float(average_precision_score(y, true)),
        "paired_weak_gt_absent_fraction": float(np.mean(weak > absent)),
        "strong_vs_absent_auroc": float(roc_auc_score(np.r_[np.zeros(absent.size, dtype=np.uint8), np.ones(strong.size, dtype=np.uint8)], np.r_[absent, strong])),
    }


def _bootstrap(absent: np.ndarray, weak: np.ndarray, strong: np.ndarray, absent_control: np.ndarray, weak_control: np.ndarray, draws: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260803); true, delta = [], []
    for _ in range(draws):
        rows = rng.integers(0, absent.size, absent.size)
        metric = _metrics(absent[rows], weak[rows], strong[rows], absent_control[rows], weak_control[rows])
        true.append(metric["true_auroc"]); delta.append(metric["true_minus_control_auroc"])
    return {"true_auroc_95_ci": [float(np.quantile(true, .025)), float(np.quantile(true, .975))], "true_minus_control_auroc_95_ci": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))]}


def run(args: argparse.Namespace) -> dict:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed")
    iq, labels, pair, receiver, target, arm = _read(args.data_path)
    true, control = _score_rows(iq)
    triplets = _triplets(pair, receiver, target, arm, labels)
    expected = 3 if args.smoke else 300
    if any(rows[0].size != expected for rows in triplets.values()):
        raise ValueError(f"P1 expected {expected} identities per target")
    if args.smoke:
        result = {"status": "completed_smoke_score_only", "experiment": "RouteA_P1_ObservabilityAdequate", "view_count": int(iq.shape[0]), "scores_finite": bool(all(np.all(np.isfinite(value)) for value in (*true.values(), *control.values()))), "metrics_reported": False, "gate_evaluated": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    else:
        values = {}; passed = []
        for name, (absent_rows, weak_rows, strong_rows) in triplets.items():
            metric = _metrics(true[name][absent_rows], true[name][weak_rows], true[name][strong_rows], control[name][absent_rows], control[name][weak_rows])
            bootstrap = _bootstrap(true[name][absent_rows], true[name][weak_rows], true[name][strong_rows], control[name][absent_rows], control[name][weak_rows], args.bootstrap_draws)
            checks = {"true_auroc_ge_0p75": metric["true_auroc"] >= .75, "true_minus_control_auroc_ge_0p10": metric["true_minus_control_auroc"] >= .10, "delta_ci_lower_gt_zero": bootstrap["true_minus_control_auroc_95_ci"][0] > 0, "paired_weak_gt_absent_fraction_ge_0p65": metric["paired_weak_gt_absent_fraction"] >= .65, "strong_auroc_not_below_weak_by_more_than_0p02": metric["strong_vs_absent_auroc"] >= metric["true_auroc"] - .02}
            values[name] = {"identity_count": int(absent_rows.size), **metric, "bootstrap": {"draws": args.bootstrap_draws, **bootstrap}, "checks": checks, "passed": bool(all(checks.values()))}; passed.append(bool(all(checks.values())))
        result = {"status": "completed_no_fit_observable", "experiment": "RouteA_P1_ObservabilityAdequate", "evidence_role": "external_protocol_h0_development_only", "view_count": int(iq.shape[0]), "fixed_statistics": {"mft_bands_hz": MFT_BANDS_HZ, "vft_band_hz": VFT_BAND_HZ, "pulse_permutation": PULSE_PERMUTATION.tolist(), "sample_permutation_seed": 20260803}, "trained_parameters": 0, "metrics_by_target": values, "gate": {"passed": bool(all(passed)), "all_scores_finite": bool(all(np.all(np.isfinite(value)) for value in (*true.values(), *control.values())))}, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--bootstrap-draws", type=int, default=5000); parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    result = run(parser.parse_args()); print(json.dumps({"status": result["status"], "gate": result.get("gate")}, ensure_ascii=False, indent=2))
