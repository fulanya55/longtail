"""Zero-fit H0u separated-delay MFT readout versus a phase-order control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.mft_matched_filter_support import fixed_phase_permutation, lfm_reference


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0u_mft_delay_aperture_20260803_r1"
BANDS = ((1600, 3000), (3500, 5000), (5400, 7000))


def _load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        real, imag = np.asarray(handle["iq_real"]), np.asarray(handle["iq_imag"])
    iq = np.stack((real, imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.maximum(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1, keepdims=True), 1e-12))
    return iq / rms[:, :, None]


def _response(iq: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    z = torch.complex(iq[:, 0], iq[:, 1])
    length = z.shape[-1]
    template = torch.zeros(length, dtype=torch.complex64, device=z.device)
    template[: reference.numel()] = reference.to(device=z.device, dtype=torch.complex64)
    return torch.fft.ifft(torch.fft.fft(z, dim=-1) * torch.fft.fft(template).conj()[None, :], dim=-1).abs()


def separated_peak_score(iq: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    response = _response(iq, reference)
    peaks = torch.stack([response[:, start:stop].amax(dim=1) for start, stop in BANDS], dim=1)
    return torch.log(peaks.clamp_min(1e-12)).mean(dim=1)


def _scores(iq: np.ndarray, reference: torch.Tensor, control: torch.Tensor, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate, permuted = [], []
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            item = torch.from_numpy(iq[start:start + batch_size]).to(device)
            candidate.append(separated_peak_score(item, reference).cpu().numpy())
            permuted.append(separated_peak_score(item, control).cpu().numpy())
    return np.concatenate(candidate).astype(np.float64), np.concatenate(permuted).astype(np.float64)


def _bootstrap(y: np.ndarray, candidate: np.ndarray, control: np.ndarray, draws: int = 5000, seed: int = 20_260_803) -> dict[str, list[float]]:
    rng, pairs = np.random.default_rng(seed), len(y) // 2
    auroc, aupr = np.empty(draws), np.empty(draws)
    for draw in range(draws):
        index = rng.integers(0, pairs, pairs)
        joined = np.concatenate((index, index + pairs))
        auroc[draw] = roc_auc_score(y[joined], candidate[joined]) - roc_auc_score(y[joined], control[joined])
        aupr[draw] = average_precision_score(y[joined], candidate[joined]) - average_precision_score(y[joined], control[joined])
    return {"method": "paired_counterfactual_bootstrap_5000", "auroc_delta_ci95": [float(np.quantile(auroc, .025)), float(np.quantile(auroc, .975))], "aupr_delta_ci95": [float(np.quantile(aupr, .025)), float(np.quantile(aupr, .975))]}


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text(encoding="utf-8")); seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not audit.get("identity_audit_passed") or not all(audit["contracts"].values()):
        raise ValueError("H0u source audit has not passed")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    absent, present = _load(args.absent_path), _load(args.present_path)
    if absent.shape != present.shape or absent.shape != (640, 2, 10_000):
        raise ValueError(f"unexpected H0u source shape: {absent.shape} / {present.shape}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = lfm_reference(device=device); control = fixed_phase_permutation(reference)
    ca, pa = _scores(absent, reference, control, device, args.batch_size); cp, pp = _scores(present, reference, control, device, args.batch_size)
    y = np.r_[np.zeros(len(ca), dtype=np.uint8), np.ones(len(cp), dtype=np.uint8)]
    candidate, permuted = np.r_[ca, cp], np.r_[pa, pp]
    metrics = {"correct_lfm": {"auroc": float(roc_auc_score(y, candidate)), "aupr": float(average_precision_score(y, candidate))}, "phase_order_permuted_lfm": {"auroc": float(roc_auc_score(y, permuted)), "aupr": float(average_precision_score(y, permuted))}}
    delta = {key: metrics["correct_lfm"][key] - metrics["phase_order_permuted_lfm"][key] for key in ("auroc", "aupr")}
    bootstrap = _bootstrap(y, candidate, permuted); paired_rate = float(np.mean(cp > ca))
    gates = {"correct_lfm_auroc_ge_0p70": metrics["correct_lfm"]["auroc"] >= .70, "candidate_minus_control_auroc_ge_0p10": delta["auroc"] >= .10, "candidate_minus_control_auroc_ci_lower_gt_zero": bootstrap["auroc_delta_ci95"][0] > 0, "candidate_minus_control_aupr_ge_0p08": delta["aupr"] >= .08, "candidate_minus_control_aupr_ci_lower_gt_zero": bootstrap["aupr_delta_ci95"][0] > 0, "paired_present_gt_absent_rate_ge_0p65": paired_rate >= .65, "zero_fit_no_threshold_no_calibration_no_fusion": True, "source_identity_contract": True, "test_v3_sealed": True}
    result = {"status": "completed", "experiment": "RouteA_H0u_MFTDelayAperture", "evidence_role": "fresh_external_delay_aperture_paired_physical_identifiability", "statistic": {"candidate": "correct_LFM_three_separated_delay_band_log_peak_mean", "phase_order_control": "fixed_phase_order_permuted_LFM_same_bands", "bands_samples": [list(band) for band in BANDS], "trained_parameters": 0}, "metrics": metrics, "candidate_minus_control": delta, "paired_bootstrap": bootstrap, "paired_present_greater_than_absent_rate": paired_rate, "score_summary": {"candidate_absent_mean": float(ca.mean()), "candidate_present_mean": float(cp.mean()), "control_absent_mean": float(pa.mean()), "control_present_mean": float(pp.mean())}, "gates": gates, "passed": bool(all(gates.values())), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--absent-path", type=Path, default=SOURCE / "mft_absent.mat")
    parser.add_argument("--present-path", type=Path, default=SOURCE / "mft_present_weak_separated.mat")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h0u_mft_delay_aperture_generation/paired_identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0u_mft_delay_aperture/summary.json")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default=None)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
