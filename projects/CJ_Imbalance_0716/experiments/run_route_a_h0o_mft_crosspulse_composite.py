"""Run H0o's preregistered no-fit MFT-present/absent composite comparison."""
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
from cjlt.models.mft_crosspulse_coherence import crosspulse_mft_scores, lfm_pulse


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0o_mft_crosspulse_composite_20260803"


def _load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        # h5py presents MATLAB's [samples, batch] payload as [batch, samples].
        real = np.asarray(handle["iq_real"])
        imag = np.asarray(handle["iq_imag"])
    iq = np.stack((real, imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.maximum(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1, keepdims=True), 1e-12))
    return iq / rms[:, :, None]


def _score(iq: np.ndarray, reference: torch.Tensor, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate: list[np.ndarray] = []
    control: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            coherent, invariant = crosspulse_mft_scores(torch.from_numpy(iq[start : start + batch_size]).to(device), reference)
            candidate.append(coherent.cpu().numpy())
            control.append(invariant.cpu().numpy())
    return np.concatenate(candidate).astype(np.float64), np.concatenate(control).astype(np.float64)


def _bootstrap(y: np.ndarray, candidate: np.ndarray, control: np.ndarray, *, draws: int = 5000, seed: int = 20_260_804) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    # Re-sample pair indices, then carry both counterfactual arms together.
    n = len(y) // 2
    auroc = np.empty(draws, dtype=np.float64)
    aupr = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pair = rng.integers(0, n, size=n)
        index = np.concatenate((pair, pair + n))
        auroc[draw] = roc_auc_score(y[index], candidate[index]) - roc_auc_score(y[index], control[index])
        aupr[draw] = average_precision_score(y[index], candidate[index]) - average_precision_score(y[index], control[index])
    return {
        "method": "paired_counterfactual_bootstrap_5000",
        "auroc_delta_ci95": [float(np.quantile(auroc, 0.025)), float(np.quantile(auroc, 0.975))],
        "aupr_delta_ci95": [float(np.quantile(aupr, 0.025)), float(np.quantile(aupr, 0.975))],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not audit.get("identity_audit_passed") or not all(audit["contracts"].values()):
        raise ValueError("H0o provenance audit has not passed")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    absent = _load(args.absent_path)
    present = _load(args.present_path)
    if absent.shape != present.shape or absent.shape[0] != 640:
        raise ValueError("unexpected H0o source shape")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = lfm_pulse(device=device)
    candidate_absent, control_absent = _score(absent, reference, device, args.batch_size)
    candidate_present, control_present = _score(present, reference, device, args.batch_size)
    y = np.concatenate((np.zeros(len(absent), dtype=np.uint8), np.ones(len(present), dtype=np.uint8)))
    candidate = np.concatenate((candidate_absent, candidate_present))
    control = np.concatenate((control_absent, control_present))
    candidate_metrics = {"auroc": float(roc_auc_score(y, candidate)), "aupr": float(average_precision_score(y, candidate))}
    control_metrics = {"auroc": float(roc_auc_score(y, control)), "aupr": float(average_precision_score(y, control))}
    difference = {name: candidate_metrics[name] - control_metrics[name] for name in candidate_metrics}
    bootstrap = _bootstrap(y, candidate, control)
    paired_rate = float(np.mean(candidate_present > candidate_absent))
    gates = {
        "candidate_auroc_ge_0p60": candidate_metrics["auroc"] >= 0.60,
        "candidate_minus_control_auroc_ge_0p05": difference["auroc"] >= 0.05,
        "candidate_minus_control_auroc_ci_lower_gt_zero": bootstrap["auroc_delta_ci95"][0] > 0.0,
        "candidate_minus_control_aupr_ge_0p03": difference["aupr"] >= 0.03,
        "candidate_minus_control_aupr_ci_lower_gt_zero": bootstrap["aupr_delta_ci95"][0] > 0.0,
        "paired_present_gt_absent_rate_ge_0p60": paired_rate >= 0.60,
        "zero_fit_no_threshold_no_calibration_no_fusion": True,
        "source_identity_contract": True,
        "test_v3_sealed": True,
    }
    result = {
        "status": "completed",
        "experiment": "RouteA_H0o_MFTCrossPulseCompositeAttribution",
        "evidence_role": "fresh_official_generator_paired_composite_physical_identifiability",
        "statistic": {"candidate": "correct_LFM_top8_energy_weighted_complex_cross_PRI_coherence", "phase_invariant_control": "same_top8_noncoherent_magnitude_peakiness", "delay_samples": [400, 999], "top_k": 8, "trained_parameters": 0},
        "candidate": candidate_metrics,
        "phase_invariant_control": control_metrics,
        "candidate_minus_control": difference,
        "paired_bootstrap": bootstrap,
        "paired_present_greater_than_absent_rate": paired_rate,
        "score_summary": {"candidate_absent_mean": float(candidate_absent.mean()), "candidate_present_mean": float(candidate_present.mean()), "control_absent_mean": float(control_absent.mean()), "control_present_mean": float(control_present.mean())},
        "gates": gates,
        "passed": bool(all(gates.values())),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--absent-path", type=Path, default=SOURCE / "mft_absent.mat")
    parser.add_argument("--present-path", type=Path, default=SOURCE / "mft_present_weak_m12db.mat")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse_generation/paired_identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse/route_a_h0o_mft_crosspulse_summary.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
