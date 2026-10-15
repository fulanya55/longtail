"""Fixed no-fit H0p comparison of three versus eight MFT copies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.mft_delay_autocorrelation import delay_autocorrelation_effective_support, permuted_delay_autocorrelation_effective_support


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0p_mft_delay_autocorrelation_pairs_20260803"


def _load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        real = np.asarray(handle["iq_real"])
        imag = np.asarray(handle["iq_imag"])
    iq = np.stack((real, imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.maximum(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1, keepdims=True), 1e-12))
    return iq / rms[:, :, None]


def _score(iq: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate: list[np.ndarray] = []
    control: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            batch = torch.from_numpy(iq[start : start + batch_size]).to(device)
            candidate.append(delay_autocorrelation_effective_support(batch).cpu().numpy())
            control.append(permuted_delay_autocorrelation_effective_support(batch).cpu().numpy())
    return np.concatenate(candidate).astype(np.float64), np.concatenate(control).astype(np.float64)


def _bootstrap(low: np.ndarray, high: np.ndarray, low_control: np.ndarray, high_control: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20_260_805)
    n = len(low)
    y = np.concatenate((np.zeros(n, dtype=np.uint8), np.ones(n, dtype=np.uint8)))
    difference = np.empty(5000, dtype=np.float64)
    for draw in range(len(difference)):
        index = rng.integers(0, n, size=n)
        candidate = roc_auc_score(y, np.concatenate((low[index], high[index])))
        control = roc_auc_score(y, np.concatenate((low_control[index], high_control[index])))
        difference[draw] = candidate - control
    return [float(np.quantile(difference, 0.025)), float(np.quantile(difference, 0.975))]


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not audit.get("identity_audit_passed") or not all(audit["contracts"].values()):
        raise ValueError("H0p provenance audit has not passed")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    low = _load(args.low_path)
    high = _load(args.high_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    candidate_low, control_low = _score(low, device=device, batch_size=args.batch_size)
    candidate_high, control_high = _score(high, device=device, batch_size=args.batch_size)
    y = np.concatenate((np.zeros(len(low), dtype=np.uint8), np.ones(len(high), dtype=np.uint8)))
    candidate_auroc = float(roc_auc_score(y, np.concatenate((candidate_low, candidate_high))))
    control_auroc = float(roc_auc_score(y, np.concatenate((control_low, control_high))))
    ci = _bootstrap(candidate_low, candidate_high, control_low, control_high)
    paired_rate = float(np.mean(candidate_high > candidate_low))
    gates = {
        "candidate_auroc_ge_0p65": candidate_auroc >= 0.65,
        "candidate_minus_control_auroc_ge_0p10": candidate_auroc - control_auroc >= 0.10,
        "paired_bootstrap_lower_gt_zero": ci[0] > 0.0,
        "paired_8_gt_3_rate_ge_0p65": paired_rate >= 0.65,
        "zero_fit_no_threshold_no_calibration_no_fusion": True,
        "test_v3_sealed": True,
    }
    result = {
        "status": "completed",
        "experiment": "RouteA_H0p_MFTDelayAutocorrelationIdentifiability",
        "evidence_role": "fresh_single_PRI_no_reference_paired_physical_identifiability",
        "statistic": {"candidate": "nonzero_lag_1_to_250_complex_autocorrelation_Renyi2_effective_support", "control": "fixed_sample_phase_index_permutation_same_autocorrelation_support", "trained_parameters": 0},
        "candidate_auroc": candidate_auroc,
        "control_auroc": control_auroc,
        "candidate_minus_control_auroc": candidate_auroc - control_auroc,
        "candidate_minus_control_paired_bootstrap_95_ci": ci,
        "paired_8_greater_than_3_rate": paired_rate,
        "score_summary": {"candidate_count3_mean": float(candidate_low.mean()), "candidate_count8_mean": float(candidate_high.mean()), "control_count3_mean": float(control_low.mean()), "control_count8_mean": float(control_high.mean())},
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
    parser.add_argument("--low-path", type=Path, default=SOURCE / "mft_count_3.mat")
    parser.add_argument("--high-path", type=Path, default=SOURCE / "mft_count_8.mat")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h0p_mft_delay_autocorrelation_generation/paired_identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0p_mft_delay_autocorrelation/route_a_h0p_mft_delay_autocorrelation_summary.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
