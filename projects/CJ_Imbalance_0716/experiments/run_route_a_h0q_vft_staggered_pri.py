"""Run the preregistered no-fit Route-A H0q VFT staggered-PRI physical screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cjlt.models.vft_staggered_pri import record_samples, staggered_vft_contrasts


def _load_iq(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        real, imag = np.asarray(handle["iq_real"], dtype=np.float32), np.asarray(handle["iq_imag"], dtype=np.float32)
    if real.shape[1] != record_samples():
        raise ValueError(f"unexpected H0q payload shape {real.shape}")
    return np.stack((real, imag), axis=1)


def _scores(signals: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate, control = [], []
    with torch.inference_mode():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device)
            first, second = staggered_vft_contrasts(batch)
            candidate.append(first.cpu().numpy()); control.append(second.cpu().numpy())
    return np.concatenate(candidate), np.concatenate(control)


def _bootstrap(low: np.ndarray, high: np.ndarray, low_control: np.ndarray, high_control: np.ndarray, *, draws: int, seed: int) -> dict[str, object]:
    labels = np.concatenate((np.zeros(low.size, dtype=np.int8), np.ones(high.size, dtype=np.int8)))
    candidate = np.concatenate((low, high)); control = np.concatenate((low_control, high_control))
    result = {
        "candidate_auroc": float(roc_auc_score(labels, candidate)),
        "control_auroc": float(roc_auc_score(labels, control)),
        "candidate_aupr": float(average_precision_score(labels, candidate)),
        "control_aupr": float(average_precision_score(labels, control)),
        "paired_high_gt_low_fraction": float(np.mean(high > low)),
    }
    rng = np.random.default_rng(seed); auroc_delta, aupr_delta = [], []
    count = low.size
    for _ in range(draws):
        index = rng.integers(0, count, size=count)
        y = np.concatenate((np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)))
        sample_candidate = np.concatenate((low[index], high[index])); sample_control = np.concatenate((low_control[index], high_control[index]))
        auroc_delta.append(roc_auc_score(y, sample_candidate) - roc_auc_score(y, sample_control))
        aupr_delta.append(average_precision_score(y, sample_candidate) - average_precision_score(y, sample_control))
    result["candidate_minus_control_auroc"] = {"mean": result["candidate_auroc"] - result["control_auroc"], "bootstrap_95_ci": [float(np.quantile(auroc_delta, .025)), float(np.quantile(auroc_delta, .975))]}
    result["candidate_minus_control_aupr"] = {"mean": result["candidate_aupr"] - result["control_aupr"], "bootstrap_95_ci": [float(np.quantile(aupr_delta, .025)), float(np.quantile(aupr_delta, .975))]}
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not audit.get("identity_audit_passed") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("H0q source audit or test-v3 seal invalid")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    low, high = _load_iq(args.low_path), _load_iq(args.high_path)
    low_candidate, low_control = _scores(low, device=device, batch_size=args.batch_size)
    high_candidate, high_control = _scores(high, device=device, batch_size=args.batch_size)
    metrics = _bootstrap(low_candidate, high_candidate, low_control, high_control, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    auc = metrics["candidate_minus_control_auroc"]; ap = metrics["candidate_minus_control_aupr"]
    checks = {
        "candidate_auroc_ge_0p80": metrics["candidate_auroc"] >= .80,
        "candidate_minus_control_auroc_ge_0p20": auc["mean"] >= .20,
        "candidate_minus_control_auroc_ci_lower_gt_zero": auc["bootstrap_95_ci"][0] > 0,
        "candidate_minus_control_aupr_ge_0p15": ap["mean"] >= .15,
        "candidate_minus_control_aupr_ci_lower_gt_zero": ap["bootstrap_95_ci"][0] > 0,
        "paired_high_gt_low_fraction_ge_0p75": metrics["paired_high_gt_low_fraction"] >= .75,
        "no_fit": True, "test_v3_still_sealed": True,
    }
    result = {
        "status": "completed", "experiment": "RouteA_H0q_VFTStaggeredPRIPhaseConsistency", "evidence_role": "fresh_paired_staggered_pri_vft_physical_identifiability",
        "source_audit": str(args.audit), "source_sha256": {"low": audit["low_sha256"], "high": audit["high_sha256"]},
        "fixed_statistic": {"pulse_count": 8, "record_samples": record_samples(), "delay_samples": [100, 500], "frequency_hz": [1000, 7000, 250], "timestamp_permutation": [0, 5, 2, 7, 4, 1, 6, 3]},
        "bootstrap": {"draws": args.bootstrap_draws, "pair_resampling": True, "seed": args.bootstrap_seed}, "device": str(device), "metrics": metrics,
        "gate": {"passed": bool(all(checks.values())), "checks": checks}, "trained_parameters": 0, "external_refit": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    data = Path(r"C:\LJQ_code\干扰数据\h0q_vft_staggered_pri_20260803")
    root = ROOT / "outputs/tapc/route_a_h0q_vft_staggered_pri"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-path", type=Path, default=data / "vft_low_1p0_2p5kHz.mat")
    parser.add_argument("--high-path", type=Path, default=data / "vft_high_5p5_7p0kHz.mat")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h0q_vft_staggered_pri_generation/paired_identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "route_a_h0q_vft_staggered_pri_summary.json")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--bootstrap-draws", type=int, default=5000); parser.add_argument("--bootstrap-seed", type=int, default=20260803); parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    value = run(parse_args())
    print(json.dumps({"status": value["status"], "gate": value["gate"]}, ensure_ascii=False, indent=2))
