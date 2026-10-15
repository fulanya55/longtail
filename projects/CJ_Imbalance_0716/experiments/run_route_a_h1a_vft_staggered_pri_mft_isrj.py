"""Run fixed no-fit H1a weak VFT present/absent attribution screen."""
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

from cjlt.models.vft_staggered_pri import record_samples, staggered_vft_evidence


def _load_iq(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        real = np.asarray(handle["iq_real"], dtype=np.float32); imag = np.asarray(handle["iq_imag"], dtype=np.float32)
    if real.shape != (640, record_samples()):
        raise ValueError(f"unexpected H1a payload shape {real.shape}")
    return np.stack((real, imag), axis=1)


def _scores(signals: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate, control = [], []
    with torch.inference_mode():
        for start in range(0, signals.shape[0], batch_size):
            first, second = staggered_vft_evidence(torch.from_numpy(signals[start:start + batch_size]).to(device))
            candidate.append(first.cpu().numpy()); control.append(second.cpu().numpy())
    return np.concatenate(candidate), np.concatenate(control)


def _metrics(absent: np.ndarray, present: np.ndarray, absent_control: np.ndarray, present_control: np.ndarray, *, draws: int, seed: int) -> dict[str, object]:
    count = absent.size; labels = np.concatenate((np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)))
    candidate = np.concatenate((absent, present)); control = np.concatenate((absent_control, present_control))
    result = {
        "candidate_auroc": float(roc_auc_score(labels, candidate)), "control_auroc": float(roc_auc_score(labels, control)),
        "candidate_aupr": float(average_precision_score(labels, candidate)), "control_aupr": float(average_precision_score(labels, control)),
        "paired_present_gt_absent_fraction": float(np.mean(present > absent)),
    }
    rng = np.random.default_rng(seed); auc_delta, ap_delta = [], []
    for _ in range(draws):
        index = rng.integers(0, count, size=count); y = np.concatenate((np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)))
        c = np.concatenate((absent[index], present[index])); k = np.concatenate((absent_control[index], present_control[index]))
        auc_delta.append(roc_auc_score(y, c) - roc_auc_score(y, k)); ap_delta.append(average_precision_score(y, c) - average_precision_score(y, k))
    result["candidate_minus_control_auroc"] = {"mean": result["candidate_auroc"] - result["control_auroc"], "bootstrap_95_ci": [float(np.quantile(auc_delta, .025)), float(np.quantile(auc_delta, .975))]}
    result["candidate_minus_control_aupr"] = {"mean": result["candidate_aupr"] - result["control_aupr"], "bootstrap_95_ci": [float(np.quantile(ap_delta, .025)), float(np.quantile(ap_delta, .975))]}
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text(encoding="utf-8")); seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not audit.get("identity_audit_passed") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("H1a source audit or test-v3 seal invalid")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    absent, present = _load_iq(args.absent_path), _load_iq(args.present_path)
    candidate_absent, control_absent = _scores(absent, device=device, batch_size=args.batch_size)
    candidate_present, control_present = _scores(present, device=device, batch_size=args.batch_size)
    metrics = _metrics(candidate_absent, candidate_present, control_absent, control_present, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    auc, ap = metrics["candidate_minus_control_auroc"], metrics["candidate_minus_control_aupr"]
    checks = {
        "candidate_auroc_ge_0p62": metrics["candidate_auroc"] >= .62,
        "candidate_minus_control_auroc_ge_0p05": auc["mean"] >= .05,
        "candidate_minus_control_auroc_ci_lower_gt_zero": auc["bootstrap_95_ci"][0] > 0,
        "candidate_minus_control_aupr_ge_0p03": ap["mean"] >= .03,
        "candidate_minus_control_aupr_ci_lower_gt_zero": ap["bootstrap_95_ci"][0] > 0,
        "paired_present_gt_absent_fraction_ge_0p60": metrics["paired_present_gt_absent_fraction"] >= .60,
        "no_fit": True, "test_v3_still_sealed": True,
    }
    result = {
        "status": "completed", "experiment": "RouteA_H1a_VFTStaggeredPRI_MFTISRJCompositeAttribution",
        "evidence_role": "fresh_paired_project_local_staggered_pri_composite_physical_identifiability",
        "source_scope": audit["source_scope"], "source_audit": str(args.audit),
        "source_sha256": {"absent": audit["absent_sha256"], "present": audit["present_sha256"]},
        "fixed_statistic": {"pulse_count": 8, "record_samples": record_samples(), "delay_samples": [100,500], "frequency_hz": [1000,7000,250], "timestamp_permutation": [0,5,2,7,4,1,6,3], "score": "full_grid_maximum_normalized_coherence"},
        "bootstrap": {"draws": args.bootstrap_draws, "pair_resampling": True, "seed": args.bootstrap_seed}, "device": str(device), "metrics": metrics,
        "gate": {"passed": bool(all(checks.values())), "checks": checks}, "trained_parameters": 0, "external_refit": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    data = Path(r"C:\LJQ_code\干扰数据\h1a_vft_staggered_pri_mft_isrj_20260803"); root = ROOT / "outputs/tapc/route_a_h1a_vft_staggered_pri"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absent-path", type=Path, default=data / "vft_absent_mft_isrj.mat"); parser.add_argument("--present-path", type=Path, default=data / "vft_present_weak_8db_mft_isrj.mat")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h1a_vft_staggered_pri_generation/paired_identity_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "route_a_h1a_vft_staggered_pri_summary.json")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--bootstrap-draws", type=int, default=5000); parser.add_argument("--bootstrap-seed", type=int, default=20260803); parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    value = run(parse_args()); print(json.dumps({"status": value["status"], "gate": value["gate"]}, ensure_ascii=False, indent=2))
