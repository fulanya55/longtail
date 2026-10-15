"""No-fit real-versus-permuted timestamp observable test on H3 development."""
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


def _score(path: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true, permuted, pair, arm = [], [], [], []
    with h5py.File(path, "r") as h5, torch.inference_mode():
        count = int(h5["labels"].shape[0])
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            iq = np.stack((h5["iq_real"][start:stop], h5["iq_imag"][start:stop]), axis=1).astype(np.float32, copy=False)
            real_score, permuted_score = staggered_vft_evidence(torch.from_numpy(iq).to(device))
            true.append(real_score.cpu().numpy())
            permuted.append(permuted_score.cpu().numpy())
            pair.append(np.asarray(h5["pair_seed"][start:stop], dtype=np.uint64))
            arm.append(np.asarray(h5["vft_visibility_arm"][start:stop], dtype=np.uint8))
    return tuple(map(np.concatenate, (true, permuted, pair, arm)))


def _triplets(true: np.ndarray, permuted: np.ndarray, pair: np.ndarray, arm: np.ndarray) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {name: [] for name in ("absent_true", "weak_true", "strong_true", "absent_permuted", "weak_permuted", "strong_permuted")}
    for identity in np.unique(pair):
        index = np.flatnonzero(pair == identity)
        ordered = index[np.argsort(arm[index])]
        if ordered.size != 3 or not np.array_equal(arm[ordered], [0, 1, 2]):
            raise ValueError(f"invalid H3 triplet for pair_seed={identity}")
        for name, score, local in (("absent_true", true, 0), ("weak_true", true, 1), ("strong_true", true, 2), ("absent_permuted", permuted, 0), ("weak_permuted", permuted, 1), ("strong_permuted", permuted, 2)):
            values[name].append(float(score[ordered[local]]))
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def _binary_metrics(absent: np.ndarray, present: np.ndarray, absent_control: np.ndarray, present_control: np.ndarray) -> dict[str, float]:
    y = np.r_[np.zeros(absent.size, dtype=np.uint8), np.ones(present.size, dtype=np.uint8)]
    score = np.r_[absent, present]
    control = np.r_[absent_control, present_control]
    return {
        "true_auroc": float(roc_auc_score(y, score)),
        "permuted_auroc": float(roc_auc_score(y, control)),
        "true_aupr": float(average_precision_score(y, score)),
        "permuted_aupr": float(average_precision_score(y, control)),
        "true_minus_permuted_auroc": float(roc_auc_score(y, score) - roc_auc_score(y, control)),
        "true_minus_permuted_aupr": float(average_precision_score(y, score) - average_precision_score(y, control)),
        "paired_present_gt_absent_fraction": float(np.mean(present > absent)),
    }


def _bootstrap(values: dict[str, np.ndarray], *, draws: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = values["absent_true"].size
    weak_delta, strong_delta, weak_auc, strong_auc = [], [], [], []
    for _ in range(draws):
        index = rng.integers(0, n, n)
        weak = _binary_metrics(values["absent_true"][index], values["weak_true"][index], values["absent_permuted"][index], values["weak_permuted"][index])
        strong = _binary_metrics(values["absent_true"][index], values["strong_true"][index], values["absent_permuted"][index], values["strong_permuted"][index])
        weak_delta.append(weak["true_minus_permuted_auroc"])
        strong_delta.append(strong["true_minus_permuted_auroc"])
        weak_auc.append(weak["true_auroc"])
        strong_auc.append(strong["true_auroc"])
    def ci(values_: list[float]) -> list[float]:
        return [float(np.quantile(values_, 0.025)), float(np.quantile(values_, 0.975))]
    return {"weak_true_minus_permuted_auroc_95_ci": ci(weak_delta), "strong_true_minus_permuted_auroc_95_ci": ci(strong_delta), "weak_true_auroc_95_ci": ci(weak_auc), "strong_true_auroc_95_ci": ci(strong_auc)}


def run(args: argparse.Namespace) -> dict[str, object]:
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if source_audit.get("status") != "completed" or not all(source_audit["contracts"].values()):
        raise ValueError("H3 source audit is not valid")
    if any(source_audit.get(key) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("H3 source audit violates test boundary")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    true, permuted, pair, arm = _score(args.development_path, device, args.batch_size)
    values = _triplets(true, permuted, pair, arm)
    weak = _binary_metrics(values["absent_true"], values["weak_true"], values["absent_permuted"], values["weak_permuted"])
    strong = _binary_metrics(values["absent_true"], values["strong_true"], values["absent_permuted"], values["strong_permuted"])
    bootstrap = _bootstrap(values, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    checks = {
        "weak_true_auroc_ge_0p58": weak["true_auroc"] >= 0.58,
        "weak_true_minus_permuted_auroc_ge_0p040": weak["true_minus_permuted_auroc"] >= 0.040,
        "weak_true_minus_permuted_auroc_ci_lower_gt_zero": bootstrap["weak_true_minus_permuted_auroc_95_ci"][0] > 0.0,
        "strong_true_auroc_not_below_weak_by_more_than_0p010": strong["true_auroc"] >= weak["true_auroc"] - 0.010,
        "weak_paired_present_gt_absent_fraction_ge_0p56": weak["paired_present_gt_absent_fraction"] >= 0.56,
        "trained_parameters_zero": True,
        "test_v3_still_unused": True,
    }
    result = {
        "status": "completed_no_fit_observable",
        "experiment": "RouteA_H3_VFTThreeStrataIdentifiability",
        "evidence_role": "fresh_role_isolated_project_local_physical_observable_development_only",
        "development_path": str(args.development_path),
        "source_audit": str(args.source_audit),
        "source_sha256": source_audit["source_sha256"],
        "fixed_statistic": {"pulse_count": 8, "record_samples": record_samples(), "delay_samples": [100, 500], "frequency_hz": [1000, 7000, 250], "timestamp_permutation": [0, 5, 2, 7, 4, 1, 6, 3], "score": "full_grid_maximum_normalized_coherence"},
        "identity_triplet_count": int(values["absent_true"].size),
        "weak_vs_absent": weak,
        "strong_vs_absent": strong,
        "bootstrap": {"draws": args.bootstrap_draws, "pair_resampling": True, "seed": args.bootstrap_seed, **bootstrap},
        "gate": {"passed": bool(all(checks.values())), "checks": checks},
        "trained_parameters": 0,
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-path", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tapc" / "route_a_h3_vft_three_strata_observable" / "summary.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--device", default=None)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
