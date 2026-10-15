"""Run the preregistered no-fit Route-A H0n cross-PRI coherence screen."""
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
from cjlt.models.mft_crosspulse_coherence import crosspulse_mft_scores, lfm_pulse


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0n_mft_crosspulse_pairs_20260803_r2"


def _load(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        iq = np.stack((np.asarray(handle["iq_real"]), np.asarray(handle["iq_imag"])), axis=1).astype(np.float32)
        metadata = {name: np.asarray(handle[name]) for name in ("pair_seed", "receiver_noise_seed")}
    power = np.sqrt(np.maximum(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1, keepdims=True), 1e-12))
    return iq / power[:, :, None], metadata


def _score(iq: np.ndarray, *, reference: torch.Tensor, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    coherent: list[np.ndarray] = []
    invariant: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            result = crosspulse_mft_scores(torch.from_numpy(iq[start : start + batch_size]).to(device), reference)
            coherent.append(result[0].cpu().numpy())
            invariant.append(result[1].cpu().numpy())
    return np.concatenate(coherent).astype(np.float64), np.concatenate(invariant).astype(np.float64)


def _paired_bootstrap(candidate_coherent: np.ndarray, candidate_scrambled: np.ndarray, control_coherent: np.ndarray, control_scrambled: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20_260_803)
    count = len(candidate_coherent)
    labels = np.concatenate((np.ones(count, dtype=int), np.zeros(count, dtype=int)))
    difference = np.empty(5000, dtype=np.float64)
    for iteration in range(len(difference)):
        indices = rng.integers(0, count, size=count)
        candidate = roc_auc_score(labels, np.concatenate((candidate_coherent[indices], candidate_scrambled[indices])))
        control = roc_auc_score(labels, np.concatenate((control_coherent[indices], control_scrambled[indices])))
        difference[iteration] = candidate - control
    return [float(np.quantile(difference, 0.025)), float(np.quantile(difference, 0.975))]


def run(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if not receipt.get("executed") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("generation receipt invalid or sealed test boundary violated")
    coherent_iq, coherent_meta = _load(args.coherent_path)
    scrambled_iq, scrambled_meta = _load(args.scrambled_path)
    if not np.array_equal(coherent_meta["pair_seed"], scrambled_meta["pair_seed"]):
        raise ValueError("paired identities differ")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = lfm_pulse(device=device)
    candidate_coherent, control_coherent = _score(coherent_iq, reference=reference, device=device, batch_size=args.batch_size)
    candidate_scrambled, control_scrambled = _score(scrambled_iq, reference=reference, device=device, batch_size=args.batch_size)
    labels = np.concatenate((np.ones(len(candidate_coherent), dtype=int), np.zeros(len(candidate_scrambled), dtype=int)))
    candidate_auroc = float(roc_auc_score(labels, np.concatenate((candidate_coherent, candidate_scrambled))))
    control_auroc = float(roc_auc_score(labels, np.concatenate((control_coherent, control_scrambled))))
    bootstrap_ci = _paired_bootstrap(candidate_coherent, candidate_scrambled, control_coherent, control_scrambled)
    paired_rate = float(np.mean(candidate_coherent > candidate_scrambled))
    gates = {
        "candidate_auroc_ge_0p80": candidate_auroc >= 0.80,
        "candidate_minus_phase_invariant_control_ge_0p20": candidate_auroc - control_auroc >= 0.20,
        "paired_bootstrap_lower_gt_zero": bootstrap_ci[0] > 0.0,
        "paired_coherent_gt_scrambled_rate_ge_0p75": paired_rate >= 0.75,
        "no_fit_no_train_no_calibration_no_threshold": True,
        "test_v3_sealed": True,
    }
    result = {
        "status": "completed",
        "experiment": "RouteA_H0n_MFTCrossPulseCoherence",
        "evidence_role": "new_observation_physical_identifiability_only",
        "observation_protocol": {"num_pulses": 4, "samples_per_pri": 3000, "input_samples": 12000},
        "intervention": "whole_PRI_unit_phase_on_MFT_component_only",
        "statistic": {
            "candidate": "correct_LFM_top8_energy_weighted_complex_cross_PRI_coherence",
            "phase_invariant_control": "same_top8_noncoherent_magnitude_peakiness",
            "delay_samples": [400, 999],
            "top_k": 8,
            "trained_parameters": 0,
        },
        "candidate_auroc": candidate_auroc,
        "phase_invariant_control_auroc": control_auroc,
        "candidate_minus_control_auroc": candidate_auroc - control_auroc,
        "candidate_minus_control_paired_bootstrap_95_ci": bootstrap_ci,
        "paired_coherent_greater_than_scrambled_rate": paired_rate,
        "score_summary": {
            "candidate_coherent_mean": float(candidate_coherent.mean()),
            "candidate_scrambled_mean": float(candidate_scrambled.mean()),
            "control_coherent_mean": float(control_coherent.mean()),
            "control_scrambled_mean": float(control_scrambled.mean()),
        },
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
    parser.add_argument("--coherent-path", type=Path, default=SOURCE / "mft_coherent.mat")
    parser.add_argument("--scrambled-path", type=Path, default=SOURCE / "mft_scrambled.mat")
    parser.add_argument("--receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h0n_mft_crosspulse_generation/formal_generation_receipt.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0n_mft_crosspulse/route_a_h0n_mft_crosspulse_summary.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
