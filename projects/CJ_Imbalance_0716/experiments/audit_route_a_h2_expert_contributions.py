"""Audit H2 global and target-expert logits from frozen development checkpoints.

The report is a predeclared safety/attribution diagnostic.  It only reads
development-mp predictions reconstructed from completed checkpoints; it never
changes a threshold, model, data role, or promotion gate.
"""
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

from cjlt.models.multipulse_vft_expert import MODES, MultiPulseVFTExpertClassifier


def _statistic(value: np.ndarray) -> dict[str, float | int]:
    return {"count": int(value.size), "mean": float(value.mean()), "std": float(value.std(ddof=0)), "mean_abs": float(np.abs(value).mean()), "positive_fraction": float((value > 0).mean())}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _run(mode: str, seed: int, root: Path, development_path: Path, device: torch.device, batch_size: int) -> dict:
    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    if receipt["modes"] != [mode] or receipt["seeds"] != [seed] or receipt["smoke"]:
        raise ValueError(f"unexpected receipt identity: {root}")
    if any(receipt[key] for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError(f"test boundary violation: {root}")
    config = receipt["config"]
    artifact_root = root / f"{mode}_seed{seed}"
    checkpoint_path = artifact_root / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiPulseVFTExpertClassifier(mode, width=int(config["width"]), expert_dim=int(config["expert_dim"])).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    globals_, experts_, finals_, labels_, powers_ = [], [], [], [], []
    with h5py.File(development_path, "r") as h5, torch.inference_mode():
        length = int(h5["labels"].shape[0])
        for start in range(0, length, batch_size):
            stop = min(start + batch_size, length)
            iq = np.stack((h5["iq_real"][start:stop], h5["iq_imag"][start:stop]), axis=1).astype(np.float32, copy=False)
            component = model(torch.from_numpy(iq).to(device), return_components=True)
            globals_.append(component["global_logits"][:, 4].cpu().numpy())
            experts_.append(component["vft_expert_logit"].cpu().numpy())
            finals_.append(component["logits"][:, 4].cpu().numpy())
            labels_.append(np.asarray(h5["labels"][start:stop, 4], dtype=bool))
            powers_.append(np.asarray(h5["component_jnr_db"][start:stop, 4], dtype=np.float64))
    global_logit, expert_logit, final_logit = map(np.concatenate, (globals_, experts_, finals_))
    labels, power = np.concatenate(labels_), np.concatenate(powers_)
    residual_error = float(np.max(np.abs(final_logit - global_logit - expert_logit)))
    if residual_error > 1e-6:
        raise ValueError(f"VFT additive decomposition failed for {mode}/seed{seed}: {residual_error}")
    strata = {
        "weak_positive": labels & np.isfinite(power) & (power <= 8.001),
        "strong_positive": labels & np.isfinite(power) & (power > 8.001),
        "negative": ~labels,
    }
    # H2 is a weak-VFT mechanism protocol.  Some generated development roles
    # have no strong-VFT examples, so strong-positive attribution is optional
    # and must be recorded as absent rather than manufactured or treated as a
    # protocol failure.  Weak positives and VFT negatives remain required.
    required = ("weak_positive", "negative")
    if not all(strata[name].any() for name in required):
        raise ValueError(f"missing required weak-VFT/negative attribution stratum for {mode}/seed{seed}")
    observed = {name: mask for name, mask in strata.items() if mask.any()}
    return {
        "mode": mode,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "vft_additive_residual_max_abs": residual_error,
        "global_expert_correlation": _correlation(global_logit, expert_logit),
        "missing_optional_strata": [name for name, mask in strata.items() if not mask.any()],
        "strata": {name: {"global_vft_logit": _statistic(global_logit[mask]), "vft_expert_logit": _statistic(expert_logit[mask]), "final_vft_logit": _statistic(final_logit[mask])} for name, mask in observed.items()},
    }


def audit(root_template: str, development_path: Path, output: Path, device: str, batch_size: int) -> dict:
    torch_device = torch.device(device)
    records = []
    for mode in ("C2-mp", "C3-mp", "C4-mp"):
        for seed in (20260803, 20260811, 20260819, 20260827, 20260904):
            root = Path(root_template.format(mode=mode.lower().replace("-mp", ""), seed=seed))
            records.append(_run(mode, seed, root, development_path, torch_device, batch_size))
    value = {"status": "completed_development_expert_contribution_audit", "experiment": "RouteA_H2_MultiPulseSixLabel", "development_path": str(development_path), "records": records, "evidence_boundary": "Development-only diagnostic; no threshold, calibration, model selection, external source, or test-v3 access.", "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-path", type=Path, required=True)
    parser.add_argument("--root-template", default="outputs/tapc/route_a_h2_multipulse_{mode}_seed{seed}")
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h2_multipulse_expert_contributions.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = audit(args.root_template, args.development_path, args.output, args.device, args.batch_size)
    print(json.dumps({"status": result["status"], "records": len(result["records"])}, ensure_ascii=False))
