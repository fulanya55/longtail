"""H4 P0: zero-fit delay-concentration comparison of true/permuted beamspace maps."""
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
import torch.nn.functional as F

from experiments.run_route_a_h3_vft_three_strata_observable import _binary_metrics, _bootstrap, _triplets
from cjlt.models.vft_staggered_pri import staggered_vft_beamspace_maps


def _concentration(value: torch.Tensor) -> torch.Tensor:
    """Fixed five-delay local mass ratio; input is [B,3,F,D] and no fit occurs."""
    delay_mass = value[:, 2].mean(dim=1)
    local = F.avg_pool1d(delay_mass.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1) * 5.0
    return local.amax(dim=1) / delay_mass.sum(dim=1).clamp_min(1e-12)


def score(path: Path, device: torch.device, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true, perm, pairs, arms = [], [], [], []
    with h5py.File(path, "r") as h5, torch.inference_mode():
        for start in range(0, h5["labels"].shape[0], batch):
            stop = min(start + batch, h5["labels"].shape[0])
            iq = np.stack((h5["iq_real"][start:stop], h5["iq_imag"][start:stop]), axis=1).astype(np.float32, copy=False)
            real, control = staggered_vft_beamspace_maps(torch.from_numpy(iq).to(device))
            true.append(_concentration(real).cpu().numpy()); perm.append(_concentration(control).cpu().numpy())
            pairs.append(np.asarray(h5["pair_seed"][start:stop])); arms.append(np.asarray(h5["vft_visibility_arm"][start:stop]))
    return tuple(map(np.concatenate, (true, perm, pairs, arms)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--development-path", type=Path, required=True); parser.add_argument("--source-audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--bootstrap-draws", type=int, default=5000); parser.add_argument("--bootstrap-seed", type=int, default=20260804); parser.add_argument("--device", default=None)
    args = parser.parse_args(); audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or not all(audit["contracts"].values()) or any(audit.get(key) for key in ("external_used","locked_test_used","test_v2_used","test_v3_used")): raise ValueError("invalid or non-development H4 P0 source audit")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); true, perm, pair, arm = score(args.development_path, device, args.batch_size)
    values = _triplets(true, perm, pair, arm); weak = _binary_metrics(values["absent_true"], values["weak_true"], values["absent_permuted"], values["weak_permuted"]); strong = _binary_metrics(values["absent_true"], values["strong_true"], values["absent_permuted"], values["strong_permuted"]); boot = _bootstrap(values, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    checks = {"weak_true_auroc_ge_0p60": weak["true_auroc"] >= .60, "weak_true_minus_permuted_auroc_ge_0p060": weak["true_minus_permuted_auroc"] >= .060, "weak_true_minus_permuted_ci_lower_gt_zero": boot["weak_true_minus_permuted_auroc_95_ci"][0] > 0, "trained_parameters_zero": True, "test_v3_still_unused": True}
    result = {"status":"completed_no_fit_p0","experiment":"RouteA_H4_VFTBeamspaceMapP0","source_sha256":audit["source_sha256"],"fixed_statistic":{"map":"normalized_complex_delay_frequency_beamspace","channels":["real","imag","magnitude"],"delay_concentration":"frequency_mean_then_fixed_5_delay_mass_ratio","trained_parameters":0},"identity_triplet_count":int(values["absent_true"].size),"weak_vs_absent":weak,"strong_vs_absent":strong,"bootstrap":{"draws":args.bootstrap_draws,"pair_resampling":True,"seed":args.bootstrap_seed,**boot},"gate":{"passed":bool(all(checks.values())),"checks":checks},"external_used":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",encoding="utf-8"); print(json.dumps({"status":result["status"],"gate":result["gate"],"weak":weak},ensure_ascii=False))
