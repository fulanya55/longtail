"""Generate a new identity-isolated three-arm H4 P0 source (no test role)."""
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

from experiments.generate_route_a_h2_multipulse_sixlabel import DATA_ROOT, LENGTH, receiver_seed
from experiments.generate_route_a_h3_vft_three_strata_source import _sample_triplet, _sha256


ROLES = {"smoke": (30, 3_500_000_000), "p0-development-v1": (1_920, 3_600_000_000)}


def generate(role: str, count: int, base_seed: int, output: Path, receipt: Path) -> dict[str, object]:
    if count % 3:
        raise ValueError("H4 count must contain complete absent/weak/strong triplets")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    groups = count // 3
    meta: dict[str, object] = {
        "status": "completed", "experiment": "RouteA_H4_VFTBeamspaceMapP0", "role": role,
        "evidence_role": "smoke_only" if role == "smoke" else "fresh_identity_isolated_p0_development_only",
        "source_scope": "project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716",
        "sample_count": count, "triplet_count": groups, "base_pair_seed": base_seed,
        "sample_seed_range": [base_seed, base_seed + groups - 1], "record_samples": LENGTH,
        "vft_arms": ["absent", "weak", "strong"], "vft_jnr_db": [None, 8.0, 20.0],
        "test_split_generated": False, "test_files_opened": False, "external_used": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    with h5py.File(output, "w") as h5:
        real = h5.create_dataset("iq_real", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        imag = h5.create_dataset("iq_imag", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        labels = h5.create_dataset("labels", (count, 6), dtype="u1"); power = h5.create_dataset("component_jnr_db", (count, 6), dtype="f4")
        pairs = h5.create_dataset("pair_seed", (count,), dtype="u8"); receiver = h5.create_dataset("receiver_noise_seed", (count,), dtype="u8")
        arm = h5.create_dataset("vft_visibility_arm", (count,), dtype="u1")
        for group in range(groups):
            seed = base_seed + group; signals, pair_labels, pair_power = _sample_triplet(seed)
            for local, signal in enumerate(signals):
                row = 3 * group + local
                real[row], imag[row] = signal.real.astype(np.float32), signal.imag.astype(np.float32)
                labels[row], power[row], pairs[row], receiver[row], arm[row] = pair_labels[local], pair_power[local], seed, receiver_seed(seed), local
    meta["source_path"] = str(output); meta["sha256"] = _sha256(output)
    receipt.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--role", choices=ROLES, default="smoke"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--tag", default="r1")
    args = parser.parse_args(); count, base = ROLES[args.role]; root = ROOT / "outputs" / "tapc" / "route_a_h4_vft_beamspace_p0_generation"
    receipt = root / f"{args.role.replace('-', '_')}_{args.tag}_generation_receipt.json"
    if not args.execute:
        root.mkdir(parents=True, exist_ok=True); receipt.write_text(json.dumps({"status":"planned","experiment":"RouteA_H4_VFTBeamspaceMapP0","role":args.role,"sample_count":count,"test_v3_used":False}, indent=2)+"\n", encoding="utf-8"); print(receipt)
    else:
        name = f"h4_vft_beamspace_{args.role}_20260803_{args.tag}"
        value = generate(args.role, count, base, DATA_ROOT / name / "closedset_6class.mat", receipt); print(json.dumps({"status":value["status"],"sha256":value["sha256"]}, ensure_ascii=False))
