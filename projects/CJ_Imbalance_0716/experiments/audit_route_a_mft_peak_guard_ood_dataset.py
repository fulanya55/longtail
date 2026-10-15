"""Audit paired train-only MFT parameter-OOD views and their receiver identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("ID_control", "low_count", "close_spacing", "ultra_close", "weak_target")
LEVELS = (25, 35, 50)
COUNT = 630
REFERENCE_RANGES = (
    (218938474, 218963673), (221938474, 221963673), (222677743, 222702942),
    (225677743, 225702942), (232677743, 232702942), (235677743, 235702942),
    (242677743, 242702942), (245677743, 245702942), (252677743, 252702942),
    (255677743, 255702942), (262677743, 262682782), (272677743, 272702942),
    (275677743, 275702942), (282677743, 282702942), (285677743, 285702942),
    (292677743, 292702942), (295677743, 295702942), (302677743, 302702942),
    (305677743, 305702942),
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(root, prefix, regime, level):
    pattern = f"{prefix}_{regime}/level_*__jnr_{level}dB/closedset_6class_train.mat"
    hits = list(root.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one OOD file for {regime} {level}, found {len(hits)}")
    return hits[0].resolve()


def run(args):
    entries = {}
    reference_receiver = None
    for regime in REGIMES:
        entries[regime] = {}
        for level in LEVELS:
            path = _find(args.data_root, args.prefix, regime, level)
            with h5py.File(path, "r") as handle:
                receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
                labels = np.asarray(handle["label_selected"])
            if receiver.size != COUNT or COUNT not in labels.shape or np.unique(receiver).size != COUNT:
                raise ValueError(f"identity contract failed: {regime} {level}")
            if reference_receiver is None:
                reference_receiver = receiver.copy()
            if not np.array_equal(receiver, reference_receiver):
                raise ValueError("OOD views are not receiver-identity paired")
            observed = (int(receiver.min()), int(receiver.max()))
            if not all(observed[1] < lo or observed[0] > hi for lo, hi in REFERENCE_RANGES):
                raise ValueError("OOD receiver identities overlap a prior role")
            entries[regime][str(level)] = {
                "path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path),
                "sample_count": COUNT, "receiver_noise_seed_range": list(observed),
                "receiver_noise_seed_unique": COUNT,
            }
    result = {
        "status": "completed", "evidence_role": "paired_parameter_ood_train_only",
        "prefix": args.prefix, "regimes": entries, "paired_receiver_identities_across_all_views": True,
        "receiver_noise_seed_range": entries[REGIMES[0]][str(LEVELS[0])]["receiver_noise_seed_range"],
        "identity_disjoint_from_prior_roles": True, "test_files_generated": False,
        "test_files_opened": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "receiver_range": result["receiver_noise_seed_range"], "output": str(args.output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\LJQ_code\CJ_Inbalance_0724\generated_data"))
    parser.add_argument("--prefix", default="routea_mft_peak_guard_ood_20260801")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_peak_guard_parameter_ood/dataset_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

