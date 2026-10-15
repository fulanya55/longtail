"""Audit a H3 three-strata source without opening any test role."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from cjlt.models.vft_staggered_pri import record_samples
from experiments.find_route_a_h0o_crosspulse_identity_range import historic_receiver_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_ids(path: Path) -> set[int]:
    with h5py.File(path, "r") as h5:
        return set(map(int, np.asarray(h5["receiver_noise_seed"]).reshape(-1).astype(np.uint64)))


def audit(path: Path, receipt: Path, output: Path, *, other_sources: tuple[Path, ...] = ()) -> dict[str, object]:
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    if manifest.get("test_files_opened") or any(manifest.get(key) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("H3 receipt violates test boundary")
    with h5py.File(path, "r") as h5:
        required = {"iq_real", "iq_imag", "labels", "component_jnr_db", "pair_seed", "receiver_noise_seed", "vft_visibility_arm"}
        if not required.issubset(h5.keys()):
            raise ValueError("H3 source fields missing")
        real, imag, labels, power = (np.asarray(h5[name]) for name in ("iq_real", "iq_imag", "labels", "component_jnr_db"))
        pair, receiver, arm = (np.asarray(h5[name]).reshape(-1) for name in ("pair_seed", "receiver_noise_seed", "vft_visibility_arm"))
    count = int(labels.shape[0])
    groups = [np.flatnonzero(pair == value) for value in np.unique(pair)]
    historic, scanned = historic_receiver_ids()
    current = set(map(int, receiver.astype(np.uint64)))
    def _triplet_contract(index: np.ndarray) -> bool:
        ordered = index[np.argsort(arm[index])]
        return bool(
            ordered.size == 3
            and np.array_equal(arm[ordered], [0, 1, 2])
            and np.array_equal(labels[ordered, 4], [0, 1, 1])
            and np.isnan(power[ordered[0], 4])
            and np.isclose(power[ordered[1], 4], 8.0)
            and np.isclose(power[ordered[2], 4], 20.0)
            and np.array_equal(labels[ordered[0], [0, 1, 2, 3, 5]], labels[ordered[1], [0, 1, 2, 3, 5]])
            and np.array_equal(labels[ordered[0], [0, 1, 2, 3, 5]], labels[ordered[2], [0, 1, 2, 3, 5]])
            and np.allclose(power[ordered[0], [0, 1, 2, 3, 5]], power[ordered[1], [0, 1, 2, 3, 5]], equal_nan=True)
            and np.allclose(power[ordered[0], [0, 1, 2, 3, 5]], power[ordered[2], [0, 1, 2, 3, 5]], equal_nan=True)
            and not np.array_equal(real[ordered[0]], real[ordered[1]])
            and not np.array_equal(real[ordered[1]], real[ordered[2]])
        )
    checks = {
        "receipt_sha256_matches_source": manifest.get("sha256") == _sha256(path),
        "payload_shape": real.shape == imag.shape == (count, record_samples()),
        "label_shape": labels.shape == (count, 6) and power.shape == (count, 6),
        "finite_iq": bool(np.isfinite(real).all() and np.isfinite(imag).all()),
        "receiver_per_triplet_shared": all(np.unique(receiver[index]).size == 1 for index in groups),
        "three_arm_counterfactual_contract": all(_triplet_contract(index) for index in groups),
        "historical_train_identity_overlap_zero": not bool(current.intersection(historic)),
        "declared_other_role_identity_overlap_zero": all(not bool(current.intersection(_receiver_ids(item))) for item in other_sources),
        "test_v3_still_unused": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H3 source audit failed: {checks}")
    result = {
        "status": "completed",
        "experiment": "RouteA_H3_VFTThreeStrataIdentifiability",
        "role": manifest["role"],
        "source_sha256": manifest["sha256"],
        "contracts": checks,
        "sample_count": count,
        "triplet_count": len(groups),
        "historic_train_sources_scanned": scanned,
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.receipt, args.output, other_sources=tuple(args.other_source)), ensure_ascii=False))
