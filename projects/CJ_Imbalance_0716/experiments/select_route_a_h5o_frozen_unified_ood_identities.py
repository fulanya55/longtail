"""Pre-generation identity exclusion for H5o train-only OOD confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
BASE_SEED = 850_000_000
COUNT = 5_040
RECEIVER_OFFSET = 198_677_743


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def build_payload(data_root: Path = DATA_ROOT) -> dict[str, object]:
    expected = np.arange(BASE_SEED + RECEIVER_OFFSET, BASE_SEED + RECEIVER_OFFSET + COUNT, dtype=np.int64)
    rows = []
    for path in sorted(data_root.rglob("*.mat")):
        if "test" in path.name.lower() or "train" not in path.name.lower():
            continue
        try:
            receiver = _receiver(path)
        except (OSError, KeyError):
            continue
        overlap = int(np.intersect1d(expected, receiver).size)
        rows.append({"path": str(path), "sha256": _sha256(path), "receiver_overlap_count": overlap})
    conflicts = [row for row in rows if row["receiver_overlap_count"]]
    return {
        "status": "completed_pre_generation_identity_selection",
        "experiment": "RouteA_H5o_FrozenUnifiedCombinedOOD",
        "evidence_role": "pre_generation_identity_exclusion",
        "base_sample_seed": BASE_SEED,
        "sample_count": COUNT,
        "expected_receiver_noise_seed_range": [int(expected.min()), int(expected.max())],
        "readable_historic_train_sources": rows,
        "identity_overlap_count": int(sum(int(row["receiver_overlap_count"]) for row in rows)),
        "identity_conflicts": conflicts,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood_generation/pre_generation_identity_selection.json")
    args = parser.parse_args()
    payload = build_payload()
    if payload["identity_overlap_count"]:
        raise ValueError(f"H5o expected interval is not unused: {payload['identity_conflicts']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
