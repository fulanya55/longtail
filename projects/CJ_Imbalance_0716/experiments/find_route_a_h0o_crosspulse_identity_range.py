"""Preselect a fresh official-generator sample-seed block for Route-A H0o.

This script reads only existing *train*.mat receiver metadata.  It never opens
test files and never creates a data source.  The receiver-noise substream
mapping is copied verbatim from compose_jamming_sample.m.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
MODULUS = 4_294_967_296
NOISE_SALT = 2 * 2_246_822_519


def receiver_seed(sample_seed: np.ndarray) -> np.ndarray:
    return ((sample_seed.astype(np.uint64) + NOISE_SALT) % MODULUS).astype(np.uint64)


def historic_receiver_ids() -> tuple[set[int], list[dict[str, object]]]:
    ids: set[int] = set()
    sources: list[dict[str, object]] = []
    for path in DATA_ROOT.rglob("*train*.mat"):
        try:
            with h5py.File(path, "r") as handle:
                if "receiver_noise_seed" not in handle:
                    continue
                values = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.uint64)
        except (OSError, KeyError):
            continue
        ids.update(map(int, values))
        sources.append({"path": str(path), "sample_count": int(values.size), "unique_receiver_count": int(np.unique(values).size)})
    return ids, sources


def select_block(*, count: int, first_candidate: int, step: int, forbidden: set[int]) -> tuple[int, np.ndarray, int]:
    candidate = first_candidate
    examined = 0
    while candidate + count - 1 < MODULUS:
        samples = np.arange(candidate, candidate + count, dtype=np.uint64)
        receivers = receiver_seed(samples)
        examined += 1
        if len(np.unique(receivers)) == count and not forbidden.intersection(map(int, receivers)):
            return candidate, receivers, examined
        candidate += step
    raise RuntimeError("no identity-isolated block found before uint32 limit")


def run(args: argparse.Namespace) -> dict[str, object]:
    forbidden, sources = historic_receiver_ids()
    sample_base, receivers, examined = select_block(
        count=args.count,
        first_candidate=args.first_candidate,
        step=args.step,
        forbidden=forbidden,
    )
    result = {
        "status": "completed_pre_generation_identity_selection",
        "experiment": args.experiment,
        "receiver_substream_formula": "(sample_seed + 2*2246822519) mod 2^32",
        "sample_seed_base": sample_base,
        "sample_seed_range": [sample_base, sample_base + args.count - 1],
        "receiver_noise_seed_range": [int(receivers.min()), int(receivers.max())],
        "sample_count": args.count,
        "candidate_blocks_examined": examined,
        "historic_train_unique_receiver_count": len(forbidden),
        "identity_overlap_count": len(forbidden.intersection(map(int, receivers))),
        "historic_train_sources_scanned": sources,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=640)
    parser.add_argument("--first-candidate", type=int, default=2_000_000_000)
    parser.add_argument("--step", type=int, default=10_000)
    parser.add_argument("--experiment", default="RouteA_H0o_MFTCrossPulseCompositeAttribution")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse_generation/pre_generation_identity_selection.json")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
