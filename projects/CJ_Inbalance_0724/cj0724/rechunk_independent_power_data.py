"""Create sample-chunked, model-ready HDF5 caches from P1d MATLAB files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cj0724.dual_axis import POWER_LEVELS


REQUIRED_DATASETS = (
    "data_legacy",
    "label_selected",
    "parameter_values",
    "parameter_mask",
    "receiver_noise_seed",
)


def convert_file(source: Path, target: Path) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(source, "r") as source_handle, h5py.File(target, "w") as target_handle:
        for name in REQUIRED_DATASETS:
            if name not in source_handle:
                raise KeyError(f"{name} missing from {source}")
        signal = np.asarray(source_handle["data_legacy"], dtype=np.float32)
        target_handle.create_dataset(
            "data_legacy", data=signal, dtype=np.float32,
            chunks=(signal.shape[0], 1), compression=None,
        )
        for name in REQUIRED_DATASETS[1:]:
            values = np.asarray(source_handle[name])
            chunks = None
            if values.ndim == 2 and values.shape[1] > 1:
                chunks = (values.shape[0], 1)
            target_handle.create_dataset(name, data=values, chunks=chunks, compression=None)
        target_handle.attrs["source_path"] = str(source.resolve())
        target_handle.attrs["cache_purpose"] = "P1d model evaluation; sample-chunked float32 legacy I/Q"
    with h5py.File(target, "r") as handle:
        chunk = tuple(handle["data_legacy"].chunks)
        shape = tuple(handle["data_legacy"].shape)
        dtype = str(handle["data_legacy"].dtype)
    return {
        "source": str(source.resolve()), "target": str(target.resolve()),
        "shape": shape, "chunks": chunk, "dtype": dtype,
        "source_bytes": source.stat().st_size, "target_bytes": target.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="generated_data/p1d_independent_power_sweep_seed100000000",
    )
    parser.add_argument(
        "--target-root",
        default="generated_data/p1d_independent_power_sweep_seed100000000_rechunked",
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, default=list(POWER_LEVELS),
        help="Power levels to convert; directory indices follow the canonical nine-level protocol.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=("train", "test"),
        help="Dataset splits to convert. Use train only to keep a generated test split unopened.",
    )
    parser.add_argument(
        "--compact-indices", action="store_true",
        help="Map the supplied levels to level_01..N rather than to canonical nine-power indices.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source_root, target_root = project / args.source_root, project / args.target_root
    target_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    requested = [int(level) for level in args.levels]
    unknown = sorted(set(requested) - set(POWER_LEVELS))
    if unknown:
        raise ValueError(f"Unknown power levels: {unknown}")
    for compact_index, level in enumerate(requested, start=1):
        index = compact_index if args.compact_indices else list(POWER_LEVELS).index(level) + 1
        directory = f"level_{index:02d}__jnr_{level}dB"
        for split in args.splits:
            filename = f"closedset_6class_{split}.mat"
            receipts.append(convert_file(source_root / directory / filename, target_root / directory / filename))
    payload = {
        "source_root": str(source_root.resolve()), "target_root": str(target_root.resolve()),
        "splits": list(args.splits),
        "files": receipts,
        "total_source_bytes": int(sum(row["source_bytes"] for row in receipts)),
        "total_target_bytes": int(sum(row["target_bytes"] for row in receipts)),
    }
    (target_root / "rechunk_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
