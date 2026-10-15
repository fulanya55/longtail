"""Create sample-chunked caches for the nine Stage-B weak-component files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cj0724.rechunk_independent_power_data import convert_file


TARGETS = ("sawfm", "vft", "isrj")
DEFICITS = (5, 10, 15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="generated_data/p2e_stage_b_weak_components_seed1500000000")
    parser.add_argument("--target-root", default="generated_data/p2e_stage_b_weak_components_seed1500000000_rechunked")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source_root, target_root = project / args.source_root, project / args.target_root
    target_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for target in TARGETS:
        for deficit in DEFICITS:
            condition = f"{target}_weak_{deficit:02d}dB"
            receipts.append(convert_file(
                source_root / condition / "compound_dataset_imbalanced.mat",
                target_root / condition / "compound_dataset_imbalanced.mat",
            ))
    payload = {
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "conditions": [f"{target}_weak_{deficit:02d}dB" for target in TARGETS for deficit in DEFICITS],
        "files": receipts,
        "total_source_bytes": int(sum(row["source_bytes"] for row in receipts)),
        "total_target_bytes": int(sum(row["target_bytes"] for row in receipts)),
    }
    (target_root / "rechunk_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
