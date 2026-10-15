"""Create a sample-chunked cache for one MATLAB v7.3 training file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cj0724.rechunk_independent_power_data import convert_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = project / args.source
    target = project / args.target
    receipt = project / args.receipt
    if receipt.exists():
        raise FileExistsError(receipt)
    row = convert_file(source, target)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
