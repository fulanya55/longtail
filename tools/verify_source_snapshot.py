"""Verify source inventory hashes; optionally verify original source locations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-originals', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / 'docs/release/source_inventory.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    for row in rows:
        paths = [root / row['repository_path']]
        if args.check_originals:
            paths.append(Path(row['source_path']))
        for path in paths:
            if not path.is_file():
                errors.append({'path': str(path), 'error': 'missing'})
            elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                errors.append({'path': str(path), 'error': 'sha256 mismatch'})
    print(json.dumps({'files': len(rows), 'check_originals': args.check_originals,
                      'errors': errors}, ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
