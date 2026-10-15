"""Merge exactly three successful P3o-L receipts without selection."""
from __future__ import annotations
import argparse, json
from pathlib import Path

SEEDS = (2101, 2102, 2103)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", required=True); args = parser.parse_args(); root = Path(args.output_root)
    rows = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "summary.json"
        if not path.exists() or (path.parent / "exception_receipt.json").exists(): raise RuntimeError(f"missing valid P3o-L receipt: {path}")
        row = json.loads(path.read_text(encoding="utf-8"));
        if len(row.get("results", [])) != 1 or int(row["results"][0]["seed"]) != seed: raise RuntimeError(f"invalid receipt: {path}")
        rows.append(row)
    gates = {key: all(row["promotion_gates"][key] for row in rows) for key in rows[0]["promotion_gates"]}
    merged = {"experiment": "P3o-L durable-D-drive posterior-component-conditioned bounded residual", "development_passed": bool(all(gates.values())), "promotion_gates": gates, "results": [r["results"][0] for r in rows], "comparisons": [r["comparisons"][0] for r in rows], "identity_audit": rows[0]["identity_audit"], "seed_receipts": [str(root / f"seed_{s}" / "summary.json") for s in SEEDS], "test_generated": False, "test_opened": False, "locked_test_used": False}
    (root / "summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
