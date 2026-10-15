"""Promote a completed small-sample R1 run into an explicitly partial receipt."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs" / "route_b_r1_preflight_training"
RUNS = sorted(ROOT.glob("balanced_multitask_*"))
if not RUNS:
    raise FileNotFoundError(f"no R1 training runs under {ROOT}")
run = RUNS[-1]
summary = run / "per_seed_summary.csv"
rows = list(csv.DictReader(summary.open(encoding="utf-8-sig")))
for row in rows:
    row["evidence_status"] = "partial_development_smoke"
    row["train_per_combination"] = 20
    row["val_per_combination"] = 10
    row["epochs"] = 1
    row["performance_claim"] = "not_validated"
    row["locked_test_used"] = "False"
out = PROJECT / "docs" / "results" / "route_b_r1_baseline.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0])
with out.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
receipt = {
    "stage": "R1",
    "status": "partial",
    "run_dir": str(run),
    "rows": rows,
    "protocol": str(run / "protocol.json"),
    "small_sample_screen": True,
    "full_development_seeds_complete": False,
    "test_generated": False,
    "test_opened": False,
    "locked_test_used": False,
}
receipts = PROJECT / "docs" / "receipts"
receipts.mkdir(parents=True, exist_ok=True)
(receipts / "route_b_r1_result_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
report = """# Route B R1 baseline — partial development screen

Status: **partial**. One seed (`2041`), one epoch, and 20/10 samples per label combination were completed for the three parameter-head paths. This is a code/data-path and directional screen only; it is not the preregistered multi-seed performance result.

- Development split: generated from the six-class training MAT with split seed `20260826`.
- Variants: `legacy_mlcnn_global`, `legacy_mlcnn_class_conditioned`, `legacy_mlcnn_class_conditioned_detached`.
- B0 classification-only and B4 scheduled warm-up remain protocol entries but have not received a performance run in this stage.
- No test-v2 was created or opened; locked test was not used.
- The exact rows are in `docs/results/route_b_r1_baseline.csv`; the run receipt is `docs/receipts/route_b_r1_result_receipt.json`.

The next required action is full development training with the frozen R1 protocol, at least five seeds, unchanged split/loss/optimizer/threshold/budget, followed by per-parameter and paired-power audits.
"""
(PROJECT / "docs" / "reports" / "route_b_r1_partial.md").write_text(report, encoding="utf-8")
index_path = receipts / "route_b_final_evidence_index.json"
index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
index.update({"r1": str(receipts / "route_b_r1_result_receipt.json"), "r1_status": "partial"})
index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
print(json.dumps({"status": "partial", "run_dir": str(run), "rows": len(rows)}, indent=2))
