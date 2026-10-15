"""Select M4 top-k using only preregistered seed-2039 validation histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_validation(history_path: Path) -> dict:
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty history: {history_path}")
    parsed = [
        {
            "epoch": int(row["epoch"]),
            "validation_common_macro_aupr": float(
                row["validation_common_macro_aupr"]
            ),
            "validation_target_weak_auroc": float(
                row["validation_target_weak_auroc"]
            ),
        }
        for row in rows
    ]
    return max(
        parsed,
        key=lambda row: (
            row["validation_common_macro_aupr"],
            row["validation_target_weak_auroc"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k8-root", type=Path, required=True)
    parser.add_argument("--k16-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {8: args.k8_root, 16: args.k16_root}
    candidates = {}
    for topk, root in roots.items():
        history = root / "seed_2039" / "history.csv"
        config = root / "experiment_config.json"
        candidates[str(topk)] = {
            "best_validation": _best_validation(history),
            "history_path": str(history.resolve()),
            "history_sha256": _sha256(history),
            "experiment_config_sha256": _sha256(config),
        }
    selected = max(
        (8, 16),
        key=lambda topk: (
            candidates[str(topk)]["best_validation"][
                "validation_common_macro_aupr"
            ],
            candidates[str(topk)]["best_validation"][
                "validation_target_weak_auroc"
            ],
        ),
    )
    payload = {
        "status": "completed",
        "selection_role": "seed_2039_validation_only",
        "selection_key": [
            "validation_common_macro_aupr",
            "validation_target_weak_auroc_tiebreak",
        ],
        "early_metrics_read_for_selection": False,
        "calibration_metrics_read_for_selection": False,
        "threshold_metrics_read_for_selection": False,
        "candidates": candidates,
        "selected_topk": selected,
        "selected_model_key": f"m4k{selected}",
        "rejected_topk": 16 if selected == 8 else 8,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
