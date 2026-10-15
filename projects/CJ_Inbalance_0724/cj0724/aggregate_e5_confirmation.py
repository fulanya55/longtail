"""Apply the frozen two-seed confirmation and combined five-seed E5 gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "macro_aupr",
    "vft_aupr",
    "macro_f1",
    "hamming",
    "weak_recall",
    "vft_fpr",
    "delta_macro_aupr",
    "delta_vft_aupr",
    "delta_macro_f1",
    "delta_hamming",
    "delta_weak_recall",
    "delta_vft_fpr",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    result = {"runs": len(rows), "seeds": [int(row["seed"]) for row in rows]}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result["positive_vft_aupr_runs"] = sum(float(row["delta_vft_aupr"]) > 0 for row in rows)
    result["physics_weight_one_runs"] = sum(
        float(row["selected_physics_weight"]) == 1.0 for row in rows
    )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-result",
        type=Path,
        default=Path("outputs/vft_physics_fusion_20260725/result.json"),
    )
    parser.add_argument(
        "--confirmation-result",
        type=Path,
        default=Path("outputs/vft_physics_confirmation_fusion_20260725/result.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/e5_confirmation_20260725.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vft_physics_e5_five_seed_20260725"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    development = json.loads(args.development_result.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation_result.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    development_rows = development["runs"]
    confirmation_rows = confirmation["runs"]
    all_rows = development_rows + confirmation_rows
    expected_confirmation = [int(seed) for seed in protocol["confirmation_seeds"]]
    actual_confirmation = [int(row["seed"]) for row in confirmation_rows]
    if actual_confirmation != expected_confirmation:
        raise ValueError(
            f"Confirmation seeds differ: expected {expected_confirmation}, got {actual_confirmation}"
        )
    if len({int(row["seed"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate seeds found in combined E5 results")
    if any(bool(row["locked_test_used"]) for row in all_rows):
        raise ValueError("A run reports locked-test use")

    confirmation_summary = summarize(confirmation_rows)
    five_seed_summary = summarize(all_rows)
    gates = {
        "confirmation_seeds_match_frozen_protocol": actual_confirmation
        == expected_confirmation,
        "confirmation_two_of_two_vft_aupr_positive": confirmation_summary[
            "positive_vft_aupr_runs"
        ]
        == 2,
        "confirmation_mean_vft_aupr_at_least_plus_0p02": confirmation_summary[
            "delta_vft_aupr_mean"
        ]
        >= 0.02,
        "confirmation_macro_aupr_noninferior": confirmation_summary[
            "delta_macro_aupr_mean"
        ]
        >= -0.005,
        "confirmation_hamming_noninferior": confirmation_summary[
            "delta_hamming_mean"
        ]
        <= 0.005,
        "confirmation_weak_recall_noninferior": confirmation_summary[
            "delta_weak_recall_mean"
        ]
        >= -0.05,
        "confirmation_vft_fpr_noninferior": confirmation_summary[
            "delta_vft_fpr_mean"
        ]
        <= 0.0,
        "combined_five_of_five_vft_aupr_positive": five_seed_summary[
            "positive_vft_aupr_runs"
        ]
        == 5,
        "combined_five_of_five_choose_physics_weight_one": five_seed_summary[
            "physics_weight_one_runs"
        ]
        == 5,
    }
    write_csv(args.output_root / "runs.csv", all_rows)
    result = {
        "method": "e5_e0_plus_vft_matched_filter",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "development_three_seed": summarize(development_rows),
        "confirmation_two_seed": confirmation_summary,
        "combined_five_seed": five_seed_summary,
        "gates": gates,
        "development_candidate_frozen": all(gates.values()),
        "locked_test_used": False,
        "locked_test_authorized": False,
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
