"""Aggregate the preregistered five-seed B1 physical-sufficiency screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RUNS = {
    20260724: Path("outputs/b1_physics_sufficiency_rerun_20260725/result.json"),
    20260725: Path("outputs/b1_physics_sufficiency_seed20260725/result.json"),
    20260726: Path("outputs/b1_physics_sufficiency_seed20260726/result.json"),
    271828: Path("outputs/b1_physics_sufficiency_seed271828/result.json"),
    314159: Path("outputs/b1_physics_sufficiency_seed314159/result.json"),
}

FROZEN_FEATURES = {"MFT": "combined", "ISRJ": "isrj_periodicity"}


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = ("delta_aupr", "delta_fpr", "delta_macro_aupr", "delta_hamming")
    result: dict[str, object] = {"runs": len(rows), "seeds": [int(row["seed"]) for row in rows]}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"{field}_mean"] = float(values.mean())
        result[f"{field}_std"] = float(values.std(ddof=0))
        result[f"{field}_positive_runs"] = int((values > 0).sum())
        result[f"{field}_negative_runs"] = int((values < 0).sum())
    result["screen_gate_passed_runs"] = int(sum(bool(row["screen_gate_passed"]) for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/b1_physics_sufficiency_five_seed_20260725"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    selected: dict[str, list[dict[str, object]]] = {target: [] for target in FROZEN_FEATURES}
    all_rows = []
    for seed, path in DEFAULT_RUNS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["runs"]:
            enriched = {"seed": seed, **row, "source": str(path.resolve())}
            all_rows.append(enriched)
            if row["feature"] == FROZEN_FEATURES.get(row["target"]):
                selected[row["target"]].append(enriched)
    summaries = {target: _summary(rows) for target, rows in selected.items()}
    gates = {
        target: {
            "same_direction_4_of_5": summary["screen_gate_passed_runs"] >= 4,
            "fpr_reduced_4_of_5": summary["delta_fpr_negative_runs"] >= 4,
            "macro_noninferior_mean": summary["delta_macro_aupr_mean"] >= -0.01,
            "hamming_noninferior_mean": summary["delta_hamming_mean"] <= 0.005,
        }
        for target, summary in summaries.items()
    }
    for gate in gates.values():
        gate["promote_to_mechanism_router"] = bool(all(gate.values()))
    result = {
        "protocol": {
            "development_seeds": [20260724, 20260725, 20260726],
            "confirmation_seeds": [271828, 314159],
            "frozen_features": FROZEN_FEATURES,
            "selection_rule": "features frozen before reading the two confirmation-seed development results",
            "locked_test_used": False,
        },
        "selected_runs": selected,
        "summaries": summaries,
        "gates": gates,
        "promoted_targets": [target for target, gate in gates.items() if gate["promote_to_mechanism_router"]],
        "locked_test_used": False,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_root / "all_candidates.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
