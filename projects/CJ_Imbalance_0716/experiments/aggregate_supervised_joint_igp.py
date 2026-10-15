"""Aggregate completed C2 selective-gradient screening receipts."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_VARIANTS = ("C2", "C2_GP", "C2_IGP")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def aggregate(root, variants=DEFAULT_VARIANTS, seeds=(2041, 2042, 2043)):
    root = Path(root)
    receipts = []
    parameter_rows = []
    for variant in variants:
        for seed in seeds:
            run_dir = root / variant / "seed_{}".format(seed)
            receipt = _read_json(run_dir / "run_receipt.json")
            if receipt.get("status") != "completed":
                raise ValueError("incomplete receipt: {}".format(run_dir))
            if bool(receipt.get("locked_test_used")):
                raise ValueError("locked-test receipt found: {}".format(run_dir))
            receipts.append(receipt)
            metrics = _read_json(run_dir / "parameter_metrics.json")
            for row in metrics["per_target"]:
                parameter_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "target": "{}/{}".format(row["class"], row["parameter"]),
                        "normalized_mae": float(row["normalized_mae"]),
                        "mae": float(row["mae"]),
                    }
                )

    aggregate_rows = []
    for variant in variants:
        selected = [row for row in receipts if row["variant"] == variant]
        entry = {"variant": variant, "seed_count": len(selected)}
        for key in (
            "calibrated_macro_aupr",
            "calibrated_macro_f1",
            "calibrated_hamming_loss",
            "oracle_macro_target_nmae",
            "system_parameter_coverage",
            "train_seconds",
        ):
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            entry[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        negative = np.asarray(
            [
                float(row["gradient_diagnostics"]["mean_negative_gradient_fraction"])
                for row in selected
            ],
            dtype=np.float64,
        )
        projected = [
            row["gradient_diagnostics"].get("mean_projected_target_count")
            for row in selected
        ]
        entry["gradient_diagnostics"] = {
            "mean_negative_gradient_fraction": float(negative.mean()),
            "mean_projected_target_count": (
                None
                if all(value is None for value in projected)
                else float(np.mean([float(value) for value in projected if value is not None]))
            ),
        }
        aggregate_rows.append(entry)

    by_variant_seed = {
        (row["variant"], int(row["seed"])): row for row in receipts
    }
    paired_deltas = []
    for candidate in variants:
        if candidate == "C2":
            continue
        for seed in seeds:
            baseline = by_variant_seed[("C2", int(seed))]
            selected = by_variant_seed[(candidate, int(seed))]
            paired_deltas.append(
                {
                    "variant": candidate,
                    "seed": int(seed),
                    "macro_aupr_difference": float(
                        selected["calibrated_macro_aupr"]
                        - baseline["calibrated_macro_aupr"]
                    ),
                    "macro_f1_difference": float(
                        selected["calibrated_macro_f1"]
                        - baseline["calibrated_macro_f1"]
                    ),
                    "hamming_difference": float(
                        selected["calibrated_hamming_loss"]
                        - baseline["calibrated_hamming_loss"]
                    ),
                    "parameter_nmae_reduction": float(
                        baseline["oracle_macro_target_nmae"]
                        - selected["oracle_macro_target_nmae"]
                    ),
                    "split_sha256_equal": bool(
                        selected["split_sha256"] == baseline["split_sha256"]
                    ),
                }
            )

    target_aggregate = []
    for variant in variants:
        targets = sorted(
            {row["target"] for row in parameter_rows if row["variant"] == variant}
        )
        for target in targets:
            selected = [
                row
                for row in parameter_rows
                if row["variant"] == variant and row["target"] == target
            ]
            target_aggregate.append(
                {
                    "variant": variant,
                    "target": target,
                    "normalized_mae_mean": float(
                        np.mean([row["normalized_mae"] for row in selected])
                    ),
                    "mae_mean": float(np.mean([row["mae"] for row in selected])),
                }
            )

    return {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "variants": list(variants),
            "seeds": [int(seed) for seed in seeds],
            "train_samples": 4096,
            "development_samples_per_role": 512,
            "epochs": 20,
        },
        "aggregate": aggregate_rows,
        "paired_deltas_vs_c2": paired_deltas,
        "per_target_aggregate": target_aggregate,
        "runs": receipts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = aggregate(args.root)
    output = args.output or args.root / "supervised_joint_igp_summary.json"
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"]}, indent=2))
