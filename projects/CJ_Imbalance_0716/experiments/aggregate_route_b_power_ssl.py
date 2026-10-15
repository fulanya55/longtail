"""Aggregate multi-seed paired-power SSL screens."""

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("RandomInit", "SameViewAug", "RandomSource", "SameSourcePower")


def run(args):
    rows = []
    for root in args.input_roots:
        with (Path(root) / "route_b_power_ssl_summary.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.extend(payload["runs"])
    aggregate = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregate.append(
            {
                "variant": variant,
                "seed_count": len(selected),
                "linear_probe_macro_aupr_mean": float(np.mean([row["linear_probe_macro_aupr"] for row in selected])),
                "linear_probe_macro_aupr_std": float(np.std([row["linear_probe_macro_aupr"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
                "finetune_macro_aupr_mean": float(np.mean([row["finetune_macro_aupr"] for row in selected])),
                "finetune_macro_aupr_std": float(np.std([row["finetune_macro_aupr"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
                "finetune_hamming_mean": float(np.mean([row["finetune_hamming_fixed_0p5"] for row in selected])),
            }
        )
    result = {"status": "completed", "locked_test_used": False, "aggregate": aggregate, "runs": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
