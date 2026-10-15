"""Promote and audit Base-DB as the primary six-class baseline."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")


def _difference(candidate, baseline):
    return {
        "delta_macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
        "delta_tail_aupr": candidate["tail_aupr"] - baseline["tail_aupr"],
        "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "delta_hamming_loss": candidate["hamming_loss"] - baseline["hamming_loss"],
        "delta_tail_f1": candidate["tail_f1"] - baseline["tail_f1"],
    }


def _aggregate(rows, seed):
    result = {}
    for index, metric in enumerate(
        ("delta_macro_aupr", "delta_tail_aupr", "delta_macro_f1", "delta_hamming_loss", "delta_tail_f1")
    ):
        result[metric] = _mean_ci([row[metric] for row in rows], seed + index)
    result["macro_wins"] = sum(row["delta_macro_aupr"] > 0 for row in rows)
    result["tail_wins"] = sum(row["delta_tail_aupr"] > 0 for row in rows)
    return result


def run(args):
    source = _read(Path(args.source_root) / "route_a_six_class_head_db_screen_summary.json")
    runs = source["runs"]
    pairs = []
    for tail_name in CLASS_NAMES:
        for seed in args.seeds:
            selected = [
                row for row in runs
                if row["tail_label"] == tail_name and row["seed"] == seed
            ]
            values = {row["variant"]: row for row in selected}
            if "BaseASL" not in values or "BaseDB" not in values:
                raise ValueError("Missing Base-ASL/Base-DB pair")
            if values["BaseASL"]["manifest_sha256"] != values["BaseDB"]["manifest_sha256"]:
                raise ValueError("Base-DB pair uses a different manifest")
            if values["BaseDB"]["locked_test_used"]:
                raise ValueError("Locked test was used")
            pairs.append(
                {
                    "tail_label": tail_name,
                    "seed": int(seed),
                    "manifest_sha256": values["BaseDB"]["manifest_sha256"],
                    **_difference(values["BaseDB"], values["BaseASL"]),
                }
            )
    by_tail = []
    for index, tail_name in enumerate(CLASS_NAMES):
        selected = [row for row in pairs if row["tail_label"] == tail_name]
        by_tail.append(
            {
                "tail_label": tail_name,
                "pair_count": len(selected),
                "aggregate": _aggregate(selected, 14000 + index * 10),
            }
        )
    overall = _aggregate(pairs, 15000)
    gate = {
        "passed": bool(
            overall["delta_macro_aupr"]["mean"] > 0.0
            and overall["delta_tail_aupr"]["mean"] > 0.0
            and overall["macro_wins"] >= 12
            and sum(row["aggregate"]["delta_tail_aupr"]["mean"] > 0.0 for row in by_tail) >= 5
        ),
        "decision": "promote_basedb_as_primary_six_class_baseline",
        "positive_tail_rotations": sum(
            row["aggregate"]["delta_tail_aupr"]["mean"] > 0.0 for row in by_tail
        ),
    }
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "gate": gate,
        "aggregate_overall": overall,
        "aggregate_by_tail": by_tail,
        "pairs": pairs,
    }
    output = Path(args.output_root) / "route_a_six_class_basedb_baseline_summary.json"
    _write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_basedb_baseline",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

