"""Frozen development-v10 subgroup-risk audit for the U2 MFT peak guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _component_power, _write_json  # noqa: E402


MFT_INDEX = 3
OTHER = ((0, "FMB"), (1, "SawFM"), (2, "SinFM"), (4, "VFT"), (5, "ISRJ"))


def _rate(prediction, mask):
    return float(prediction[mask, MFT_INDEX].mean()) if int(mask.sum()) else float("nan")


def _margin(power, labels):
    result = np.full(len(labels), np.inf, dtype=np.float64)
    for row in np.flatnonzero(labels[:, MFT_INDEX] == 1):
        other = np.flatnonzero((labels[row] == 1) & (np.arange(labels.shape[1]) != MFT_INDEX))
        if len(other):
            result[row] = float(power[row, MFT_INDEX] - power[row, other].max())
    return result


def _seed_metrics(base_prediction, candidate_prediction, labels, receiver, margin):
    positive = labels[:, MFT_INDEX] == 1
    negative = ~positive
    order = labels.sum(axis=1)
    groups = {"negative_order": {}, "positive_margin": {}, "negative_colabel": {}, "receiver_block": {}}
    for value in range(1, 6):
        mask = negative & (order == value)
        groups["negative_order"][str(value)] = {"count": int(mask.sum()), "base": _rate(base_prediction, mask), "candidate": _rate(candidate_prediction, mask)}
    margin_masks = {
        "le_minus15": positive & (margin <= -15),
        "minus15_to_minus10": positive & (margin > -15) & (margin <= -10),
        "minus10_to_minus5": positive & (margin > -10) & (margin <= -5),
        "minus5_to_0": positive & (margin > -5) & (margin <= 0),
        "gt0_or_single": positive & (margin > 0),
    }
    for name, mask in margin_masks.items():
        groups["positive_margin"][name] = {"count": int(mask.sum()), "base": _rate(base_prediction, mask), "candidate": _rate(candidate_prediction, mask)}
    for index, name in OTHER:
        mask = negative & (labels[:, index] == 1) & (order >= 4)
        groups["negative_colabel"][name] = {"count": int(mask.sum()), "base": _rate(base_prediction, mask), "candidate": _rate(candidate_prediction, mask)}
    for block in range(5):
        block_mask = (receiver % 5) == block
        negative_mask = block_mask & negative & (order >= 4)
        weak_mask = block_mask & positive & (margin <= -10)
        groups["receiver_block"][str(block)] = {
            "hard_negative_count": int(negative_mask.sum()), "weak_positive_count": int(weak_mask.sum()),
            "hard_fpr_base": _rate(base_prediction, negative_mask), "hard_fpr_candidate": _rate(candidate_prediction, negative_mask),
            "weak_recall_base": _rate(base_prediction, weak_mask), "weak_recall_candidate": _rate(candidate_prediction, weak_mask),
        }
    return groups


def _aggregate(rows):
    output = {}
    for family in ("negative_order", "positive_margin", "negative_colabel"):
        output[family] = {}
        for group_index, group in enumerate(rows[0][family]):
            delta = [row[family][group]["candidate"] - row[family][group]["base"] for row in rows]
            output[family][group] = {
                "count": int(rows[0][family][group]["count"]),
                "base": _mean_ci([row[family][group]["base"] for row in rows], 20409000 + group_index),
                "candidate": _mean_ci([row[family][group]["candidate"] for row in rows], 20409100 + group_index),
                "delta": _mean_ci(delta, 20409200 + group_index),
                "beneficial_seeds": int(sum(value < 0 for value in delta)) if family != "positive_margin" else int(sum(value >= 0 for value in delta)),
            }
    output["receiver_block"] = {}
    for block in rows[0]["receiver_block"]:
        fpr_delta = [row["receiver_block"][block]["hard_fpr_candidate"] - row["receiver_block"][block]["hard_fpr_base"] for row in rows]
        recall_delta = [row["receiver_block"][block]["weak_recall_candidate"] - row["receiver_block"][block]["weak_recall_base"] for row in rows]
        output["receiver_block"][block] = {
            "hard_fpr_delta": _mean_ci(fpr_delta, 20409300 + int(block)),
            "weak_recall_delta": _mean_ci(recall_delta, 20409400 + int(block)),
        }
    return output


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["test_files_opened"]:
        raise ValueError("development-v10 audit failed")
    data_path = Path(audit["development_v10_path"])
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    power = _component_power(parameters)
    margin = _margin(power, labels)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    rows = []
    for seed in args.seeds:
        root = args.prediction_root / f"seed_{seed}"
        base = np.load(root / "K0_U2Frozen/predictions.npz")
        candidate = np.load(root / "K1_AlignedPeakNegativeGuard/predictions.npz")
        if not np.array_equal(base["targets"], labels) or not np.array_equal(candidate["targets"], labels):
            raise AssertionError("prediction targets differ from development-v10")
        rows.append(_seed_metrics(base["predictions"].astype(bool), candidate["predictions"].astype(bool), labels, receiver, margin))
    aggregate = _aggregate(rows)
    supported_margin = [entry for entry in aggregate["positive_margin"].values() if entry["count"] >= 50]
    checks = {
        "order4_fpr_reduced": aggregate["negative_order"]["4"]["delta"]["mean"] < 0,
        "order5_fpr_reduced": aggregate["negative_order"]["5"]["delta"]["mean"] < 0,
        "worst_supported_power_recall_delta_ge_minus0p05": min(entry["delta"]["mean"] for entry in supported_margin) >= -0.05,
        "all_receiver_blocks_reduce_hard_fpr": all(entry["hard_fpr_delta"]["mean"] < 0 for entry in aggregate["receiver_block"].values()),
        "four_of_five_colabels_reduce_fpr": sum(entry["delta"]["mean"] < 0 for entry in aggregate["negative_colabel"].values()) >= 4,
    }
    result = {
        "status": "completed", "experiment": "RouteA_U2_MFT_PeakGuard_v10_Subgroups",
        "evidence_role": "read_only_frozen_development_v10_subgroup_audit", "rows": rows,
        "aggregate": aggregate, "risk_gate": {"checks": checks, "passed": bool(all(checks.values()))},
        "external_refit": False, "locked_test_used": False, "test_v3_used": False, "test_v10_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"status": "completed", "gate": result["risk_gate"], "output": str(args.output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v10_generation/identity_audit.json")
    parser.add_argument("--prediction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_development_v10_frozen")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_evidence_bundle/subgroup_audit_v10.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

