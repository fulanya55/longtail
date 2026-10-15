"""Aggregate the frozen frequency experiment and apply its preregistered gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SEEDS = (260821, 260822, 260823, 260824, 260825)
PRIMARY = ("FB", "LabelIF20", "ComboIF20")
SECONDARY = ("LabelIF10", "ComboIF10", "LabelIF50", "ComboIF50")


FORMAL_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_formal"
OUTPUT_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_frequency_gate"
BASELINES = ("F0_B0_Focal", "F1_B0_DB", "F2_Ordinary_cRT")
CANDIDATE = "F3_RBC_cRT"
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 3930200001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_rows() -> dict[tuple[int, str], dict[str, Any]]:
    rows = {}
    for seed in SEEDS:
        for regime in PRIMARY + SECONDARY:
            path = FORMAL_ROOT / f"seed_{seed}" / regime / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["status"] != "completed":
                raise ValueError(f"incomplete formal cell: {seed}/{regime}")
            rows[(seed, regime)] = payload
    return rows


def macro(row, arm, view="VH") -> float:
    return float(row["results"][arm]["by_view"][view]["macro_aupr"])


def label_metric(row, arm, label, view="VH") -> float:
    return float(row["results"][arm]["by_view"][view]["per_label_aupr"][label])


def baseline_arm(row) -> str:
    return max(BASELINES, key=lambda arm: macro(row, arm))


def mean_summary(values) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
        "positive_count": int((array > 0).sum()),
        "nonnegative_count": int((array >= 0).sum()),
    }


def weighted_average_precision(truth, scores, bootstrap_counts):
    truth = np.asarray(truth, dtype=np.float64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    weights = bootstrap_counts[:, order].astype(np.float64, copy=False)
    positive_weights = weights * truth[order][None, :]
    cumulative_total = np.cumsum(weights, axis=1)
    cumulative_positive = np.cumsum(positive_weights, axis=1)
    precision = cumulative_positive / np.maximum(cumulative_total, 1.0)
    return (precision * positive_weights).sum(axis=1) / np.maximum(positive_weights.sum(axis=1), 1.0)


def receiver_bootstrap(rows):
    items = []
    for seed in SEEDS:
        for regime in ("LabelIF20", "ComboIF20"):
            row = rows[(seed, regime)]
            selected_baseline = baseline_arm(row)
            path = FORMAL_ROOT / f"seed_{seed}" / regime / "evaluation_predictions.npz"
            prediction = np.load(path)
            items.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "baseline": selected_baseline,
                    "targets": prediction["targets"].astype(np.uint8),
                    "candidate": prediction[f"{CANDIDATE}_VH_logits"].astype(np.float64),
                    "baseline_logits": prediction[f"{selected_baseline}_VH_logits"].astype(np.float64),
                    "prediction_sha256": sha256(path),
                }
            )
    sample_count = len(items[0]["targets"])
    if any(len(item["targets"]) != sample_count for item in items):
        raise ValueError("bootstrap evaluation identity counts differ")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    item_deltas = np.empty((BOOTSTRAP_REPETITIONS, len(items)), dtype=np.float32)
    batch_size = 100
    for start in range(0, BOOTSTRAP_REPETITIONS, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPETITIONS)
        current = stop - start
        sampled = rng.randint(0, sample_count, size=(current, sample_count))
        counts = np.zeros((current, sample_count), dtype=np.uint16)
        row_index = np.repeat(np.arange(current), sample_count)
        np.add.at(counts, (row_index, sampled.reshape(-1)), 1)
        for item_index, item in enumerate(items):
            per_label = []
            for label_index in range(len(CLASS_NAMES)):
                candidate_ap = weighted_average_precision(
                    item["targets"][:, label_index], item["candidate"][:, label_index], counts
                )
                baseline_ap = weighted_average_precision(
                    item["targets"][:, label_index], item["baseline_logits"][:, label_index], counts
                )
                per_label.append(candidate_ap - baseline_ap)
            item_deltas[start:stop, item_index] = np.mean(per_label, axis=0)
    aggregate = item_deltas.mean(axis=1)
    label_values = item_deltas[:, [index for index, item in enumerate(items) if item["regime"] == "LabelIF20"]].mean(axis=1)
    combo_values = item_deltas[:, [index for index, item in enumerate(items) if item["regime"] == "ComboIF20"]].mean(axis=1)

    def describe(values):
        return {
            "mean": float(np.mean(values)),
            "95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "repetitions": BOOTSTRAP_REPETITIONS,
        }

    return {
        "seed": BOOTSTRAP_SEED,
        "identity_count_per_cell": sample_count,
        "pairing": "same receiver identity bootstrap indices for F3 and frozen replicatewise best baseline",
        "items": [{key: value for key, value in item.items() if key not in ("targets", "candidate", "baseline_logits")} for item in items],
        "combined_primary_if20": describe(aggregate),
        "label_if20": describe(label_values),
        "combo_if20": describe(combo_values),
    }, item_deltas, aggregate


def aggregate(rows):
    cell_rows = []
    for seed in SEEDS:
        for regime in PRIMARY + SECONDARY:
            row = rows[(seed, regime)]
            best = baseline_arm(row)
            candidate_macro = macro(row, CANDIDATE)
            best_macro = macro(row, best)
            per_label_delta = {
                label: label_metric(row, CANDIDATE, label) - label_metric(row, best, label)
                for label in CLASS_NAMES
            }
            tail_labels = row["three_least_frequent_labels"]
            cell_rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "best_baseline": best,
                    "candidate_macro_aupr": candidate_macro,
                    "best_baseline_macro_aupr": best_macro,
                    "macro_aupr_delta": candidate_macro - best_macro,
                    "three_least_frequent_labels": tail_labels,
                    "three_tail_mean_aupr_delta": float(np.mean([per_label_delta[label] for label in tail_labels])),
                    "per_label_aupr_delta": per_label_delta,
                }
            )
    lookup = {(row["seed"], row["regime"]): row for row in cell_rows}
    primary_seed = []
    for seed in SEEDS:
        primary_seed.append(
            {
                "seed": seed,
                "if20_mean_delta": float(np.mean([lookup[(seed, regime)]["macro_aupr_delta"] for regime in ("LabelIF20", "ComboIF20")])),
                "all_primary_mean_delta": float(np.mean([lookup[(seed, regime)]["macro_aupr_delta"] for regime in PRIMARY])),
                "vs_f2_mean_delta": float(np.mean([macro(rows[(seed, regime)], CANDIDATE) - macro(rows[(seed, regime)], "F2_Ordinary_cRT") for regime in PRIMARY])),
                "vs_random_mean_delta": float(np.mean([macro(rows[(seed, regime)], CANDIDATE) - macro(rows[(seed, regime)], "F3_random_pair_rank") for regime in PRIMARY])),
            }
        )
    if20 = [lookup[(seed, regime)]["macro_aupr_delta"] for seed in SEEDS for regime in ("LabelIF20", "ComboIF20")]
    fb = [lookup[(seed, "FB")]["macro_aupr_delta"] for seed in SEEDS]
    all_primary_tail = [lookup[(seed, regime)]["three_tail_mean_aupr_delta"] for seed in SEEDS for regime in PRIMARY]
    primary_label_safety = {
        f"{regime}/{label}": mean_summary([lookup[(seed, regime)]["per_label_aupr_delta"][label] for seed in SEEDS])
        for regime in PRIMARY
        for label in CLASS_NAMES
    }
    secondary = {
        regime: mean_summary([lookup[(seed, regime)]["macro_aupr_delta"] for seed in SEEDS])
        for regime in SECONDARY
    }
    controls = {
        "vs_f2_all_primary": mean_summary([row["vs_f2_mean_delta"] for row in primary_seed]),
        "vs_random_all_primary": mean_summary([row["vs_random_mean_delta"] for row in primary_seed]),
        "vs_no_rank_all_primary": mean_summary([macro(rows[(seed, regime)], CANDIDATE) - macro(rows[(seed, regime)], "F3_no_rank") for seed in SEEDS for regime in PRIMARY]),
        "vs_no_anchor_all_primary": mean_summary([macro(rows[(seed, regime)], CANDIDATE) - macro(rows[(seed, regime)], "F3_no_anchor") for seed in SEEDS for regime in PRIMARY]),
    }
    bootstrap, item_deltas, aggregate_bootstrap = receiver_bootstrap(rows)
    gates = {
        "G1_if20_macro_mean_at_least_plus_0p005": float(np.mean(if20)) >= 0.005,
        "G2_receiver_bootstrap_ci_lower_above_zero": bootstrap["combined_primary_if20"]["95_ci"][0] > 0.0,
        "G3_at_least_4_of_5_seed_if20_means_positive": sum(row["if20_mean_delta"] > 0 for row in primary_seed) >= 4,
        "G4_three_least_labels_mean_at_least_plus_0p010": float(np.mean(all_primary_tail)) >= 0.010,
        "G5_fb_macro_mean_at_least_minus_0p005": float(np.mean(fb)) >= -0.005,
        "G6_every_primary_label_mean_at_least_minus_0p010": all(value["mean"] >= -0.010 for value in primary_label_safety.values()),
        "G7_if10_if50_each_at_least_3_of_5_nonnegative": all(value["nonnegative_count"] >= 3 for value in secondary.values()),
        "G8a_f3_beats_f2_mean_and_4_of_5": controls["vs_f2_all_primary"]["mean"] > 0 and controls["vs_f2_all_primary"]["positive_count"] >= 4,
        "G8b_f3_beats_random_mean_and_4_of_5": controls["vs_random_all_primary"]["mean"] > 0 and controls["vs_random_all_primary"]["positive_count"] >= 4,
    }
    return {
        "status": "completed",
        "decision": "frequency_gate_passed" if all(gates.values()) else "frequency_reconstruction_failed",
        "cell_rows": cell_rows,
        "primary_if20_macro_delta": mean_summary(if20),
        "primary_per_seed": primary_seed,
        "three_tail_primary_delta": mean_summary(all_primary_tail),
        "fb_macro_delta": mean_summary(fb),
        "primary_label_safety": primary_label_safety,
        "secondary_direction": secondary,
        "controls": controls,
        "receiver_bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }, item_deltas, aggregate_bootstrap


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    result, item_deltas, aggregate_bootstrap = aggregate(rows)
    summary_path = OUTPUT_ROOT / "frequency_gate_summary.json"
    write_json(summary_path, result)
    np.savez_compressed(
        OUTPUT_ROOT / "receiver_bootstrap_samples.npz",
        item_deltas=item_deltas,
        aggregate_delta=aggregate_bootstrap,
        seed=np.asarray([BOOTSTRAP_SEED], dtype=np.int64),
    )
    receipt = {
        "status": "completed",
        "decision": result["decision"],
        "summary_sha256": sha256(summary_path),
        "bootstrap_samples_sha256": sha256(OUTPUT_ROOT / "receiver_bootstrap_samples.npz"),
        "script_sha256": sha256(Path(__file__)),
        "rawsaw_trained": False,
        "joint_2x2_run": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_ROOT / "run_receipt.json", receipt)
    print(json.dumps({
        "decision": result["decision"],
        "primary_if20_macro_delta": result["primary_if20_macro_delta"],
        "bootstrap": result["receiver_bootstrap"]["combined_primary_if20"],
        "controls": result["controls"],
        "gates": result["gates"],
        "failed_gates": result["failed_gates"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
