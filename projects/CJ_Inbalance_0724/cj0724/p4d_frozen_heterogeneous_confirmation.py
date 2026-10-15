"""Frozen P4c evaluation on two fresh identity pools without fitting or calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset, read_labels
from cj0724.dual_axis import read_identity_metadata
from cj0724.multitask import PARAMETER_COLUMNS, build_multitask_model
from cj0724.multitask_benchmark import _combination_ids
from cj0724.paper_core_evidence import load_normalizer
from cj0724.tuned_multitask_sweep import _collect_outputs, _compact, _metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _target_key(row: dict[str, object]) -> str:
    return f"{row['class_name']}.{row['name']}"


def paired_comparisons(
    evaluations: list[dict[str, object]],
) -> list[dict[str, object]]:
    indexed = {
        (int(row["data_seed"]), int(row["model_seed"]), str(row["model"])): row
        for row in evaluations
    }
    pairs: list[dict[str, object]] = []
    data_seeds = sorted({key[0] for key in indexed})
    model_seeds = sorted({key[1] for key in indexed})
    for data_seed in data_seeds:
        for model_seed in model_seeds:
            shared = indexed[(data_seed, model_seed, "legacy_mlcnn_global")]
            routed = indexed[(data_seed, model_seed, "mlcnn_iq_grouped_task_routed")]
            shared_metrics = shared["metrics"]
            routed_metrics = routed["metrics"]
            shared_nmae = float(shared_metrics["parameter_macro_nmae"])
            routed_nmae = float(routed_metrics["parameter_macro_nmae"])
            shared_targets = {
                _target_key(row): row for row in shared["per_target"]
            }
            routed_targets = {
                _target_key(row): row for row in routed["per_target"]
            }
            target_deltas = {
                key: float(routed_targets[key]["normalized_mae"])
                - float(shared_targets[key]["normalized_mae"])
                for key in shared_targets
            }
            pairs.append({
                "data_seed": data_seed,
                "model_seed": model_seed,
                "shared_parameter_macro_nmae": shared_nmae,
                "routed_parameter_macro_nmae": routed_nmae,
                "relative_nmae_reduction": (shared_nmae - routed_nmae) / shared_nmae,
                "parameter_macro_pearson_delta": float(routed_metrics["parameter_macro_pearson"])
                - float(shared_metrics["parameter_macro_pearson"]),
                "macro_aupr_delta": float(routed_metrics["macro_aupr"])
                - float(shared_metrics["macro_aupr"]),
                "subset_acc_delta": float(routed_metrics["subset_acc"])
                - float(shared_metrics["subset_acc"]),
                "per_target_nmae_delta": target_deltas,
            })
    return pairs


def audit_promotion(
    pairs: list[dict[str, object]], preregister: dict[str, object]
) -> dict[str, object]:
    gates = preregister["promotion_gates"]
    minimum_pair = float(
        gates["each_of_6_paired_route_parameter_macro_nmae_reduction_at_least"]
    )
    minimum_mean = float(gates["mean_route_parameter_macro_nmae_reduction_at_least"])
    maximum_aupr_drop = float(gates["each_paired_route_macro_aupr_drop_at_most"])
    maximum_subset_drop = float(gates["each_paired_route_subset_accuracy_drop_at_most"])
    target_limit = float(
        gates[
            "no_target_worse_in_all_3_model_seeds_within_either_data_seed_by_more_than_absolute_nmae"
        ]
    )
    reductions = [float(pair["relative_nmae_reduction"]) for pair in pairs]
    pair_gates = {
        "all_6_relative_nmae_reduction_at_least_threshold": all(
            value >= minimum_pair for value in reductions
        ),
        "mean_relative_nmae_reduction_at_least_threshold": float(np.mean(reductions))
        >= minimum_mean,
        "all_6_macro_aupr_drop_within_limit": all(
            float(pair["macro_aupr_delta"]) >= -maximum_aupr_drop for pair in pairs
        ),
        "all_6_subset_drop_within_limit": all(
            float(pair["subset_acc_delta"]) >= -maximum_subset_drop for pair in pairs
        ),
        "all_6_parameter_macro_pearson_nonlower": all(
            float(pair["parameter_macro_pearson_delta"]) >= 0.0 for pair in pairs
        ),
    }
    systematic_failures: list[dict[str, object]] = []
    for data_seed in sorted({int(pair["data_seed"]) for pair in pairs}):
        selected = [pair for pair in pairs if int(pair["data_seed"]) == data_seed]
        targets = sorted(selected[0]["per_target_nmae_delta"])
        for target in targets:
            deltas = [float(pair["per_target_nmae_delta"][target]) for pair in selected]
            if all(delta > target_limit for delta in deltas):
                systematic_failures.append({
                    "data_seed": data_seed,
                    "target": target,
                    "deltas": deltas,
                    "threshold": target_limit,
                })
    pair_gates["no_systematic_target_failure"] = not systematic_failures
    sawfm_replication = []
    for data_seed in sorted({int(pair["data_seed"]) for pair in pairs}):
        deltas = [
            float(pair["per_target_nmae_delta"]["SawFM.sweep_period"])
            for pair in pairs
            if int(pair["data_seed"]) == data_seed
        ]
        sawfm_replication.append({
            "data_seed": data_seed,
            "deltas": deltas,
            "worse_in_3_of_3": all(delta > 0.0 for delta in deltas),
            "mean_delta": float(np.mean(deltas)),
        })
    return {
        "thresholds": {
            "minimum_pair_relative_nmae_reduction": minimum_pair,
            "minimum_mean_relative_nmae_reduction": minimum_mean,
            "maximum_aupr_drop": maximum_aupr_drop,
            "maximum_subset_drop": maximum_subset_drop,
            "systematic_target_absolute_nmae_limit": target_limit,
        },
        "aggregate": {
            "mean_relative_nmae_reduction": float(np.mean(reductions)),
            "minimum_relative_nmae_reduction": float(np.min(reductions)),
            "maximum_relative_nmae_reduction": float(np.max(reductions)),
        },
        "gates": pair_gates,
        "systematic_target_failures": systematic_failures,
        "sawfm_sweep_period_watch": sawfm_replication,
        "promotion_passed": all(pair_gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="outputs/p4c_heterogeneous_route_variable_jnr_20260801/tuned_multitask_20260801_063924",
    )
    parser.add_argument(
        "--preregister",
        default="configs/p4d_frozen_heterogeneous_route_two_data_seed_preregister_20260801.json",
    )
    parser.add_argument("--data-files", nargs=2, required=True)
    parser.add_argument(
        "--excluded-identity-files", nargs="*", default=[]
    )
    parser.add_argument(
        "--output-root", default="outputs/p4d_frozen_heterogeneous_route_two_data_seed_20260801"
    )
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project = Path.cwd()
    run_root = project / args.run_root
    preregister_path = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    normalizer = load_normalizer(protocol["normalizer"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    selected_rows = [
        row for row in summary
        if row["model"] in ("legacy_mlcnn_global", "mlcnn_iq_grouped_task_routed")
        and row.get("checkpoint_objective") == "parameter"
        and int(row["seed"]) in (3041, 3042, 3043)
    ]
    if len(selected_rows) != 6:
        raise RuntimeError(f"Expected six frozen P4c rows, found {len(selected_rows)}")

    excluded_sets: list[set[int]] = []
    for raw_path in args.excluded_identity_files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project / path
        identities, _ = read_identity_metadata(path)
        excluded_sets.append(set(np.asarray(identities, dtype=np.int64).tolist()))

    evaluations: list[dict[str, object]] = []
    data_audits: list[dict[str, object]] = []
    seen_identity_sets: list[set[int]] = []
    for data_file_raw in args.data_files:
        data_file = Path(data_file_raw)
        if not data_file.is_absolute():
            data_file = project / data_file
        identities, labels = read_identity_metadata(data_file)
        identities = np.asarray(identities, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.uint8)
        identity_set = set(identities.tolist())
        data_seed = int(data_file.parent.name.rsplit("seed", 1)[-1].split("_", 1)[0])
        overlap_excluded = sum(len(identity_set & prior) for prior in excluded_sets)
        overlap_other_new = sum(len(identity_set & prior) for prior in seen_identity_sets)
        combo_count = int(np.unique(_combination_ids(labels)).size)
        if identities.size != 6300 or len(identity_set) != 6300:
            raise RuntimeError(f"Expected 6300 unique identities in {data_file}")
        if combo_count != 63 or overlap_excluded or overlap_other_new:
            raise RuntimeError(
                f"Identity/composition audit failed for {data_file}: "
                f"combinations={combo_count}, excluded_overlap={overlap_excluded}, "
                f"new_overlap={overlap_other_new}"
            )
        seen_identity_sets.append(identity_set)
        data_audits.append({
            "data_seed": data_seed,
            "data_file": str(data_file),
            "sample_count": int(identities.size),
            "unique_identity_count": int(len(identity_set)),
            "identity_sha256": identity_hash(identities),
            "combination_count": combo_count,
            "overlap_with_excluded": overlap_excluded,
            "overlap_with_other_new_pool": overlap_other_new,
        })

        dataset = V73IQDataset(data_file, np.arange(identities.size, dtype=np.int64), PARAMETER_COLUMNS)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for row in selected_rows:
            model = build_multitask_model(
                row["model"], protocol["width"], protocol["dropout"],
                protocol["head_dim"], protocol["parameter_hidden_dim"],
                protocol["legacy_attention"],
            ).to(device)
            checkpoint_path = Path(row["checkpoint"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = project / checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            started = time.perf_counter()
            outputs = _collect_outputs(
                model, loader, device, normalizer, float(protocol["parameter_weight"])
            )
            metrics = _metrics(outputs, np.asarray(json.loads(row["thresholds"])), normalizer)
            elapsed = time.perf_counter() - started
            evaluations.append({
                "data_seed": data_seed,
                "model": row["model"],
                "model_seed": int(row["seed"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "thresholds": json.loads(row["thresholds"]),
                "metrics": _compact(metrics),
                "per_target": metrics["parameter"]["per_target"],
                "evaluation_seconds": elapsed,
                "parameter_count": int(row["parameters"]),
            })
            del model, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()

    pairs = paired_comparisons(evaluations)
    promotion = audit_promotion(pairs, preregister)
    payload = {
        "experiment": preregister["experiment"],
        "preregister": str(preregister_path),
        "preregister_sha256": sha256_file(preregister_path),
        "source_run_root": str(run_root),
        "evaluation_contract": {
            "training_on_external_data": False,
            "normalizer_refit": False,
            "threshold_recalibration": False,
            "checkpoint_selection_on_external_data": False,
            "locked_test_used": False,
        },
        "data_audit": data_audits,
        "evaluations": evaluations,
        "paired_comparisons": pairs,
        "promotion_audit": promotion,
        "interpretation": (
            "Frozen heterogeneous route confirmed on both new identity pools."
            if promotion["promotion_passed"]
            else "Frozen heterogeneous route failed at least one preregistered external-confirmation gate."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "data_audit": data_audits,
        "promotion_audit": promotion,
        "output": str(output_root / "summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
