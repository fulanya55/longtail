"""Confirm the trainable Raw-only SawFM frontend on new split seeds."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _development_pairs, _indices_for_identities, _level_lookup, _stratified_triplets,
)
from experiments.run_route_a_stable_expert_visibility_calibration import _split_three_by_combo  # noqa: E402
from experiments.run_route_a_trainable_raw_stage1_sawfm_frontend import _evaluate, _forward, _score_model, _tensor, _write_history  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


VARIANTS = {
    "R0_FullPhysics": ("full", False),
    "R1_EnvelopeOnly": ("envelope", False),
    "R2_PhaseOnly": ("phase", False),
    "R3_TimeShuffled": ("full", True),
}


def _extract_raw(path, indices, normalization, batch_size):
    dataset = V73SignalDataset(
        path=path, indices=np.asarray(indices, dtype=np.int64),
        class_mode="six", normalization=normalization,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    real, imag = [], []
    try:
        for batch in loader:
            signal = batch["signal"].numpy().astype(np.float16)
            real.append(signal[:, 0:1]); imag.append(signal[:, 1:2])
    finally:
        dataset.close()
    return {"raw_real": np.concatenate(real), "raw_imag": np.concatenate(imag)}


def _paired_group(path, positive_indices, negative_indices, normalization, batch_size, name):
    positive = _extract_raw(path, positive_indices, normalization, batch_size)
    negative = _extract_raw(path, negative_indices, normalization, batch_size)
    return {
        "name": str(name),
        "raw_real": np.concatenate([positive["raw_real"], negative["raw_real"]]),
        "raw_imag": np.concatenate([positive["raw_imag"], negative["raw_imag"]]),
        "targets": np.concatenate([
            np.ones(len(positive_indices), dtype=np.float32),
            np.zeros(len(negative_indices), dtype=np.float32),
        ]),
    }


def _apply_time_permutation(groups, permutation):
    result = []
    for group in groups:
        updated = dict(group)
        updated["raw_real"] = group["raw_real"][..., permutation]
        updated["raw_imag"] = group["raw_imag"][..., permutation]
        result.append(updated)
    return result


def _concat(groups):
    return {
        key: np.concatenate([group[key] for group in groups], axis=0)
        for key in ("raw_real", "raw_imag", "targets")
    }


def _train(variant, descriptor_mode, shuffled, fit_groups, validation_groups, seed, device, args):
    train_seed = 521000 + int(seed) + list(VARIANTS).index(variant) * 1000
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.hidden_channels, topk=args.topk,
        dropout=args.dropout, descriptor_mode=descriptor_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fit = _concat(fit_groups)
    rng = np.random.default_rng(train_seed)
    best_state, best_key, best_epoch, stale = None, None, 0, 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train(); order = rng.permutation(len(fit["targets"])); total = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            indices = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = _forward(model, fit, indices, device)
                loss = F.binary_cross_entropy_with_logits(logits, _tensor(fit["targets"], indices, device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip); optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
        validation = _evaluate(validation_groups, _score_model(model, validation_groups, device, args.batch_size))
        history.append({
            "epoch": epoch, "train_loss": total / len(order),
            "validation_mean_power_auroc": validation["mean_power_auroc"],
            "validation_worst_power_auroc": validation["worst_power_auroc"],
            "validation_mean_power_aupr": validation["mean_power_aupr"],
        })
        key = (validation["mean_power_auroc"], validation["mean_power_aupr"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(args.patience): break
    model.load_state_dict(best_state)
    return model, best_epoch, history, train_seed


def _run_seed(args, config, labels, identities, seed, device):
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    selection = _stratified_triplets(plan["train_indices"], labels, identities, args.pairs_per_combo, 531000 + seed)
    fit_positions, validation_positions, scale_positions = _split_three_by_combo(selection, labels, 532000 + seed)
    development = _development_pairs(plan["early_stop_indices"], labels, identities, 533000 + seed)
    role_identity_sets = {
        "fit": set(selection["positive_identities"][fit_positions].tolist() + selection["hard_negative_identities"][fit_positions].tolist()),
        "validation": set(selection["positive_identities"][validation_positions].tolist() + selection["hard_negative_identities"][validation_positions].tolist()),
        "scale": set(selection["positive_identities"][scale_positions].tolist() + selection["hard_negative_identities"][scale_positions].tolist()),
        "development": set(development["positive_identities"].tolist() + development["hard_negative_identities"].tolist()),
    }
    overlaps = {
        f"{first}_{second}": len(role_identity_sets[first].intersection(role_identity_sets[second]))
        for first_index, first in enumerate(role_identity_sets)
        for second in list(role_identity_sets)[first_index + 1:]
    }
    if any(overlaps.values()): raise ValueError(f"identity overlap: {overlaps}")
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    level_info = _level_lookup(power_files, args.levels)

    def role_groups(positions):
        groups = []
        groups.append(_paired_group(
            _data_path(config), selection["positive_base_indices"][positions],
            selection["hard_negative_base_indices"][positions], config["normalization"],
            args.feature_batch_size, "root",
        ))
        for level in args.levels:
            groups.append(_paired_group(
                power_files[int(level)],
                _indices_for_identities(level_info[int(level)], selection["positive_identities"][positions]),
                _indices_for_identities(level_info[int(level)], selection["hard_negative_identities"][positions]),
                config["normalization"], args.feature_batch_size, str(level),
            ))
        return groups

    fit = role_groups(fit_positions)
    validation = [group for group in role_groups(validation_positions) if group["name"] != "root"]
    scale = [group for group in role_groups(scale_positions) if group["name"] != "root"]
    development_groups = []
    for level in args.levels:
        development_groups.append(_paired_group(
            power_files[int(level)],
            _indices_for_identities(level_info[int(level)], development["positive_identities"]),
            _indices_for_identities(level_info[int(level)], development["hard_negative_identities"]),
            config["normalization"], args.feature_batch_size, str(level),
        ))
    permutation = np.random.default_rng(534000 + seed).permutation(3000)
    rows = []
    seed_root = args.output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
    for variant, (descriptor_mode, shuffled) in VARIANTS.items():
        fit_values, validation_values, scale_values, development_values = fit, validation, scale, development_groups
        if shuffled:
            fit_values = _apply_time_permutation(fit, permutation)
            validation_values = _apply_time_permutation(validation, permutation)
            scale_values = _apply_time_permutation(scale, permutation)
            development_values = _apply_time_permutation(development_groups, permutation)
        model, best_epoch, history, train_seed = _train(
            variant, descriptor_mode, shuffled, fit_values, validation_values, seed, device, args
        )
        validation_metrics = _evaluate(validation_values, _score_model(model, validation_values, device, args.batch_size))
        scale_metrics = _evaluate(scale_values, _score_model(model, scale_values, device, args.batch_size))
        development_metrics = _evaluate(development_values, _score_model(model, development_values, device, args.batch_size))
        row = {
            "seed": seed, "variant": variant, "descriptor_mode": descriptor_mode,
            "time_shuffled": shuffled, "best_epoch": best_epoch, "train_seed": train_seed,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "validation": validation_metrics, "scale_confirmation": scale_metrics, "development": development_metrics,
            "validation_development_auroc_gap": float(validation_metrics["mean_power_auroc"] - development_metrics["mean_power_auroc"]),
            "split_sha256": plan["split_sha256"], "identity_overlaps": overlaps, "locked_test_used": False,
        }
        rows.append(row)
        output = seed_root / variant; output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best_frontend.pth"
        torch.save({"model_state_dict": model.state_dict(), **row}, checkpoint)
        _write_history(output / "history.csv", history); _write(output / "metrics.json", row)
        _write(output / "run_receipt.json", {
            "status": "completed", "seed": seed, "variant": variant,
            "checkpoint_sha256": _sha256(checkpoint), "split_sha256": plan["split_sha256"],
            "identity_overlaps": overlaps, "locked_test_used": False,
        })
    _write(seed_root / "identity_manifest.json", {
        "seed": seed, "split_sha256": plan["split_sha256"], "selection_sha256": selection["selection_sha256"],
        "development_sha256": development["selection_sha256"], "role_counts": {key: len(value) for key, value in role_identity_sets.items()},
        "identity_overlaps": overlaps, "locked_test_used": False,
    })
    return rows


def _aggregate(rows):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            role: {
                metric: _mean_ci(
                    [row[role][metric] for row in selected],
                    541000 + variant_index * 100 + role_index * 20 + metric_index,
                )
                for metric_index, metric in enumerate(("mean_power_auroc", "worst_power_auroc", "mean_power_aupr"))
            }
            for role_index, role in enumerate(("validation", "scale_confirmation", "development"))
        }
        result[variant]["validation_development_auroc_gap"] = _mean_ci(
            [row["validation_development_auroc_gap"] for row in selected], 542000 + variant_index
        )
    return result


def _gate(rows):
    indexed = {(row["seed"], row["variant"]): row for row in rows}
    per_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        full = indexed[(seed, "R0_FullPhysics")]
        shuffled = indexed[(seed, "R3_TimeShuffled")]
        delta = full["development"]["mean_power_auroc"] - shuffled["development"]["mean_power_auroc"]
        checks = {
            "development_mean_auroc_0p95": full["development"]["mean_power_auroc"] >= 0.95,
            "development_worst_auroc_0p90": full["development"]["worst_power_auroc"] >= 0.90,
            "scale_mean_auroc_0p95": full["scale_confirmation"]["mean_power_auroc"] >= 0.95,
            "generalization_gap_0p10": full["validation_development_auroc_gap"] <= 0.10,
            "time_structure_delta_0p05": delta >= 0.05,
        }
        per_seed.append({"seed": seed, "time_structure_delta": float(delta), "checks": checks, "passed": all(checks.values())})
    return {"per_seed": per_seed, "passed_seed_count": int(sum(item["passed"] for item in per_seed)), "passed": all(item["passed"] for item in per_seed)}


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels): raise ValueError("identity labels drifted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds: rows.extend(_run_seed(args, config, labels, identities, int(seed), device))
    aggregate = _aggregate(rows); gate = _gate(rows)
    recognition_passed = all(
        row["development"]["mean_power_auroc"] >= 0.95
        and row["development"]["worst_power_auroc"] >= 0.90
        and row["scale_confirmation"]["mean_power_auroc"] >= 0.95
        for row in rows if row["variant"] == "R0_FullPhysics"
    )
    if gate["passed"]:
        next_decision = "design_bounded_h2_raw_frontend_reconnection"
    elif recognition_passed:
        next_decision = "keep_raw_frontend_but_claim_distribution_not_temporal_mechanism"
    else:
        next_decision = "stop_raw_frontend_new_split_confirmation_failed"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {"seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS), "new_split_confirmation": True},
        "per_run": rows, "aggregate": aggregate, "promotion_gate": gate,
        "recognition_gate_passed": bool(recognition_passed), "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_raw_only_new_split_confirmation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_only_new_split_confirmation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
