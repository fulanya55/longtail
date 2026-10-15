"""Combine the frozen visibility parameter calibrator with residual classification gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import POWER_LEVELS, discover_power_train_files, read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _collect, _saw_prediction
from cj0724.models import SideRoutedPhysicsResidualHead, WeakVisibilityResidualHead
from cj0724.multitask import PARAMETER_COLUMNS, build_multitask_model
from cj0724.paper_core_evidence import load_normalizer
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.power_robustness_calibration import (
    _concat_role,
    apply_parameter_calibrator,
    evaluate_method,
    fit_parameter_calibrator,
    prepare_record,
)
from cj0724.run_physics_parameter_specialists import _positive, _stratified_limit


@torch.no_grad()
def residual_logits(
    model: torch.nn.Module,
    residual_head: WeakVisibilityResidualHead,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    residual_head.eval()
    parts = []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(iq)
            base_logits = output[0] if isinstance(output, tuple) else output
            corrected = residual_head(iq, base_logits)
        parts.append(corrected.float().cpu().numpy())
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--residual-root", default="outputs/weak_visibility_residual_full_20260725")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/weak_visibility_joint_eval_20260725")
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    residual_root = Path(args.residual_root)
    specialist_root = Path(args.specialist_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    residual_protocol = json.loads((residual_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles_npz = np.load(run_root / "role_indices.npz")
    roles = {name: roles_npz[name].astype(np.int64) for name in roles_npz.files}
    files = discover_power_train_files(args.power_root)
    root_seeds, root_labels = read_identity_metadata(protocol["data_file"])
    power_seeds, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_seeds, power_seeds) or not np.array_equal(root_labels, power_labels):
        raise RuntimeError("root and power identities differ")
    normalizer = load_normalizer(protocol["normalizer"])
    mean = normalizer.mean.numpy().reshape(12).astype(np.float64)
    scale = normalizer.scale.numpy().reshape(12).astype(np.float64)
    with h5py.File(protocol["data_file"], "r") as handle:
        root_parameter_values = np.asarray(handle["parameter_values"], dtype=np.float64)
        root_parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)

    selected_rows = [
        row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed"
        and row.get("checkpoint_objective") == "parameter"
    ]
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for selected_row in selected_rows:
        seed = int(selected_row["seed"])
        residual_result_path = residual_root / args.profile / f"seed_{seed}" / "result.json"
        residual_checkpoint_path = residual_root / args.profile / f"seed_{seed}" / "best.pt"
        residual_result = json.loads(residual_result_path.read_text(encoding="utf-8"))
        residual_checkpoint = torch.load(residual_checkpoint_path, map_location=device, weights_only=False)
        model = build_multitask_model(
            selected_row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
            protocol["parameter_hidden_dim"], protocol["legacy_attention"],
        ).to(device)
        model.load_state_dict(
            torch.load(selected_row["checkpoint"], map_location=device, weights_only=False)["model_state"]
        )
        model.eval()
        if residual_checkpoint.get("head") == "side_physics":
            residual_head = SideRoutedPhysicsResidualHead(
                width=8, hidden_dim=64, dropout=0.1,
                residual_bound=float(residual_checkpoint["profile_config"]["residual_bound"]),
            ).to(device)
        else:
            residual_head = WeakVisibilityResidualHead(
                width=12, hidden_dim=64, dropout=0.1,
                residual_bound=float(residual_checkpoint["profile_config"]["residual_bound"]),
            ).to(device)
        residual_head.load_state_dict(residual_checkpoint["model_state"])
        residual_head.eval()
        saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw.load_state_dict(
            torch.load(
                specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt",
                map_location=device,
                weights_only=True,
            )
        )
        saw.eval()
        train_positive = _positive(roles["train"], root_labels, root_parameter_mask, 1, 14)
        saw_train = _stratified_limit(
            train_positive, root_parameter_values[14], 2048, seed + 101, "scalar"
        )
        saw_mean = float(np.mean(root_parameter_values[14, saw_train]))
        saw_scale = float(np.std(root_parameter_values[14, saw_train]))

        records = {}
        for role in ("early", "tune", "confirm"):
            by_level = {}
            for level in POWER_LEVELS:
                dataset = V73IQDataset(files[level], roles[role], PARAMETER_COLUMNS)
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    pin_memory=device.type == "cuda",
                )
                record = _collect(model, loader, device, normalizer)
                record["prediction"][:, 1, 1] = _saw_prediction(
                    saw, loader, device, saw_mean, saw_scale
                )
                corrected_logits = residual_logits(model, residual_head, loader, device)
                record = prepare_record(record, mean, scale, level)
                record["residual_logits"] = corrected_logits
                by_level[level] = record
                dataset.close()
            records[role] = by_level

        combined_early = _concat_role(records["early"])
        base_thresholds = np.asarray(json.loads(selected_row["thresholds"]), dtype=np.float32)
        tune_base_probability = {
            level: 1.0 / (1.0 + np.exp(-np.clip(records["tune"][level]["logits"], -30, 30)))
            for level in POWER_LEVELS
        }
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            calibrators = fit_parameter_calibrator(combined_early, "visibility", alpha)
            for gamma in (0.25, 0.5, 0.75, 1.0):
                tune_prediction = {
                    level: apply_parameter_calibrator(
                        records["tune"][level], calibrators, "visibility", gamma
                    )
                    for level in POWER_LEVELS
                }
                _, metrics = evaluate_method(
                    records["tune"], tune_prediction, tune_base_probability,
                    base_thresholds, mean, scale,
                )
                candidate = (metrics["nine_level_mean_oracle_parameter_nmae"], alpha, gamma, calibrators)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        confirm_prediction = {
            level: apply_parameter_calibrator(
                records["confirm"][level], best[3], "visibility", best[2]
            )
            for level in POWER_LEVELS
        }
        base_probability = {
            level: 1.0 / (1.0 + np.exp(-np.clip(records["confirm"][level]["logits"], -30, 30)))
            for level in POWER_LEVELS
        }
        residual_probability = {
            level: 1.0 / (1.0 + np.exp(-np.clip(records["confirm"][level]["residual_logits"], -30, 30)))
            for level in POWER_LEVELS
        }
        _, base_metrics = evaluate_method(
            records["confirm"], confirm_prediction, base_probability, base_thresholds, mean, scale
        )
        residual_thresholds = np.asarray(residual_result["selected_thresholds"], dtype=np.float32)
        level_rows, residual_metrics = evaluate_method(
            records["confirm"], confirm_prediction, residual_probability,
            residual_thresholds, mean, scale,
        )
        results.append(
            {
                "seed": seed,
                "parameter_alpha": float(best[1]),
                "parameter_gamma": float(best[2]),
                "base": base_metrics,
                "residual_joint": residual_metrics,
                "residual_per_level": level_rows,
            }
        )
        del model, residual_head, saw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {"seed_count": len(results)}
    for method in ("base", "residual_joint"):
        for metric in (
            "nine_level_mean_oracle_parameter_nmae",
            "nine_level_mean_predicted_gate_parameter_nmae",
            "nine_level_mean_miss_penalty_parameter_nmae",
            "weak_recall_10db",
            "weak_recall_50db",
            "nine_level_mean_false_positive_gate_rate",
        ):
            values = [row[method][metric] for row in results]
            aggregate[f"{method}_{metric}_mean"] = float(np.mean(values))
            aggregate[f"{method}_{metric}_std"] = float(np.std(values))
    aggregate["three_seed_miss_nmae_same_direction"] = bool(
        all(
            row["residual_joint"]["nine_level_mean_miss_penalty_parameter_nmae"]
            < row["base"]["nine_level_mean_miss_penalty_parameter_nmae"]
            for row in results
        )
    )
    aggregate["relative_miss_nmae_reduction"] = float(
        1.0
        - aggregate["residual_joint_nine_level_mean_miss_penalty_parameter_nmae_mean"]
        / aggregate["base_nine_level_mean_miss_penalty_parameter_nmae_mean"]
    )
    payload = {
        "protocol": {
            "parameter_calibrator": "visibility residual Ridge; fit early, alpha/gamma selected tune",
            "classification_gate": "balanced bounded IQ-Grouped residual; thresholds frozen from calibration/tune",
            "classification_profile": args.profile,
            "identity_roles": residual_protocol["role_hashes"],
            "locked_test_used": False,
        },
        "aggregate": aggregate,
        "per_seed": results,
    }
    (output_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
