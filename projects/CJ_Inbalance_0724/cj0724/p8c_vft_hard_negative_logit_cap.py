"""P8c: equal-compute random BCE versus VFT difficult-negative logit-cap training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from cj0724.dual_axis import (
    CLASS_NAMES, audit_power_alignment, build_identity_roles, discover_power_train_files,
    read_identity_metadata, roles_receipt,
)
from cj0724.dual_axis_train import metrics_from_probabilities, predict, select_f1_thresholds, set_seed
from cj0724.models import build_benchmark_model
from cj0724.p8a_multi_axis_longtail_atlas import build_manifest, sha256_file
from cj0724.p8b_axis_stratified_longtail import (
    POWERS, VFT, FixedViewDataset, ReplacementBatchSampler, _bootstrap_vft_aupr_delta,
    _json_safe, _loader, _metric_summary,
)


def hard_negative_logit_cap_loss(
    logits: torch.Tensor, labels: torch.Tensor, penalty_weight: float, cap_max_logit: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Ordinary multi-label BCE plus a cap only for the frozen VFT hard-negative relation."""
    base = functional.binary_cross_entropy_with_logits(logits, labels)
    hard = (
        (labels[:, VFT] < 0.5)
        & (labels[:, 2] > 0.5)
        & (labels[:, 3] > 0.5)
        & (labels[:, 5] > 0.5)
    )
    count = int(hard.sum().item())
    if count == 0 or float(penalty_weight) == 0.0:
        return base, logits.new_zeros(()), count
    cap = torch.relu(logits[hard, VFT] - float(cap_max_logit)).square().mean()
    return base + float(penalty_weight) * cap, cap, count


def _train_epoch(
    model: nn.Module, loader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, penalty_weight: float, cap_max_logit: float,
) -> dict:
    model.train(); total_loss = total_bce = total_cap = 0.0; total_samples = total_hard = 0
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, cap, hard_count = hard_negative_logit_cap_loss(model(iq), labels, penalty_weight, cap_max_logit)
            bce = functional.binary_cross_entropy_with_logits(model(iq), labels) if False else loss - float(penalty_weight) * cap
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        n = int(labels.shape[0]); total_samples += n; total_hard += hard_count
        total_loss += float(loss.detach()) * n; total_bce += float(bce.detach()) * n; total_cap += float(cap.detach()) * n
    denominator = max(total_samples, 1)
    return {"loss": total_loss / denominator, "bce": total_bce / denominator, "cap": total_cap / denominator, "hard_rows": total_hard}


def _predict_power(model: nn.Module, files: dict[int, Path], labels: np.ndarray, indices: np.ndarray, level: int, device: torch.device):
    dataset = FixedViewDataset(files, labels, indices, np.full(len(indices), int(level), dtype=np.int16))
    true, probability, weak, sample_indices, _ = predict(model, _loader(dataset, 64), device)
    dataset.close()
    return true, probability, weak, sample_indices


def _train_one(name: str, seed: int, files: dict[int, Path], labels: np.ndarray, train_manifest: dict, roles: dict[str, np.ndarray], cfg: dict, output: Path, device: torch.device) -> dict:
    set_seed(seed)
    selected = np.asarray(train_manifest["selected_indices"], dtype=np.int64)
    levels = np.asarray(train_manifest["view_level_by_selected_position_db"], dtype=np.int16)
    train_dataset = FixedViewDataset(files, labels, selected, levels)
    optimization = cfg["models"]["optimization"]
    sampler = ReplacementBatchSampler(len(selected), int(optimization["batch_size"]), int(optimization["steps_per_epoch"]), seed)
    train_loader = _loader(train_dataset, sampler=sampler)
    early_dataset = FixedViewDataset(files, labels, roles["early_stop"], np.full(len(roles["early_stop"]), 10, dtype=np.int16))
    early_loader = _loader(early_dataset, 64)
    model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimization["lr"]), weight_decay=float(optimization["weight_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_cfg = cfg["models"]["loss"]
    penalty_weight = float(loss_cfg["penalty_weight"]) if name.startswith("C1_") else 0.0
    cap_max_logit = float(loss_cfg["cap_max_logit"])
    best_key, best_epoch, best_state = (-np.inf, -np.inf, -np.inf), 0, None
    history = []
    for epoch in range(1, int(optimization["epochs"]) + 1):
        sampler.set_epoch(epoch)
        train = _train_epoch(model, train_loader, optimizer, scaler, device, penalty_weight, cap_max_logit)
        y, p, weak, _, _ = predict(model, early_loader, device)
        report = metrics_from_probabilities(y, p, np.full(6, 0.5), weak)
        key = (float(report["macro_aupr"]), float(report["per_label_aupr"][VFT]), -epoch)
        history.append({"epoch": epoch, **train, "macro_aupr": key[0], "vft_aupr": key[1]})
        if key > best_key:
            best_key, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    calibration_dataset = FixedViewDataset(files, labels, roles["calibration"], np.full(len(roles["calibration"]), 10, dtype=np.int16))
    cy, cp, _, _, _ = predict(model, _loader(calibration_dataset, 64), device); calibration_dataset.close()
    thresholds = select_f1_thresholds(cy, cp)
    powers = {}; raw = {}
    for power in POWERS:
        y, p, weak, idx = _predict_power(model, files, labels, roles["development"], power, device)
        powers[str(power)] = _metric_summary(y, p, thresholds, weak)
        raw[str(power)] = {"labels": y, "probability": p, "indices": idx}
    checkpoint = output / f"{name}__seed{seed}.pt"
    torch.save({"model_state": model.state_dict(), "thresholds": thresholds, "best_epoch": best_epoch, "history": history}, checkpoint)
    np.savez_compressed(output / f"{name}__seed{seed}_development.npz", **{f"{power}_{key}": value for power, values in raw.items() for key, value in values.items()})
    train_dataset.close(); early_dataset.close()
    return {"name": name, "seed": int(seed), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": int(best_epoch), "thresholds": thresholds.tolist(), "history": history, "powers": powers, "raw_10": raw["10"]}


def _gates(pairs: list[dict]) -> dict:
    rows = [pair["delta"] for pair in pairs]
    non_vft = [all(value <= 0.02 for value in row["non_vft_fpr_delta"]) for row in rows]
    return {
        "all_three_seeds_vft_10db_aupr_nonnegative_vs_C0": all(row["vft_10db_aupr_delta"] >= 0.0 for row in rows),
        "mean_vft_10db_aupr_gain_vs_C0_min": float(np.mean([row["vft_10db_aupr_delta"] for row in rows])) >= 0.01,
        "all_three_seeds_macro_aupr_delta_vs_C0_min": all(row["macro_10db_aupr_delta"] >= -0.005 for row in rows),
        "all_three_seeds_hamming_delta_vs_C0_max": all(row["hamming_10db_delta"] <= 0.005 for row in rows),
        "all_three_seeds_vft_hard_negative_fpr_delta_vs_C0_max": all(row["hard_fpr_10db_delta"] <= 0.0 for row in rows),
        "mean_vft_hard_negative_fpr_delta_vs_C0_max": float(np.mean([row["hard_fpr_10db_delta"] for row in rows])) <= -0.005,
        "all_three_seeds_worst_power_vft_fpr_delta_vs_C0_max": all(row["worst_power_vft_fpr_delta"] <= 0.0 for row in rows),
        "all_three_seeds_vft_separation_delta_vs_C0_min": all(row["vft_separation_10db_delta"] >= 0.0 for row in rows),
        "no_non_vft_label_has_systemic_fpr_increase_over_0p02": all(non_vft),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.preregister.read_text(encoding="utf-8")); freeze = json.loads(args.method_freeze.read_text(encoding="utf-8")); audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    if sha256_file(args.preregister) != freeze["preregister_sha256"] or sha256_file(Path(__file__)) != freeze["runner_sha256"]:
        raise RuntimeError("P8c preregistration or runner hash mismatch")
    if sha256_file(Path(__file__).with_name("audit_p8c_vft_hard_negative_logit_cap.py")) != freeze["audit_runner_sha256"]:
        raise RuntimeError("P8c data-audit runner hash mismatch")
    if audit["preregister_sha256"] != sha256_file(args.preregister) or not audit["valid_for_p8c_training"]:
        raise RuntimeError("P8c valid train-only data audit is required")
    if args.output_root.exists():
        raise FileExistsError(f"Output exists: {args.output_root}")
    files = discover_power_train_files(args.data_root)
    if any("test" in str(path).lower() for path in files.values()):
        raise RuntimeError("P8c permits only train paths")
    if str(args.data_root.resolve()) != audit["data_root"]:
        raise RuntimeError("P8c audit data root mismatch")
    _, labels = read_identity_metadata(files[30])
    roles = build_identity_roles(labels, int(cfg["roles"]["split_seed"]))
    manifest = build_manifest(labels, roles["optimization"], "mixed_lt", cfg["roles"]["training_manifest"]["label_targets"], "combo_lt", 1325000003, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); args.output_root.mkdir(parents=True)
    runs = []
    for seed in cfg["models"]["model_seeds_development"]:
        c0 = _train_one("C0_random_BCE", int(seed), files, labels, manifest, roles, cfg, args.output_root, device)
        c1 = _train_one("C1_random_BCE_plus_vft_hard_negative_logit_cap", int(seed), files, labels, manifest, roles, cfg, args.output_root, device)
        a, b = c0["powers"]["10"], c1["powers"]["10"]
        worst = max(c1["powers"][str(power)]["per_label_fpr"][VFT] - c0["powers"][str(power)]["per_label_fpr"][VFT] for power in POWERS)
        delta = {"vft_10db_aupr_delta": b["per_label_aupr"][VFT] - a["per_label_aupr"][VFT], "macro_10db_aupr_delta": b["macro_aupr"] - a["macro_aupr"], "hamming_10db_delta": b["hamming_loss"] - a["hamming_loss"], "hard_fpr_10db_delta": b["vft_hard_negative_fpr"] - a["vft_hard_negative_fpr"], "vft_separation_10db_delta": b["vft_separation"] - a["vft_separation"], "worst_power_vft_fpr_delta": worst, "non_vft_fpr_delta": [b["per_label_fpr"][i] - a["per_label_fpr"][i] for i in range(6) if i != VFT]}
        bootstrap = _bootstrap_vft_aupr_delta(c0["raw_10"]["labels"], c0["raw_10"]["probability"], c1["raw_10"]["probability"], int(seed) + 500)
        c0.pop("raw_10"); c1.pop("raw_10"); runs.append({"seed": int(seed), "C0": c0, "C1": c1, "delta": delta, "bootstrap_vft_10db_aupr": bootstrap})
    gates = _gates(runs)
    result = {"experiment": cfg["experiment"], "preregister_sha256": sha256_file(args.preregister), "method_freeze_sha256": sha256_file(args.method_freeze), "data_audit_sha256": sha256_file(args.data_audit), "train_only_alignment": audit_power_alignment(files), "roles": roles_receipt(roles, labels, int(cfg["roles"]["split_seed"])), "manifest": manifest, "runs": runs, "gates": gates, "development_passed": bool(all(gates.values())), "verdict": "promote_to_two_model_seed_extension" if all(gates.values()) else "stop_no_confirmation_data", "test_generated": False, "test_opened": False, "locked_test_used": False}
    (args.output_root / "result.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": result["development_passed"], "gates": gates, "verdict": result["verdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
