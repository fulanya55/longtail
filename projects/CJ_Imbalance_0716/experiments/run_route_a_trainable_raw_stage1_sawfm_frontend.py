"""Train a small Raw-IQ/Stage1 SawFM frontend with the backbone frozen."""

import argparse
import copy
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

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import _indices_for_identities  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


VARIANTS = {
    "N0_RawOnly": "raw",
    "N1_Stage1Only": "stage1",
    "N2_RawStage1": "raw_stage1",
}


def _extract_sequences(backbone, path, indices, normalization, device, batch_size):
    dataset = V73SignalDataset(
        path=path, indices=np.asarray(indices, dtype=np.int64),
        class_mode="six", normalization=normalization,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = {key: [] for key in ("raw_real", "raw_imag", "stage1_real", "stage1_imag")}
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    maps = backbone.forward_structure_maps(signal)
                    stage1_real = F.avg_pool1d(maps["stage1_real"], 4, 4)
                    stage1_imag = F.avg_pool1d(maps["stage1_imag"], 4, 4)
                output["raw_real"].append(signal[:, 0:1].detach().cpu().to(torch.float16).numpy())
                output["raw_imag"].append(signal[:, 1:2].detach().cpu().to(torch.float16).numpy())
                output["stage1_real"].append(stage1_real.detach().cpu().to(torch.float16).numpy())
                output["stage1_imag"].append(stage1_imag.detach().cpu().to(torch.float16).numpy())
    finally:
        dataset.close()
    return {key: np.concatenate(values, axis=0) for key, values in output.items()}


def _wrap_base_extract(original):
    def augmented(args, config, backbone, base_head, h2_checkpoint, indices, device):
        result = original(args, config, backbone, base_head, h2_checkpoint, indices, device)
        result.update(_extract_sequences(
            backbone, _data_path(config), indices, config["normalization"],
            device, args.feature_batch_size,
        ))
        return result
    return augmented


def _wrap_power_extract(original):
    def augmented(args, config, backbone, base_head, h2_checkpoint, power_path, level_info, identities, device):
        result = original(
            args, config, backbone, base_head, h2_checkpoint,
            power_path, level_info, identities, device,
        )
        indices = _indices_for_identities(level_info, identities)
        result.update(_extract_sequences(
            backbone, power_path, indices, config["normalization"],
            device, args.feature_batch_size,
        ))
        return result
    return augmented


def _make_groups(payloads, role, levels):
    groups = []
    for level in ["root"] + [str(value) for value in levels]:
        positive = payloads[(role, level, "positive")]
        negative = payloads[(role, level, "negative")]
        group = {"name": level}
        for key in ("raw_real", "raw_imag", "stage1_real", "stage1_imag"):
            group[key] = np.concatenate([positive[key], negative[key]], axis=0)
        group["targets"] = np.concatenate([
            np.ones(len(positive["raw_real"]), dtype=np.float32),
            np.zeros(len(negative["raw_real"]), dtype=np.float32),
        ])
        groups.append(group)
    return groups


def _development_groups(payloads, levels):
    groups = []
    for level in levels:
        positive = payloads[("development", str(level), "positive")]
        negative = payloads[("development", str(level), "negative")]
        group = {"name": str(level)}
        for key in ("raw_real", "raw_imag", "stage1_real", "stage1_imag"):
            group[key] = np.concatenate([positive[key], negative[key]], axis=0)
        group["targets"] = np.concatenate([
            np.ones(len(positive["raw_real"]), dtype=np.float32),
            np.zeros(len(negative["raw_real"]), dtype=np.float32),
        ])
        groups.append(group)
    return groups


def _concat_groups(groups):
    result = {}
    for key in ("raw_real", "raw_imag", "stage1_real", "stage1_imag", "targets"):
        result[key] = np.concatenate([group[key] for group in groups], axis=0)
    result["domains"] = np.concatenate([
        np.full(len(group["targets"]), group["name"], dtype=object) for group in groups
    ])
    return result


def _tensor(values, indices, device):
    return torch.from_numpy(values[indices]).to(device=device, dtype=torch.float32)


def _forward(model, arrays, indices, device):
    stage1_real = (
        _tensor(arrays["stage1_real"], indices, device)
        if "stage1_real" in arrays else None
    )
    stage1_imag = (
        _tensor(arrays["stage1_imag"], indices, device)
        if "stage1_imag" in arrays else None
    )
    return model(
        _tensor(arrays["raw_real"], indices, device),
        _tensor(arrays["raw_imag"], indices, device),
        stage1_real,
        stage1_imag,
    )


def _score_model(model, groups, device, batch_size):
    model.eval()
    scores = {}
    with torch.no_grad():
        for group in groups:
            values = []
            for start in range(0, len(group["targets"]), int(batch_size)):
                indices = np.arange(start, min(start + int(batch_size), len(group["targets"])))
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = _forward(model, group, indices, device)
                values.append(logits.cpu().float().numpy())
            scores[group["name"]] = np.concatenate(values)
    return scores


def _evaluate(groups, scores):
    by_level = {}
    for group in groups:
        targets = group["targets"].astype(np.uint8)
        values = scores[group["name"]]
        by_level[group["name"]] = {
            "auroc": float(roc_auc_score(targets, values)),
            "aupr": float(average_precision_score(targets, values)),
            "positive_count": int(targets.sum()),
            "negative_count": int((1 - targets).sum()),
        }
    metrics = list(by_level.values())
    return {
        "by_level": by_level,
        "mean_power_auroc": float(np.mean([item["auroc"] for item in metrics])),
        "worst_power_auroc": float(np.min([item["auroc"] for item in metrics])),
        "mean_power_aupr": float(np.mean([item["aupr"] for item in metrics])),
    }


def _train_variant(variant, mode, fit_groups, validation_groups, seed, device, args):
    train_seed = 491000 + int(seed) + list(VARIANTS).index(variant) * 1000
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = RawStage1SawFMFrontend(
        mode=mode, hidden_channels=args.hidden_channels,
        topk=args.topk, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit = _concat_groups(fit_groups)
    rng = np.random.default_rng(train_seed)
    best_state = None
    best_epoch = 0
    best_key = None
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(len(fit["targets"]))
        total_loss = 0.0
        count = 0
        for start in range(0, len(order), int(args.batch_size)):
            indices = order[start : start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = _forward(model, fit, indices, device)
                targets = _tensor(fit["targets"], indices, device)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            count += len(indices)
        validation_scores = _score_model(model, validation_groups, device, args.batch_size)
        validation = _evaluate(validation_groups, validation_scores)
        history.append({
            "epoch": int(epoch), "train_loss": total_loss / max(count, 1),
            "validation_mean_power_auroc": validation["mean_power_auroc"],
            "validation_worst_power_auroc": validation["worst_power_auroc"],
            "validation_mean_power_aupr": validation["mean_power_aupr"],
        })
        key = (validation["mean_power_auroc"], validation["mean_power_aupr"], -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("No frontend checkpoint was selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history, train_seed


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    replay_args = copy.copy(args)
    replay_args.output_root = args.output_root / "frozen_replay"
    original_base = stable._extract_base_indices
    original_power = stable._extract_power_identities
    try:
        stable._extract_base_indices = _wrap_base_extract(original_base)
        stable._extract_power_identities = _wrap_power_extract(original_power)
        _, audit = selective._run_seed(replay_args, config, labels, identities, seed, device, prior_summary)
    finally:
        stable._extract_base_indices = original_base
        stable._extract_power_identities = original_power
    payloads = selective._PAYLOADS
    fit = _make_groups(payloads, "fit", args.levels)
    validation = [group for group in _make_groups(payloads, "threshold", args.levels) if group["name"] != "root"]
    scale = [group for group in _make_groups(payloads, "scale", args.levels) if group["name"] != "root"]
    development = _development_groups(payloads, args.levels)
    rows = []
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    for variant, mode in VARIANTS.items():
        model, best_epoch, history, train_seed = _train_variant(
            variant, mode, fit, validation, seed, device, args
        )
        validation_metrics = _evaluate(validation, _score_model(model, validation, device, args.batch_size))
        scale_metrics = _evaluate(scale, _score_model(model, scale, device, args.batch_size))
        development_metrics = _evaluate(development, _score_model(model, development, device, args.batch_size))
        counts = {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        }
        row = {
            "seed": int(seed), "variant": variant, "mode": mode,
            "best_epoch": int(best_epoch), "train_seed": int(train_seed),
            "parameter_counts": counts,
            "validation": validation_metrics, "scale_confirmation": scale_metrics,
            "development": development_metrics,
            "validation_development_auroc_gap": float(validation_metrics["mean_power_auroc"] - development_metrics["mean_power_auroc"]),
            "identity_overlap": int(audit["identity_overlap"]), "locked_test_used": False,
        }
        rows.append(row)
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_frontend.pth"
        torch.save({
            "model_state_dict": model.state_dict(), "variant": variant, "mode": mode,
            "best_epoch": best_epoch, "train_seed": train_seed,
            "parameter_counts": counts,
        }, checkpoint_path)
        _write_history(output_dir / "history.csv", history)
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed", "seed": int(seed), "variant": variant,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "f2_checkpoint_sha256": audit["f2_checkpoint_sha256"],
            "identity_overlap": int(audit["identity_overlap"]), "locked_test_used": False,
        })
    _write(seed_root / "source_and_identity_audit.json", audit)
    return rows, audit


def _aggregate(rows):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            role: {
                metric: _mean_ci(
                    [row[role][metric] for row in selected],
                    501000 + variant_index * 100 + role_index * 20 + metric_index,
                )
                for metric_index, metric in enumerate(("mean_power_auroc", "worst_power_auroc", "mean_power_aupr"))
            }
            for role_index, role in enumerate(("validation", "scale_confirmation", "development"))
        }
        result[variant]["validation_development_auroc_gap"] = _mean_ci(
            [row["validation_development_auroc_gap"] for row in selected], 502000 + variant_index
        )
        result[variant]["parameter_count"] = int(selected[0]["parameter_counts"]["total"])
    return result


def _promotion_gate(rows, audits):
    n0 = {row["seed"]: row for row in rows if row["variant"] == "N0_RawOnly"}
    n2 = {row["seed"]: row for row in rows if row["variant"] == "N2_RawStage1"}
    per_seed = []
    complement = []
    for seed, row in sorted(n2.items()):
        development = row["development"]
        checks = {
            "development_mean_auroc_0p90": development["mean_power_auroc"] >= 0.90,
            "development_worst_auroc_0p85": development["worst_power_auroc"] >= 0.85,
            "development_50db_auroc_0p85": development["by_level"]["50"]["auroc"] >= 0.85,
            "scale_mean_auroc_0p90": row["scale_confirmation"]["mean_power_auroc"] >= 0.90,
            "generalization_gap_0p15": row["validation_development_auroc_gap"] <= 0.15,
            "parameters_under_20k": row["parameter_counts"]["total"] < 20000,
        }
        delta = development["mean_power_auroc"] - n0[seed]["development"]["mean_power_auroc"]
        complement.append({"seed": int(seed), "delta_n2_minus_n0": float(delta), "passed": delta >= 0.01})
        per_seed.append({"seed": int(seed), "checks": checks, "passed": all(checks.values())})
    audit_pass = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    frontend_count = int(sum(item["passed"] for item in per_seed))
    complement_count = int(sum(item["passed"] for item in complement))
    return {
        "per_seed": per_seed, "frontend_passed_seed_count": frontend_count,
        "stage1_complement": complement, "stage1_complement_passed_seed_count": complement_count,
        "audit_passed": audit_pass,
        "frontend_passed": frontend_count >= 2 and audit_pass,
        "stage1_complement_passed": complement_count >= 2,
    }


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    prior_summary = _read(args.prior_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device, prior_summary)
        rows.extend(seed_rows); audits.append(audit)
    aggregate = _aggregate(rows)
    gate = _promotion_gate(rows, audits)
    n0_pass = sum(
        row["development"]["mean_power_auroc"] >= 0.90
        and row["development"]["worst_power_auroc"] >= 0.85
        and row["scale_confirmation"]["mean_power_auroc"] >= 0.90
        for row in rows if row["variant"] == "N0_RawOnly"
    ) >= 2
    if gate["frontend_passed"] and gate["stage1_complement_passed"]:
        next_decision = "confirm_trainable_raw_stage1_frontend_on_new_model_seeds"
    elif n0_pass:
        next_decision = "confirm_simpler_trainable_raw_only_frontend_on_new_model_seeds"
    else:
        next_decision = "stop_trainable_early_frontend_oracle_did_not_transfer"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS),
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
            "frozen_f2": True, "connected_to_six_class": False,
        },
        "audits": audits, "per_run": rows, "aggregate": aggregate,
        "promotion_gate": gate, "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_trainable_raw_stage1_sawfm_frontend_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--frozen-visibility-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration")
    parser.add_argument("--prior-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration" / "route_a_stable_expert_visibility_calibration_summary.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_trainable_raw_stage1_sawfm_frontend")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
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
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
