"""Frequency-only Route-A reconstruction with frozen-feature RB-C cRT heads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.v73_dataset import read_labels
from cjlt.losses.classification import DistributionBalancedLoss, MultiLabelClassificationLoss
from cjlt.models.backbone_adapter import build_backbone
from cjlt.models.frequency_reconstruction import (
    FrozenLinearHead,
    build_pair_map,
    classifier_state,
    cpu_state,
    label_balanced_bce,
    normalized_anchor_loss,
    pair_ranking_loss,
)
from experiments.run_route_a_dual_axis_sawfm_r2_formal import (
    core_loss_value,
    dataset_loader,
    predict_core,
    sha256,
    state_sha256,
    write_history,
    write_json,
)


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SEEDS = (260821, 260822, 260823, 260824, 260825)
DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\dual-axis-frequency-development-v2")
VIEW_PATHS = {
    "VH": DATA_ROOT / "sawfm_level_05__jnr_30dB/closedset_6class_train.mat",
    "VL": DATA_ROOT / "sawfm_level_03__jnr_20dB/closedset_6class_train.mat",
}
AUDIT_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage1_data_audit"
SEAL_PATH = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
REGIME_PATHS = {
    "FB": AUDIT_ROOT / "frequency_balanced_manifest.json",
    "LabelIF10": AUDIT_ROOT / "label_if10_manifest.json",
    "LabelIF20": AUDIT_ROOT / "label_if20_manifest.json",
    "LabelIF50": AUDIT_ROOT / "label_if50_manifest.json",
    "ComboIF10": AUDIT_ROOT / "combination_if10_manifest.json",
    "ComboIF20": AUDIT_ROOT / "combination_if20_manifest.json",
    "ComboIF50": AUDIT_ROOT / "combination_if50_manifest.json",
}
PRIMARY = ("FB", "LabelIF20", "ComboIF20")
SECONDARY = ("LabelIF10", "ComboIF10", "LabelIF50", "ComboIF50")
BASE_ARMS = ("F0_B0_Focal", "F1_B0_DB", "F2_Ordinary_cRT", "F3_RBC_cRT")
CONTROL_ARMS = ("F3_no_rank", "F3_no_anchor", "F3_random_pair_rank")


def metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    per_label = {
        name: float(average_precision_score(targets[:, index], logits[:, index]))
        for index, name in enumerate(CLASS_NAMES)
    }
    return {"macro_aupr": float(np.mean(list(per_label.values()))), "per_label_aupr": per_label}


def uniform_sampler(sample_count: int, draws: int, seed: int) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        torch.ones(int(sample_count), dtype=torch.double),
        num_samples=int(draws),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def train_core_pair(path, train_indices, early_indices, initial_state, seed, device, args):
    models = {
        "focal": build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device),
        "db": build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device),
    }
    for model in models.values():
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    train_labels = read_labels(path)[train_indices]
    criteria = {
        "focal": MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0).to(device),
        "db": DistributionBalancedLoss(train_labels.sum(axis=0), len(train_labels)).to(device),
    }
    optimizers = {}
    for name, model in models.items():
        classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
        optimizers[name] = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if id(p) not in classifier_ids], "lr": 5e-5},
                {"params": list(model.classifier.parameters()), "lr": 5e-4},
            ],
            weight_decay=1e-4,
        )
    dataset, loader = dataset_loader(
        path,
        train_indices,
        args.batch_size,
        sampler=uniform_sampler(len(train_indices), args.draws_per_epoch, seed + 73000),
    )
    best_state = {name: None for name in models}
    best_key = {name: None for name in models}
    best_epoch = {name: 0 for name in models}
    stale = {name: 0 for name in models}
    active = {name: True for name in models}
    histories = {name: [] for name in models}
    try:
        for epoch in range(1, args.epochs + 1):
            for name, model in models.items():
                if active[name]:
                    model.train()
            losses = {name: [] for name in models}
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                for name, model in models.items():
                    if not active[name]:
                        continue
                    optimizers[name].zero_grad(set_to_none=True)
                    loss = core_loss_value(criteria[name], model(signal), targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite {name} core loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"non-finite {name} core gradient")
                    optimizers[name].step()
                    losses[name].append(float(loss.detach().cpu()))
            for name, model in models.items():
                if not active[name]:
                    continue
                early = predict_core(model, path, early_indices, device, args.eval_batch_size)
                current = metrics(early["targets"], early["logits"])
                key = (current["macro_aupr"], current["per_label_aupr"]["SawFM"])
                histories[name].append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses[name])),
                        "early_macro_aupr": key[0],
                        "early_sawfm_aupr": key[1],
                        "draws_per_epoch": int(args.draws_per_epoch),
                        "backbone_lr": 5e-5,
                        "classifier_lr": 5e-4,
                    }
                )
                if best_key[name] is None or key > best_key[name]:
                    best_key[name] = key
                    best_epoch[name] = epoch
                    best_state[name] = cpu_state(model)
                    stale[name] = 0
                else:
                    stale[name] += 1
                    if stale[name] >= args.patience:
                        active[name] = False
            if not any(active.values()):
                break
    finally:
        dataset.close()
    for name, model in models.items():
        if best_state[name] is None:
            raise RuntimeError(f"{name} core did not produce a checkpoint")
        model.load_state_dict(best_state[name], strict=True)
    return models, best_state, best_epoch, histories


def extract_features(model, path, indices, device, batch_size):
    dataset, loader = dataset_loader(path, indices, batch_size)
    features, targets, sample_ids = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                _, current = model(signal, return_features=True)
                features.append(current.detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "features": np.concatenate(features),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def train_head(name, train, early, anchor, seed, device, args, pair_maps):
    anchor_weight, anchor_bias = anchor
    head = FrozenLinearHead(anchor_weight, anchor_bias).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    train_features = torch.as_tensor(train["features"], device=device)
    train_targets = torch.as_tensor(train["targets"], dtype=torch.float32, device=device)
    early_features = torch.as_tensor(early["features"], device=device)
    positive_counts = train_targets.sum(dim=0)
    hard_map = torch.as_tensor(pair_maps["hard"], dtype=torch.long, device=device)
    random_map = torch.as_tensor(pair_maps["random"], dtype=torch.long, device=device)
    use_rank = name in ("F3_RBC_cRT", "F3_no_anchor", "F3_random_pair_rank")
    use_anchor = name in ("F3_RBC_cRT", "F3_no_rank", "F3_random_pair_rank")
    pair_map = random_map if name == "F3_random_pair_rank" else hard_map
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.RandomState(int(seed) + 180000)
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        order = rng.choice(len(train_features), size=args.draws_per_epoch, replace=True)
        epoch_losses = []
        for start in range(0, len(order), args.head_batch_size):
            positions = torch.as_tensor(order[start : start + args.head_batch_size], dtype=torch.long, device=device)
            targets = train_targets[positions]
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_features[positions])
            balanced = label_balanced_bce(logits, targets, positive_counts, len(train_features))
            rank = pair_ranking_loss(head, train_features, positions, targets, pair_map, 0.2) if use_rank else balanced * 0.0
            anchor_loss = normalized_anchor_loss(
                head, anchor_weight.to(device), anchor_bias.to(device)
            ) if use_anchor else balanced * 0.0
            loss = balanced + 0.25 * rank + 1e-3 * anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {name} loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite {name} gradient")
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.no_grad():
            early_logits = head(early_features).detach().cpu().numpy()
        current = metrics(early["targets"], early_logits)
        key = (current["macro_aupr"], current["per_label_aupr"]["SawFM"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "early_macro_aupr": key[0],
                "early_sawfm_aupr": key[1],
                "rank_weight": 0.25 if use_rank else 0.0,
                "anchor_weight": 1e-3 if use_anchor else 0.0,
                "pair_mode": "random" if name == "F3_random_pair_rank" else "hard" if use_rank else "none",
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = cpu_state(head)
            stale = 0
        else:
            stale += 1
            if stale >= args.head_patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    head.load_state_dict(best_state, strict=True)
    return head, best_state, best_epoch, history


def predict_head(head, feature_payload, device):
    head.eval()
    with torch.no_grad():
        logits = head(torch.as_tensor(feature_payload["features"], device=device)).detach().cpu().numpy()
    return logits


def validate_sources():
    audit = json.loads((AUDIT_ROOT / "data_audit.json").read_text(encoding="utf-8"))
    if not audit["checks"]["stage2_smoke_admitted"]:
        raise ValueError("Stage 1 did not admit modeling")
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if not (
        seal["status"] == "sealed"
        and seal["test_v3"]["access_authorized"] is False
        and seal["test_v3"]["analysis_allowed"] is False
        and seal["test_v3"]["used"] is False
    ):
        raise ValueError("test-v3 seal is not intact")
    hashes = {}
    for view, path in VIEW_PATHS.items():
        current = sha256(path)
        expected = audit["files"]["high_visibility" if view == "VH" else "low_visibility"]["sha256"]
        if current != expected:
            raise ValueError(f"{view} source hash drift")
        hashes[view] = current
    return audit, hashes


def arms_for_regime(regime: str) -> tuple[str, ...]:
    return BASE_ARMS + CONTROL_ARMS if regime in PRIMARY else BASE_ARMS


def run_seed_regime(args, regime, seed, roles, device, source_hashes):
    output = args.output_root / f"seed_{seed}" / regime
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "run_receipt.json"
    summary_path = output / "summary.json"
    manifest_path = REGIME_PATHS[regime]
    script_hash = sha256(Path(__file__))
    if args.resume and receipt_path.is_file() and summary_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "completed" and receipt.get("script_sha256") == script_hash:
            print(json.dumps({"seed": seed, "regime": regime, "status": "resumed"}), flush=True)
            return json.loads(summary_path.read_text(encoding="utf-8"))
    started = time.time()
    base_receipt = {
        "seed": int(seed),
        "regime": regime,
        "script_sha256": script_hash,
        "manifest_sha256": sha256(manifest_path),
        "source_hashes": source_hashes,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_indices = np.asarray(manifest["selected_indices"], dtype=np.int64)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        template = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
        initial_state = copy.deepcopy(template.state_dict())
        initial_hash = state_sha256(initial_state)
        models, states, best_epochs, histories = train_core_pair(
            VIEW_PATHS["VH"], train_indices, roles["early"], initial_state, seed, device, args
        )
        focal = models["focal"]
        db = models["db"]
        focal_path = output / "F0_B0_Focal.pt"
        db_path = output / "F1_B0_DB.pt"
        torch.save({"model_state_dict": states["focal"], "best_epoch": best_epochs["focal"], "initial_state_sha256": initial_hash}, focal_path)
        torch.save({"model_state_dict": states["db"], "best_epoch": best_epochs["db"], "initial_state_sha256": initial_hash}, db_path)
        write_history(output / "F0_B0_Focal_history.csv", histories["focal"])
        write_history(output / "F1_B0_DB_history.csv", histories["db"])

        train_features = extract_features(focal, VIEW_PATHS["VH"], train_indices, device, args.eval_batch_size)
        early_features = extract_features(focal, VIEW_PATHS["VH"], roles["early"], device, args.eval_batch_size)
        evaluation_features = {
            view: extract_features(focal, path, roles["evaluation"], device, args.eval_batch_size)
            for view, path in VIEW_PATHS.items()
        }
        pair_maps = {
            "hard": build_pair_map(train_features["targets"], seed + 210000, "hard"),
            "random": build_pair_map(train_features["targets"], seed + 220000, "random"),
        }
        anchor = classifier_state(focal)
        heads = {}
        head_paths = {}
        head_histories = {}
        for arm in arms_for_regime(regime):
            if arm in ("F0_B0_Focal", "F1_B0_DB"):
                continue
            head, state, epoch, history = train_head(
                arm, train_features, early_features, anchor, seed, device, args, pair_maps
            )
            heads[arm] = head
            path = output / f"{arm}.pt"
            torch.save({"model_state_dict": state, "best_epoch": epoch, "focal_anchor_sha256": sha256(focal_path)}, path)
            write_history(output / f"{arm}_history.csv", history)
            head_paths[arm] = {"path": str(path), "sha256": sha256(path), "best_epoch": epoch}
            head_histories[arm] = history

        predictions: dict[str, np.ndarray] = {}
        results = {arm: {"by_view": {}} for arm in arms_for_regime(regime)}
        for view in ("VH", "VL"):
            focal_result = predict_core(focal, VIEW_PATHS[view], roles["evaluation"], device, args.eval_batch_size)
            db_result = predict_core(db, VIEW_PATHS[view], roles["evaluation"], device, args.eval_batch_size)
            if not np.array_equal(focal_result["sample_ids"], evaluation_features[view]["sample_ids"]):
                raise AssertionError("evaluation feature identity alignment failed")
            logits_by_arm = {
                "F0_B0_Focal": focal_result["logits"],
                "F1_B0_DB": db_result["logits"],
            }
            logits_by_arm.update({arm: predict_head(head, evaluation_features[view], device) for arm, head in heads.items()})
            for arm, logits in logits_by_arm.items():
                results[arm]["by_view"][view] = metrics(focal_result["targets"], logits)
                predictions[f"{arm}_{view}_logits"] = logits.astype(np.float32)
        np.savez_compressed(
            output / "evaluation_predictions.npz",
            sample_ids=evaluation_features["VH"]["sample_ids"],
            targets=evaluation_features["VH"]["targets"],
            **predictions,
        )
        train_counts = train_features["targets"].sum(axis=0).astype(int)
        summary = {
            "status": "completed",
            "evidence_level": "smoke_only_not_performance_evidence" if args.mode == "smoke" else "five_seed_new_identity_development",
            "seed": int(seed),
            "regime": regime,
            "train_identity_count": int(len(train_indices)),
            "train_label_counts": dict(zip(CLASS_NAMES, train_counts.tolist())),
            "three_least_frequent_labels": [CLASS_NAMES[index] for index in np.argsort(train_counts)[:3]],
            "draws_per_epoch": int(args.draws_per_epoch),
            "core_initial_state_sha256": initial_hash,
            "best_epochs": {"F0_B0_Focal": best_epochs["focal"], "F1_B0_DB": best_epochs["db"], **{name: value["best_epoch"] for name, value in head_paths.items()}},
            "parameter_counts": {
                "B0": int(sum(parameter.numel() for parameter in focal.parameters())),
                "cRT_head": int(sum(parameter.numel() for parameter in next(iter(heads.values())).parameters())),
            },
            "checkpoints": {
                "F0_B0_Focal": {"path": str(focal_path), "sha256": sha256(focal_path)},
                "F1_B0_DB": {"path": str(db_path), "sha256": sha256(db_path)},
                **head_paths,
            },
            "pair_map_sha256": {
                name: hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()
                for name, value in pair_maps.items()
            },
            "results": results,
            "predictions_sha256": sha256(output / "evaluation_predictions.npz"),
            "elapsed_seconds": float(time.time() - started),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        write_json(summary_path, summary)
        write_json(receipt_path, {**base_receipt, "status": "completed", "summary_sha256": sha256(summary_path), "elapsed_seconds": summary["elapsed_seconds"]})
        print(json.dumps({"seed": seed, "regime": regime, "status": "completed", "best_epochs": summary["best_epochs"], "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False), flush=True)
        return summary
    except Exception as exception:
        write_json(receipt_path, {**base_receipt, "status": "aborted_engineering_failure", "error_type": type(exception).__name__, "error": str(exception), "traceback": traceback.format_exc(), "elapsed_seconds": float(time.time() - started)})
        raise


def formal_protocol(args) -> bool:
    return bool(
        args.mode == "formal"
        and tuple(args.seeds) == SEEDS
        and tuple(args.regimes) == PRIMARY + SECONDARY
        and args.epochs == 36
        and args.patience == 8
        and args.batch_size == 64
        and args.draws_per_epoch == 2400
        and args.head_epochs == 100
        and args.head_patience == 15
        and args.head_batch_size == 256
    )


def run(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    terminal = args.output_root / "terminal_receipt.json"
    started = time.time()
    try:
        audit, source_hashes = validate_sources()
        if args.mode == "formal" and not formal_protocol(args):
            raise ValueError("formal arguments do not match the frozen preregistration")
        labels = read_labels(VIEW_PATHS["VH"])
        if not np.array_equal(labels, read_labels(VIEW_PATHS["VL"])):
            raise ValueError("paired labels drifted")
        saved = np.load(AUDIT_ROOT / "role_indices.npz")
        roles = {name: saved[name].astype(np.int64) for name in ("early", "calibration", "threshold", "evaluation")}
        if args.mode == "smoke":
            rng = np.random.RandomState(26082000)
            for name in roles:
                values = roles[name].copy()
                rng.shuffle(values)
                roles[name] = values[: args.smoke_role_count]
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("frequency reconstruction requires CUDA")
        config = {
            "status": "frozen" if args.mode == "formal" else "smoke",
            "mode": args.mode,
            "seeds": list(args.seeds),
            "regimes": list(args.regimes),
            "primary": list(PRIMARY),
            "secondary": list(SECONDARY),
            "arms": {regime: list(arms_for_regime(regime)) for regime in args.regimes},
            "core": {"model": "dfms_iq", "width": 16, "feature_dim": 128, "dropout": 0.2, "epochs": args.epochs, "patience": args.patience, "draws_per_epoch": args.draws_per_epoch, "batch_size": args.batch_size, "optimizer": "AdamW", "backbone_lr": 5e-5, "classifier_lr": 5e-4, "weight_decay": 1e-4, "gradient_clip": 5.0},
            "head": {"epochs": args.head_epochs, "patience": args.head_patience, "batch_size": args.head_batch_size, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "rank_weight": 0.25, "rank_margin": 0.2, "anchor_weight": 1e-3},
            "selection": "VH early macro-AUPR, SawFM AUPR tie-break",
            "evaluation": "one pass per preregistered arm on new evaluation role; hard gate uses VH raw-logit AUPR",
            "source_hashes": source_hashes,
            "stage1_audit_sha256": sha256(AUDIT_ROOT / "data_audit.json"),
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        write_json(args.output_root / "resolved_config.json", config)
        rows = []
        for seed in args.seeds:
            for regime in args.regimes:
                rows.append(run_seed_regime(args, regime, int(seed), roles, device, source_hashes))
        index = [{"seed": row["seed"], "regime": row["regime"], "summary": str(args.output_root / f"seed_{row['seed']}" / row["regime"] / "summary.json"), "status": row["status"]} for row in rows]
        write_json(args.output_root / "run_index.json", {"status": "completed", "mode": args.mode, "runs": index, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
        receipt = {"status": "completed", "decision": "admit_formal_analysis" if args.mode == "formal" else "admit_formal_training", "run_count": len(rows), "elapsed_seconds": float(time.time() - started), "script_sha256": sha256(Path(__file__)), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
        write_json(terminal, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return rows
    except Exception as exception:
        write_json(terminal, {"status": "aborted_engineering_failure", "error_type": type(exception).__name__, "error": str(exception), "traceback": traceback.format_exc(), "elapsed_seconds": float(time.time() - started), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--draws-per-epoch", type=int, default=None)
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument("--head-patience", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--smoke-role-count", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.mode == "formal":
        args.output_root = args.output_root or ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_formal"
        args.seeds = args.seeds or list(SEEDS)
        args.regimes = args.regimes or list(PRIMARY + SECONDARY)
        args.epochs = 36 if args.epochs is None else args.epochs
        args.patience = 8 if args.patience is None else args.patience
        args.draws_per_epoch = 2400 if args.draws_per_epoch is None else args.draws_per_epoch
        args.head_epochs = 100 if args.head_epochs is None else args.head_epochs
        args.head_patience = 15 if args.head_patience is None else args.head_patience
    else:
        args.output_root = args.output_root or ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_smoke"
        args.seeds = args.seeds or [260821]
        args.regimes = args.regimes or ["LabelIF20"]
        args.epochs = 2 if args.epochs is None else args.epochs
        args.patience = 2 if args.patience is None else args.patience
        args.draws_per_epoch = 256 if args.draws_per_epoch is None else args.draws_per_epoch
        args.head_epochs = 3 if args.head_epochs is None else args.head_epochs
        args.head_patience = 2 if args.head_patience is None else args.head_patience
    unknown = set(args.regimes) - set(REGIME_PATHS)
    if unknown:
        parser.error(f"unknown regimes: {sorted(unknown)}")
    return args


if __name__ == "__main__":
    run(parse_args())
