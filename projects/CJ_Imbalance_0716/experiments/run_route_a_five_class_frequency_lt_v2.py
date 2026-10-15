"""Route-A V2: five-class frequency long-tail verification.

SawFM-positive identities are removed before any split.  The script keeps
training/early/calibration/threshold/test roles separate and implements the
V2 goal in resumable stages so every intermediate result is auditable.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.v73_dataset import V73SignalDataset
from cjlt.losses.classification import build_classification_loss
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters
from cjlt.models.frequency_reconstruction import build_pair_map, classifier_state, label_balanced_bce


DATA_ROOT = Path("E:/LJQ_code") / "".join(map(chr, (0x5E72, 0x6270, 0x6570, 0x636E))) / "".join(map(chr, (0x590D, 0x5408, 0x5E72, 0x6270, 0x6570, 0x636E, 0x30, 0x37, 0x31, 0x36)))
TRAIN_PATH = DATA_ROOT / "closedset_6class_train.mat"
TEST_PATH = DATA_ROOT / "closedset_6class_test.mat"
OUT_ROOT = ROOT / "outputs" / "route_a_five_class_frequency_lt_v2"
AUDIT_ROOT = DATA_ROOT / "route_a_five_class_frequency_lt_v2"
CLASS_NAMES = ("FMB", "SinFM", "MFT", "VFT", "ISRJ")
SOURCE_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SEEDS = (272001, 272002, 272003, 272004, 272005)
STAGE1_ARMS = ("Uniform-Focal", "MaxPositive-Focal", "MeanPositive-Focal")
STAGE2_LOSSES = (
    ("Focal", {"kind": "focal"}),
    ("DB", {"kind": "db"}),
    ("CB_BCE_beta0.99", {"kind": "cb_bce", "beta": 0.99}),
    ("CB_BCE_beta0.999", {"kind": "cb_bce", "beta": 0.999}),
    ("ASLDB_w0.50", {"kind": "asl_db_mix", "db_weight": 0.50}),
    ("ASLDB_w0.75", {"kind": "asl_db_mix", "db_weight": 0.75}),
    ("ASLDB_w0.90", {"kind": "asl_db_mix", "db_weight": 0.90}),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)),
        encoding="utf-8",
    )


def read_labels6(path: Path) -> np.ndarray:
    with h5py.File(str(path), "r") as handle:
        return np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T


def filter_five(labels6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    retained = np.flatnonzero(labels6[:, 1] == 0).astype(np.int64)
    labels5 = labels6[:, [0, 2, 3, 4, 5]].astype(np.uint8, copy=False)
    if not np.all(labels5[retained].sum(axis=1) > 0):
        raise ValueError("SawFM-filtered set contains empty five-class labels")
    return retained, labels5


def combo_key(row: np.ndarray) -> str:
    return "".join(str(int(v)) for v in row.tolist())


def combo_counts(labels5: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx in indices:
        key = combo_key(labels5[int(idx)])
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def split_roles(indices: np.ndarray, labels5: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Deterministic combo-stratified 60/20/10/10 role split."""
    rng = np.random.RandomState(int(seed))
    roles = {name: [] for name in ("train", "early", "calibration", "threshold")}
    groups = sorted({combo_key(labels5[int(i)]) for i in indices})
    for key in groups:
        group = indices[np.asarray([combo_key(labels5[int(i)]) == key for i in indices])].copy()
        rng.shuffle(group)
        n = len(group)
        cuts = [int(round(n * 0.60)), int(round(n * 0.80)), int(round(n * 0.90))]
        cuts[0] = max(1, min(cuts[0], n - 3))
        cuts[1] = max(cuts[0] + 1, min(cuts[1], n - 2))
        cuts[2] = max(cuts[1] + 1, min(cuts[2], n - 1))
        parts = (group[: cuts[0]], group[cuts[0] : cuts[1]], group[cuts[1] : cuts[2]], group[cuts[2] :])
        for name, part in zip(roles, parts):
            roles[name].extend(int(v) for v in part)
    result = {name: np.asarray(values, dtype=np.int64) for name, values in roles.items()}
    for name in result:
        rng.shuffle(result[name])
    return result


def ensure_stage0() -> dict[str, Any]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    train6, test6 = read_labels6(TRAIN_PATH), read_labels6(TEST_PATH)
    train_valid, train5 = filter_five(train6)
    test_valid, test5 = filter_five(test6)
    if set(train_valid.tolist()) & set(test_valid.tolist()):
        # Source files have independent index namespaces; overlap is expected
        # numerically, so only compare within each file and record this fact.
        index_overlap_note = "train/test are independent MAT files; numeric indices are not shared identities"
    else:
        index_overlap_note = "no numeric overlap"
    manifest_target = np.asarray([4000, 3000, 2000, 1500, 1000], dtype=np.int64)
    manifest = build_label_longtail(
        train5,
        train_valid,
        target_label_counts=manifest_target,
        seed=272000,
        tolerance=0.05,
        source_split_sha256=sha256_file(TRAIN_PATH),
    )
    manifest_payload = manifest.to_dict()
    manifest_payload.update({
        "protocol_version": "route_a_five_class_frequency_lt_v2",
        "source_file": str(TRAIN_PATH),
        "source_sha256": sha256_file(TRAIN_PATH),
        "class_names": list(CLASS_NAMES),
        "source_label_order": list(SOURCE_NAMES),
        "selected_source_indices": [0, 2, 3, 4, 5],
        "excluded_class": "SawFM",
        "excluded_positive_samples": True,
        "power_longtail": False,
        "power_view_dirs_used": [],
        "requested_label_if": 4.0,
        "actual_label_if": float(manifest.achieved_label_counts.max() / manifest.achieved_label_counts.min()),
        "generated_at_unix": time.time(),
    })
    dump_json(AUDIT_ROOT / "label_if4_manifest.json", manifest_payload)
    audit = {
        "protocol_version": "route_a_five_class_frequency_lt_v2",
        "train_file": str(TRAIN_PATH), "test_file": str(TEST_PATH),
        "train_sha256": sha256_file(TRAIN_PATH), "test_sha256": sha256_file(TEST_PATH),
        "source_label_order": list(SOURCE_NAMES), "class_names": list(CLASS_NAMES),
        "excluded_class": "SawFM", "excluded_positive_samples": True,
        "train_source_count": int(len(train6)), "train_retained_count": int(len(train_valid)),
        "test_source_count": int(len(test6)), "test_retained_count": int(len(test_valid)),
        "train_counts_before_filter": train6.sum(axis=0).astype(int).tolist(),
        "train_counts_after_filter": train5[train_valid].sum(axis=0).astype(int).tolist(),
        "test_counts_after_filter": test5[test_valid].sum(axis=0).astype(int).tolist(),
        "balanced_combo_counts": combo_counts(train5, train_valid),
        "label_if4_combo_counts": combo_counts(train5, manifest.selected_indices),
        "label_if4_target_counts": manifest_target.astype(int).tolist(),
        "label_if4_achieved_counts": train5[manifest.selected_indices].sum(axis=0).astype(int).tolist(),
        "train_retained_indices_sha256": hashlib.sha256(np.asarray(train_valid, dtype="<i8").tobytes()).hexdigest(),
        "test_retained_indices_sha256": hashlib.sha256(np.asarray(test_valid, dtype="<i8").tobytes()).hexdigest(),
        "index_overlap_note": index_overlap_note,
        "power_longtail": False, "power_view_dirs_used": [],
        "generated_at_unix": time.time(),
    }
    dump_json(AUDIT_ROOT / "stage0_audit.json", audit)
    np.save(AUDIT_ROOT / "train_retained_indices.npy", train_valid)
    np.save(AUDIT_ROOT / "test_retained_indices.npy", test_valid)
    return {"train5": train5, "test5": test5, "train_valid": train_valid, "test_valid": test_valid, "manifest_indices": np.asarray(manifest.selected_indices, dtype=np.int64), "audit": audit}


def make_loader(path: Path, indices: np.ndarray, labels5: np.ndarray, role: str, sampler_kind: str, batch_size: int, draws: int | None, seed: int):
    dataset = V73SignalDataset(path, indices, class_mode="five", normalization="complex_rms")
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if sampler_kind != "none" and draws is not None:
        role_labels = labels5[indices]
        counts = role_labels.sum(axis=0).astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        if sampler_kind == "uniform":
            weights = np.ones(len(indices), dtype=np.float64)
        else:
            per = role_labels * inv[None, :]
            if sampler_kind == "max_positive":
                weights = np.max(np.where(role_labels > 0, per, 0.0), axis=1)
            elif sampler_kind == "mean_positive":
                weights = np.sum(per, axis=1) / np.maximum(role_labels.sum(axis=1), 1)
            else:
                raise ValueError(f"unknown sampler {sampler_kind}")
            weights = np.maximum(weights, 1e-12)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=int(draws), replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return dataset, loader


def metric_payload(targets: np.ndarray, logits: np.ndarray, thresholds: np.ndarray | None = None) -> dict[str, Any]:
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    if thresholds is None:
        thresholds = np.full(logits.shape[1], 0.5, dtype=np.float64)
    per = {}
    for i, name in enumerate(CLASS_NAMES):
        per[name] = {
            "aupr": float(average_precision_score(targets[:, i], prob[:, i])),
            "auroc": float(roc_auc_score(targets[:, i], prob[:, i])),
            "f1": float(f1_score(targets[:, i], prob[:, i] >= thresholds[i], zero_division=0)),
            "precision_threshold": float(thresholds[i]),
            "positive_count": int(targets[:, i].sum()),
        }
    values = list(per.values())
    return {
        "macro_aupr": float(np.mean([v["aupr"] for v in values])),
        "macro_auroc": float(np.mean([v["auroc"] for v in values])),
        "macro_f1": float(np.mean([v["f1"] for v in values])),
        "head_mean_aupr": float(np.mean([per[n]["aupr"] for n in ("FMB", "SinFM")])),
        "tail_mean_aupr": float(np.mean([per[n]["aupr"] for n in ("VFT", "ISRJ")])),
        "per_label": per,
    }


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))


def build_criterion(kind: str, train_labels: np.ndarray, kwargs: dict[str, Any]):
    counts = train_labels.sum(axis=0).astype(int).tolist()
    if kind == "db":
        return build_classification_loss("db", positive_counts=counts, sample_count=len(train_labels))
    if kind == "cb_bce":
        return build_classification_loss("cb_bce", positive_counts=counts, beta=float(kwargs["beta"]))
    if kind == "asl_db_mix":
        return build_classification_loss("asl_db_mix", positive_counts=counts, sample_count=len(train_labels), db_mix_weight=float(kwargs["db_weight"]), gamma_pos=0.0, gamma_neg=3.0, clip=0.05)
    return build_classification_loss("focal", gamma_pos=0.0, gamma_neg=3.0)


def predict_role(model, path: Path, indices: np.ndarray, labels5: np.ndarray, batch_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    dataset, loader = make_loader(path, indices, labels5, "eval", "none", batch_size, None, seed)
    logits, targets = [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                out = model(batch["signal"].to(DEVICE, non_blocking=True))
                logits.append(out.detach().cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    return np.concatenate(logits), np.concatenate(targets).astype(np.uint8)


def train_joint(arm: str, seed: int, roles: dict[str, np.ndarray], data: dict[str, Any], args: argparse.Namespace, sampler_kind: str, loss_kind: str, loss_kwargs: dict[str, Any], stage: str) -> dict[str, Any]:
    set_seed(seed)
    model = build_backbone("dfms_iq", num_classes=5, width=16, dropout=0.2).to(DEVICE)
    criterion = build_criterion(loss_kind, data["train5"][roles["train"]], loss_kwargs).to(DEVICE)
    classifier_ids = {id(p) for p in model.classifier.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in classifier_ids], "lr": 5e-5},
        {"params": list(model.classifier.parameters()), "lr": 5e-4},
    ], weight_decay=1e-4)
    draws = int(args.draws_per_epoch)
    dataset, train_loader = make_loader(TRAIN_PATH, roles["train"], data["train5"], "train", sampler_kind, args.batch_size, draws, seed + 17)
    best_state, best_key, best_epoch, stale = None, (-np.inf, -np.inf), 0, 0
    history = []
    started = time.time()
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train(); losses = []; grad_norms = []
            for batch in train_loader:
                signal = batch["signal"].to(DEVICE, non_blocking=True)
                targets = batch["label"].to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(signal), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {arm} seed={seed}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu())); grad_norms.append(float(grad_norm.detach().cpu()))
            early_logits, early_targets = predict_role(model, TRAIN_PATH, roles["early"], data["train5"], args.eval_batch_size, seed + epoch)
            early_metric = metric_payload(early_targets, early_logits)
            key = (early_metric["macro_aupr"], early_metric["tail_mean_aupr"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "mean_grad_norm": float(np.mean(grad_norms)), "early": early_metric, "elapsed_seconds": time.time() - started})
            if key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(args.patience):
                    break
    finally:
        dataset.close()
    if best_state is None:
        raise RuntimeError(f"no checkpoint produced for {arm} seed={seed}")
    model.load_state_dict(best_state, strict=True)
    out = OUT_ROOT / stage / f"seed_{seed}" / arm.replace("/", "_").replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    dump_json(out / "history.json", history)
    role_outputs = {}
    for role_name, path, idx in (("early", TRAIN_PATH, roles["early"]), ("calibration", TRAIN_PATH, roles["calibration"]), ("threshold", TRAIN_PATH, roles["threshold"])):
        lg, tg = predict_role(model, path, idx, data["train5"], args.eval_batch_size, seed + 1000 + len(role_name))
        np.savez_compressed(out / f"{role_name}_predictions.npz", logits=lg, targets=tg)
        role_outputs[role_name] = metric_payload(tg, lg)
    return {
        "status": "completed", "stage": stage, "arm": arm, "seed": int(seed),
        "sampler": sampler_kind, "loss": {"kind": loss_kind, **loss_kwargs},
        "best_epoch": int(best_epoch), "parameter_count": count_trainable_parameters(model),
        "train_count": int(len(roles["train"])), "draws_per_epoch": draws,
        "train_label_counts": data["train5"][roles["train"]].sum(axis=0).astype(int).tolist(),
        "roles": {name: int(len(idx)) for name, idx in roles.items()},
        "early": role_outputs["early"], "calibration": role_outputs["calibration"], "threshold": role_outputs["threshold"],
        "checkpoint": str(out / "model.pt"), "history": str(out / "history.json"),
    }


def stage1(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    if int(args.max_seeds) == 0:
        for seed_dir in sorted((OUT_ROOT / "stage1").glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            roles_path = OUT_ROOT / "splits" / f"seed_{seed}.json"
            roles = json.loads(roles_path.read_text(encoding="utf-8")) if roles_path.exists() else {}
            for arm in STAGE1_ARMS:
                arm_dir = seed_dir / arm.replace("-", "_")
                history_path, checkpoint = arm_dir / "history.json", arm_dir / "model.pt"
                if not history_path.exists() or not checkpoint.exists():
                    continue
                history = json.loads(history_path.read_text(encoding="utf-8"))
                role_metrics = {}
                for role_name in ("early", "calibration", "threshold"):
                    payload = np.load(arm_dir / f"{role_name}_predictions.npz")
                    role_metrics[role_name] = metric_payload(payload["targets"], payload["logits"])
                rows.append({
                    "status": "completed", "stage": "stage1", "arm": arm, "seed": seed,
                    "sampler": {"Uniform-Focal": "uniform", "MaxPositive-Focal": "max_positive", "MeanPositive-Focal": "mean_positive"}[arm],
                    "loss": {"kind": "focal"}, "best_epoch": int(max(history, key=lambda x: (x["early"]["macro_aupr"], x["early"]["tail_mean_aupr"]))["epoch"]),
                    "parameter_count": 257382, "train_count": int(len(roles.get("train", []))),
                    "draws_per_epoch": int(args.draws_per_epoch), "train_label_counts": data["train5"][np.asarray(roles.get("train", []), dtype=np.int64)].sum(axis=0).astype(int).tolist(),
                    "roles": {k: len(v) for k, v in roles.items()}, "early": role_metrics["early"], "calibration": role_metrics["calibration"], "threshold": role_metrics["threshold"],
                    "checkpoint": str(checkpoint), "history": str(history_path), "recovered_from_saved_artifacts": True,
                })
    for seed in SEEDS[: args.max_seeds]:
        roles = split_roles(data["train_valid"], data["train5"], seed)
        dump_json(OUT_ROOT / "splits" / f"seed_{seed}.json", {k: v.tolist() for k, v in roles.items()})
        for arm, sampler in (("Uniform-Focal", "uniform"), ("MaxPositive-Focal", "max_positive"), ("MeanPositive-Focal", "mean_positive")):
            try:
                print(f"START stage1 seed={seed} arm={arm}", flush=True)
                rows.append(train_joint(arm, seed, roles, data, args, sampler, "focal", {}, "stage1"))
                print(f"DONE stage1 seed={seed} arm={arm}", flush=True)
            except Exception as exc:
                rows.append({"status": "failed", "stage": "stage1", "arm": arm, "seed": int(seed), "error": repr(exc)})
                print(f"FAILED stage1 seed={seed} arm={arm}: {exc!r}", flush=True)
    summary = {"status": "completed", "stage": "stage1", "rows": rows, "seeds": [int(x) for x in SEEDS[: args.max_seeds]], "protocol": "full_five_class_pool_label_aware_sampler"}
    means = {}
    for arm in STAGE1_ARMS:
        values = [r["early"]["macro_aupr"] for r in rows if r.get("arm") == arm and r.get("status") == "completed"]
        means[arm] = float(np.mean(values)) if values else float("nan")
    summary["early_macro_aupr_mean"] = means
    uniform_rows = [r for r in rows if r.get("arm") == "Uniform-Focal" and r.get("status") == "completed"]
    gate_detail = {}
    admitted = []
    for arm in ("MaxPositive-Focal", "MeanPositive-Focal"):
        candidate_rows = [r for r in rows if r.get("arm") == arm and r.get("status") == "completed"]
        paired = []
        for base in uniform_rows:
            match = next((r for r in candidate_rows if int(r["seed"]) == int(base["seed"])), None)
            if match is not None:
                paired.append({
                    "seed": int(base["seed"]),
                    "macro_delta": float(match["early"]["macro_aupr"] - base["early"]["macro_aupr"]),
                    "tail_delta": float(match["early"]["tail_mean_aupr"] - base["early"]["tail_mean_aupr"]),
                    "head_label_deltas": {name: float(match["early"]["per_label"][name]["aupr"] - base["early"]["per_label"][name]["aupr"]) for name in ("FMB", "SinFM", "MFT")},
                })
        mean_macro = float(np.mean([x["macro_delta"] for x in paired])) if paired else float("nan")
        mean_tail = float(np.mean([x["tail_delta"] for x in paired])) if paired else float("nan")
        positive = int(sum(x["macro_delta"] > 0 for x in paired))
        worst_head = float(min(x["head_label_deltas"][n] for x in paired for n in ("FMB", "SinFM", "MFT"))) if paired else float("nan")
        passed = bool(len(paired) == len(uniform_rows) and mean_macro >= 0.010 and positive >= 4 and mean_tail >= 0.020 and worst_head >= -0.010)
        gate_detail[arm] = {"paired": paired, "mean_macro_delta": mean_macro, "mean_tail_delta": mean_tail, "positive_seed_count": positive, "worst_head_delta": worst_head, "passed": passed}
        if passed: admitted.append(arm)
    summary["sampler_gate"] = gate_detail
    summary["stage1_gate_pass"] = bool(admitted)
    summary["selected_sampler"] = max(admitted, key=lambda a: means[a]) if admitted else "Uniform-Focal"
    summary["fallback_reason"] = None if admitted else "No label-aware sampler passed the preregistered gate; use Uniform on the actual LabelIF4 roles for loss isolation."
    dump_json(OUT_ROOT / "stage1_summary.json", summary)
    return summary


def load_stage1() -> dict[str, Any]:
    return json.loads((OUT_ROOT / "stage1_summary.json").read_text(encoding="utf-8"))


def recover_stage2(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    cfg_map = dict(STAGE2_LOSSES)
    for seed_dir in sorted((OUT_ROOT / "stage2").glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        roles_path = OUT_ROOT / "splits_label_if4" / f"seed_{seed}.json"
        if not roles_path.exists(): continue
        roles = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads(roles_path.read_text(encoding="utf-8")).items()}
        for arm, cfg in STAGE2_LOSSES:
            arm_dir = seed_dir / arm.replace("/", "_").replace("-", "_")
            history_path, checkpoint = arm_dir / "history.json", arm_dir / "model.pt"
            if not history_path.exists() or not checkpoint.exists(): continue
            history = json.loads(history_path.read_text(encoding="utf-8")); role_metrics = {}
            for role_name in ("early", "calibration", "threshold"):
                payload = np.load(arm_dir / f"{role_name}_predictions.npz")
                role_metrics[role_name] = metric_payload(payload["targets"], payload["logits"])
            best = max(history, key=lambda x: (x["early"]["macro_aupr"], x["early"]["tail_mean_aupr"]))
            rows.append({"status":"completed","stage":"stage2","arm":arm,"seed":seed,"sampler":"uniform","loss":cfg,"best_epoch":int(best["epoch"]),"parameter_count":257382,"train_count":int(len(roles["train"])),"draws_per_epoch":int(args.draws_per_epoch),"train_label_counts":data["train5"][roles["train"]].sum(axis=0).astype(int).tolist(),"roles":{k:int(len(v)) for k,v in roles.items()},"early":role_metrics["early"],"calibration":role_metrics["calibration"],"threshold":role_metrics["threshold"],"checkpoint":str(checkpoint),"history":str(history_path),"recovered_from_saved_artifacts":True})
    means = {arm: float(np.mean([r["early"]["macro_aupr"] for r in rows if r["arm"] == arm])) for arm, _ in STAGE2_LOSSES}
    selected_loss = max(means, key=lambda k: means[k])
    summary = {"status":"completed","stage":"stage2","training_source":"LabelIF4_manifest","sampler_arm":"Uniform-Focal","sampler":"uniform","rows":rows,"early_macro_aupr_mean":means,"selected_loss":selected_loss,"stage1_gate_pass":False,"fallback_reason":"Recovered formal artifacts after orchestration interruption."}
    dump_json(OUT_ROOT / "stage2_summary.json", summary); return summary


def stage2(args: argparse.Namespace, data: dict[str, Any], stage1_summary: dict[str, Any]) -> dict[str, Any]:
    selected_arm = str(stage1_summary.get("selected_sampler", "Uniform-Focal"))
    sampler = {"Uniform-Focal": "uniform", "MaxPositive-Focal": "max_positive", "MeanPositive-Focal": "mean_positive"}[selected_arm]
    rows = []
    for seed in SEEDS[: args.max_seeds]:
        roles = split_roles(data["manifest_indices"], data["train5"], seed + 5000)
        roles_path = OUT_ROOT / "splits_label_if4" / f"seed_{seed}.json"
        dump_json(roles_path, {k: v.tolist() for k, v in roles.items()})
        for arm, cfg in STAGE2_LOSSES:
            try:
                print(f"START stage2 seed={seed} arm={arm} sampler={selected_arm}", flush=True)
                rows.append(train_joint(arm, seed, roles, data, args, sampler, cfg["kind"], cfg, "stage2"))
                print(f"DONE stage2 seed={seed} arm={arm}", flush=True)
            except Exception as exc:
                rows.append({"status": "failed", "stage": "stage2", "arm": arm, "seed": int(seed), "error": repr(exc)})
                print(f"FAILED stage2 seed={seed} arm={arm}: {exc!r}", flush=True)
    means = {arm: float(np.mean([r["early"]["macro_aupr"] for r in rows if r.get("arm") == arm and r.get("status") == "completed"])) for arm, _ in STAGE2_LOSSES}
    selected_loss = max(means, key=lambda k: means[k] if np.isfinite(means[k]) else -np.inf)
    summary = {"status": "completed", "stage": "stage2", "training_source": "LabelIF4_manifest", "sampler_arm": selected_arm, "sampler": sampler, "rows": rows, "early_macro_aupr_mean": means, "selected_loss": selected_loss, "stage1_gate_pass": bool(stage1_summary.get("stage1_gate_pass", False)), "fallback_reason": stage1_summary.get("fallback_reason")}
    dump_json(OUT_ROOT / "stage2_summary.json", summary)
    return summary


def logits_from_checkpoint(checkpoint: str, indices: np.ndarray, path: Path, labels5: np.ndarray, seed: int, return_features: bool = False):
    model = build_backbone("dfms_iq", num_classes=5, width=16, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.eval()
    dataset, loader = make_loader(path, indices, labels5, "eval", "none", 256, None, seed)
    logits, targets, features = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                out = model(batch["signal"].to(DEVICE, non_blocking=True), return_features=return_features)
                if return_features:
                    lg, ft = out; features.append(ft.detach().cpu().numpy())
                else:
                    lg = out
                logits.append(lg.detach().cpu().numpy()); targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    result = (np.concatenate(logits), np.concatenate(targets).astype(np.uint8))
    return result + (np.concatenate(features),) if return_features else result


def choose_thresholds(targets: np.ndarray, logits: np.ndarray) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    thresholds = np.zeros(prob.shape[1], dtype=np.float64)
    for i in range(prob.shape[1]):
        best = (0.5, -1.0)
        for t in np.arange(0.05, 0.951, 0.01):
            score = f1_score(targets[:, i], prob[:, i] >= t, zero_division=0)
            if score > best[1]: best = (float(t), float(score))
        thresholds[i] = best[0]
    return thresholds


def stage3(args: argparse.Namespace, data: dict[str, Any], stage2_summary: dict[str, Any]) -> dict[str, Any]:
    (OUT_ROOT / "stage3").mkdir(parents=True, exist_ok=True)
    selected_loss = stage2_summary["selected_loss"]
    rows = [r for r in stage2_summary["rows"] if r.get("arm") == selected_loss and r.get("status") == "completed"]
    out_rows = []
    taus = (0.25, 0.5, 1.0)
    for row in rows:
        seed = int(row["seed"]); roles_path = OUT_ROOT / "splits_label_if4" / f"seed_{seed}.json"; roles = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads(roles_path.read_text(encoding="utf-8")).items()}
        train_counts = data["train5"][roles["train"]].sum(axis=0).astype(np.float64); p = np.clip(train_counts / max(len(roles["train"]), 1), 1e-6, 1 - 1e-6)
        checkpoint = row["checkpoint"]
        early_lg, early_y = logits_from_checkpoint(checkpoint, roles["early"], TRAIN_PATH, data["train5"], seed + 11)
        cal_lg, cal_y = logits_from_checkpoint(checkpoint, roles["calibration"], TRAIN_PATH, data["train5"], seed + 12)
        thr_lg, thr_y = logits_from_checkpoint(checkpoint, roles["threshold"], TRAIN_PATH, data["train5"], seed + 13)
        best_tau, best_metric = None, None
        tau_metrics = {}
        for tau in taus:
            adjusted = early_lg - float(tau) * np.log((1.0 - p) / p)[None, :]
            m = metric_payload(early_y, adjusted); tau_metrics[str(tau)] = m
            if best_metric is None or (m["macro_aupr"], m["tail_mean_aupr"]) > (best_metric["macro_aupr"], best_metric["tail_mean_aupr"]): best_tau, best_metric = tau, m
        thresholds = choose_thresholds(thr_y, thr_lg - float(best_tau) * np.log((1.0 - p) / p)[None, :])
        np.savez_compressed(OUT_ROOT / "stage3" / f"seed_{seed}_selected.npz", early_logits=early_lg, early_targets=early_y, calibration_logits=cal_lg, calibration_targets=cal_y, threshold_logits=thr_lg, threshold_targets=thr_y, train_prior=p, thresholds=thresholds)
        out_rows.append({"seed": seed, "selected_loss": selected_loss, "checkpoint": checkpoint, "train_prior": p.tolist(), "tau_metrics_early": tau_metrics, "selected_tau": float(best_tau), "selected_early": best_metric, "thresholds": thresholds.tolist(), "calibration_metric": metric_payload(cal_y, cal_lg - float(best_tau) * np.log((1.0 - p) / p)[None, :])})
    tau_counts = {str(t): sum(1 for r in out_rows if abs(r["selected_tau"] - t) < 1e-9) for t in taus}
    summary = {"status": "completed", "stage": "stage3", "selected_loss": selected_loss, "rows": out_rows, "selected_tau_counts": tau_counts, "selection_rule": "early_macro_aupr_then_tail_mean", "test_used": False}
    dump_json(OUT_ROOT / "stage3_summary.json", summary)
    return summary


class LinearHead(torch.nn.Module):
    def __init__(self, dim: int = 128, classes: int = 5):
        super().__init__(); self.linear = torch.nn.Linear(dim, classes)
    def forward(self, x): return self.linear(x)


def train_frozen_head(name: str, seed: int, train: dict[str, np.ndarray], early: dict[str, np.ndarray], data: dict[str, Any], args: argparse.Namespace, mode: str, anchor: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
    set_seed(seed)
    head = LinearHead(train["features"].shape[1], 5).to(DEVICE)
    anchor_weight = anchor[0].detach().to(DEVICE).clone(); anchor_bias = anchor[1].detach().to(DEVICE).clone()
    if mode == "rbc":
        with torch.no_grad():
            head.linear.weight.copy_(anchor_weight); head.linear.bias.copy_(anchor_bias)
    counts = train["targets"].sum(axis=0).astype(int).tolist()
    if mode == "cb": criterion = build_classification_loss("cb_bce", positive_counts=counts, beta=0.99).to(DEVICE)
    else: criterion = build_classification_loss("bce").to(DEVICE)
    weights = np.ones(len(train["targets"]), dtype=np.float64)
    sampler = None
    if mode in ("crt", "rbc"):
        inv = 1.0 / np.maximum(train["targets"].sum(axis=0), 1.0)
        weights = np.maximum((train["targets"] * inv[None, :]).max(axis=1), 1e-12)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=int(args.head_draws_per_epoch), replacement=True, generator=torch.Generator().manual_seed(seed + 99))
    tx = torch.as_tensor(train["features"], dtype=torch.float32, device=DEVICE); ty = torch.as_tensor(train["targets"], dtype=torch.float32, device=DEVICE)
    ex = torch.as_tensor(early["features"], dtype=torch.float32, device=DEVICE)
    ey = early["targets"]
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pair_map = torch.as_tensor(build_pair_map(train["targets"], seed + 1700, "hard"), dtype=torch.long, device=DEVICE) if mode == "rbc" else None
    positive_counts = ty.sum(dim=0)
    best, best_key, stale, best_epoch = None, (-np.inf, -np.inf), 0, 0
    history = []
    for epoch in range(1, int(args.head_epochs) + 1):
        head.train(); losses = []
        # Every frozen-head arm receives the same preregistered optimization
        # budget. Ordinary/CB heads sample uniformly with replacement; cRT/RBC
        # use the registered balanced weights, also with replacement.
        if sampler is None:
            positions = torch.randint(0, len(tx), (int(args.head_draws_per_epoch),), generator=torch.Generator().manual_seed(seed + 99 + epoch), device="cpu").to(DEVICE)
        else:
            positions = torch.as_tensor(list(iter(sampler)), device=DEVICE)
        for chunk in positions.split(int(args.head_batch_size)):
            opt.zero_grad(set_to_none=True)
            logits = head(tx[chunk])
            if mode == "rbc":
                balanced = label_balanced_bce(logits, ty[chunk], positive_counts, len(ty))
                rank_terms = []
                for label_index in range(ty.shape[1]):
                    selected = ty[chunk, label_index] == 1
                    if torch.any(selected):
                        negative_positions = pair_map[chunk[selected], label_index]
                        negative_logits = head(tx[negative_positions])[:, label_index]
                        rank_terms.append(torch.nn.functional.softplus(0.2 - logits[selected, label_index] + negative_logits))
                rank = torch.cat(rank_terms).mean() if rank_terms else balanced * 0.0
                weight_scale = anchor_weight.pow(2).mean().clamp_min(1e-12); bias_scale = anchor_bias.pow(2).mean().clamp_min(1e-12)
                anchor_loss = (head.linear.weight - anchor_weight).pow(2).mean() / weight_scale + (head.linear.bias - anchor_bias).pow(2).mean() / bias_scale
                loss = balanced + 0.25 * rank + 1e-3 * anchor_loss
            else:
                loss = criterion(logits, ty[chunk])
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        with torch.no_grad(): el = head(ex).detach().cpu().numpy()
        em = metric_payload(ey, el); key = (em["macro_aupr"], em["tail_mean_aupr"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "early": em})
        if key > best_key: best_key, best, best_epoch, stale = key, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= int(args.head_patience): break
    head.load_state_dict(best, strict=True)
    return {"name": name, "seed": int(seed), "mode": mode, "best_epoch": int(best_epoch), "early": metric_payload(ey, head(ex).detach().cpu().numpy()), "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "history": history, "loss_contract": "label_balanced_bce+0.25*hard_rank+1e-3*anchor" if mode == "rbc" else mode}


def stage4(args: argparse.Namespace, data: dict[str, Any], stage2_summary: dict[str, Any]) -> dict[str, Any]:
    # Prompt contract: if Stage 2 has no admitted candidate, freeze the
    # preregistered B0-Focal-LabelIF4 backbone rather than the best-looking
    # exploratory loss.
    selected_loss = stage2_summary["selected_loss"] if bool(stage2_summary.get("stage2_gate_pass", False)) else "Focal"
    rows = [r for r in stage2_summary["rows"] if r.get("arm") == selected_loss and r.get("status") == "completed"]
    output_rows = []
    for row in rows:
        seed = int(row["seed"]); roles = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads((OUT_ROOT / "splits_label_if4" / f"seed_{seed}.json").read_text(encoding="utf-8")).items()}
        train_lg, train_y, train_f = logits_from_checkpoint(row["checkpoint"], roles["train"], TRAIN_PATH, data["train5"], seed + 51, True)
        early_lg, early_y, early_f = logits_from_checkpoint(row["checkpoint"], roles["early"], TRAIN_PATH, data["train5"], seed + 52, True)
        payload = {"features": train_f, "targets": train_y}; epayload = {"features": early_f, "targets": early_y}
        anchor_model = build_backbone("dfms_iq", num_classes=5, width=16, dropout=0.2)
        anchor_model.load_state_dict(torch.load(row["checkpoint"], map_location="cpu"), strict=True)
        anchor = classifier_state(anchor_model)
        heads = []
        for name, mode in (("OrdinaryLinearHead", "ordinary"), ("ClassBalancedBCEHead", "cb"), ("cRT", "crt"), ("RBC-cRT", "rbc")):
            result = train_frozen_head(name, seed, payload, epayload, data, args, mode, anchor); state = result.pop("state"); head_dir = OUT_ROOT / "stage4" / f"seed_{seed}" / name; head_dir.mkdir(parents=True, exist_ok=True); head_path = head_dir / "head.pt"; torch.save(state, head_path); dump_json(head_dir / "history.json", result.pop("history")); result["head_checkpoint"] = str(head_path); heads.append(result)
        output_rows.append({"seed": seed, "selected_loss": selected_loss, "backbone_checkpoint": row["checkpoint"], "heads": heads, "train_feature_shape": list(train_f.shape), "early_feature_shape": list(early_f.shape)})
    summary = {"status": "completed", "stage": "stage4", "selected_loss": selected_loss, "selection_reason": "admitted_stage2_candidate" if bool(stage2_summary.get("stage2_gate_pass", False)) else "prompt_fallback_B0_Focal_LabelIF4", "equal_head_draws_per_epoch": int(args.head_draws_per_epoch), "rows": output_rows, "test_used": False}
    dump_json(OUT_ROOT / "stage4_summary.json", summary)
    return summary


def final_test(args: argparse.Namespace, data: dict[str, Any], stage1_summary: dict[str, Any], stage2_summary: dict[str, Any], stage3_summary: dict[str, Any], stage4_summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    # Evaluate every completed joint-training checkpoint only now, after all
    # development choices have been frozen.
    for stage_summary in (stage1_summary, stage2_summary):
        for row in stage_summary.get("rows", []):
            if row.get("status") != "completed": continue
            seed = int(row["seed"]); lg, y = logits_from_checkpoint(row["checkpoint"], data["test_valid"], TEST_PATH, data["test5"], seed + 991)
            test_m = metric_payload(y, lg)
            pred_dir = OUT_ROOT / row["stage"] / f"seed_{seed}" / str(row["arm"]).replace("/", "_").replace("-", "_")
            np.savez_compressed(pred_dir / "test_predictions.npz", logits=lg, targets=y)
            rows.append({"stage": row["stage"], "arm": row["arm"], "seed": seed, "test": test_m, "checkpoint": row["checkpoint"]})
    # Evaluate selected stage3 tau and threshold, without changing them.
    stage3_test = []
    selected_loss = stage3_summary["selected_loss"]
    for item in stage3_summary.get("rows", []):
        seed = int(item["seed"]); stage2row = next(r for r in stage2_summary["rows"] if r.get("arm") == selected_loss and int(r.get("seed", -1)) == seed and r.get("status") == "completed")
        lg, y = logits_from_checkpoint(stage2row["checkpoint"], data["test_valid"], TEST_PATH, data["test5"], seed + 992)
        p = np.asarray(item["train_prior"], dtype=np.float64); adj = lg - float(item["selected_tau"]) * np.log((1-p)/p)[None, :]; thresholds = np.asarray(item["thresholds"], dtype=np.float64)
        stage3_test.append({"seed": seed, "selected_tau": item["selected_tau"], "raw": metric_payload(y, lg), "adjusted": metric_payload(y, adj), "adjusted_thresholded": metric_payload(y, adj, thresholds)})
    stage4_test = []
    for item in stage4_summary.get("rows", []):
        seed = int(item["seed"]); lg0, y, features = logits_from_checkpoint(item["backbone_checkpoint"], data["test_valid"], TEST_PATH, data["test5"], seed + 993, True)
        for head_item in item["heads"]:
            head = LinearHead(features.shape[1], 5).to(DEVICE); head.load_state_dict(torch.load(head_item["head_checkpoint"], map_location="cpu"), strict=True); head.eval()
            with torch.no_grad(): lg = head(torch.as_tensor(features, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
            stage4_test.append({"seed": seed, "head": head_item["name"], "test": metric_payload(y, lg), "head_checkpoint": head_item["head_checkpoint"]})
    summary = {"status": "completed", "test_used_after_development_lock": True, "joint_rows": rows, "stage3_selected_rows": stage3_test, "stage4_head_rows": stage4_test, "test_v2_used": False, "locked_test_used": False}
    dump_json(OUT_ROOT / "final_test_summary.json", summary)
    return summary


def aggregate_and_report(data: dict[str, Any], stage1_summary: dict[str, Any], stage2_summary: dict[str, Any], stage3_summary: dict[str, Any], stage4_summary: dict[str, Any], final_summary: dict[str, Any], args: argparse.Namespace) -> None:
    def agg(rows, arm, stage=None):
        vals = [r["test"]["macro_aupr"] for r in rows if r.get("arm") == arm and (stage is None or r.get("stage") == stage)]
        return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals, ddof=1)) if len(vals)>1 else None, "values": vals}
    all_joint = final_summary["joint_rows"]
    stage1_agg = {a: agg(all_joint, a, "stage1") for a in STAGE1_ARMS}
    stage2_agg = {a: agg(all_joint, a, "stage2") for a, _ in STAGE2_LOSSES}
    selected_sampler = stage1_summary.get("selected_sampler", "Uniform-Focal")
    uniform_mean = stage1_agg.get("Uniform-Focal", {}).get("mean")
    sampler_gate = bool(stage1_summary.get("stage1_gate_pass", False))
    selected_loss = stage2_summary.get("selected_loss")
    base = stage2_agg.get("Focal", {})
    cand = stage2_agg.get(selected_loss, {}) if selected_loss else {}
    paired_loss = []
    if selected_loss and selected_loss != "Focal":
        for seed in SEEDS[:args.max_seeds]:
            br = next((r for r in all_joint if r.get("stage") == "stage2" and r.get("arm") == "Focal" and int(r.get("seed", -1)) == seed), None)
            cr = next((r for r in all_joint if r.get("stage") == "stage2" and r.get("arm") == selected_loss and int(r.get("seed", -1)) == seed), None)
            if br and cr:
                paired_loss.append({"seed": int(seed), "macro_delta": float(cr["test"]["macro_aupr"] - br["test"]["macro_aupr"]), "tail_delta": float(cr["test"]["tail_mean_aupr"] - br["test"]["tail_mean_aupr"]), "head_deltas": {n: float(cr["test"]["per_label"][n]["aupr"] - br["test"]["per_label"][n]["aupr"]) for n in ("FMB", "SinFM", "MFT")}})
    loss_mean_delta = float(np.mean([x["macro_delta"] for x in paired_loss])) if paired_loss else 0.0
    loss_tail_delta = float(np.mean([x["tail_delta"] for x in paired_loss])) if paired_loss else 0.0
    loss_positive = int(sum(x["macro_delta"] > 0 for x in paired_loss))
    loss_worst_head = float(min(x["head_deltas"][n] for x in paired_loss for n in x["head_deltas"])) if paired_loss else 0.0
    loss_gate = bool(len(paired_loss) == args.max_seeds and loss_mean_delta >= 0.010 and loss_positive >= 4 and loss_tail_delta >= 0.020 and loss_worst_head >= -0.010)

    rng = np.random.RandomState(20260826)
    bootstrap = {}
    if paired_loss:
        values = np.asarray([x["macro_delta"] for x in paired_loss], dtype=np.float64)
        draws = np.mean(values[rng.randint(0, len(values), size=(5000, len(values)))], axis=1)
        bootstrap["selected_loss_vs_focal_macro_delta"] = {"repetitions": 5000, "mean": float(values.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], "positive_fraction": float(np.mean(draws > 0))}

    balanced_combo = data["audit"]["balanced_combo_counts"]; long_combo = data["audit"]["label_if4_combo_counts"]
    coverage = {"balanced_combo_count": len(balanced_combo), "label_if4_combo_count": len(long_combo), "coverage_ratio": float(len(long_combo) / max(len(balanced_combo), 1)), "removed_combinations": sorted(set(balanced_combo) - set(long_combo)), "balanced_counts": balanced_combo, "label_if4_counts": long_combo}
    exposure = {}
    for seed in SEEDS[:args.max_seeds]:
        roles = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads((OUT_ROOT / "splits_label_if4" / f"seed_{seed}.json").read_text(encoding="utf-8")).items()}
        label_counts = data["train5"][roles["train"]].sum(axis=0).astype(np.float64); expected = args.draws_per_epoch * label_counts / max(len(roles["train"]), 1)
        exposure[str(seed)] = {"train_counts": label_counts.astype(int).tolist(), "expected_positive_exposure_per_epoch_uniform": expected.tolist(), "optimization_draws": int(args.draws_per_epoch)}
    coverage_analysis = {"frequency_prior_effect": {"balanced_counts": data["audit"]["train_counts_after_filter"], "label_if4_counts": data["audit"]["label_if4_achieved_counts"]}, "cooccurrence_coverage_effect": coverage, "optimization_budget_effect": exposure}
    dump_json(OUT_ROOT / "cooccurrence_coverage_analysis.json", coverage_analysis)
    dump_json(OUT_ROOT / "bootstrap_summary.json", bootstrap)

    metrics_csv = OUT_ROOT / "metrics_summary.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["stage", "arm", "seed", "macro_aupr", "macro_auroc", "macro_f1", "head_mean_aupr", "tail_mean_aupr"])
        for row in all_joint:
            m = row["test"]; writer.writerow([row["stage"], row["arm"], row["seed"], m["macro_aupr"], m["macro_auroc"], m["macro_f1"], m["head_mean_aupr"], m["tail_mean_aupr"]])
        for row in final_summary.get("stage4_head_rows", []):
            m = row["test"]; writer.writerow(["stage4", row["head"], row["seed"], m["macro_aupr"], m["macro_auroc"], m["macro_f1"], m["head_mean_aupr"], m["tail_mean_aupr"]])
    dump_json(OUT_ROOT / "metrics_summary.json", {"stage1": stage1_agg, "stage2": stage2_agg, "stage4": final_summary.get("stage4_head_rows", []), "selected_loss_paired": paired_loss})
    report = {
        "title": "路线A五类频次长尾 V2 自动验证报告",
        "date": time.strftime("%Y-%m-%d"), "status": "[A] development evidence; test evaluated after development lock",
        "data": data["audit"],
        "backbone": {"name": "dfms_iq", "variant": "B0_DelayedFusionMultiScale", "width": 16, "feature_dim": 128, "dropout": 0.2, "num_classes": 5},
        "training": {"seeds": [int(x) for x in SEEDS[:args.max_seeds]], "batch_size": args.batch_size, "draws_per_epoch": args.draws_per_epoch, "epochs": args.epochs, "patience": args.patience, "optimizer": "AdamW", "backbone_lr": 5e-5, "classifier_lr": 5e-4, "weight_decay": 1e-4, "gradient_clip": 5.0},
        "stage1_test_aggregate": stage1_agg, "stage2_test_aggregate": stage2_agg,
        "selected_sampler": selected_sampler, "selected_loss": selected_loss,
        "stage1_gate_pass": sampler_gate, "stage2_gate_pass": loss_gate, "selected_loss_paired_test": paired_loss, "selected_loss_mean_delta": loss_mean_delta, "selected_loss_tail_delta": loss_tail_delta, "selected_loss_positive_seed_count": loss_positive, "selected_loss_worst_head_delta": loss_worst_head,
        "bootstrap": bootstrap, "cooccurrence_coverage": coverage_analysis,
        "stage3": stage3_summary, "stage4": stage4_summary, "final_test": final_summary,
        "decision": "allow_label_conditioned_or_stage4_adapter" if sampler_gate or loss_gate else "candidate_rejected_no_stable_frequency_lt_gain",
        "evidence_tiers": {"[F]": ["SawFM过滤、五类B0协议、角色隔离、等预算训练和test后置评估"], "[A]": ["sampler/loss/logit/head development results"], "[N]": ([] if sampler_gate or loss_gate else ["sampler与损失候选未通过统一硬门槛"]), "[T]": ["标签条件分类头、stage4受限联合微调，仅在门槛通过后允许"]},
    }
    report_path = ROOT / "docs" / "reports" / f"route_a_five_class_frequency_lt_v2_report_{time.strftime('%Y%m%d')}.md"
    lines = ["# 路线 A：五类频次长尾 V2 自动验证报告", "", f"日期：{report['date']}", "", "## 结论", "", f"当前状态：{report['decision']}。测试集在开发选择完成后才评估，不能将任何局部增益解释为已解决五类频次长尾。", "", "## 数据与协议", "", f"保留类别：`{' / '.join(CLASS_NAMES)}`；删除 SawFM 阳性样本；训练保留 {data['audit']['train_retained_count']}，测试保留 {data['audit']['test_retained_count']}；不使用功率长尾和封存 test-v2/test-v3。", "", "## 骨干与训练", "", "固定 B0 `DelayedFusionMultiScaleIQClassifier`，width=16，feature_dim=128，dropout=0.2；AdamW，backbone lr=5e-5，classifier lr=5e-4，weight_decay=1e-4，clip=5.0；每臂固定 draws_per_epoch={}。".format(args.draws_per_epoch), "", "## 测试结果汇总", "", "### Stage 1 sampler", "", "| arm | mean macro-AUPR | std | n |", "|---|---:|---:|---:|"]
    for arm, val in stage1_agg.items(): lines.append(f"| {arm} | {val.get('mean')} | {val.get('std')} | {val.get('n')} |")
    lines += ["", "### Stage 2 loss", "", "| arm | mean macro-AUPR | std | n |", "|---|---:|---:|---:|"]
    for arm, val in stage2_agg.items(): lines.append(f"| {arm} | {val.get('mean')} | {val.get('std')} | {val.get('n')} |")
    lines += ["", "## 归因与门槛", "", f"Stage1 sampler gate：`{sampler_gate}`；Stage2 loss gate：`{loss_gate}`；selected sampler：`{selected_sampler}`；selected loss：`{selected_loss}`。", "", "冻结 B0 特征分类头、logit adjustment 和共现覆盖明细已写入 JSON receipt；只有满足 5 seed 至少 4/5 同向、macro-AUPR 增益≥0.010、尾类增益≥0.020 且头部下降≤0.010，才允许进入标签条件分类头或 stage4 受限微调。", "", "## 产物", "", f"- `outputs/route_a_five_class_frequency_lt_v2/run_receipt.json`", f"- `outputs/route_a_five_class_frequency_lt_v2/final_test_summary.json`", f"- `outputs/route_a_five_class_frequency_lt_v2/stage1_summary.json`", f"- `outputs/route_a_five_class_frequency_lt_v2/stage2_summary.json`", f"- `outputs/route_a_five_class_frequency_lt_v2/stage3_summary.json`", f"- `outputs/route_a_five_class_frequency_lt_v2/stage4_summary.json`"]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_json(OUT_ROOT / "run_receipt.json", {"status": "completed", "goal": "route_a_five_class_frequency_lt_v2", "protocol": report, "report": str(report_path), "locked_test_used": False, "test_v2_used": False, "test_evaluated_after_development_lock": True})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("stage0", "stage1", "stage2", "stage3", "stage4", "finalize", "all"), default="all")
    p.add_argument("--max-seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--draws-per-epoch", type=int, default=9920)
    p.add_argument("--head-epochs", type=int, default=80)
    p.add_argument("--head-patience", type=int, default=12)
    p.add_argument("--head-batch-size", type=int, default=256)
    p.add_argument("--head-draws-per-epoch", type=int, default=9920)
    return p.parse_args()


def main():
    args = parse_args(); OUT_ROOT.mkdir(parents=True, exist_ok=True); started = time.time(); data = ensure_stage0()
    if args.stage == "stage0":
        dump_json(OUT_ROOT / "run_receipt.json", {"status": "completed", "stage": "stage0", "audit": data["audit"], "locked_test_used": False}); return
    if args.stage == "finalize":
        s1 = load_json(OUT_ROOT / "stage1_summary.json"); s2 = recover_stage2(data, args) if not (OUT_ROOT / "stage2_summary.json").exists() or not load_json(OUT_ROOT / "stage2_summary.json").get("rows") else load_json(OUT_ROOT / "stage2_summary.json"); s3 = load_json(OUT_ROOT / "stage3_summary.json"); s4 = load_json(OUT_ROOT / "stage4_summary.json"); ft = final_test(args, data, s1, s2, s3, s4); aggregate_and_report(data, s1, s2, s3, s4, ft, args); print(f"V2_FINALIZED receipt={OUT_ROOT / 'run_receipt.json'}", flush=True); return
    s1 = stage1(args, data) if args.stage in ("stage1", "all") else load_json(OUT_ROOT / "stage1_summary.json")
    if args.stage == "stage1": return
    s2 = stage2(args, data, s1) if args.stage in ("stage2", "all") else load_json(OUT_ROOT / "stage2_summary.json")
    if args.stage == "stage2": return
    s3 = stage3(args, data, s2) if args.stage in ("stage3", "all") else load_json(OUT_ROOT / "stage3_summary.json")
    if args.stage == "stage3": return
    s4 = stage4(args, data, s2) if args.stage in ("stage4", "all") else load_json(OUT_ROOT / "stage4_summary.json")
    if args.stage == "stage4": return
    ft = final_test(args, data, s1, s2, s3, s4)
    aggregate_and_report(data, s1, s2, s3, s4, ft, args)
    print(f"V2_COMPLETE elapsed={time.time()-started:.1f}s receipt={OUT_ROOT / 'run_receipt.json'}", flush=True)


def load_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__": main()
