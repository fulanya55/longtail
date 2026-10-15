"""Formal, no-threshold RAAT-H0/v1.1 diagnostic on a fresh train-only source."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
import torch.nn.functional as F

from cjlt.models.raat_local_map import RAATLocalMapNet, extract_raat_map, parameter_count
from cjlt.data.v73_dataset import read_labels
from cjlt.data.parameter_schema import get_parameter_specs


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"MFT": 3, "VFT": 4, "ISRJ": 5}
SEEDS = (2041, 2042, 2043, 2044, 2045)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_iq(path: Path, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle["data_legacy"][:, ids], dtype=np.float32).T
    iq = raw.reshape(len(ids), 2, 3000)
    rms = np.sqrt(np.maximum(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1), 1e-12))
    result = np.ascontiguousarray(iq / rms[:, None, None], dtype=np.float32)
    if not np.isfinite(result).all(): raise FloatingPointError("non-finite I/Q")
    return result


def label_only_view(labels: np.ndarray, iq: np.ndarray, ids: np.ndarray, target: int) -> np.ndarray:
    """Match positives/negatives without using weak masks or component metadata."""
    others = [index for index in range(6) if index != target]
    energy = np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1)
    rank = np.argsort(np.argsort(energy, kind="stable"), kind="stable")
    quintile = np.minimum(4, (rank * 5) // max(len(ids), 1))
    rows: list[int] = []
    for composition in sorted({tuple(labels[item, others].tolist()) for item in ids}):
        key = np.asarray([tuple(labels[item, others].tolist()) == composition for item in ids])
        for q in range(5):
            local = np.flatnonzero(key & (quintile == q))
            positive = local[labels[ids[local], target] == 1]
            negative = local[labels[ids[local], target] == 0]
            keep = min(len(positive), len(negative))
            rows.extend(ids[positive[:keep]].tolist()); rows.extend(ids[negative[:keep]].tolist())
    selected = np.asarray(sorted(set(rows)), dtype=np.int64)
    if len(selected) < 64 or len(np.unique(labels[selected, target])) != 2:
        raise ValueError(f"label-only matched view is degenerate for target {target}")
    return selected


def standardize(fit: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = fit.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.maximum(fit.std(axis=(0, 2, 3), keepdims=True), 1e-5).astype(np.float32)
    return ((values - mean) / std).astype(np.float32), mean, std


def extract_maps(iq: np.ndarray, target: str, reverse: bool, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            tensor = torch.from_numpy(iq[start:start + batch_size]).to(device)
            rows.append(extract_raat_map(tensor, target, reverse_chirp=reverse).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def initial_state(seed: int) -> tuple[dict[str, torch.Tensor], str]:
    set_seed(seed)
    state = {key: value.detach().cpu().clone() for key, value in RAATLocalMapNet().state_dict().items()}
    digest = hashlib.sha256(b"".join(value.numpy().tobytes() for value in state.values())).hexdigest()
    return state, digest


def train_epoch30(xfit: np.ndarray, yfit: np.ndarray, seed: int, initial: dict[str, torch.Tensor], device: torch.device, args: argparse.Namespace) -> tuple[nn.Module, list[dict], list[int]]:
    set_seed(seed)
    model = RAATLocalMapNet().to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    generator = torch.Generator().manual_seed(seed + 991)
    history, order_hashes = [], []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(xfit), generator=generator).numpy(); losses = []
        order_hashes.append(int.from_bytes(hashlib.sha256(order.tobytes()).digest()[:8], "big"))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            xb = torch.from_numpy(xfit[index]).to(device); yb = torch.from_numpy(yfit[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses))})
    return model, history, order_hashes


def score(model: nn.Module, maps: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(maps), batch_size): rows.append(model(torch.from_numpy(maps[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def metric(y: np.ndarray, value: np.ndarray) -> dict:
    return {"count": int(len(y)), "auroc": float(roc_auc_score(y, value)), "aupr": float(average_precision_score(y, value))}


def paired_hierarchical_bootstrap(records: list[dict], metric_name: str, seed: int) -> dict:
    """Resample seeds and, within each selected seed, paired endpoint samples."""
    seed_values = []
    for record in records:
        seed_values.append(metric(record["y"], record["aligned"])[metric_name] - metric(record["y"], record["wrong"])[metric_name])
    rng = np.random.default_rng(seed); draws = []
    for _ in range(10000):
        values = []
        for seed_index in rng.integers(0, len(records), size=len(records)):
            record = records[int(seed_index)]
            for _attempt in range(32):
                selected = rng.integers(0, len(record["y"]), size=len(record["y"]))
                truth = record["y"][selected]
                if np.unique(truth).size == 2:
                    values.append(metric(truth, record["aligned"][selected])[metric_name] - metric(truth, record["wrong"][selected])[metric_name])
                    break
            else:
                raise RuntimeError("bootstrap sample lost a class repeatedly")
        draws.append(float(np.mean(values)))
    return {"mean": float(np.mean(seed_values)), "seed_values": [float(item) for item in seed_values], "positive_seed_count": int(np.sum(np.asarray(seed_values) > 0)), "bootstrap_95_ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])], "bootstrap": "paired_hierarchical_10000_seed_then_within_seed_sample"}


def weak_mask_for_evaluation(data_path: Path, labels: np.ndarray) -> np.ndarray:
    """Isolated, post-prediction evaluation-only weak annotation.

    The function is deliberately called after all maps, normalizers, models and
    scores have been created.  Its output never enters any training function.
    """
    power_columns = tuple(get_parameter_specs(name)[0].column_index for name in ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"))
    with h5py.File(data_path, "r") as handle:
        values = np.asarray(handle["parameter_values"], dtype=np.float32).T
    powers = values[:, power_columns]
    weak = np.zeros_like(labels, dtype=bool)
    for row, active in enumerate(labels.astype(bool)):
        indices = np.flatnonzero(active)
        if len(indices) < 2 or not np.isfinite(powers[row, indices]).all():
            continue
        for index in indices:
            other = indices[indices != index]
            weak[row, index] = bool(powers[row, index] - powers[row, other].max() <= -10.0)
    return weak


def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    audit, seal, stage0 = read_json(args.audit), read_json(args.test_v3_manifest), read_json(args.stage0_manifest)
    if not audit.get("identity_audit_passed") or audit.get("test_files_opened"): raise ValueError("RAAT source audit failed")
    if sha256(args.data_path) != audit["sha256"]: raise ValueError("data hash differs from audited source")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 seal not intact")
    if stage0["new_data_generated"] or stage0["implementation_started"]: pass
    if stage0["candidate"] != "Reference-Aligned Ambiguity Topology (RAAT)": raise ValueError("wrong Stage 0 manifest")
    return audit, seal, stage0


def target_run(args: argparse.Namespace, target_name: str, labels: np.ndarray, roles: dict[str, np.ndarray], device: torch.device) -> dict:
    target = TARGETS[target_name]
    union = np.unique(np.concatenate([roles[name] for name in ("fit", "validation", "early")]))
    iq = load_iq(args.data_path, union); position = {int(identifier): offset for offset, identifier in enumerate(union)}
    role_positions = {name: np.asarray([position[int(identifier)] for identifier in ids], dtype=np.int64) for name, ids in roles.items()}
    selected = {name: label_only_view(labels, iq[role_positions[name]], roles[name], target) for name in roles}
    selected_positions = {name: np.asarray([position[int(identifier)] for identifier in ids], dtype=np.int64) for name, ids in selected.items()}
    arms = {"C_RaatAligned": False, "B_WrongReference": True}
    arm_maps: dict[str, np.ndarray] = {}
    map_audit: dict[str, dict] = {}
    for arm, reverse in arms.items():
        maps = extract_maps(iq, target_name, reverse, device, args.feature_batch_size)
        arm_maps[arm] = maps
        map_audit[arm] = {"shape": list(maps.shape), "finite": bool(np.isfinite(maps).all()), "mean": float(maps.mean()), "std": float(maps.std()), "nonzero_fraction": float(np.count_nonzero(maps) / maps.size), "map_sha256": hashlib.sha256(maps.tobytes()).hexdigest()}
    map_l2 = float(np.linalg.norm(arm_maps["C_RaatAligned"] - arm_maps["B_WrongReference"]))
    if map_l2 <= 0 or not all(row["finite"] for row in map_audit.values()): raise ValueError("reference-destruction map audit failed")
    rows = []
    for seed in args.seeds:
        variants = {}
        initial, initial_hash = initial_state(seed)
        order_reference = None
        for arm in arms:
            fit_all, early_all = selected_positions["fit"], selected_positions["early"]
            normalized, mean, std = standardize(arm_maps[arm][fit_all], arm_maps[arm])
            model, history, order_hashes = train_epoch30(normalized[fit_all], labels[selected["fit"], target].astype(np.float32), seed, initial, device, args)
            if order_reference is None: order_reference = order_hashes
            if order_hashes != order_reference: raise ValueError("arms lack matched batch order")
            values = score(model, normalized[early_all], device, args.eval_batch_size)
            y_all = labels[selected["early"], target]
            variants[arm] = {"scores": values, "all": metric(y_all, values), "normalization_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(), "epoch30_history": history, "batch_order_hashes": order_hashes, "initial_state_sha256": initial_hash}
        rows.append({"seed": int(seed), "variants": variants})
    # Metadata appears only after epoch-30 scores for both arms have been written.
    weak = weak_mask_for_evaluation(args.data_path, labels)
    y_all = labels[selected["early"], target]
    weak_positive = (y_all == 1) & weak[selected["early"], target]; negative = y_all == 0
    strong_positive = (y_all == 1) & ~weak[selected["early"], target]
    if weak_positive.sum() < 160 or negative.sum() < 160 or strong_positive.sum() < 2: raise ValueError(f"insufficient post-prediction endpoint cases for {target_name}")
    weak_records, strong_records = [], []
    for row in rows:
        weak_y = np.r_[np.ones(weak_positive.sum(), dtype=np.uint8), np.zeros(negative.sum(), dtype=np.uint8)]
        strong_y = np.r_[np.ones(strong_positive.sum(), dtype=np.uint8), np.zeros(negative.sum(), dtype=np.uint8)]
        weak_record = {"y": weak_y}; strong_record = {"y": strong_y}
        for arm in arms:
            values = row["variants"][arm].pop("scores")
            weak_score = np.r_[values[weak_positive], values[negative]]
            strong_score = np.r_[values[strong_positive], values[negative]]
            row["variants"][arm]["weak"] = metric(weak_y, weak_score)
            row["variants"][arm]["strong"] = metric(strong_y, strong_score)
            key = "aligned" if arm == "C_RaatAligned" else "wrong"
            weak_record[key] = weak_score; strong_record[key] = strong_score
        weak_records.append(weak_record); strong_records.append(strong_record)
    weak_auroc = paired_hierarchical_bootstrap(weak_records, "auroc", 711 + target)
    weak_aupr = paired_hierarchical_bootstrap(weak_records, "aupr", 721 + target)
    strong_auroc = paired_hierarchical_bootstrap(strong_records, "auroc", 811 + target)
    gate = {"weak_auroc_mean_ge_0p015": weak_auroc["mean"] >= .015, "weak_auroc_ci_lower_gt_0": weak_auroc["bootstrap_95_ci"][0] > 0, "weak_auroc_positive_4of5": weak_auroc["positive_seed_count"] >= 4, "strong_auroc_each_ge_minus_0p010": bool(np.all(np.asarray(strong_auroc["seed_values"]) >= -.010)), "map_destruction_passed": map_l2 > 0}
    return {"target": target_name, "index": target, "matched_role_counts": {name: int(len(ids)) for name, ids in selected.items()}, "map_audit": {"arms": map_audit, "aligned_minus_wrong_l2": map_l2}, "per_seed": rows, "delta": {"weak_auroc": weak_auroc, "weak_aupr": weak_aupr, "strong_auroc": strong_auroc}, "gate": {"checks": gate, "passed": bool(all(gate.values()))}}


def run(args: argparse.Namespace) -> dict:
    audit, _, _ = validate_inputs(args)
    labels = read_labels(args.data_path)
    role_file = np.load(audit["role_indices_path"], allow_pickle=False); roles = {name: np.asarray(role_file[name], dtype=np.int64) for name in ("fit", "validation", "early")}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if parameter_count() != 7425: raise ValueError("frozen RAAT parameter count drifted")
    results = {name: target_run(args, name, labels, roles, device) for name in TARGETS}
    passing = [name for name, value in results.items() if value["gate"]["passed"]]
    result = {"status": "completed", "experiment": "RouteA_RAAT_H0_v1_1", "evidence_role": "fresh_identity_isolated_train_only_H0", "audit_sha256": sha256(args.audit), "data_sha256": sha256(args.data_path), "seeds": [int(value) for value in args.seeds], "architecture": "RAATLocalMapNet", "parameter_count": parameter_count(), "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs": args.epochs, "checkpoint": "epoch_30_only"}, "results": results, "passing_targets": passing, "promotion": "permit_H1_for_passing_targets" if len(passing) >= 2 else "stop_RAAC_before_H1_H2", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_raat_used": False}
    write_json(args.output_root / "route_a_raat_h0_v1_summary.json", result)
    print(json.dumps({"passing_targets": passing, "promotion": result["promotion"], "output": str(args.output_root)}, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_raat_h0_v1\closedset_6class_train.mat"))
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_raat_h0_generation/identity_audit.json")
    parser.add_argument("--stage0-manifest", type=Path, default=ROOT / "outputs/tapc/route_a_raat_stage0_freeze/stage0_preimplementation_manifest_v1_1.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tapc/route_a_raat_h0_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
