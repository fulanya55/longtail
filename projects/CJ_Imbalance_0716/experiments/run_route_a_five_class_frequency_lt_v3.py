"""Route-A V3: identity-isolated five-class frequency long-tail protocol.

Uses train-only sources from 0731_v3 (development) and 0731_v2
(confirmation).  Long-tail manifests are solved at the combination level so
all 31 non-empty five-label combinations remain present; only combination
counts are changed.  The contaminated 0716 test file is never opened.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path("E:/LJQ_code/CJ_Imbalance_0716")
DATA = Path("E:/LJQ_code/干扰数据")
DEV_ROOT = DATA / "复合干扰数据0731_v3"
CONF_ROOT = DATA / "复合干扰数据0731_v2"
DEV_TRAIN = DEV_ROOT / "closedset_6class_train.mat"
CONF_TRAIN = CONF_ROOT / "closedset_6class_train.mat"
OUT = ROOT / "outputs" / "route_a_five_class_frequency_lt_v3"
AUDIT = DATA / "route_a_five_class_frequency_lt_v3"
CLASS_NAMES = ("FMB", "SinFM", "MFT", "VFT", "ISRJ")
SOURCE_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGET = np.asarray([4000, 3000, 2000, 1500, 1000], dtype=np.int64)
SEEDS = (273001, 273002, 273003, 273004, 273005)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")


def read_labels(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        y = np.asarray(h["label_selected"][:], dtype=np.uint8)
    if y.ndim != 2:
        raise ValueError(f"unexpected labels shape {y.shape}")
    if y.shape[0] == 6:
        y = y.T
    elif y.shape[1] != 6:
        raise ValueError(f"unexpected labels shape {y.shape}")
    return y


def filter_five(y6: np.ndarray):
    keep = np.flatnonzero(y6[:, 1] == 0).astype(np.int64)
    y5 = y6[:, [0, 2, 3, 4, 5]].astype(np.uint8, copy=False)
    if not np.all(y5[keep].sum(1) > 0):
        raise ValueError("SawFM filtering produced an empty five-class row")
    return keep, y5


def key(row: np.ndarray) -> str:
    return "".join(str(int(v)) for v in row.tolist())


def combo_groups(y5: np.ndarray, indices: np.ndarray):
    groups = {}
    for i in indices.tolist():
        groups.setdefault(key(y5[int(i)]), []).append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in sorted(groups.items())}


def solve_combo_counts(y5: np.ndarray, indices: np.ndarray, target: np.ndarray, seed: int):
    groups = combo_groups(y5, indices)
    keys = sorted(groups)
    if len(keys) != 31:
        raise ValueError(f"source has {len(keys)} combinations, expected 31")
    A = np.asarray([[int(k[j]) for k in keys] for j in range(5)], dtype=float)
    # Keep every combination represented while meeting exact marginal counts.
    rng = np.random.RandomState(seed)
    c = rng.uniform(-1e-7, 1e-7, size=len(keys))
    result = milp(c=c, integrality=np.ones(len(keys)), bounds=Bounds(np.ones(len(keys)), np.asarray([len(groups[k]) for k in keys], dtype=float)), constraints=LinearConstraint(A, target, target), options={"time_limit": 60})
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")
    counts = np.rint(result.x).astype(np.int64)
    if np.any(counts < 1) or not np.array_equal(A.astype(np.int64) @ counts, target):
        raise RuntimeError("invalid solved combination counts")
    selected = []
    for k, n in zip(keys, counts.tolist()):
        pool = groups[k].copy()
        rng.shuffle(pool)
        selected.extend(pool[:n].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected, {k: int(n) for k, n in zip(keys, counts.tolist())}


def make_source_manifest(path: Path, label_name: str, seed: int):
    y6 = read_labels(path)
    keep, y5 = filter_five(y6)
    selected, combo_counts = solve_combo_counts(y5, keep, TARGET, seed)
    payload = {
        "protocol_version": "route_a_five_class_frequency_lt_v3",
        "role": label_name,
        "source_file": str(path),
        "source_sha256": sha256(path),
        "class_names": list(CLASS_NAMES),
        "source_label_order": list(SOURCE_NAMES),
        "excluded_class": "SawFM",
        "excluded_positive_samples": True,
        "power_longtail": False,
        "target_label_counts": TARGET.tolist(),
        "achieved_label_counts": y5[selected].sum(0).astype(int).tolist(),
        "all_source_five_count": int(len(keep)),
        "selected_count": int(len(selected)),
        "combination_count": int(len(combo_counts)),
        "combination_counts": combo_counts,
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
    }
    return y5, keep, selected, payload


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dev_y5, dev_keep, dev_selected, dev_manifest = make_source_manifest(DEV_TRAIN, "development", 273000)
    conf_y5, conf_keep, conf_selected, conf_manifest = make_source_manifest(CONF_TRAIN, "confirmation_holdout", 274000)
    dump(AUDIT / "development_manifest.json", dev_manifest)
    dump(AUDIT / "confirmation_manifest.json", conf_manifest)
    dump(AUDIT / "protocol_audit.json", {
        "protocol_version": "route_a_five_class_frequency_lt_v3",
        "status": "ready_for_development_training",
        "development_source": str(DEV_TRAIN),
        "confirmation_source": str(CONF_TRAIN),
        "development_source_sha256": sha256(DEV_TRAIN),
        "confirmation_source_sha256": sha256(CONF_TRAIN),
        "test_files_opened": [],
        "test_reused": False,
        "development_combinations": len(dev_manifest["combination_counts"]),
        "confirmation_combinations": len(conf_manifest["combination_counts"]),
        "development_target_counts": dev_manifest["achieved_label_counts"],
        "confirmation_target_counts": conf_manifest["achieved_label_counts"],
        "identity_isolation": "separate 0731_v3 and 0731_v2 train-only MAT sources; no numeric-index matching",
    })
    np.save(AUDIT / "development_five_indices.npy", dev_keep)
    np.save(AUDIT / "development_longtail_indices.npy", dev_selected)
    np.save(AUDIT / "confirmation_five_indices.npy", conf_keep)
    np.save(AUDIT / "confirmation_longtail_indices.npy", conf_selected)
    # Reuse the audited V2 trainer only for development roles.  Its final_test
    # function is deliberately not called, so no test file is opened.
    spec = importlib.util.spec_from_file_location("route_a_v2_impl", ROOT / "experiments" / "run_route_a_five_class_frequency_lt_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(v2)
    v2.TRAIN_PATH = DEV_TRAIN
    v2.TEST_PATH = CONF_TRAIN  # defensive only; final_test is never invoked
    v2.OUT_ROOT = OUT
    v2.AUDIT_ROOT = AUDIT
    data = {"train5": dev_y5, "test5": conf_y5, "train_valid": dev_keep, "test_valid": conf_keep, "manifest_indices": dev_selected, "audit": {"train_retained_count": int(len(dev_keep)), "test_retained_count": int(len(conf_keep)), "train_counts_after_filter": dev_y5[dev_keep].sum(0).astype(int).tolist(), "label_if4_achieved_counts": dev_y5[dev_selected].sum(0).astype(int).tolist(), "balanced_combo_counts": {k:int(len(v)) for k,v in combo_groups(dev_y5, dev_keep).items()}, "label_if4_combo_counts": dev_manifest["combination_counts"]}}
    class Args:
        max_seeds = 5; epochs = 18; patience = 6; batch_size = 64; eval_batch_size = 256; draws_per_epoch = 9920
        head_epochs = 12; head_patience = 4; head_batch_size = 128; head_draws_per_epoch = 9920
    args = Args()
    # First gate is a five-seed B0/Focal Uniform baseline.  The sampler
    # comparison and all loss/head arms are launched only after this baseline
    # has a complete, auditable receipt; this keeps the new protocol cheap to
    # interrupt and prevents unnecessary exploratory branches.
    rows = []
    for seed in v2.SEEDS[:5]:
        roles = v2.split_roles(data["train_valid"], data["train5"], int(seed))
        dump(OUT / "splits" / f"seed_{seed}.json", {k: v.tolist() for k, v in roles.items()})
        print(f"START development baseline seed={seed}", flush=True)
        rows.append(v2.train_joint("Uniform-Focal", int(seed), roles, data, args, "uniform", "focal", {}, "stage1"))
        print(f"DONE development baseline seed={seed}", flush=True)
    vals = [r["early"]["macro_aupr"] for r in rows]
    s1 = {"status": "completed", "stage": "stage1", "rows": rows, "early_macro_aupr_mean": {"Uniform-Focal": float(np.mean(vals))}, "test_used": False, "stage1_gate_pass": False}
    dump(OUT / "stage1_summary.json", s1)
    dump(OUT / "development_run_receipt.json", {"status": "completed_development_baseline_only", "stage1": s1, "test_evaluated": False, "confirmation_evaluated": False, "protocol_audit": str(AUDIT / "protocol_audit.json")})
    print(json.dumps({"status": "completed_development_baseline_only", "stage1_gate": s1.get("stage1_gate_pass"), "baseline_early_macro_aupr": s1.get("early_macro_aupr_mean", {}).get("Uniform-Focal"), "test_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
