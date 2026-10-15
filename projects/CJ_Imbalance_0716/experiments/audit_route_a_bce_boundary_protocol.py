"""Audit the effective severity of the Route-A label/combo long-tail protocol.

Read-only with respect to source data and frozen checkpoints.  The audit
reproduces the P3 fit split, reports effective IF/CF, and probes feasibility of
fixed-budget label targets using the existing MILP manifest builder.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.split_builder import combo_key
from experiments.run_route_a_c1_baseline import (
    CLASS_NAMES,
    SELECTED_INDICES,
    pure_six_indices,
)


def discover_data_root() -> Path:
    # Build the Unicode path without relying on console code-page handling.
    parts = [
        "".join(chr(c) for c in [0x8bba, 0x6587, 0x76f8, 0x5173]),
        "".join(chr(c) for c in [0x5e72, 0x6270, 0x6570, 0x636e]),
        "".join(chr(c) for c in [0x590d, 0x5408, 0x5e72, 0x6270, 0x6570, 0x636e]) + "0716",
        "".join(chr(c) for c in [0x4e5d, 0x7c7b]),
    ]
    root = Path("E:/").joinpath(*parts)
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile(labels: np.ndarray, indices: np.ndarray) -> dict:
    y = labels[indices]
    label_counts = y.sum(axis=0).astype(int)
    combos = Counter(combo_key(row) for row in y)
    combo_values = np.asarray(list(combos.values()), dtype=float)
    return {
        "sample_count": int(len(indices)),
        "label_counts": label_counts.tolist(),
        "effective_if": float(label_counts.max() / max(label_counts.min(), 1)),
        "unique_combinations": int(len(combos)),
        "combo_min": int(combo_values.min()) if len(combo_values) else 0,
        "combo_max": int(combo_values.max()) if len(combo_values) else 0,
        "effective_cf": float(combo_values.max() / max(combo_values.min(), 1)) if len(combo_values) else 0.0,
    }


def main() -> None:
    output = PROJECT_ROOT / "outputs" / "route_a_bce_boundary_protocol_audit_20260901"
    output.mkdir(parents=True, exist_ok=True)
    data_root = discover_data_root()
    train_30 = next(
        p for p in data_root.rglob("closedset_9class_train.mat") if "level_05" in str(p)
    )
    pure = pure_six_indices(train_30)
    perm = np.random.default_rng(260828).permutation(pure)
    fit = np.sort(perm[:4096])
    val = np.sort(perm[4096:6144])
    with h5py.File(str(train_30), "r") as h:
        labels = np.asarray(h["known_label"][:, pure], dtype=np.uint8).T[:, list(SELECTED_INDICES)]
    fit_local = np.searchsorted(pure, fit)
    val_local = np.searchsorted(pure, val)

    existing = []
    p3_root = PROJECT_ROOT / "outputs" / "route_a_p3_dual_axis_longtail_20260828"
    for path in sorted(p3_root.glob("longtail_manifest_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = np.asarray(payload.get("achieved_label_counts") or payload.get("label_counts"), dtype=float)
        combo_counts = np.asarray(list(payload["combo_counts"].values()), dtype=float)
        existing.append({
            "manifest": str(path.resolve()),
            "selected_count": int(payload.get("selected_count", payload.get("achieved_sample_count"))),
            "label_counts": counts.astype(int).tolist(),
            "effective_if": float(counts.max() / counts.min()),
            "unique_combinations": int(len(combo_counts)),
            "combo_min": int(combo_counts.min()),
            "combo_max": int(combo_counts.max()),
            "effective_cf": float(combo_counts.max() / combo_counts.min()),
        })

    feasibility = []
    # Small fixed-budget probes: these are feasibility checks only, not model runs.
    for target_sample_count in (800, 1200, 1600):
        tail = 200 if target_sample_count == 800 else 400
        for requested_if in (1, 2, 3, 4, 5):
            target = np.rint((tail * requested_if) * requested_if ** (-np.arange(6) / 5.0)).astype(int)
            row = {
                "target_sample_count": target_sample_count,
                "requested_if": requested_if,
                "target_label_counts": target.tolist(),
                "target_sum": int(target.sum()),
            }
            try:
                manifest = build_label_longtail(
                    labels[fit_local], np.arange(len(fit_local)), target, 2039,
                    tolerance=0.05, source_split_sha256="audit-only",
                    target_sample_count=target_sample_count, sample_count_tolerance=0.0,
                )
                achieved = manifest.achieved_label_counts.astype(int)
                row.update({
                    "feasible": True,
                    "achieved_sample_count": int(len(manifest.selected_indices)),
                    "achieved_label_counts": achieved.tolist(),
                    "effective_if": float(achieved.max() / achieved.min()),
                })
            except Exception as exc:
                row.update({"feasible": False, "error": f"{type(exc).__name__}: {exc}"})
            feasibility.append(row)

    report = {
        "schema": "route-a-bce-boundary-protocol-audit-v1",
        "status": "completed",
        "locked_test_used": False,
        "test_files_opened": False,
        "frozen_e5_modified": False,
        "data_root": str(data_root.resolve()),
        "train_30_path": str(train_30.resolve()),
        "train_30_sha256": sha256_file(train_30),
        "selected_classes": list(CLASS_NAMES),
        "split": {
            "split_seed": 260828,
            "fit_count": int(len(fit)),
            "val_count": int(len(val)),
            "fit_indices_sha256": hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes()).hexdigest(),
            "val_indices_sha256": hashlib.sha256(np.asarray(val, dtype="<i8").tobytes()).hexdigest(),
        },
        "source_profile": profile(labels, np.arange(len(labels))),
        "fit_profile": profile(labels, fit_local),
        "val_profile": profile(labels, val_local),
        "existing_p3_manifests": existing,
        "fixed_budget_label_target_feasibility": feasibility,
        "interpretation": {
            "effective_if_warning": "Existing P3 manifests must be judged by achieved counts, not nominal IF=5.",
            "next_gate": "Construct development-v2 manifests with fixed sample budget and audited effective IF/CF before any loss comparison.",
        },
    }
    out_json = output / "audit.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {
        "status": "completed",
        "audit": str(out_json.resolve()),
        "locked_test_used": False,
        "test_files_opened": False,
        "frozen_e5_modified": False,
    }
    (output / "terminal_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "audit": str(out_json.resolve()), "existing": existing}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
