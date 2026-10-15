"""Build audited fixed-budget development-v2 manifests for BCE boundary screening.

This stage creates label-only and combination-only long-tail conditions.  It
does not train a model and never opens test/locked-test files.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import build_label_longtail, profile_multi_axis_imbalance
from cjlt.data.split_builder import combo_key
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, pure_six_indices


def discover_data_root() -> Path:
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


def sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def build_combo_fixed(labels: np.ndarray, indices: np.ndarray, cf: float, seed: int, target_n: int) -> tuple[np.ndarray, dict]:
    """Select exactly target_n samples with an audited approximate CF."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups[combo_key(labels[idx])].append(int(idx))
    names = sorted(groups)
    rng = np.random.RandomState(int(seed))
    order = names.copy()
    rng.shuffle(order)
    # Assign a geometric profile and adjust to exactly target_n.
    max_available = min(len(v) for v in groups.values())
    best = None
    for maximum in range(2, max_available + 1):
        raw = np.rint(maximum * float(cf) ** (-np.arange(len(names)) / max(len(names) - 1, 1))).astype(int)
        raw = np.maximum(raw, 1)
        delta = abs(int(raw.sum()) - int(target_n))
        if best is None or delta < best[0]:
            best = (delta, raw)
    counts_order = best[1].copy()
    # Correct the total while respecting [1, available] bounds.
    while counts_order.sum() < target_n:
        candidates = [i for i, name in enumerate(order) if counts_order[i] < len(groups[name])]
        if not candidates:
            raise ValueError("combo target sample count exceeds available data")
        i = candidates[int(rng.randint(len(candidates)))]
        counts_order[i] += 1
    while counts_order.sum() > target_n:
        candidates = [i for i in range(len(order)) if counts_order[i] > 1]
        if not candidates:
            raise ValueError("combo target sample count cannot be reduced")
        i = candidates[int(rng.randint(len(candidates)))]
        counts_order[i] -= 1
    selected: list[int] = []
    selected_counts: dict[str, int] = {}
    for i, name in enumerate(order):
        candidates = np.asarray(groups[name], dtype=np.int64).copy()
        rng.shuffle(candidates)
        chosen = candidates[: int(counts_order[i])]
        selected.extend(chosen.tolist())
        selected_counts[name] = int(len(chosen))
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    values = np.asarray(list(selected_counts.values()), dtype=float)
    meta = {
        "protocol": "combo_lt_fixed_budget",
        "requested_cf": float(cf),
        "achieved_cf": float(values.max() / values.min()),
        "combo_counts": {k: int(selected_counts[k]) for k in sorted(selected_counts)},
        "selected_count": int(len(selected_array)),
    }
    return selected_array, meta


def main() -> None:
    out = PROJECT_ROOT / "outputs" / "route_a_bce_boundary_v1_20260901"
    out.mkdir(parents=True, exist_ok=True)
    root = discover_data_root()
    source = next(p for p in root.rglob("closedset_9class_train.mat") if "level_05" in str(p))
    pure = pure_six_indices(source)
    perm = np.random.default_rng(260828).permutation(pure)
    fit = np.sort(perm[:4096])
    val = np.sort(perm[4096:6144])
    with h5py.File(str(source), "r") as h:
        labels_pure = np.asarray(h["known_label"][:, pure], dtype=np.uint8).T
    labels = labels_pure[:, list(SELECTED_INDICES)]
    fit_local = np.searchsorted(pure, fit)
    val_local = np.searchsorted(pure, val)
    labels_fit = labels[fit_local]
    labels_val = labels[val_local]
    role = {
        "schema": "route-a-bce-boundary-role-v1",
        "selected_classes": list(CLASS_NAMES),
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "split_seed": 260828,
        "fit_count": int(len(fit)),
        "val_count": int(len(val)),
        "fit_indices_sha256": sha256_indices(fit),
        "val_indices_sha256": sha256_indices(val),
        "test": {"used": False, "locked_test_used": False},
        "frozen_e5_modified": False,
    }
    (out / "role_manifest.json").write_text(json.dumps(role, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    # Three-seed screen plus two fresh confirmation seeds.  Existing manifests
    # are deterministic and will be regenerated byte-for-byte for the original
    # seeds.
    seeds = (2039, 2040, 2041, 2042, 2043)
    # Label-only: fixed N=1600, exact targets are feasible through IF=3.
    targets = {
        "label_IF1": [400, 400, 400, 400, 400, 400],
        "label_IF2": [800, 696, 606, 528, 459, 400],
        "label_IF3": [1200, 963, 773, 621, 498, 400],
    }
    for seed in seeds:
        for name, target in targets.items():
            if name == "label_IF1":
                # The exact equal-label MILP is unnecessarily hard at N=1600;
                # use a deterministic random balanced anchor and audit its
                # achieved counts.  This is a balanced baseline, not a claim
                # of exact target counts.
                rng = np.random.RandomState(int(seed))
                local = np.arange(len(labels_fit), dtype=np.int64)
                rng.shuffle(local)
                selected_local = local[:1600]
                selected = fit[selected_local]
            else:
                m = build_label_longtail(
                    labels_fit, np.arange(len(labels_fit)), target, int(seed), tolerance=0.05,
                    source_split_sha256=sha256_indices(fit), target_sample_count=1600,
                    sample_count_tolerance=0.0,
                )
                selected = fit[m.selected_indices]
            counts = labels[np.searchsorted(pure, selected)].sum(axis=0).astype(int)
            payload = {
                "schema": "route-a-bce-boundary-manifest-v1",
                "condition": name,
                "axis": "label_only",
                "seed": int(seed),
                "requested_if": float(max(target) / min(target)),
                "target_label_counts": list(map(int, target)),
                "achieved_label_counts": counts.tolist(),
                "effective_if": float(counts.max() / counts.min()),
                "selected_indices": selected.astype(int).tolist(),
                "selected_count": int(len(selected)),
                "selected_indices_sha256": sha256_indices(selected),
                "source_fit_indices_sha256": sha256_indices(fit),
                "combo_counts": Counter(combo_key(row) for row in labels[np.searchsorted(pure, selected)]),
                "locked_test_used": False,
            }
            payload["combo_counts"] = {str(k): int(v) for k, v in sorted(payload["combo_counts"].items())}
            path = out / f"{name}__seed_{seed}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(payload)
    # Combination-only: fixed N=1600, approximate requested CF is audited.
    for seed in seeds:
        for cf in (1, 2, 4, 8):
            selected_local, combo_meta = build_combo_fixed(labels_fit, np.arange(len(labels_fit)), cf, int(seed), 1600)
            selected = fit[selected_local]
            y = labels[np.searchsorted(pure, selected)]
            counts = y.sum(axis=0).astype(int)
            name = f"combo_CF{cf}"
            payload = {
                "schema": "route-a-bce-boundary-manifest-v1",
                "condition": name,
                "axis": "combo_only",
                "seed": int(seed),
                "requested_cf": float(cf),
                "achieved_cf": combo_meta["achieved_cf"],
                "selected_indices": selected.astype(int).tolist(),
                "selected_count": int(len(selected)),
                "selected_indices_sha256": sha256_indices(selected),
                "label_counts": counts.tolist(),
                "effective_if": float(counts.max() / counts.min()),
                "combo_counts": combo_meta["combo_counts"],
                "locked_test_used": False,
            }
            path = out / f"{name}__seed_{seed}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(payload)
    summary = {
        "schema": "route-a-bce-boundary-manifest-summary-v1",
        "status": "completed",
        "locked_test_used": False,
        "frozen_e5_modified": False,
        "conditions": {
            "label_only": sorted({r["condition"] for r in records if r["axis"] == "label_only"}),
            "combo_only": sorted({r["condition"] for r in records if r["axis"] == "combo_only"}),
        },
        "records": [
            {
                "condition": r["condition"], "axis": r["axis"], "seed": r["seed"],
                "selected_count": r["selected_count"],
                "effective_if": r.get("effective_if"),
                "requested_if": r.get("requested_if"),
                "requested_cf": r.get("requested_cf"),
                "achieved_cf": r.get("achieved_cf"),
            } for r in records
        ],
        "next_step": "Train the common BCE anchor on these audited manifests before comparing long-tail losses.",
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "terminal_receipt.json").write_text(json.dumps({"status":"completed","manifest_summary":str((out/"manifest_summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"n":len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
