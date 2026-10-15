"""Formal R1 data audit and identity-role preparation for Route A."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import build_combo_longtail, build_label_longtail
from cjlt.data.split_builder import combo_key
from cjlt.data.v73_dataset import inspect_v73_layout, read_labels, read_parameters


DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1")
LOW_PATH = DATA_ROOT / "sawfm_level_03__jnr_20dB/closedset_6class_train.mat"
HIGH_PATH = DATA_ROOT / "sawfm_level_05__jnr_30dB/closedset_6class_train.mat"
SWEEP_MANIFEST = DATA_ROOT / "power_sweep_manifest.csv"
FORMAL_RECEIPT = ROOT / "outputs/route_a_dual_axis_next/r1_data_generation/formal_generation_receipt.json"
PREFLIGHT_AUDIT = ROOT / "outputs/route_a_dual_axis_next/r1_data_generation/preflight_identity_audit.json"
OUTPUT_DIR = ROOT / "outputs/route_a_dual_axis_next/r1_data_audit"

CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
ROLE_SEED = 26080702
ROLE_COUNTS_PER_COMBO = {
    "optimization": 240,
    "early": 40,
    "calibration": 40,
    "threshold": 40,
    "evaluation": 40,
}
FB_TARGET_LABEL_COUNTS = np.asarray([1200] * 6, dtype=np.int64)
LT_TARGET_LABEL_COUNTS = np.asarray([1600, 400, 1213, 919, 696, 528], dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indices_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_cell_vectors(path: Path, keys: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for key in keys:
            refs = handle[key][0, :]
            result[key] = [np.asarray(handle[ref]).reshape(-1) for ref in refs]
    return result


def build_roles(labels: np.ndarray, seed: int = ROLE_SEED) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(np.asarray(labels, dtype=np.uint8)):
        groups.setdefault(combo_key(row), []).append(index)
    rng = np.random.RandomState(int(seed))
    roles = {name: [] for name in ROLE_COUNTS_PER_COMBO}
    expected = sum(ROLE_COUNTS_PER_COMBO.values())
    for name in sorted(groups):
        values = np.asarray(groups[name], dtype=np.int64)
        if len(values) != expected:
            raise ValueError(f"combination {name} has {len(values)} identities, expected {expected}")
        rng.shuffle(values)
        start = 0
        for role, count in ROLE_COUNTS_PER_COMBO.items():
            roles[role].extend(values[start : start + count].tolist())
            start += count
    return {
        name: np.asarray(values, dtype=np.int64)
        for name, values in roles.items()
    }


def validate_roles(roles: dict[str, np.ndarray], sample_count: int) -> dict[str, Any]:
    names = list(roles)
    intersections = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            intersections[f"{left}__{right}"] = int(
                len(np.intersect1d(roles[left], roles[right]))
            )
    merged = np.concatenate([roles[name] for name in names])
    return {
        "counts": {name: int(len(roles[name])) for name in names},
        "sha256": {name: indices_sha256(roles[name]) for name in names},
        "pairwise_intersection_counts": intersections,
        "union_count": int(len(np.unique(merged))),
        "covers_all_identities": int(len(np.unique(merged))) == int(sample_count),
        "all_pairwise_disjoint": all(value == 0 for value in intersections.values()),
    }


def power_contract(labels: np.ndarray) -> dict[str, Any]:
    keys = (
        "component_target_power_dB",
        "relative_power_dB",
        "weak_component_mask",
        "component_seeds",
    )
    low = read_cell_vectors(LOW_PATH, keys)
    high = read_cell_vectors(HIGH_PATH, keys)
    target_ok = True
    relative_ok = True
    weak_ok = True
    component_seed_ok = True
    low_weak_saw_count = 0
    high_weak_count = 0
    for index, label in enumerate(labels):
        active = np.flatnonzero(label)
        expected_low_target = np.asarray(
            [20.0 if value == 1 else 30.0 for value in active], dtype=np.float64
        )
        expected_high_target = np.full(len(active), 30.0, dtype=np.float64)
        expected_low_relative = np.zeros(len(active), dtype=np.float64)
        expected_low_weak = np.zeros(len(active), dtype=bool)
        if len(active) > 1 and 1 in active:
            saw_position = int(np.flatnonzero(active == 1)[0])
            expected_low_relative[saw_position] = -10.0
            expected_low_weak[saw_position] = True
        expected_high_weak = np.zeros(len(active), dtype=bool)

        target_ok &= np.array_equal(low[keys[0]][index], expected_low_target)
        target_ok &= np.array_equal(high[keys[0]][index], expected_high_target)
        relative_ok &= np.array_equal(low[keys[1]][index], expected_low_relative)
        relative_ok &= np.array_equal(high[keys[1]][index], np.zeros(len(active)))
        weak_ok &= np.array_equal(low[keys[2]][index].astype(bool), expected_low_weak)
        weak_ok &= np.array_equal(high[keys[2]][index].astype(bool), expected_high_weak)
        component_seed_ok &= np.array_equal(low[keys[3]][index], high[keys[3]][index])
        low_weak_saw_count += int(expected_low_weak.sum())
        high_weak_count += int(expected_high_weak.sum())
    return {
        "only_sawfm_target_power_changes_as_preregistered": bool(target_ok),
        "relative_power_truth_matches_0_and_minus10db": bool(relative_ok),
        "weak_mask_matches_sawfm_compound_only": bool(weak_ok),
        "component_seeds_equal_across_views": bool(component_seed_ok),
        "low_view_weak_sawfm_identity_count": int(low_weak_saw_count),
        "high_view_weak_component_count": int(high_weak_count),
    }


def signal_probe(path: Path, indices: tuple[int, ...]) -> dict[str, Any]:
    rows = []
    with h5py.File(path, "r") as handle:
        for index in indices:
            signal = np.asarray(handle["data_legacy"][:, index], dtype=np.float64)
            power = float(np.mean(signal[:3000] ** 2 + signal[3000:] ** 2))
            rows.append({
                "index": int(index),
                "all_finite": bool(np.isfinite(signal).all()),
                "rms": float(np.sqrt(max(power, 0.0))),
                "nonzero": bool(np.any(signal != 0.0)),
            })
    return {
        "rows": rows,
        "passed": all(row["all_finite"] and row["nonzero"] for row in rows),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_audit() -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    formal = json.loads(FORMAL_RECEIPT.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_AUDIT.read_text(encoding="utf-8"))
    labels_low = read_labels(LOW_PATH)
    labels_high = read_labels(HIGH_PATH)
    with h5py.File(LOW_PATH, "r") as low_handle, h5py.File(HIGH_PATH, "r") as high_handle:
        receiver_low = np.asarray(low_handle["receiver_noise_seed"][0, :], dtype=np.int64)
        receiver_high = np.asarray(high_handle["receiver_noise_seed"][0, :], dtype=np.int64)
        target_fields_equal = True
        for key in ("target_delay_s", "target_range_m", "target_present", "unknown_present"):
            left = low_handle[key][:]
            right = high_handle[key][:]
            if np.issubdtype(left.dtype, np.floating):
                target_fields_equal &= np.array_equal(left, right, equal_nan=True)
            else:
                target_fields_equal &= np.array_equal(left, right)

    low_values, low_mask = read_parameters(LOW_PATH)
    high_values, high_mask = read_parameters(HIGH_PATH)
    roles = build_roles(labels_low)
    role_audit = validate_roles(roles, len(labels_low))
    optimization_sha = role_audit["sha256"]["optimization"]
    fb = build_label_longtail(
        labels_low,
        roles["optimization"],
        FB_TARGET_LABEL_COUNTS,
        seed=26080703,
        tolerance=0.0,
        source_split_sha256=optimization_sha,
        target_sample_count=2400,
        sample_count_tolerance=0.0,
    )
    lt = build_label_longtail(
        labels_low,
        roles["optimization"],
        LT_TARGET_LABEL_COUNTS,
        seed=26080704,
        tolerance=0.0,
        source_split_sha256=optimization_sha,
        target_sample_count=2400,
        sample_count_tolerance=0.0,
    )
    combo = build_combo_longtail(
        labels_low,
        roles["optimization"],
        imbalance_factor=4.0,
        seed=26080705,
        source_split_sha256=optimization_sha,
        min_samples=3,
    )
    power = power_contract(labels_low)
    label_counts = labels_low.sum(axis=0).astype(int)

    audit = {
        "status": "completed",
        "experiment": "RouteA_DualAxis_SawFM_DevelopmentV1_DataAudit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "new_identity_development_only",
        "files": {
            "low_visibility": {
                "path": str(LOW_PATH), "bytes": LOW_PATH.stat().st_size,
                "sha256": sha256(LOW_PATH), "layout": inspect_v73_layout(LOW_PATH),
            },
            "high_visibility": {
                "path": str(HIGH_PATH), "bytes": HIGH_PATH.stat().st_size,
                "sha256": sha256(HIGH_PATH), "layout": inspect_v73_layout(HIGH_PATH),
            },
            "power_sweep_manifest": {
                "path": str(SWEEP_MANIFEST), "sha256": sha256(SWEEP_MANIFEST),
            },
            "formal_generation_receipt": {
                "path": str(FORMAL_RECEIPT), "sha256": sha256(FORMAL_RECEIPT),
            },
        },
        "identity_and_label_contract": {
            "sample_count": int(len(labels_low)),
            "class_names": list(CLASS_NAMES),
            "label_counts": dict(zip(CLASS_NAMES, label_counts.tolist())),
            "labels_equal_across_views": bool(np.array_equal(labels_low, labels_high)),
            "labels_sha256": array_sha256(labels_low, "u1"),
            "receiver_noise_seed_equal_across_views": bool(np.array_equal(receiver_low, receiver_high)),
            "receiver_noise_seed_unique": int(len(np.unique(receiver_low))) == len(receiver_low),
            "receiver_noise_seed_sha256": array_sha256(receiver_low, "<i8"),
            "receiver_noise_seed_first": int(receiver_low[0]),
            "receiver_noise_seed_last": int(receiver_low[-1]),
            "target_and_unknown_fields_equal_across_views": bool(target_fields_equal),
            "known_historical_seed_collision_count": len(preflight["identity_collisions"]),
        },
        "power_contract": power,
        "schema_and_finite_checks": {
            "parameter_masks_equal_across_views": bool(np.array_equal(low_mask, high_mask)),
            "low_active_parameters_finite": bool(np.isfinite(low_values[low_mask]).all()),
            "high_active_parameters_finite": bool(np.isfinite(high_values[high_mask]).all()),
            "low_signal_probe": signal_probe(LOW_PATH, (0, 400, 2400, 12599, 25199)),
            "high_signal_probe": signal_probe(HIGH_PATH, (0, 400, 2400, 12599, 25199)),
        },
        "roles": role_audit,
        "optimization_manifests": {
            "frequency_balanced": fb.to_dict(),
            "label_long_tail_primary": lt.to_dict(),
            "combination_long_tail_secondary": combo.to_dict(),
        },
        "generation_receipt_status": formal["status"],
        "generation_paired_seed_verified": formal["paired_seed_verified"],
        "checks": {},
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    checks = {
        "formal_generation_completed": formal["status"] == "completed" and formal["executed"] is True,
        "two_views_have_25200_samples": len(labels_low) == len(labels_high) == 25200,
        "identity_pairing_exact": (
            audit["identity_and_label_contract"]["labels_equal_across_views"]
            and audit["identity_and_label_contract"]["receiver_noise_seed_equal_across_views"]
            and audit["identity_and_label_contract"]["receiver_noise_seed_unique"]
            and audit["identity_and_label_contract"]["target_and_unknown_fields_equal_across_views"]
        ),
        "historical_identity_collision_free": len(preflight["identity_collisions"]) == 0,
        "sawfm_visibility_intervention_exact": all(
            power[name]
            for name in (
                "only_sawfm_target_power_changes_as_preregistered",
                "relative_power_truth_matches_0_and_minus10db",
                "weak_mask_matches_sawfm_compound_only",
                "component_seeds_equal_across_views",
            )
        ),
        "roles_complete_disjoint_and_paired": (
            role_audit["covers_all_identities"] and role_audit["all_pairwise_disjoint"]
        ),
        "fb_manifest_exact_2400": (
            fb.achieved_sample_count == 2400
            and np.array_equal(fb.achieved_label_counts, FB_TARGET_LABEL_COUNTS)
        ),
        "label_lt_manifest_exact_2400": (
            lt.achieved_sample_count == 2400
            and np.array_equal(lt.achieved_label_counts, LT_TARGET_LABEL_COUNTS)
        ),
        "schema_and_finite_checks_pass": (
            audit["schema_and_finite_checks"]["low_active_parameters_finite"]
            and audit["schema_and_finite_checks"]["high_active_parameters_finite"]
            and audit["schema_and_finite_checks"]["low_signal_probe"]["passed"]
            and audit["schema_and_finite_checks"]["high_signal_probe"]["passed"]
        ),
        "r2_smoke_admitted": False,
    }
    checks["r2_smoke_admitted"] = all(value for key, value in checks.items() if key != "r2_smoke_admitted")
    audit["checks"] = checks
    combo_payload = combo.to_dict()
    combo_payload["achieved_sample_count"] = int(len(combo.selected_indices))
    combo_payload["selection_note"] = (
        "Secondary IF=4 robustness manifest; formal comparisons must match the "
        "primary protocol's optimizer update count."
    )
    audit["optimization_manifests"]["combination_long_tail_secondary"] = combo_payload
    artifacts = {
        "fb": fb.to_dict(),
        "lt": lt.to_dict(),
        "combo": combo_payload,
    }
    return audit, roles, artifacts


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    audit, roles, manifests = build_audit()
    audit_path = OUTPUT_DIR / "data_audit.json"
    write_json(audit_path, audit)
    np.savez_compressed(OUTPUT_DIR / "role_indices.npz", **roles)
    write_json(OUTPUT_DIR / "role_manifest.json", audit["roles"])
    write_json(OUTPUT_DIR / "frequency_balanced_manifest.json", manifests["fb"])
    write_json(OUTPUT_DIR / "label_long_tail_manifest.json", manifests["lt"])
    write_json(OUTPUT_DIR / "combination_long_tail_manifest.json", manifests["combo"])
    receipt = {
        "status": "completed" if audit["checks"]["r2_smoke_admitted"] else "stopped",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "admit_r2_smoke" if audit["checks"]["r2_smoke_admitted"] else "stop_before_r2",
        "data_audit": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "data_audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__)),
        "new_data_generated": True,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_DIR / "run_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "decision": receipt["decision"],
        "checks": audit["checks"],
        "role_counts": audit["roles"]["counts"],
        "fb_counts": manifests["fb"]["achieved_label_counts"],
        "lt_counts": manifests["lt"]["achieved_label_counts"],
        "combo_lt_sample_count": manifests["combo"]["achieved_sample_count"],
    }, indent=2, ensure_ascii=False))
    if not audit["checks"]["r2_smoke_admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
